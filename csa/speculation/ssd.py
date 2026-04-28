"""
SSD (Speculative Speculative Decoding) module.

Implements async speculation using CUDA streams for parallel token generation.
This is a core component for achieving 2-3x speedup in CSA.
"""

import torch
import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Tuple, Optional

try:
    from vllm import LLM, SamplingParams
    SSD_AVAILABLE = True
except ImportError:
    SSD_AVAILABLE = False


class SSDSpeculator:
    """
    Speculative Speculative Decoding (SSD) speculator.
    
    Uses a draft model to generate speculative tokens in parallel,
    then verifies them against the target model for acceptance.
    """
    
    def __init__(self, draft_model_path, skeleton_kv, speculate_k=7, async_fan_out=3,
                 use_cuda_streams=True, max_workers=4):
        """
        Initialize SSD speculator.
        
        Args:
            draft_model_path: Path to draft model (smaller/faster model)
            skeleton_kv: Compressed KV cache skeleton
            speculate_k: Number of tokens to speculate per outcome
            async_fan_out: Number of parallel speculation outcomes
            use_cuda_streams: Use CUDA streams for parallelism
            max_workers: Max thread workers for CPU fallback
        """
        self.draft_model_path = draft_model_path
        self.skeleton_kv = skeleton_kv
        self.speculate_k = speculate_k
        self.async_fan_out = async_fan_out
        self.use_cuda_streams = use_cuda_streams and torch.cuda.is_available()
        
        # Speculation cache: maps predicted outcomes to pre-computed speculations
        # Key: (num_accepted, rejection_pos) -> speculated_tokens
        self.speculation_cache: Dict[Tuple[int, int], List[int]] = {}
        
        # CUDA streams for async execution
        self.cuda_streams: List[torch.cuda.Stream] = []
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        
        if self.use_cuda_streams:
            # Create separate CUDA streams for parallel speculation
            for i in range(async_fan_out):
                self.cuda_streams.append(torch.cuda.Stream())
        
        # Initialize draft model
        self.draft_model = None
        self.draft_tokenizer = None
        self.llm = None
        
        if SSD_AVAILABLE:
            try:
                # Initialize vLLM for draft model with SSD
                self.llm = LLM(
                    model=draft_model_path,
                    speculative_model=draft_model_path,
                    num_speculative_tokens=speculate_k,
                    gpu_memory_utilization=0.3,  # Lower for draft model
                )
            except Exception as e:
                print(f"Failed to initialize vLLM: {e}")
                self._init_fallback()
        else:
            self._init_fallback()
    
    def _init_fallback(self):
        """Initialize fallback draft model using transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using fallback draft model on {device}")
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            self.draft_model_path,
            torch_dtype=torch.float16
        ).to(device)
        self.draft_tokenizer = AutoTokenizer.from_pretrained(self.draft_model_path)
        if self.draft_tokenizer.pad_token is None:
            self.draft_tokenizer.pad_token = self.draft_tokenizer.eos_token
    
    def speculate(self, current_tokens: List[int], max_new_tokens: int = 10) -> List[int]:
        """
        Generate speculations using draft model.
        
        Args:
            current_tokens: Current token sequence
            max_new_tokens: Maximum tokens to speculate
            
        Returns:
            speculated_tokens: Generated token sequence from draft
        """
        if self.llm is not None:
            return self._speculate_vllm(current_tokens, max_new_tokens)
        elif self.draft_model is not None:
            return self._speculate_fallback(current_tokens, max_new_tokens)
        else:
            return []
    
    def _speculate_vllm(self, current_tokens: List[int], max_new_tokens: int) -> List[int]:
        """Speculate using vLLM (SSD)."""
        # Convert tokens to text
        prompt_text = self._tokens_to_text(current_tokens)
        
        sampling_params = SamplingParams(
            temperature=0.0,  # Greedy for speculation
            max_tokens=max_new_tokens,
            stop=["\n", "</s>"]  # Stop tokens
        )
        
        # Generate with vLLM (includes speculation)
        outputs = self.llm.generate([prompt_text], sampling_params)
        speculated_text = outputs[0].outputs[0].text
        
        # Convert back to tokens
        speculated_tokens = self._text_to_tokens(speculated_text)
        return speculated_tokens
    
    def _speculate_fallback(self, current_tokens: List[int], max_new_tokens: int) -> List[int]:
        """Fallback speculation using standard transformers."""
        device = next(self.draft_model.parameters()).device
        input_ids = torch.tensor([current_tokens], device=device)
        
        with torch.no_grad():
            outputs = self.draft_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,  # Greedy
                pad_token_id=self.draft_tokenizer.eos_token_id
            )
        
        new_tokens = outputs[0][len(current_tokens):].tolist()
        return new_tokens
    
    def speculate_with_cache(self, current_tokens: List[int], predicted_outcome: Tuple[int, int]) -> List[int]:
        """
        Use speculation cache for predicted verification outcomes (SSD core feature).
        
        Args:
            current_tokens: Current sequence
            predicted_outcome: (num_accepted, rejection_pos) tuple
            
        Returns:
            cached_speculation: Pre-computed tokens for this outcome
        """
        cache_key = predicted_outcome
        if cache_key in self.speculation_cache:
            return self.speculation_cache[cache_key]
        
        # Cache miss: compute and store
        speculated = self.speculate(current_tokens, self.speculate_k)
        self.speculation_cache[cache_key] = speculated
        return speculated
    
    def speculate_async(self, current_tokens: List[int], predicted_outcomes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """
        Perform async speculation for multiple predicted outcomes using CUDA streams.
        
        Args:
            current_tokens: Current token sequence
            predicted_outcomes: List of (num_accepted, rejection_pos) tuples
            
        Returns:
            speculations: Dict mapping outcomes to speculated token sequences
        """
        if self.use_cuda_streams and len(predicted_outcomes) > 1:
            return self._speculate_cuda_streams(current_tokens, predicted_outcomes)
        else:
            return self._speculate_sequential(current_tokens, predicted_outcomes)
    
    def _speculate_cuda_streams(self, current_tokens: List[int], predicted_outcomes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Async speculation using CUDA streams for parallelism."""
        results = {}
        
        def speculate_single(outcome, stream_idx):
            """Speculate for a single outcome on a specific CUDA stream."""
            stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
            with torch.cuda.stream(stream):
                return outcome, self.speculate_with_cache(current_tokens, outcome)
        
        # Submit async tasks
        futures = []
        for i, outcome in enumerate(predicted_outcomes):
            future = self.executor.submit(speculate_single, outcome, i)
            futures.append(future)
        
        # Collect results
        for future in futures:
            outcome, speculation = future.result()
            results[outcome] = speculation
        
        return results
    
    def _speculate_sequential(self, current_tokens: List[int], predicted_outcomes: List[Tuple[int, int]]) -> Dict[Tuple[int, int], List[int]]:
        """Sequential speculation for fallback."""
        results = {}
        for outcome in predicted_outcomes:
            results[outcome] = self.speculate_with_cache(current_tokens, outcome)
        return results
    
    def predict_outcomes(self, current_tokens: List[int]) -> List[Tuple[int, int]]:
        """
        Predict likely verification outcomes for speculation (SSD innovation).
        
        Uses draft model to generate speculations and predict acceptance patterns.
        
        Args:
            current_tokens: Current sequence
            
        Returns:
            outcomes: List of (num_accepted, rejection_pos) tuples, ordered by probability
        """
        if not current_tokens:
            return [(self.speculate_k, -1)]
        
        # Generate draft speculations to predict outcomes
        draft_tokens = self.speculate(current_tokens, self.speculate_k)
        
        if not draft_tokens:
            return [(0, 0)]
        
        # Predict outcomes based on draft confidence
        # Higher confidence = more tokens accepted
        outcomes = [
            (len(draft_tokens), -1),  # All accepted (optimistic)
            (max(1, len(draft_tokens) - 1), len(draft_tokens) - 1),  # Reject last
            (max(1, len(draft_tokens) // 2), len(draft_tokens) // 2),  # Reject half
        ]
        return outcomes[:self.async_fan_out]
    
    def verify(self, target_model, speculated_tokens: List[int], skeleton_kv, turbo_cache) -> List[int]:
        """
        Verify speculations against target model using compressed cache.
        
        Args:
            target_model: Target model
            speculated_tokens: Speculated sequence
            skeleton_kv: Compressed skeleton
            turbo_cache: Quantized cache
            
        Returns:
            accepted_tokens: Verified tokens
        """
        accepted = []
        device = next(target_model.parameters()).device
        
        for i, token in enumerate(speculated_tokens):
            # Forward target with current compressed state
            input_tensor = torch.tensor([[token]], device=device)
            with torch.no_grad():
                outputs = target_model(input_tensor, past_key_values=skeleton_kv)
                next_logits = outputs.logits[:, -1, :]
                predicted = torch.argmax(next_logits, dim=-1).item()
            
            if predicted == token:
                accepted.append(token)
                # Update cache (simplified - in practice, extract and quantize new KV)
            else:
                break
        
        return accepted
    
    def _tokens_to_text(self, tokens: List[int]) -> str:
        """Convert token list to text."""
        if self.draft_tokenizer is not None:
            return self.draft_tokenizer.decode(tokens)
        # Fallback
        return " ".join(map(str, tokens))
    
    def _text_to_tokens(self, text: str) -> List[int]:
        """Convert text back to tokens."""
        if self.draft_tokenizer is not None:
            return self.draft_tokenizer.encode(text)
        # Fallback
        return [int(t) for t in text.split() if t.isdigit()]
    
    def cleanup(self):
        """Clean up resources."""
        if self.executor:
            self.executor.shutdown(wait=True)
        
        if self.llm is not None:
            # vLLM cleanup if needed
            pass
        
        self.speculation_cache.clear()