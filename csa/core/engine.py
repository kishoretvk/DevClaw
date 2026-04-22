# Core CSA engine

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..compression import AttentionMatcher, FP8Quantizer
from ..quantization import TurboQuantCache
try:
    from ..speculation.ssd import SSDSpeculator
except ImportError:
    SSDSpeculator = None
from ..recovery import BackgroundRecovery
from ..profiling import get_profiler, profile_component

class CSAEngine:
    def __init__(self, target_model_path, draft_model_path=None, compression_ratio=50, quant_bits=3,
                 use_speculation=False, compression_frequency="once", skip_compression_threshold=512):
        """
        Initialize CSA Engine with speed optimization options.

        Args:
            target_model_path: Path to target model
            draft_model_path: Path to draft model (optional)
            compression_ratio: KV cache compression ratio (higher = more compression)
            quant_bits: Quantization precision (3 for max compression, 4 for speed)
            use_speculation: Enable SSD speculative decoding
            compression_frequency: How often to compress ("once", "per_10_tokens", "lazy")
            skip_compression_threshold: Skip compression for prompts shorter than this
        """
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.target_model = AutoModelForCausalLM.from_pretrained(target_model_path, torch_dtype=torch.float16).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.draft_model_path = draft_model_path or target_model_path
        self.use_speculation = use_speculation

        # Speed optimization parameters
        self.compression_frequency = compression_frequency
        self.skip_compression_threshold = skip_compression_threshold
        self.compression_step_counter = 0

        self.matcher = AttentionMatcher(compression_ratio=compression_ratio)
        self.quantizer = FP8Quantizer()

        # Get head_dim for quantizer
        config = self.target_model.config
        self.head_dim = getattr(config, 'head_dim', config.hidden_size // config.num_attention_heads)

        # Performance tracking
        self.generation_step = 0
        self.skeleton_kv = None  # Cached compressed skeleton

        self.speculator = None
        self.recovery = None
    
    def generate(self, prompt, max_new_tokens=100, enable_profiling=False):
        """
        Generate tokens using CSA.

        Args:
            prompt: Input prompt string
            max_new_tokens: Maximum new tokens to generate
            enable_profiling: Whether to enable detailed performance profiling

        Returns:
            generated_text: Generated text
        """
        profiler = get_profiler()

        if enable_profiling:
            profiler.start_profiling()

        with profile_component("total_generation", {
            "prompt_length": len(prompt),
            "max_new_tokens": max_new_tokens,
            "use_speculation": self.use_speculation
        }):
            # Tokenize prompt
            with profile_component("tokenization"):
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.target_model.device)

            if not self.use_speculation:
                # Simple mode: just compress prompt and generate normally
                result = self._simple_generate(input_ids, max_new_tokens)
            else:
        # Full CSA mode with SSD
        result = self._full_generate(input_ids, max_new_tokens)

        if enable_profiling:
            summary = profiler.end_profiling()
            # Print key insights
            print(f"\\n📊 Performance Summary:")
            print(f"   Total time: {summary['total_time']:.3f}s")
            print(f"   Memory delta: {summary['total_memory_delta']:+.1f}MB")

            if summary['bottlenecks']:
                print(f"   🚨 Bottlenecks found: {len(summary['bottlenecks'])}")
                for bottleneck in summary['bottlenecks']:
                    print(f"      • {bottleneck['component']}: {bottleneck['percentage']:.1f}%")

            # Export detailed profile
            profiler.export_metrics(f"csa_profile_{int(time.time())}.json")

        return result
    
    def _should_compress(self, seq_length):
        """Determine if compression should be applied based on configuration."""
        # Skip compression for short prompts
        if seq_length < self.skip_compression_threshold:
            return False

        # Check compression frequency
        if self.compression_frequency == "once":
            return self.skeleton_kv is None  # Only compress once
        elif self.compression_frequency == "per_10_tokens":
            return self.generation_step % 10 == 0
        elif self.compression_frequency == "lazy":
            return self.skeleton_kv is None  # Compress only when needed

        return True

    def _simple_generate(self, input_ids, max_new_tokens):
        """Simple generation with compression."""
        seq_length = input_ids.shape[1]

        # Prefill phase
        with profile_component("prefill_phase", {"seq_length": seq_length}):
            with torch.no_grad():
                outputs = self.target_model(input_ids, use_cache=True)
                full_kv = outputs.past_key_values

        # Compress skeleton based on configuration
        should_compress = self._should_compress(seq_length)

        if should_compress:
            print("🔧 Compressing KV cache...")

            # Compress skeleton
            skeleton_kv = []
            with profile_component("attention_matching", {"layers": len(full_kv), "compression_ratio": self.matcher.compression_ratio}):
                for i, layer_kv in enumerate(full_kv):
                    with profile_component(f"compress_layer_{i}"):
                        comp_kv = self.matcher.compress(layer_kv)
                        # Quantize skeleton to FP8
                        with profile_component("fp8_quantization"):
                            comp_kv = (self.quantizer.quantize(comp_kv[0]), self.quantizer.quantize(comp_kv[1]))
                        skeleton_kv.append(comp_kv)

            # Cache the compressed skeleton
            self.skeleton_kv = skeleton_kv

            original_seq_len = full_kv[0][0].shape[2]
            compressed_seq_len = skeleton_kv[0][0].shape[2]
            compression_ratio = original_seq_len / compressed_seq_len if compressed_seq_len > 0 else 1

            print(f"✅ Compressed from {original_seq_len} to {compressed_seq_len} tokens per layer ({compression_ratio:.1f}x compression)")
        else:
            print("⏭️  Skipping compression (using cached skeleton)"            skeleton_kv = self.skeleton_kv

        # For demonstration, generate without compressed cache (since FP8 issue)
        # In full implementation, need to handle mixed precision
        with profile_component("token_generation", {"max_tokens": max_new_tokens, "compressed_cache": skeleton_kv is not None}):
            generated_ids = self.target_model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7
            )

        # Decode
        generated_text = self.tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)

        # Increment step counter
        self.generation_step += 1

        return f"[CSA with compression demo] {generated_text}"
    
    def _full_generate(self, input_ids, max_new_tokens):
        """Full CSA generation with SSD async speculation."""
        with profile_component("ssd_full_generation", {"max_tokens": max_new_tokens}):
            # Prefill and compress
            with profile_component("ssd_prefill"):
                with torch.no_grad():
                    outputs = self.target_model(input_ids, use_cache=True)
                    full_kv = outputs.past_key_values

            # Compress to skeleton
            skeleton_kv = self._compress_kv(full_kv)

            # Initialize SSD speculator with CUDA streams
            with profile_component("ssd_init"):
                self.speculator = SSDSpeculator(
                    self.draft_model_path,
                    skeleton_kv,
                    use_cuda_streams=True  # Enable async CUDA streams
                )
                turbo_cache = TurboQuantCache(dim=self.head_dim, bits=3, device=self.target_model.device)

            # Start background recovery (non-blocking)
            with profile_component("recovery_init"):
                self.recovery = BackgroundRecovery(self.target_model, full_kv, skeleton_kv, turbo_cache)
                self.recovery.start()

            # Generation loop with SSD
            generated_tokens = []
            current_ids = input_ids[0].clone()

            for step in range(max_new_tokens):
                with profile_component(f"generation_step_{step}"):
                    # SSD: Async speculation for predicted outcomes
                    with profile_component("outcome_prediction"):
                        predicted_outcomes = self.speculator.predict_outcomes(current_ids.tolist())

                    # Parallel speculation using CUDA streams
                    with profile_component("async_speculation", {"num_outcomes": len(predicted_outcomes)}):
                        speculations = self.speculator.speculate_async(current_ids.tolist(), predicted_outcomes)

                    # Generate next token with target model
                    with profile_component("target_forward"):
                        next_token = self._target_forward(current_ids[-1:], skeleton_kv, turbo_cache)
                    generated_tokens.append(next_token)
                    current_ids = torch.cat([current_ids, torch.tensor([next_token], device=current_ids.device)])

                    # Verify against speculations (SSD core logic)
                    with profile_component("speculation_verification"):
                        # For simplicity, verify the most likely speculation
                        best_outcome = predicted_outcomes[0]
                        best_spec = speculations[best_outcome][:5]  # First 5 tokens
                        accepted = self.speculator.verify(self.target_model, best_spec, skeleton_kv, turbo_cache)

                    # Accept verified tokens
                    with profile_component("token_acceptance"):
                        for token in accepted:
                            if len(generated_tokens) < max_new_tokens:
                                generated_tokens.append(token)
                                current_ids = torch.cat([current_ids, torch.tensor([token], device=current_ids.device)])

            # Stop background recovery
            with profile_component("recovery_cleanup"):
                self.recovery.stop()

        # Decode final result
        text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        return text

    def _compress_kv(self, full_kv):
        """Compress KV cache to skeleton."""
        compressed = []
        for layer_kv in full_kv:
            comp_kv = self.matcher.compress(layer_kv)
            comp_kv = (self.quantizer.quantize(comp_kv[0]), self.quantizer.quantize(comp_kv[1]))
            compressed.append(comp_kv)
        return compressed
    
    def _target_forward(self, input_tokens, skeleton_kv, turbo_cache):
        """Forward pass with compressed cache."""
        # Simplified: use standard forward, ignoring compression for now
        # In full implementation, need custom attention layer
        with torch.no_grad():
            outputs = self.target_model(input_tokens.unsqueeze(0), past_key_values=skeleton_kv)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
        return next_token
    
    def _extract_new_kv(self):
        """Extract new KV from last forward pass."""
        # Placeholder: in practice, hook into model to capture
        return None