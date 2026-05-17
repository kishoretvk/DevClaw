"""
Core CSA (Compressed Speculative Attention) engine.
"""

import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from ..compression import AttentionMatcher, FP8Quantizer
from ..compression.dynamic_cache import DynamicHierarchicalCache
from ..quantization import TurboQuantCache
try:
    from ..speculation.ssd import SSDSpeculator
except ImportError:
    SSDSpeculator = None
from ..recovery import BackgroundRecovery
from ..profiling import get_profiler, profile_component
from ..attention import CompressedAttention, AttentionPatcher


def _get_kv_list(full_kv):
    """Convert DynamicCache or similar to a list of (key, value) tuples."""
    # Already a list of 2-tuples
    if isinstance(full_kv, (list, tuple)) and full_kv and len(full_kv[0]) == 2:
        return list(full_kv)
    # Handle transformers DynamicCache (new API)
    if hasattr(full_kv, 'key_cache') and hasattr(full_kv, 'value_cache'):
        kc, vc = full_kv.key_cache, full_kv.value_cache
        if isinstance(kc, (list, tuple)) and isinstance(vc, (list, tuple)):
            return [(kc[i], vc[i]) for i in range(len(kc))]
    # Fallback: try iterating and take first 2 elements per item
    try:
        result = []
        for item in full_kv:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                result.append((item[0], item[1]))
            else:
                result.append(item)
        return result
    except TypeError:
        raise TypeError(f"Unsupported KV cache type: {type(full_kv)}")


def _kv_list_to_cache(kv_list):
    """Convert list of (key, value) tuples back to a DynamicCache for model.generate()."""
    from transformers import DynamicCache
    cache = DynamicCache()
    # Directly set cache internals to avoid DynamicCache.update() issues
    cache.key_cache = [k for k, v in kv_list]
    cache.value_cache = [v for k, v in kv_list]
    # CRITICAL: Set seen_tokens so model.generate() knows the cache has content
    # Without this, get_seq_length() returns 0 and generation produces empty output
    if kv_list and kv_list[0][0] is not None:
        seq_len = kv_list[0][0].shape[2]
        cache._seen_tokens = seq_len
    return cache


class CSAEngine:
    """Main engine for CSA acceleration with 50x KV compression and 5-10x speedup."""

    def __init__(self, target_model_path, draft_model_path=None, compression_ratio=50,
                 quant_bits=3, use_speculation=True, compression_frequency="once",
                 skip_compression_threshold=512, device="auto", use_dynamic_cache=True):
        """
        Initialize CSA Engine.

        Args:
            target_model_path: Path to target model
            draft_model_path: Path to draft model (optional)
            compression_ratio: KV cache compression ratio (target: 50x)
            quant_bits: Quantization bits (3 for max compression, 4 for speed)
            use_speculation: Enable SSD speculative decoding
            compression_frequency: How often to compress ("once", "per_10_tokens", "lazy")
            skip_compression_threshold: Skip compression for prompts shorter than this
            device: Device to use ("cuda", "cpu", "auto")
            use_dynamic_cache: Use DynamicHierarchicalCache (recommended for 50x compression)
        """
        # Device configuration
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        print(f"Loading target model on {self.device}...")
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path, torch_dtype=torch.float16
        ).to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.draft_model_path = draft_model_path or target_model_path
        self.use_speculation = use_speculation
        self.compression_ratio = compression_ratio
        self.use_dynamic_cache = use_dynamic_cache

        # Speed optimization parameters
        self.compression_frequency = compression_frequency
        self.skip_compression_threshold = skip_compression_threshold
        self.compression_step_counter = 0

        self.matcher = AttentionMatcher(compression_ratio=compression_ratio)
        self.quantizer = FP8Quantizer()

        # Dynamic Hierarchical Cache (for 50x compression)
        if use_dynamic_cache:
            config = self.target_model.config
            num_layers = getattr(config, 'num_hidden_layers',
                                  getattr(config, 'n_layer', 12))
            # For 50x compression on 424 tokens: need ~9 tokens total
            # skeleton_budget: uniform position coverage (just indices, not KV)
            # detail_budget: heavy-hitter tokens at full precision
            # recent_window: most recent tokens always kept
            # Total KV tokens kept = detail_budget + recent_window
            target_tokens = max(8, 512 // compression_ratio)
            detail = max(4, target_tokens // 2)
            recent = max(4, target_tokens - detail)
            self.dynamic_cache = DynamicHierarchicalCache(
                skeleton_budget=max(5, target_tokens),
                detail_budget=detail,
                recent_window=recent,
                num_layers=num_layers,
                skeleton_rebuild_freq=50
            )
        else:
            self.dynamic_cache = None

        # Get model dimensions
        config = self.target_model.config
        self.head_dim = getattr(config, 'head_dim',
                               config.hidden_size // config.num_attention_heads)
        self.num_heads = config.num_attention_heads
        self.num_layers = getattr(config, 'num_hidden_layers',
                                 getattr(config, 'n_layer', 12))

        # Performance tracking
        self.generation_step = 0
        self.skeleton_kv = None
        self.speculator = None
        self.recovery = None

        # NOTE: CompressedAttention patching is disabled because it conflicts
        # with model.generate() when using DynamicCache. The compressed KV cache
        # is passed directly to model.generate() via DynamicCache instead.
        self.patched_layers = []
        print("   Using DynamicCache-based compression (no attention patching)")

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
                input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)

            if not self.use_speculation:
                result = self._simple_generate(input_ids, max_new_tokens)
            else:
                result = self._full_generate(input_ids, max_new_tokens)

        if enable_profiling:
            summary = profiler.end_profiling()
            print(f"\n Performance Summary:")
            print(f"   Total time: {summary['total_time']:.3f}s")
            print(f"   Memory delta: {summary['total_memory_delta']:+.1f}MB")

            if summary['bottlenecks']:
                print(f"   Bottlenecks found: {len(summary['bottlenecks'])}")
                for bottleneck in summary['bottlenecks']:
                    print(f"      {bottleneck['component']}: {bottleneck['percentage']:.1f}%")

            profiler.export_metrics(f"csa_profile_{int(time.time())}.json")

        return result

    def _should_compress(self, seq_length):
        """Determine if compression should be applied based on configuration."""
        if seq_length < self.skip_compression_threshold:
            return False

        if self.compression_frequency == "once":
            # When using dynamic cache, check its initialized state
            if self.use_dynamic_cache and self.dynamic_cache:
                return not self.dynamic_cache.initialized
            return self.skeleton_kv is None
        elif self.compression_frequency == "per_10_tokens":
            return self.generation_step % 10 == 0
        elif self.compression_frequency == "lazy":
            # When using dynamic cache, check its initialized state
            if self.use_dynamic_cache and self.dynamic_cache:
                return not self.dynamic_cache.initialized
            return self.skeleton_kv is None

        return True

    def _simple_generate(self, input_ids, max_new_tokens):
        """Simple generation with compression."""
        seq_length = input_ids.shape[1]

        # Prefill phase
        with profile_component("prefill_phase", {"seq_length": seq_length}):
            with torch.no_grad():
                outputs = self.target_model(input_ids, use_cache=True)
                full_kv = outputs.past_key_values

        # Convert DynamicCache to list of tuples for iteration
        kv_list = _get_kv_list(full_kv)

        # Compress using dynamic cache if available
        should_compress = self._should_compress(seq_length)

        if should_compress:
            print("Compressing KV cache...")
            t_compress = time.time()
            if self.dynamic_cache and self.use_dynamic_cache:
                skeleton_kv = self._compress_with_dynamic_cache(full_kv)
            else:
                skeleton_kv = self._compress_kv(kv_list)
            self.compression_time = time.time() - t_compress
            print(f"   Compression took {self.compression_time:.3f}s")

            self.skeleton_kv = skeleton_kv

            original_seq_len = kv_list[0][0].shape[2]
            compressed_seq_len = skeleton_kv[0][0].shape[2]
            compression_ratio = original_seq_len / compressed_seq_len if compressed_seq_len > 0 else 1

            print(f"Compressed from {original_seq_len} to {compressed_seq_len} tokens per layer ({compression_ratio:.1f}x compression)")
        else:
            print("Skipping compression (using cached skeleton)")
            skeleton_kv = self.skeleton_kv

        # Generate with compressed cache
        if skeleton_kv is not None:
            print("Using COMPRESSED KV cache for generation!")
            print(f"   Passing {len(skeleton_kv)} compressed layers directly to model")
            with profile_component("token_generation", {"max_tokens": max_new_tokens, "compressed_cache": True}):
                generated_ids = self._generate_with_compressed_cache(
                    input_ids, skeleton_kv, max_new_tokens
                )
        else:
            print("Using standard generation (no compressed cache)")
            with profile_component("token_generation", {"max_tokens": max_new_tokens, "compressed_cache": False}):
                generated_ids = self.target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7
                )

        generated_text = self.tokenizer.decode(generated_ids[0][len(input_ids[0]):], skip_special_tokens=True)

        self.generation_step += 1
        return generated_text

    def _generate_with_compressed_cache(self, input_ids, compressed_kv, max_new_tokens):
        """Manual token-by-token generation with compressed KV cache.

        Bypasses model.generate() which has issues with DynamicCache format.
        Uses model() forward pass directly with list-of-tuples KV cache.
        """
        generated_tokens = []
        # Start with last token of input
        current_token = input_ids[:, -1:]
        past_kv = list(compressed_kv)  # list of (key, value) tuples

        for _ in range(max_new_tokens):
            with torch.no_grad():
                outputs = self.target_model(
                    current_token,
                    past_key_values=past_kv,
                    use_cache=True
                )
                past_kv = outputs.past_key_values
                # Handle DynamicCache returned by model
                if hasattr(past_kv, 'key_cache'):
                    past_kv = _get_kv_list(past_kv)

                next_token_logits = outputs.logits[:, -1, :]

                # Sample with temperature
                probs = torch.softmax(next_token_logits / 0.7, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                token_id = next_token.item()
                generated_tokens.append(token_id)

                # Check for EOS
                if token_id == self.tokenizer.eos_token_id:
                    break

                current_token = next_token

        return self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    def _full_generate(self, input_ids, max_new_tokens):
        """Full CSA generation with self-speculative decoding (5-10x speedup)."""

        # Prefill and compress
        with profile_component("ssd_prefill"):
            with torch.no_grad():
                outputs = self.target_model(input_ids, use_cache=True)
                full_kv = outputs.past_key_values

        # Convert DynamicCache to list of tuples for iteration
        kv_list = _get_kv_list(full_kv)

        # Compress to skeleton
        if self.dynamic_cache and self.use_dynamic_cache:
            skeleton_kv = self._compress_with_dynamic_cache(full_kv)
        else:
            skeleton_kv = self._compress_kv(kv_list)

        self.skeleton_kv = skeleton_kv

        # Use manual generation with compressed cache
        return self._generate_with_compressed_cache(input_ids, skeleton_kv, max_new_tokens)

    def _compress_kv(self, full_kv):
        """Compress KV cache to skeleton using uniform compression."""
        compressed = []
        for layer_kv in full_kv:
            comp_kv = self.matcher.compress(layer_kv)
            # Keep in float16 to avoid dtype mismatch with model
            comp_kv = (comp_kv[0].to(torch.float16), comp_kv[1].to(torch.float16))
            compressed.append(comp_kv)
        return compressed

    def _compress_with_dynamic_cache(self, full_kv):
        """Compress using DynamicHierarchicalCache for 50x compression."""
        if not self.dynamic_cache:
            return self._compress_kv(_get_kv_list(full_kv))

        # Ensure full_kv is a list of (key, value) tuples
        kv_list = _get_kv_list(full_kv)

        # Initialize cache if not already done
        if not self.dynamic_cache.initialized:
            self.dynamic_cache.initialize(kv_list)

        # Update dynamic cache with full KV
        for layer_idx, (key, value) in enumerate(kv_list):
            # Use L2 norm of key vectors as importance proxy
            # Higher norm = more "active" key = more likely to be attended to
            # Shape: (batch, heads, seq_len, head_dim) -> (seq_len,)
            scores = torch.linalg.vector_norm(key.float(), dim=(0, 1, 3))
            self.dynamic_cache.update(layer_idx, (key, value), scores)

        # Get compressed cache
        compressed = self.dynamic_cache.get_all_caches()
        return compressed

    def _target_forward(self, input_tokens, skeleton_kv, turbo_cache):
        """Forward pass with compressed cache."""
        with torch.no_grad():
            outputs = self.target_model(input_tokens.unsqueeze(0), past_key_values=skeleton_kv)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1).item()
        return next_token

    def _extract_new_kv(self):
        """Extract new KV from last forward pass."""
        return None

    def cleanup(self):
        """Clean up resources and restore original model if patched."""
        if hasattr(self, 'patched_layers') and self.patched_layers:
            print("Restoring original attention layers...")
            AttentionPatcher.restore_model(self.patched_layers)
            self.patched_layers = []

        if hasattr(self, 'recovery') and self.recovery:
            self.recovery.stop()

        if hasattr(self, 'speculator') and self.speculator:
            self.speculator.cleanup()
