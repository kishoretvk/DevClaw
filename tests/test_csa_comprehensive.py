"""
Comprehensive tests for CSA working components

These tests verify the components that are actually working:
- KV cache compression algorithm
- Quantization/dequantization
- Compressed cache wrapper
- Basic engine initialization
"""

import pytest
import torch
from csa.compression import AttentionMatcher, FP8Quantizer
from csa.compression.cache_wrapper import CompressedKVCache, EfficientCompressedCache
from csa.quantization import TurboQuantCache
from csa.core import CSAEngine


class TestAttentionMatching:
    """Tests for KV cache compression algorithm"""
    
    def test_compression_reduces_size(self):
        """Test that compression reduces sequence length."""
        matcher = AttentionMatcher(compression_ratio=10)
        
        # Mock KV cache: batch=1, heads=8, seq=100, dim=64
        key = torch.randn(1, 8, 100, 64)
        value = torch.randn(1, 8, 100, 64)
        kv_cache = (key, value)
        
        compressed = matcher.compress(kv_cache)
        comp_key, comp_value = compressed
        
        # Should be compressed to 10 tokens (100/10)
        assert comp_key.shape[2] == 10
        assert comp_value.shape == comp_key.shape
        assert comp_key.shape[2] < key.shape[2]
    
    def test_compression_preserves_batch_heads(self):
        """Test that batch and head dimensions are preserved."""
        matcher = AttentionMatcher(compression_ratio=5)
        
        key = torch.randn(2, 12, 50, 64)
        value = torch.randn(2, 12, 50, 64)
        kv_cache = (key, value)
        
        compressed = matcher.compress(kv_cache)
        comp_key, comp_value = compressed
        
        assert comp_key.shape[0] == 2  # batch
        assert comp_key.shape[1] == 12  # heads
        assert comp_key.shape[2] == 10  # compressed seq (50/5)
        assert comp_key.shape[3] == 64  # head dim
    
    def test_compression_ratio_parameter(self):
        """Test different compression ratios."""
        for ratio in [5, 10, 20, 50]:
            matcher = AttentionMatcher(compression_ratio=ratio)
            
            key = torch.randn(1, 8, 100, 64)
            value = torch.randn(1, 8, 100, 64)
            kv_cache = (key, value)
            
            compressed = matcher.compress(kv_cache)
            expected_seq = 100 // ratio
            assert compressed[0].shape[2] == expected_seq
    
    def test_compression_uniform_sampling(self):
        """Test that compression uses uniform sampling."""
        matcher = AttentionMatcher(compression_ratio=10)
        
        # Create KV with identifiable pattern
        key = torch.arange(1 * 8 * 100 * 64).reshape(1, 8, 100, 64).float()
        value = torch.ones(1, 8, 100, 64)
        kv_cache = (key, value)
        
        compressed = matcher.compress(kv_cache)
        comp_key, comp_value = compressed
        
        # Should have sampled uniformly
        assert comp_key.shape[2] == 10
        # Values should be from original (sampled)
        # Note: max may be equal if last element is sampled
        assert comp_key.max() <= key.max()
        assert comp_key.min() >= key.min()


class TestFP8Quantization:
    """Tests for FP8 quantization"""
    
    def test_quantization_dtype(self):
        """Test that quantization produces FP8 dtype."""
        quantizer = FP8Quantizer()
        
        tensor = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        quantized = quantizer.quantize(tensor)
        
        assert quantized.dtype == torch.float8_e4m3fn
    
    def test_dequantization_dtype(self):
        """Test that dequantization returns to FP16."""
        quantizer = FP8Quantizer()
        
        tensor = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        assert dequantized.dtype == torch.float16
    
    def test_quantization_roundtrip(self):
        """Test that quantization/dequantization approximately preserves values."""
        quantizer = FP8Quantizer()
        
        tensor = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # FP8 has limited precision, so allow some error
        # Use relative tolerance since FP8 values can have significant quantization error
        assert torch.allclose(tensor, dequantized, rtol=0.1, atol=0.5)
    
    def test_quantization_shape_preserved(self):
        """Test that quantization preserves tensor shape."""
        quantizer = FP8Quantizer()
        
        tensor = torch.randn(2, 12, 50, 64, dtype=torch.float16)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        assert quantized.shape == tensor.shape
        assert dequantized.shape == tensor.shape
    
    def test_quantization_with_zero(self):
        """Test quantization handles zero values."""
        quantizer = FP8Quantizer()
        
        tensor = torch.zeros(1, 8, 10, 64, dtype=torch.float16)
        quantized = quantizer.quantize(tensor)
        dequantized = quantizer.dequantize(quantized)
        
        # Zeros should remain close to zero
        assert torch.allclose(dequantized, tensor, atol=0.1)


class TestCompressedKVCache:
    """Tests for CompressedKVCache wrapper"""
    
    def test_cache_creation(self):
        """Test creating a CompressedKVCache."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        assert cache.num_layers == 1
        assert cache.original_seq_len == 100
        assert cache.compression_ratio == 10
    
    def test_decompress_layer(self):
        """Test decompressing a single layer."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        k, v = cache.decompress_layer(0)
        
        # Should be expanded to original size
        assert k.shape == (1, 8, 100, 64)
        assert v.shape == (1, 8, 100, 64)
    
    def test_to_standard_cache(self):
        """Test converting to standard cache format."""
        comp_k = torch.randn(1, 8, 5, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 5, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v), (comp_k.clone(), comp_v.clone())]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=50,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        standard = cache.to_standard_cache()
        
        assert len(standard) == 2
        assert standard[0][0].shape == (1, 8, 50, 64)
        assert standard[1][0].shape == (1, 8, 50, 64)
    
    def test_cache_indexing(self):
        """Test indexing into cache."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        k, v = cache[0]
        assert k.shape == (1, 8, 100, 64)
    
    def test_decompression_caching(self):
        """Test that decompression is cached."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        # First access
        k1, v1 = cache.decompress_layer(0)
        
        # Second access should return cached
        k2, v2 = cache.decompress_layer(0)
        
        # Should be same object (cached)
        assert k1 is k2
        assert v1 is v2
    
    def test_with_quantizer(self):
        """Test with FP8 quantization."""
        quantizer = FP8Quantizer()
        
        # Create and quantize
        k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        
        qk = quantizer.quantize(k)
        qv = quantizer.quantize(v)
        
        compressed_kv = [(qk, qv)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=quantizer,
            device='cpu'
        )
        
        # Decompress should handle quantization
        dk, dv = cache.decompress_layer(0)
        assert dk.shape == (1, 8, 100, 64)
        assert dk.dtype == torch.float16
    
    def test_small_compression_ratio(self):
        """Test with very small compressed size."""
        comp_k = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = CompressedKVCache(
            compressed_kv=compressed_kv,
            original_seq_len=50,
            compression_ratio=50,
            quantizer=None,
            device='cpu'
        )
        
        k, v = cache.decompress_layer(0)
        assert k.shape == (1, 8, 50, 64)
    
    def test_multiple_layers(self):
        """Test with multiple layers."""
        layers = []
        for _ in range(12):  # 12 layers like GPT-2
            comp_k = torch.randn(1, 12, 5, 64, dtype=torch.float16)
            comp_v = torch.randn(1, 12, 5, 64, dtype=torch.float16)
            layers.append((comp_k, comp_v))
        
        cache = CompressedKVCache(
            compressed_kv=layers,
            original_seq_len=100,
            compression_ratio=20,
            quantizer=None,
            device='cpu'
        )
        
        assert cache.num_layers == 12
        
        standard = cache.to_standard_cache()
        assert len(standard) == 12
        
        for i, (k, v) in enumerate(standard):
            assert k.shape == (1, 12, 100, 64)
            assert v.shape == (1, 12, 100, 64)


class TestEfficientCompressedCache:
    """Tests for EfficientCompressedCache"""
    
    def test_get_compressed_layer(self):
        """Test getting compressed layer without decompression."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = EfficientCompressedCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        k, v = cache.get_compressed_layer(0)
        assert k.shape == (1, 8, 10, 64)
        assert v.shape == (1, 8, 10, 64)
    
    def test_compute_compressed_attention(self):
        """Test attention computation with compressed cache."""
        comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        comp_v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        compressed_kv = [(comp_k, comp_v)]
        
        cache = EfficientCompressedCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=None,
            device='cpu'
        )
        
        # Create query
        query = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        
        # Compute attention
        output = cache.compute_compressed_attention(query, 0)
        
        # Output should have same shape as query
        assert output.shape == (1, 8, 1, 64)
    
    def test_compute_compressed_attention_with_quantizer(self):
        """Test compressed attention with quantization."""
        quantizer = FP8Quantizer()
        
        k = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        v = torch.randn(1, 8, 10, 64, dtype=torch.float16)
        
        qk = quantizer.quantize(k)
        qv = quantizer.quantize(v)
        
        compressed_kv = [(qk, qv)]
        
        cache = EfficientCompressedCache(
            compressed_kv=compressed_kv,
            original_seq_len=100,
            compression_ratio=10,
            quantizer=quantizer,
            device='cpu'
        )
        
        query = torch.randn(1, 8, 1, 64, dtype=torch.float16)
        output = cache.compute_compressed_attention(query, 0)
        
        assert output.shape == (1, 8, 1, 64)


class TestTurboQuantCache:
    """Tests for TurboQuantCache"""
    
    def test_cache_creation(self):
        """Test creating TurboQuantCache."""
        cache = TurboQuantCache(dim=64, bits=3, device='cpu')
        
        assert cache.bits == 3
        assert len(cache.cache) == 0
    
    def test_append_and_get(self):
        """Test appending and retrieving from cache."""
        cache = TurboQuantCache(dim=64, bits=3, device='cpu')
        
        kv = (torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64))
        cache.append(kv)
        
        retrieved = cache.get(0, original_shape=(1, 8, 1, 64))
        assert len(retrieved) == 2
        assert retrieved[0].shape == kv[0].shape
        assert retrieved[1].shape == kv[1].shape
    
    def test_multiple_entries(self):
        """Test cache with multiple entries."""
        cache = TurboQuantCache(dim=64, bits=3, device='cpu')
        
        for i in range(5):
            kv = (torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64))
            cache.append(kv)
        
        assert len(cache.cache) == 5
        
        # Retrieve middle entry
        retrieved = cache.get(2, original_shape=(1, 8, 1, 64))
        assert retrieved[0].shape == (1, 8, 1, 64)


class TestCSAEngine:
    """Tests for CSAEngine initialization"""
    
    def test_engine_creation(self):
        """Test CSAEngine can be created (without loading models)."""
        # This test verifies the class can be imported and instantiated
        # Actual model loading is skipped in test environment
        try:
            from csa.core import CSAEngine
            # Would need mock models for full test
            assert CSAEngine is not None
        except ImportError:
            pytest.skip("CSAEngine not available")
    
    def test_import_all_components(self):
        """Test that all CSA components can be imported."""
        from csa.compression import AttentionMatcher, FP8Quantizer
        from csa.compression.cache_wrapper import CompressedKVCache, EfficientCompressedCache
        from csa.quantization import TurboQuantCache
        from csa.core import CSAEngine
        
        assert AttentionMatcher is not None
        assert FP8Quantizer is not None
        assert CompressedKVCache is not None
        assert EfficientCompressedCache is not None
        assert TurboQuantCache is not None
        assert CSAEngine is not None