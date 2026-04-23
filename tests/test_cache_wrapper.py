"""
Tests for CompressedKVCache wrapper
"""

import pytest
import torch
from csa.compression.cache_wrapper import CompressedKVCache, EfficientCompressedCache
from csa.compression import FP8Quantizer


def test_compressed_kv_cache_creation():
    """Test creating a CompressedKVCache."""
    # Create mock compressed KV
    comp_k = torch.randn(1, 8, 10, 64, dtype=torch.float16)  # 10 compressed tokens
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


def test_decompress_layer():
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


def test_to_standard_cache():
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


def test_cache_indexing():
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


def test_decompression_caching():
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


def test_efficient_compressed_cache():
    """Test EfficientCompressedCache."""
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
    
    # Test getting compressed layer
    k, v = cache.get_compressed_layer(0)
    assert k.shape == (1, 8, 10, 64)
    assert v.shape == (1, 8, 10, 64)


def test_compute_compressed_attention():
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


def test_with_quantizer():
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


def test_small_compression_ratio():
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


def test_multiple_layers():
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
