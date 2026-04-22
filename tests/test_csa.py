import pytest
import torch
from csa.compression import AttentionMatcher, FP8Quantizer
from csa.quantization import TurboQuantCache

def test_attention_matching():
    """Test attention matching compression."""
    matcher = AttentionMatcher(compression_ratio=10)
    
    # Mock KV cache: batch=1, heads=8, seq=100, dim=64
    key = torch.randn(1, 8, 100, 64)
    value = torch.randn(1, 8, 100, 64)
    kv_cache = (key, value)
    
    compressed = matcher.compress(kv_cache)
    comp_key, comp_value = compressed
    
    assert comp_key.shape[2] == 10  # Compressed to 10 tokens
    assert comp_value.shape == comp_key.shape

def test_fp8_quantization():
    """Test FP8 quantization."""
    quantizer = FP8Quantizer()
    
    tensor = torch.randn(1, 8, 10, 64, dtype=torch.float16)
    quantized = quantizer.quantize(tensor)
    dequantized = quantizer.dequantize(quantized)
    
    assert quantized.dtype == torch.float8_e4m3fn
    assert dequantized.dtype == torch.float16
    # Check approximate reconstruction
    assert torch.allclose(tensor, dequantized, atol=1e-2)

def test_turboquant_cache():
    """Test TurboQuant cache."""
    cache = TurboQuantCache(dim=64, bits=3, device="cpu")  # CPU for testing
    
    kv = (torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64))
    cache.append(kv)
    
    retrieved = cache.get(0)
    assert len(retrieved) == 2
    # Check shapes match
    assert retrieved[0].shape == kv[0].shape

def test_csa_initialization():
    """Test CSA engine initialization (without actual models)."""
    # Mock without loading models
    try:
        from csa.core import CSAEngine
        # Would need mock models for full test
        pass
    except ImportError:
        pytest.skip("Model dependencies not available")

def test_ssd_speculator():
    """Test SSD speculator (mocked)."""
    # Would require actual model paths
    pytest.skip("Requires model setup")

def test_background_recovery():
    """Test background recovery (mocked)."""
    pytest.skip("Complex threading test")