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
    # Check approximate reconstruction (FP8 has limited precision)
    # Use relative tolerance since FP8 values can have significant quantization error
    assert torch.allclose(tensor, dequantized, rtol=0.1, atol=0.5)

def test_turboquant_cache():
    """Test TurboQuant cache."""
    cache = TurboQuantCache(dim=64, bits=3, device="cpu")  # CPU for testing
    
    kv = (torch.randn(1, 8, 1, 64), torch.randn(1, 8, 1, 64))
    cache.append(kv)
    
    # Get with original shape to reshape back
    retrieved = cache.get(0, original_shape=(1, 8, 1, 64))
    assert len(retrieved) == 2
    # Check shapes match
    assert retrieved[0].shape == kv[0].shape

@pytest.mark.skip(reason="Requires model download - run manually with -m integration")
def test_csa_initialization():
    """
    Integration test: CSA engine initialization with actual model.

    Note: This is an integration test requiring model downloads.
    Skip with: pytest tests/test_csa.py -m "not integration"
    """
    from csa.core import CSAEngine

    # Test with gpt2 (small model, fast download)
    engine = CSAEngine(
        target_model_path="gpt2",
        compression_ratio=50,
        quant_bits=3,
        use_speculation=False,
        compression_frequency="once",
        device="cpu",
        use_dynamic_cache=True
    )

    assert engine is not None
    assert engine.compression_ratio == 50
    assert engine.target_model is not None

def test_ssd_speculator():
    """Test SSD speculator class structure (without model loading)."""
    from csa.speculation.ssd import SelfSpeculativeDecoder

    # Verify the class exists and has required methods
    assert SelfSpeculativeDecoder is not None
    assert hasattr(SelfSpeculativeDecoder, '__init__')
    assert hasattr(SelfSpeculativeDecoder, 'draft')
    assert hasattr(SelfSpeculativeDecoder, 'verify')

def test_background_recovery():
    """Test background recovery class structure."""
    from csa.recovery.recovery import BackgroundRecovery

    # Verify the class exists and has required methods
    assert BackgroundRecovery is not None
    assert hasattr(BackgroundRecovery, '__init__')
    # Note: Full threading tests would require more complex setup