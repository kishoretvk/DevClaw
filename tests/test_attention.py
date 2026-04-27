"""
Tests for CompressedAttention and AttentionPatcher.
"""

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csa.attention import CompressedAttention, AttentionPatcher


class TestCompressedAttention:
    """Tests for CompressedAttention module."""
    
    def test_module_creation(self):
        """Test creating CompressedAttention module."""
        # Create a simple mock attention
        class MockAttention(torch.nn.Module):
            def forward(self, query, key=None, value=None, past_key_values=None, **kwargs):
                return query, None
        
        mock_attn = MockAttention()
        attn = CompressedAttention(
            original_attention=mock_attn,
            head_dim=64,
            num_heads=8,
            compression_ratio=10,
            device='cpu'
        )
        
        assert attn.head_dim == 64
        assert attn.num_heads == 8
        assert attn.compression_ratio == 10
        assert not attn.use_compressed
    
    def test_enable_disable_compressed(self):
        """Test enabling/disabling compressed mode."""
        class MockAttention(torch.nn.Module):
            def forward(self, query, key=None, value=None, past_key_values=None, **kwargs):
                return query, None
        
        mock_attn = MockAttention()
        attn = CompressedAttention(
            original_attention=mock_attn,
            head_dim=64,
            num_heads=8,
            compression_ratio=10,
            device='cpu'
        )
        
        # Initially disabled
        assert not attn.use_compressed
        
        # Enable
        attn.enable_compressed_mode()
        assert attn.use_compressed
        
        # Disable
        attn.disable_compressed_mode()
        assert not attn.use_compressed
    
    def test_compress_kv(self):
        """Test compressing KV cache."""
        class MockAttention(torch.nn.Module):
            def forward(self, query, key=None, value=None, past_key_values=None, **kwargs):
                return query, None
        
        mock_attn = MockAttention()
        attn = CompressedAttention(
            original_attention=mock_attn,
            head_dim=64,
            num_heads=8,
            compression_ratio=10,
            device='cpu'
        )
        
        # Create mock KV cache
        kv_cache = [
            (torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64)),
            (torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64))
        ]
        
        # Compress
        compressed = attn.compress_kv(kv_cache)
        
        assert len(compressed) == 2  # 2 layers
        assert compressed[0][0].shape[2] == 10  # Compressed to 10 tokens
        assert compressed[1][0].shape[2] == 10


class TestAttentionPatcher:
    """Tests for AttentionPatcher."""
    
    def test_detect_model_type_gpt2(self):
        """Test detecting GPT-2 model type."""
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        model_type = AttentionPatcher.detect_model_type(model)
        assert model_type == 'gpt2'
    
    def test_patch_gpt2(self):
        """Test patching GPT-2 model."""
        model = AutoModelForCausalLM.from_pretrained("gpt2")
        
        # Patch model
        patched_layers = AttentionPatcher.patch_model(
            model,
            compression_ratio=10,
            device='cpu'
        )
        
        assert len(patched_layers) > 0
        
        # Verify attention is now CompressedAttention
        block = model.transformer.h[0]
        assert isinstance(block.attn, CompressedAttention)
        
        # Restore
        AttentionPatcher.restore_model(patched_layers)
        
        # Verify restored
        assert not isinstance(block.attn, CompressedAttention)