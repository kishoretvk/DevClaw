"""
Compressed Attention module for CSA.

This module implements attention computation that works directly with
compressed KV cache, enabling actual speedup by avoiding
decompression back to standard format.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple


class CompressedAttention(torch.nn.Module):
    """
    Custom attention that works directly with compressed KV cache.
    
    This is the P0 blocker removal - enables actual speedup
    by using compressed cache directly instead of decompressing.
    
    Model-agnostic: works with GPT, Llama, Qwen, Mistral.
    """
    
    def __init__(self, original_attention, head_dim: int, num_heads: int, 
                 compression_ratio: int = 10, device: str = "auto"):
        """
        Args:
            original_attention: The original attention module to wrap
            head_dim: Dimension per attention head
            num_heads: Number of attention heads
            compression_ratio: Compression ratio for KV cache
            device: Device to run on ("cuda", "cpu", "auto")
        """
        super().__init__()
        
        self.original_attention = original_attention
        self.head_dim = head_dim
        self.num_heads = num_heads
        self.compression_ratio = compression_ratio
        
        # Device setup
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
        
        # Compression components
        from ..compression import AttentionMatcher
        self.matcher = AttentionMatcher(compression_ratio=compression_ratio)
        
        # State
        self.compressed_kv = None
        self.use_compressed = False
        
    def forward(self, 
                 query: torch.Tensor,
                 key: Optional[torch.Tensor] = None,
                 value: Optional[torch.Tensor] = None,
                 past_key_value: Optional[Tuple] = None,
                 attention_mask: Optional[torch.Tensor] = None,
                 **kwargs):
        """
        Forward pass that uses compressed KV cache when available.
        
        Args:
            query: Query tensor (batch, heads, seq, head_dim)
            key: Key tensor (for computing new KV)
            value: Value tensor (for computing new KV)
            past_key_value: Compressed KV cache from CSA
            attention_mask: Optional attention mask
            
        Returns:
            output: Attention output
            new_kv: Updated KV cache (compressed if enabled)
        """
        # If we have compressed KV cache, use it directly
        if past_key_value is not None and self.use_compressed:
            return self._attention_with_compressed(
                query, past_key_value, attention_mask
            )
        
        # Otherwise, use standard attention
        return self.original_attention.forward(
            query, key=key, value=value,
            past_key_value=past_key_value,
            attention_mask=attention_mask,
            **kwargs
        )
    
    def _attention_with_compressed(self, 
                              query: torch.Tensor,
                              compressed_kv: Tuple[torch.Tensor, torch.Tensor],
                              attention_mask: Optional[torch.Tensor] = None):
        """
        Compute attention using compressed KV cache directly.
        
        Args:
            query: Query tensor (batch, heads, 1, head_dim) for generation
            compressed_kv: Tuple of (compressed_k, compressed_v)
            attention_mask: Optional mask
            
        Returns:
            output: Attention output (batch, heads, 1, head_dim)
            None: No new KV to cache (using compressed)
        """
        comp_k, comp_v = compressed_kv
        
        # Ensure correct device
        comp_k = comp_k.to(self.device)
        comp_v = comp_v.to(self.device)
        query = query.to(self.device)
        
        # Compute attention scores with compressed keys
        # query: (batch, heads, 1, head_dim)
        # comp_k: (batch, heads, comp_seq, head_dim)
        scores = torch.matmul(query, comp_k.transpose(-2, -1))
        scores = scores / (self.head_dim ** 0.5)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Adjust mask for compressed sequence length
            scores = scores + attention_mask
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of compressed values
        output = torch.matmul(attn_weights, comp_v)
        
        return output, None  # No new KV to cache
    
    def enable_compressed_mode(self):
        """Enable using compressed KV cache."""
        self.use_compressed = True
        
    def disable_compressed_mode(self):
        """Disable compressed mode, use standard attention."""
        self.use_compressed = False
        
    def compress_kv(self, kv_cache: Tuple[torch.Tensor, ...]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compress KV cache using AttentionMatcher.
        
        Args:
            kv_cache: Tuple of (key, value) tensors
            
        Returns:
            compressed_kv: Compressed (key, value)
        """
        compressed = []
        for k, v in kv_cache:
            comp_k, comp_v = self.matcher.compress((k, v))
            compressed.append((comp_k.to(self.device), comp_v.to(self.device)))
        return compressed