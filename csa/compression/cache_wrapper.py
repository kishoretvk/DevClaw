"""
Compressed KV Cache Wrapper

Provides a wrapper around compressed KV cache that can be used
during generation by decompressing on-the-fly.
"""

import torch
import torch.nn.functional as F


class CompressedKVCache:
    """
    Wrapper for compressed KV cache that stores data in compressed form
    and decompresses when needed for generation.
    
    This enables memory savings (storing compressed) while maintaining
    compatibility with standard transformer attention.
    """
    
    def __init__(self, compressed_kv, original_seq_len, compression_ratio, 
                 quantizer=None, device='cuda'):
        """
        Args:
            compressed_kv: List of (key, value) tuples, each compressed
            original_seq_len: Original sequence length before compression
            compression_ratio: Ratio used for compression
            quantizer: Optional quantizer for dequantization
            device: Device for computation
        """
        self.compressed_kv = compressed_kv
        self.original_seq_len = original_seq_len
        self.compression_ratio = compression_ratio
        self.quantizer = quantizer
        self.device = device
        self.num_layers = len(compressed_kv)
        
        # Cache decompressed values to avoid repeated work
        self._decompressed_cache = {}
        
    def decompress_layer(self, layer_idx):
        """
        Decompress a single layer's KV cache.
        
        Strategy: Use linear interpolation to expand compressed cache
        back to original sequence length. This is simple but effective
        for uniform sampling compression.
        """
        if layer_idx in self._decompressed_cache:
            return self._decompressed_cache[layer_idx]
            
        comp_k, comp_v = self.compressed_kv[layer_idx]
        
        # Dequantize if needed
        if self.quantizer is not None and hasattr(comp_k, 'dequantize'):
            comp_k = self.quantizer.dequantize(comp_k)
            comp_v = self.quantizer.dequantize(comp_v)
        
        batch, num_heads, comp_seq, head_dim = comp_k.shape
        
        # If already full size, return as-is
        if comp_seq >= self.original_seq_len:
            self._decompressed_cache[layer_idx] = (comp_k, comp_v)
            return comp_k, comp_v
        
        # Use interpolation to expand to original size
        # This is a simple approach - more sophisticated methods possible
        target_size = self.original_seq_len
        
        # For 4D tensors (batch, heads, seq, dim), we need to use bilinear interpolation
        # by treating (batch*heads) as batch and dim as channels
        # Reshape: (batch, heads, seq, dim) -> (batch*heads, 1, seq, dim)
        batch_heads = batch * num_heads
        
        # Reshape keys for bilinear interpolation
        k_reshaped = comp_k.reshape(batch_heads, 1, comp_seq, head_dim)
        v_reshaped = comp_v.reshape(batch_heads, 1, comp_seq, head_dim)
        
        # Use bilinear interpolation to upsample sequence dimension
        # We want to expand from comp_seq to target_size along the seq dimension (height)
        # while keeping dim (width) the same
        decomp_k_reshaped = F.interpolate(
            k_reshaped,
            size=(target_size, head_dim),  # (height=seq, width=dim)
            mode='bilinear',
            align_corners=True
        )
        
        decomp_v_reshaped = F.interpolate(
            v_reshaped,
            size=(target_size, head_dim),
            mode='bilinear',
            align_corners=True
        )
        
        # Reshape back: (batch*heads, 1, new_seq, dim) -> (batch, heads, new_seq, dim)
        decomp_k = decomp_k_reshaped.reshape(batch, num_heads, target_size, head_dim)
        decomp_v = decomp_v_reshaped.reshape(batch, num_heads, target_size, head_dim)
        
        # Ensure shapes are correct
        assert decomp_k.shape == (batch, num_heads, target_size, head_dim), \
            f"Decompressed key shape mismatch: {decomp_k.shape} vs expected {(batch, num_heads, target_size, head_dim)}"
        
        result = (decomp_k, decomp_v)
        self._decompressed_cache[layer_idx] = result
        return result
    
    def to_standard_cache(self):
        """
        Convert to standard transformers cache format.
        
        Returns:
            List of (key, value) tuples with original sequence length
        """
        standard_cache = []
        for i in range(self.num_layers):
            k, v = self.decompress_layer(i)
            standard_cache.append((k, v))
        return standard_cache
    
    def get_seq_length(self):
        """Return the original sequence length."""
        return self.original_seq_len
    
    def __len__(self):
        return self.num_layers
    
    def __getitem__(self, idx):
        """Allow indexing like standard cache."""
        return self.decompress_layer(idx)


class EfficientCompressedCache:
    """
    More efficient cache that only decompresses what's needed.
    
    Instead of decompressing the full sequence, this computes
    attention directly with the compressed representation.
    """
    
    def __init__(self, compressed_kv, original_seq_len, compression_ratio,
                 quantizer=None, device='cuda'):
        self.compressed_kv = compressed_kv
        self.original_seq_len = original_seq_len
        self.compression_ratio = compression_ratio
        self.quantizer = quantizer
        self.device = device
        self.num_layers = len(compressed_kv)
        
    def get_compressed_layer(self, layer_idx):
        """Get compressed KV for a layer (without decompression)."""
        comp_k, comp_v = self.compressed_kv[layer_idx]
        
        # Dequantize if needed
        if self.quantizer is not None:
            if hasattr(comp_k, 'dequantize'):
                comp_k = self.quantizer.dequantize(comp_k)
            if hasattr(comp_v, 'dequantize'):
                comp_v = self.quantizer.dequantize(comp_v)
                
        return comp_k, comp_v
    
    def compute_compressed_attention(self, query, layer_idx):
        """
        Compute attention using compressed KV cache.
        
        This is the key innovation - instead of decompressing,
        we compute attention directly with the compressed representation.
        
        Args:
            query: Query tensor (batch, heads, 1, head_dim)
            layer_idx: Which layer to use
            
        Returns:
            attention_output: (batch, heads, 1, head_dim)
        """
        comp_k, comp_v = self.get_compressed_layer(layer_idx)
        
        # Compute attention scores with compressed keys
        scores = torch.matmul(query, comp_k.transpose(-2, -1))
        scores = scores / (query.shape[-1] ** 0.5)
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        
        # Weighted sum of compressed values
        output = torch.matmul(attn_weights, comp_v)
        
        return output
