# TurboQuant quantization module

import torch
from pyturboquant.core import MSEQuantizer

class TurboQuantCache:
    def __init__(self, dim, bits=3, seed=42, device="cuda"):
        self.bits = bits
        self.k_quantizer = MSEQuantizer(dim=dim, bits=bits, seed=seed, device=device)
        self.v_quantizer = MSEQuantizer(dim=dim, bits=bits, seed=seed, device=device)
        self.cache = []  # List of (quantized_k, quantized_v)
    
    def append(self, kv):
        """
        Quantize and append new KV to cache.
        
        Args:
            kv: Tuple of (key, value) tensors, shape (batch, heads, 1, dim)
        """
        k, v = kv
        # Flatten batch and heads for quantization
        k_flat = k.view(-1, k.shape[-1])  # (batch*heads, dim)
        v_flat = v.view(-1, v.shape[-1])
        
        quantized_k = self.k_quantizer.quantize(k_flat)
        quantized_v = self.v_quantizer.quantize(v_flat)
        
        self.cache.append((quantized_k, quantized_v))
    
    def get(self, idx):
        """
        Get dequantized KV at index.
        
        Args:
            idx: Index in cache
        
        Returns:
            kv: Dequantized (key, value) tensors
        """
        quantized_k, quantized_v = self.cache[idx]
        k = self.k_quantizer.dequantize(quantized_k)
        v = self.v_quantizer.dequantize(quantized_v)
        return (k, v)
    
    def apply_residual(self, idx, residual_k, residual_v):
        """
        Apply residual correction to cached KV.
        
        Args:
            idx: Index to correct
            residual_k, residual_v: Residual tensors to add
        """
        # For simplicity, add residual to dequantized and re-quantize
        k, v = self.get(idx)
        k_corrected = k + residual_k
        v_corrected = v + residual_v
        
        # Re-quantize
        quantized_k = self.k_quantizer.quantize(k_corrected)
        quantized_v = self.v_quantizer.quantize(v_corrected)
        
        self.cache[idx] = (quantized_k, quantized_v)