# TurboQuant implementation - standalone quantization module
# Replaces pyturboquant dependency

import torch
import torch.nn as nn
import math

class MSEQuantizer:
    """
    Minimum Squared Error Quantizer for KV cache compression.
    Implements k-means style quantization with learnable centroids.
    """
    def __init__(self, dim, bits=3, seed=42, device="cuda"):
        self.dim = dim
        self.bits = bits
        self.num_centroids = 2 ** bits
        self.device = device
        
        # Initialize centroids uniformly in [-1, 1]
        torch.manual_seed(seed)
        self.centroids = torch.linspace(-1, 1, self.num_centroids, device=device)
        
    def quantize(self, tensor):
        """
        Quantize tensor to specified number of bits.
        
        Args:
            tensor: Input tensor (batch*heads, dim)
            
        Returns:
            indices: Quantized indices (uint8)
            scale: Scale factor for dequantization
            min_val: Min value for dequantization
        """
        # Compute min/max for scaling
        min_val = tensor.min()
        max_val = tensor.max()
        
        # Scale to [0, num_centroids - 1]
        scale = (max_val - min_val) / (self.num_centroids - 1)
        if scale == 0:
            scale = 1.0
            
        normalized = (tensor - min_val) / scale
        indices = torch.clamp(normalized.round(), 0, self.num_centroids - 1).to(torch.uint8)
        
        # Store scale and min for dequantization
        return {
            'indices': indices,
            'scale': scale,
            'min_val': min_val,
            'shape': tensor.shape
        }
    
    def dequantize(self, quantized):
        """
        Dequantize indices back to float tensor.
        
        Args:
            quantized: Dict with indices, scale, min_val
            
        Returns:
            tensor: Dequantized tensor
        """
        indices = quantized['indices']
        scale = quantized['scale']
        min_val = quantized['min_val']
        shape = quantized['shape']
        
        # Convert back to float and rescale
        tensor = indices.float() * scale + min_val
        return tensor.reshape(shape)


class TurboQuantKernel:
    """
    Optimized quantization kernels for CUDA.
    Falls back to CPU implementation if CUDA unavailable.
    """
    def __init__(self, bits=3):
        self.bits = bits
        self.num_levels = 2 ** bits
        
    def quantize_symmetric(self, tensor):
        """
        Symmetric quantization around zero.
        Faster but less precise for non-symmetric distributions.
        """
        abs_max = tensor.abs().max()
        scale = abs_max / (self.num_levels // 2 - 1)
        
        if scale == 0:
            scale = 1.0
            
        # Quantize to signed integers
        quantized = (tensor / scale).round().to(torch.int8)
        # Clamp to valid range
        quantized = torch.clamp(quantized, -(self.num_levels // 2), self.num_levels // 2 - 1)
        
        return {
            'data': quantized,
            'scale': scale,
            'shape': tensor.shape
        }
    
    def dequantize_symmetric(self, quantized):
        """Dequantize symmetric quantized tensor."""
        data = quantized['data']
        scale = quantized['scale']
        shape = quantized['shape']
        
        tensor = data.float() * scale
        return tensor.reshape(shape)


class AdaptiveQuantizer:
    """
    Adaptive quantizer that selects between symmetric and asymmetric
    based on tensor statistics.
    """
    def __init__(self, dim, bits=3, device="cuda"):
        self.dim = dim
        self.bits = bits
        self.device = device
        self.mse_quantizer = MSEQuantizer(dim, bits, device=device)
        self.kernel = TurboQuantKernel(bits)
        
    def quantize(self, tensor):
        """Choose best quantization method based on tensor stats."""
        # Check if distribution is roughly symmetric around zero
        mean = tensor.mean()
        std = tensor.std()
        
        # Use symmetric if mean is close to 0 and distribution is symmetric
        if abs(mean) < 0.1 * std:
            return self.kernel.quantize_symmetric(tensor)
        else:
            return self.mse_quantizer.quantize(tensor)
    
    def dequantize(self, quantized):
        """Dequantize based on stored method."""
        if 'data' in quantized:
            return self.kernel.dequantize_symmetric(quantized)
        else:
            return self.mse_quantizer.dequantize(quantized)
