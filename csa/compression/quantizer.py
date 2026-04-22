# FP8 quantization for skeleton

import torch

class FP8Quantizer:
    def __init__(self, dtype=torch.float8_e4m3fn):
        self.dtype = dtype
    
    def quantize(self, tensor):
        """
        Quantize tensor to FP8.
        
        Args:
            tensor: Input tensor (FP16/FP32)
        
        Returns:
            quantized_tensor: FP8 quantized tensor
        """
        # Use torch's built-in FP8 quantization
        return tensor.to(dtype=self.dtype)
    
    def dequantize(self, quantized_tensor):
        """
        Dequantize FP8 tensor back to FP16.
        
        Args:
            quantized_tensor: FP8 tensor
        
        Returns:
            tensor: Dequantized tensor
        """
        return quantized_tensor.to(dtype=torch.float16)