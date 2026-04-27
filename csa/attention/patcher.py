"""
Attention Patcher for multi-model support.

This module provides utilities to patch different model architectures
(GPT-2, Llama, Qwen, Mistral) to use CompressedAttention.
"""

import torch
from typing import List, Tuple, Optional
from ..attention.compressed_attention import CompressedAttention


class AttentionPatcher:
    """Patches model attention to use CompressedAttention."""
    
    @staticmethod
    def detect_model_type(model) -> str:
        """
        Detect if model is GPT, Llama, Qwen, or Mistral.
        
        Returns:
            Model type string: 'gpt2', 'llama', 'qwen', 'mistral', or 'unknown'
        """
        # Check model config for type hints
        if hasattr(model.config, 'model_type'):
            model_type = model.config.model_type.lower()
            if 'gpt2' in model_type or 'gpt-2' in model_type:
                return 'gpt2'
            elif 'llama' in model_type:
                return 'llama'
            elif 'qwen' in model_type:
                return 'qwen'
            elif 'mistral' in model_type:
                return 'mistral'
        
        # Fallback: check model class name
        model_class = model.__class__.__name__.lower()
        if 'gpt2' in model_class or 'gpt-2' in model_class:
            return 'gpt2'
        elif 'llama' in model_class:
            return 'llama'
        elif 'qwen' in model_class:
            return 'qwen'
        elif 'mistral' in model_class:
            return 'mistral'
        
        # Check architecture
        if hasattr(model, 'transformer'):
            if hasattr(model.transformer, 'h'):  # GPT-2 style
                return 'gpt2'
            elif hasattr(model.transformer, 'layers'):  # Llama/Qwen style
                # Need to distinguish between Llama and Qwen
                if hasattr(model.config, 'num_key_value_heads'):  # Qwen2
                    return 'qwen'
                else:
                    return 'llama'
        
        return 'unknown'
    
    @staticmethod
    def patch_model(model, compression_ratio: int = 10, device: str = "auto") -> List[Tuple]:
        """
        Patch all attention layers in model to use CompressedAttention.
        
        Args:
            model: The transformer model to patch
            compression_ratio: Compression ratio for KV cache
            device: Device to run on
            
        Returns:
            List of patched layers for potential restoration
        """
        model_type = AttentionPatcher.detect_model_type(model)
        
        if model_type == 'gpt2':
            return AttentionPatcher._patch_gpt2(model, compression_ratio, device)
        elif model_type == 'llama':
            return AttentionPatcher._patch_llama(model, compression_ratio, device)
        elif model_type == 'qwen':
            return AttentionPatcher._patch_qwen(model, compression_ratio, device)
        elif model_type == 'mistral':
            return AttentionPatcher._patch_mistral(model, compression_ratio, device)
        else:
            raise ValueError(f"Unsupported model type: {model_type}. "
                             f"Supported: GPT-2, Llama, Qwen, Mistral")
    
    @staticmethod
    def _patch_gpt2(model, compression_ratio: int, device: str) -> List[Tuple]:
        """Patch GPT-2 attention layers."""
        patched_layers = []
        
        # GPT-2 structure: model.transformer.h[].attn
        for block in model.transformer.h:
            original_attn = block.attn
            head_dim = model.config.n_embd // model.config.n_head
            
            # Create compressed attention
            compressed_attn = CompressedAttention(
                original_attention=original_attn,
                head_dim=head_dim,
                num_heads=model.config.n_head,
                compression_ratio=compression_ratio,
                device=device
            )
            
            # Replace attention
            block.attn = compressed_attn
            patched_layers.append((block, 'attn', original_attn))
        
        return patched_layers
    
    @staticmethod
    def _patch_llama(model, compression_ratio: int, device: str) -> List[Tuple]:
        """Patch Llama attention layers (handles RoPE, GQA)."""
        patched_layers = []
        
        # Llama structure: model.model.layers[].self_attn
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        
        for layer in model.model.layers:
            original_attn = layer.self_attn
            
            # Create compressed attention
            compressed_attn = CompressedAttention(
                original_attention=original_attn,
                head_dim=head_dim,
                num_heads=num_heads,
                compression_ratio=compression_ratio,
                device=device
            )
            
            # Replace attention
            layer.self_attn = compressed_attn
            patched_layers.append((layer, 'self_attn', original_attn))
        
        return patched_layers
    
    @staticmethod
    def _patch_qwen(model, compression_ratio: int, device: str) -> List[Tuple]:
        """Patch Qwen attention layers."""
        patched_layers = []
        
        # Qwen2 structure: model.model.layers[].self_attn
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        
        for layer in model.model.layers:
            original_attn = layer.self_attn
            
            # Create compressed attention
            compressed_attn = CompressedAttention(
                original_attention=original_attn,
                head_dim=head_dim,
                num_heads=num_heads,
                compression_ratio=compression_ratio,
                device=device
            )
            
            # Replace attention
            layer.self_attn = compressed_attn
            patched_layers.append((layer, 'self_attn', original_attn))
        
        return patched_layers
    
    @staticmethod
    def _patch_mistral(model, compression_ratio: int, device: str) -> List[Tuple]:
        """Patch Mistral attention layers (handles sliding window)."""
        patched_layers = []
        
        # Mistral structure: model.model.layers[].self_attn
        num_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // num_heads
        
        for layer in model.model.layers:
            original_attn = layer.self_attn
            
            # Create compressed attention
            compressed_attn = CompressedAttention(
                original_attention=original_attn,
                head_dim=head_dim,
                num_heads=num_heads,
                compression_ratio=compression_ratio,
                device=device
            )
            
            # Replace attention
            layer.self_attn = compressed_attn
            patched_layers.append((layer, 'self_attn', original_attn))
        
        return patched_layers
    
    @staticmethod
    def restore_model(patched_layers: List[Tuple]):
        """
        Restore original attention layers.
        
        Args:
            patched_layers: List of (parent, attr_name, original_module) tuples
        """
        for parent, attr_name, original_module in patched_layers:
            setattr(parent, attr_name, original_module)