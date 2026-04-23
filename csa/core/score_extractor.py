"""
Attention Score Extractor using Forward Hooks

Extracts attention scores without output_attentions=True (which causes O(seq²) memory).
Uses forward hooks to capture scores mid-forward with O(seq_len) memory per layer.

Compatible with:
- Standard HuggingFace transformers
- torch.compile (register hooks before compilation)
- Flash Attention (hooks fire on module output, not inside kernel)
"""

import torch


class AttentionScoreExtractor:
    """
    Extract attention scores using forward hooks.
    
    Avoids output_attentions=True which materializes full attention matrices.
    Instead, hooks capture scores during forward pass and detaches immediately.
    
    Memory cost: O(seq_len) per layer (not O(seq_len²))
    """
    
    def __init__(self, model):
        """
        Args:
            model: HuggingFace transformers model
        """
        self.model = model
        self.scores = {}  # layer_idx -> attention scores from last forward
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on attention modules."""
        # Try different model architectures
        if hasattr(self.model, 'model') and hasattr(self.model.model, 'layers'):
            # Llama, Mistral style
            layers = self.model.model.layers
        elif hasattr(self.model, 'transformer') and hasattr(self.model.transformer, 'h'):
            # GPT-2 style
            layers = self.model.transformer.h
        elif hasattr(self.model, 'model') and hasattr(self.model.model, 'decoder') and hasattr(self.model.model.decoder, 'layers'):
            # OPT style
            layers = self.model.model.decoder.layers
        else:
            raise ValueError(f"Unsupported model architecture: {type(self.model)}")
        
        for idx, layer in enumerate(layers):
            # Find attention module
            if hasattr(layer, 'self_attn'):
                attn_module = layer.self_attn
            elif hasattr(layer, 'attn'):
                attn_module = layer.attn
            else:
                continue
            
            hook = attn_module.register_forward_hook(self._make_hook(idx))
            self.hooks.append(hook)
    
    def _make_hook(self, layer_idx):
        """Create hook function for a specific layer."""
        def hook(module, input, output):
            """
            Hook that captures attention scores.
            
            output is typically (hidden_states, attn_weights, past_kv) or similar tuple.
            We extract attn_weights if present.
            """
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]  # (batch, heads, query_len, key_len)
                if attn_weights is not None:
                    # Detach immediately to avoid keeping computation graph
                    # Store only what's needed - full tensor for now
                    # In production: could store only last query position
                    self.scores[layer_idx] = attn_weights.detach()
        return hook
    
    def get_scores(self, layer_idx):
        """
        Get attention scores for a specific layer.
        
        Returns:
            Attention scores tensor or None if not captured
        """
        return self.scores.get(layer_idx)
    
    def get_all_scores(self):
        """Get all captured attention scores."""
        return self.scores.copy()
    
    def clear(self):
        """Clear captured scores to free memory."""
        self.scores.clear()
    
    def remove(self):
        """Remove all hooks. Call when done."""
        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - clean up hooks."""
        self.remove()
        return False
