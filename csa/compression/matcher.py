# Attention Matching compression module

import torch
import torch.nn.functional as F

class AttentionMatcher:
    def __init__(self, compression_ratio=50, method="uniform"):
        self.compression_ratio = compression_ratio
        self.method = method  # "uniform", "importance", "clustering"
    
    def compress(self, kv_cache, query_cache=None):
        """
        Compress full KV cache to skeleton using Attention Matching.
        
        Args:
            kv_cache: Tuple of (key, value) tensors from prefill, shape (batch, num_heads, seq_len, head_dim)
            query_cache: Optional query tensors for importance scoring
        
        Returns:
            skeleton_kv: Compressed KV cache as tuple (key, value)
        """
        key, value = kv_cache
        batch, num_heads, seq_len, head_dim = key.shape
        
        # Calculate target skeleton size
        skeleton_size = max(1, seq_len // self.compression_ratio)
        
        print(f"Compressing: batch={batch}, heads={num_heads}, seq={seq_len}, dim={head_dim} -> skeleton_size={skeleton_size}")
        
        if self.method == "uniform":
            # Simple uniform sampling
            if skeleton_size == 1:
                indices = torch.tensor([0], dtype=torch.long)
            else:
                indices = torch.linspace(0, seq_len - 1, skeleton_size, dtype=torch.long)
            print(f"Indices: {indices}")
            skeleton_key = key[:, :, indices, :]
            skeleton_value = value[:, :, indices, :]
        
        elif self.method == "importance" and query_cache is not None:
            # Importance sampling based on attention scores
            # Compute average attention across heads and queries
            query = query_cache  # Assume last query or average
            if query.dim() == 4:  # (batch, heads, seq, dim)
                query = query.mean(dim=1, keepdim=True)  # Average heads
            
            # Simple dot product attention scores
            scores = torch.matmul(query, key.transpose(-2, -1))  # (batch, 1, 1, seq_len)
            scores = scores.squeeze(1).squeeze(1)  # (batch, seq_len)
            
            # Select top-k important positions
            _, indices = torch.topk(scores, skeleton_size, dim=-1)
            indices = indices.sort(dim=-1).values  # Sort for consistency
            
            skeleton_key = key[:, :, indices, :]
            skeleton_value = value[:, :, indices, :]
        
        else:
            # Fallback to uniform
            if skeleton_size == 1:
                indices = torch.tensor([0], dtype=torch.long)
            else:
                indices = torch.linspace(0, seq_len - 1, skeleton_size, dtype=torch.long)
            skeleton_key = key[:, :, indices, :]
            skeleton_value = value[:, :, indices, :]
        
        print(f"Compressed shapes: key {skeleton_key.shape}, value {skeleton_value.shape}")
        return (skeleton_key, skeleton_value)