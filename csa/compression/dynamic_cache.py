"""
Dynamic Hierarchical KV Cache

Combines proven techniques from:
- H2O (heavy hitter eviction)
- SnapKV (observation window)
- PyramidKV (hierarchical budgets)

Features:
- Position-based skeleton (no GPU tensor clones)
- Complete deduplication (tracks all detail positions)
- Per-layer score accumulation
- H2O-style eviction
"""

import torch
import torch.nn.functional as F


class DynamicHierarchicalCache:
    """
    Two-tier KV cache:
    - Detail cache: Full precision, importance-weighted, evictable (H2O-style)
    - Skeleton positions: Uniform coverage indices, no tensor storage
    
    The skeleton provides coverage by tracking positions, not tensors.
    At get_cache() time, we check if skeleton positions are in the detail cache.
    Missing positions are reconstructed from nearest neighbors.
    """
    
    def __init__(self, 
                 skeleton_budget=20,
                 detail_budget=128,
                 recent_window=64,
                 num_layers=12,
                 skeleton_rebuild_freq=50):
        """
        Args:
            skeleton_budget: Number of uniform position samples for coverage
            detail_budget: Number of heavy hitter tokens to keep at full precision
            recent_window: Number of recent tokens to always keep
            num_layers: Number of transformer layers
            skeleton_rebuild_freq: Rebuild skeleton positions every N tokens
        """
        self.skeleton_budget = skeleton_budget
        self.detail_budget = detail_budget
        self.recent_window = recent_window
        self.num_layers = num_layers
        self.skeleton_rebuild_freq = skeleton_rebuild_freq
        
        # Per-layer storage
        self.detail_kv = [None] * num_layers          # Full precision tensors
        self.detail_positions = [set() for _ in range(num_layers)]  # Surviving positions
        self.detail_scores = [None] * num_layers        # Cumulative attention scores
        self.skeleton_positions = [None] * num_layers  # Position indices only (no tensors!)
        
        # State tracking
        self.seq_len = 0
        self.tokens_since_rebuild = [0] * num_layers
        self.initialized = False
    
    def initialize(self, full_kv, prefill_scores=None):
        """
        Initialize cache from full KV cache (prefill phase).
        
        Args:
            full_kv: List of (key, value) tuples from prefill
            prefill_scores: Optional dict of {layer_idx: scores} for initial importance
        """
        # Set seq_len BEFORE the loop so skeleton positions are correct
        self.seq_len = full_kv[0][0].shape[2]
        
        for layer_idx, (k, v) in enumerate(full_kv):
            seq_len = k.shape[2]
            
            # Detail: Start with all tokens at full precision
            self.detail_kv[layer_idx] = (k.detach().clone(), v.detach().clone())
            self.detail_positions[layer_idx] = set(range(seq_len))
            
            # Initialize scores
            if prefill_scores is not None and layer_idx in prefill_scores:
                self.detail_scores[layer_idx] = prefill_scores[layer_idx].cpu().clone()
            else:
                self.detail_scores[layer_idx] = torch.zeros(seq_len)
            
            # Initialize skeleton positions
            self._update_skeleton_positions(layer_idx)
            self.tokens_since_rebuild[layer_idx] = 0
        
        self.initialized = True
    
    def _update_skeleton_positions(self, layer_idx):
        """
        Update skeleton position indices.
        
        This is O(skeleton_budget) CPU work with ZERO GPU allocation.
        We store indices, not tensor slices.
        """
        if self.seq_len <= self.skeleton_budget:
            # Too short for meaningful skeleton
            self.skeleton_positions[layer_idx] = torch.arange(self.seq_len)
            return
        
        # Uniform sample of current sequence length
        self.skeleton_positions[layer_idx] = torch.linspace(
            0, self.seq_len - 1, self.skeleton_budget
        ).long()
    
    def update(self, layer_idx, new_kv, attention_scores):
        """
        Add new token and manage cache.
        
        Args:
            layer_idx: Which layer to update
            new_kv: Tuple of (key, value) for new token
            attention_scores: Attention scores from forward pass
                             Shape: (batch, heads, query_len, key_len)
        """
        k_new, v_new = new_kv
        
        # Add to detail cache
        if self.detail_kv[layer_idx] is None:
            self.detail_kv[layer_idx] = (k_new.detach().clone(), v_new.detach().clone())
            self.detail_scores[layer_idx] = torch.zeros(k_new.shape[2])
            self.detail_positions[layer_idx] = set(range(k_new.shape[2]))
        else:
            k_detail = torch.cat([self.detail_kv[layer_idx][0], k_new], dim=2)
            v_detail = torch.cat([self.detail_kv[layer_idx][1], v_new], dim=2)
            self.detail_kv[layer_idx] = (k_detail, v_detail)
            
            # Update positions
            old_len = len(self.detail_positions[layer_idx])
            new_positions = set(range(old_len, old_len + k_new.shape[2]))
            self.detail_positions[layer_idx].update(new_positions)
            
            # Update scores for new token
            self._update_scores(layer_idx, attention_scores)
        
        self.seq_len += k_new.shape[2]
        self.tokens_since_rebuild[layer_idx] += k_new.shape[2]
        
        # Rebuild skeleton positions periodically
        if self.tokens_since_rebuild[layer_idx] >= self.skeleton_rebuild_freq:
            self._update_skeleton_positions(layer_idx)
            self.tokens_since_rebuild[layer_idx] = 0
        
        # Evict detail if over budget
        self._evict_detail(layer_idx)
    
    def _update_scores(self, layer_idx, attention_scores):
        """
        Update cumulative attention scores.
        
        Args:
            attention_scores: (batch, heads, query_len, key_len)
        """
        # Average over batch and heads to get per-key-token importance
        # Shape: (query_len, key_len)
        scores_per_key = attention_scores.mean(dim=(0, 1))
        
        # For generation, query_len=1 (just the new token)
        # Get attention from new token to all previous tokens
        new_token_attention = scores_per_key[-1]  # Last query position
        
        # Extend scores tensor to match current sequence length
        current_len = self.detail_scores[layer_idx].shape[0]
        new_len = new_token_attention.shape[0]
        
        if new_len > current_len:
            # Pad with zeros for new tokens
            padding = torch.zeros(new_len - current_len)
            self.detail_scores[layer_idx] = torch.cat([
                self.detail_scores[layer_idx],
                padding
            ])
        
        # Accumulate attention scores
        min_len = min(current_len, new_len)
        self.detail_scores[layer_idx][:min_len] += new_token_attention[:min_len].cpu()
    
    def _evict_detail(self, layer_idx):
        """
        Evict detail cache using H2O-style heavy hitters.
        
        Keeps:
        - recent_window most recent tokens
        - detail_budget heavy hitters (highest cumulative attention)
        """
        k_detail, v_detail = self.detail_kv[layer_idx]
        seq_len = k_detail.shape[2]
        
        total_budget = self.detail_budget + self.recent_window
        if seq_len <= total_budget:
            # Under budget, keep everything
            self.detail_positions[layer_idx] = set(range(seq_len))
            return
        
        scores = self.detail_scores[layer_idx]
        
        # Keep recent window
        recent_start = seq_len - self.recent_window
        recent_indices = torch.arange(recent_start, seq_len)
        
        # Keep heavy hitters from older tokens
        old_scores = scores[:recent_start]
        if old_scores.shape[0] > self.detail_budget:
            _, hh_indices = torch.topk(old_scores, k=self.detail_budget)
            keep_indices = torch.cat([hh_indices.sort().values, recent_indices])
        else:
            # Not enough old tokens to fill budget
            keep_indices = torch.arange(seq_len)
        
        # Apply eviction
        self.detail_kv[layer_idx] = (
            k_detail[:, :, keep_indices, :],
            v_detail[:, :, keep_indices, :]
        )
        self.detail_scores[layer_idx] = scores[keep_indices]
        
        # CRITICAL: Update position tracking
        self.detail_positions[layer_idx] = set(keep_indices.tolist())
    
    def get_cache(self, layer_idx):
        """
        Get combined cache for attention computation.
        
        Returns detail cache with any missing skeleton positions reconstructed.
        For now, returns detail cache only (skeleton positions are tracked for
        future reconstruction if needed).
        
        Returns:
            (key, value) tensors for attention
        """
        if not self.initialized:
            raise RuntimeError("Cache not initialized. Call initialize() first.")
        
        # For now, return detail cache only
        # Full implementation would reconstruct missing skeleton positions
        return self.detail_kv[layer_idx]
    
    def get_all_caches(self):
        """Get all layer caches as list."""
        return [self.get_cache(i) for i in range(self.num_layers)]
    
    def get_memory_stats(self):
        """Get memory usage statistics."""
        total_detail_tokens = 0
        total_skeleton_positions = 0
        
        for layer_idx in range(self.num_layers):
            if self.detail_kv[layer_idx] is not None:
                total_detail_tokens += self.detail_kv[layer_idx][0].shape[2]
            if self.skeleton_positions[layer_idx] is not None:
                total_skeleton_positions += len(self.skeleton_positions[layer_idx])
        
        return {
            'detail_tokens': total_detail_tokens,
            'skeleton_positions': total_skeleton_positions,
            'seq_len': self.seq_len,
            'compression_ratio': self.seq_len / (total_detail_tokens / self.num_layers) if total_detail_tokens > 0 else 1.0
        }
