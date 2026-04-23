"""
Tests for DynamicHierarchicalCache
"""

import pytest
import torch
from csa.compression.dynamic_cache import DynamicHierarchicalCache


def test_cache_initialization():
    """Test cache initialization."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=12
    )
    
    # Create mock full KV cache (12 layers, 1000 tokens)
    full_kv = []
    for _ in range(12):
        k = torch.randn(1, 8, 1000, 64, dtype=torch.float16)
        v = torch.randn(1, 8, 1000, 64, dtype=torch.float16)
        full_kv.append((k, v))
    
    cache.initialize(full_kv)
    
    assert cache.initialized
    assert cache.seq_len == 1000
    assert len(cache.detail_positions[0]) == 1000  # All positions kept initially
    assert cache.skeleton_positions[0] is not None
    assert len(cache.skeleton_positions[0]) == 20  # Skeleton budget


def test_position_based_skeleton():
    """Test that skeleton stores positions, not tensors."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=1
    )
    
    full_kv = [(torch.randn(1, 8, 1000, 64), torch.randn(1, 8, 1000, 64))]
    cache.initialize(full_kv)
    
    # Skeleton should be position indices only
    skeleton = cache.skeleton_positions[0]
    assert isinstance(skeleton, torch.Tensor)
    assert skeleton.dtype == torch.long
    assert skeleton.shape[0] == 20
    
    # Positions should be uniform sample
    assert skeleton[0].item() == 0
    assert skeleton[-1].item() == 999


def test_detail_eviction():
    """Test H2O-style detail eviction."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=10,  # Small for testing
        recent_window=5,
        num_layers=1
    )
    
    # Initialize with 20 tokens
    full_kv = [(torch.randn(1, 8, 20, 64), torch.randn(1, 8, 20, 64))]
    cache.initialize(full_kv)
    
    # Add 10 more tokens (total 30, budget=15)
    for i in range(10):
        k_new = torch.randn(1, 8, 1, 64)
        v_new = torch.randn(1, 8, 1, 64)
        
        # Mock attention scores (make some positions more important)
        scores = torch.zeros(1, 8, 1, 20 + i + 1)
        scores[0, 0, 0, 0] = 1.0  # Position 0 is important
        scores[0, 0, 0, 5] = 0.8  # Position 5 is important
        
        cache.update(0, (k_new, v_new), scores)
    
    # Check that detail cache was evicted
    k_detail, v_detail = cache.detail_kv[0]
    total_budget = cache.detail_budget + cache.recent_window
    assert k_detail.shape[2] <= total_budget
    
    # Check position tracking
    assert len(cache.detail_positions[0]) <= total_budget


def test_score_accumulation():
    """Test that scores accumulate correctly."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=1
    )
    
    full_kv = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))]
    cache.initialize(full_kv)
    
    # Add token with attention scores
    k_new = torch.randn(1, 8, 1, 64)
    v_new = torch.randn(1, 8, 1, 64)
    
    scores = torch.zeros(1, 8, 1, 11)
    scores[0, 0, 0, 0] = 1.0
    
    cache.update(0, (k_new, v_new), scores)
    
    # Score at position 0 should be accumulated
    assert cache.detail_scores[0][0] == 1.0
    
    # Add another token
    k_new2 = torch.randn(1, 8, 1, 64)
    v_new2 = torch.randn(1, 8, 1, 64)
    
    scores2 = torch.zeros(1, 8, 1, 12)
    scores2[0, 0, 0, 0] = 0.5
    
    cache.update(0, (k_new2, v_new2), scores2)
    
    # Score at position 0 should be accumulated (1.0 + 0.5)
    assert cache.detail_scores[0][0] == 1.5


def test_memory_stats():
    """Test memory statistics."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=10,
        recent_window=5,
        num_layers=2
    )
    
    full_kv = [
        (torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64)),
        (torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64))
    ]
    cache.initialize(full_kv)
    
    stats = cache.get_memory_stats()
    
    assert stats['seq_len'] == 100
    assert stats['detail_tokens'] == 200  # 100 per layer × 2 layers
    assert stats['skeleton_positions'] == 40  # 20 per layer × 2 layers
    assert stats['compression_ratio'] == 1.0  # No eviction yet


def test_get_cache():
    """Test getting cache for attention."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=1
    )
    
    full_kv = [(torch.randn(1, 8, 100, 64), torch.randn(1, 8, 100, 64))]
    cache.initialize(full_kv)
    
    k, v = cache.get_cache(0)
    
    assert k.shape == (1, 8, 100, 64)
    assert v.shape == (1, 8, 100, 64)


def test_uninitialized_error():
    """Test that get_cache raises error if not initialized."""
    cache = DynamicHierarchicalCache(num_layers=1)
    
    with pytest.raises(RuntimeError, match="not initialized"):
        cache.get_cache(0)


def test_per_layer_independence():
    """Test that each layer has independent cache."""
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=10,
        recent_window=5,
        num_layers=2
    )
    
    full_kv = [
        (torch.randn(1, 8, 20, 64), torch.randn(1, 8, 20, 64)),
        (torch.randn(1, 8, 20, 64), torch.randn(1, 8, 20, 64))
    ]
    cache.initialize(full_kv)
    
    # Add tokens to layer 0 only
    k_new = torch.randn(1, 8, 1, 64)
    v_new = torch.randn(1, 8, 1, 64)
    scores = torch.zeros(1, 8, 1, 21)
    
    cache.update(0, (k_new, v_new), scores)
    
    # Layer 0 should have 21 tokens
    assert cache.detail_kv[0][0].shape[2] == 21
    
    # Layer 1 should still have 20 tokens
    assert cache.detail_kv[1][0].shape[2] == 20
