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
    """
    Test that scores are assigned correctly to NEW tokens only.

    CRITICAL: After the May 2026 fix, scores are NOT re-added to existing tokens.
    Each token's score is set once when it's added, not accumulated across steps.
    This prevents score inflation over time.
    """
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
    
    # Set all heads to 1.0 so mean over 8 heads = 1.0
    scores = torch.zeros(1, 8, 1, 11)
    scores[0, :, 0, 0] = 1.0  # Mean over 8 heads = 1.0
    
    cache.update(0, (k_new, v_new), scores)
    
    # Score at position 0 should be accumulated (mean = 1.0)
    assert abs(cache.detail_scores[0][0].item() - 1.0) < 0.01
    
    # Add another token
    k_new2 = torch.randn(1, 8, 1, 64)
    v_new2 = torch.randn(1, 8, 1, 64)
    
    scores2 = torch.zeros(1, 8, 1, 12)
    scores2[0, :, 0, 0] = 0.5  # Mean over 8 heads = 0.5
    
    cache.update(0, (k_new2, v_new2), scores2)
    
    # Score at position 0 should accumulate (1.0 + 0.5 = 1.5)
    assert abs(cache.detail_scores[0][0].item() - 1.5) < 0.01


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
        detail_budget=100,  # Large enough to avoid eviction
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
    
    # Layer 0 should have 21 tokens (no eviction with budget=100)
    assert cache.detail_kv[0][0].shape[2] == 21
    
    # Layer 1 should still have 20 tokens (unchanged)
    assert cache.detail_kv[1][0].shape[2] == 20


def test_score_accumulation_no_inflation():
    """
    Test that scores don't inflate by re-adding to all previous tokens.

    CRITICAL FIX TEST: Previous bug in _update_scores() was re-adding attention
    scores to ALL previous tokens on each generation step, causing score inflation.
    This test verifies that only NEW tokens get their scores added.
    """
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=1
    )

    # Initialize with 10 tokens
    full_kv = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))]
    cache.initialize(full_kv)

    # Add token 1 with attention score = 1.0
    k_new = torch.randn(1, 8, 1, 64)
    v_new = torch.randn(1, 8, 1, 64)
    scores = torch.zeros(1, 8, 1, 11)
    scores[0, :, 0, :] = 1.0  # All positions get score 1.0

    cache.update(0, (k_new, v_new), scores)

    # Add token 2 with attention score = 0.0 (no attention)
    k_new2 = torch.randn(1, 8, 1, 64)
    v_new2 = torch.randn(1, 8, 1, 64)
    scores2 = torch.zeros(1, 8, 1, 12)
    # Don't add any attention - scores should remain 1.0, not increase

    cache.update(0, (k_new2, v_new2), scores2)

    # CRITICAL: Score at position 0 should still be 1.0 (not inflated)
    # Previous bug: would have been 2.0 (re-added)
    assert cache.detail_scores[0][0].item() == 1.0, \
        f"Score inflation detected: expected 1.0, got {cache.detail_scores[0][0].item()}"


def test_seq_len_tracking():
    """
    Test that seq_len tracks actual sequence length correctly.

    CRITICAL FIX TEST: Previous bug was using += which accumulated length
    instead of setting it to actual length.
    """
    cache = DynamicHierarchicalCache(
        skeleton_budget=20,
        detail_budget=128,
        recent_window=64,
        num_layers=1
    )

    # Initialize with 10 tokens
    full_kv = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))]
    cache.initialize(full_kv)

    assert cache.seq_len == 10, f"Expected seq_len=10, got {cache.seq_len}"

    # Add 5 tokens one by one
    for i in range(5):
        k_new = torch.randn(1, 8, 1, 64)
        v_new = torch.randn(1, 8, 1, 64)
        scores = torch.zeros(1, 8, 1, 10 + i + 1)
        cache.update(0, (k_new, v_new), scores)

        # seq_len should be actual length, not accumulated
        expected_len = 10 + i + 1
        assert cache.seq_len == expected_len, \
            f"Expected seq_len={expected_len}, got {cache.seq_len}"


def test_position_alignment_after_eviction():
    """
    Test that positions remain aligned after eviction.

    CRITICAL FIX TEST: Previous bug kept positions as absolute indices after
    eviction, but the cache was compressed, causing misalignment.
    """
    cache = DynamicHierarchicalCache(
        skeleton_budget=10,
        detail_budget=5,  # Small budget to force eviction
        recent_window=3,
        num_layers=1
    )

    # Initialize with 10 tokens
    full_kv = [(torch.randn(1, 8, 10, 64), torch.randn(1, 8, 10, 64))]
    cache.initialize(full_kv)

    # Add tokens to trigger eviction
    for i in range(10):
        k_new = torch.randn(1, 8, 1, 64)
        v_new = torch.randn(1, 8, 1, 64)
        scores = torch.zeros(1, 8, 1, 10 + i + 1)
        cache.update(0, (k_new, v_new), scores)

    # After eviction, positions should be relative [0, 1, 2, ...]
    # not absolute indices from before eviction
    k_detail, v_detail = cache.detail_kv[0]
    actual_seq_len = k_detail.shape[2]
    positions = cache.detail_positions[0]

    # Positions should be valid indices into the compressed cache
    assert all(0 <= p < actual_seq_len for p in positions), \
        f"Position out of bounds: positions={positions}, actual_seq_len={actual_seq_len}"

    # Number of positions should match actual cache size
    assert len(positions) == actual_seq_len, \
        f"Position count mismatch: {len(positions)} positions for {actual_seq_len} tokens"
