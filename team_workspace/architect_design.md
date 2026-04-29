# CSA Architecture Design — Path to 50x Compression + 5-10x Speedup

**Architect**: Krishna (TheExploreEcho)  
**Date**: April 28, 2026  
**Status**: ACTIVE DESIGN — Implementation Required  

---

## 1. EXECUTIVE SUMMARY

The CSA framework has working compression (5-50x verified) and FP8 quantization, but the SSD (Speculative Speculative Decoding) module is critically misdesigned and the compression algorithm is naive. Achieving 50x compression AND 5-10x speedup requires three major architectural corrections:

1. **Fix SSD speculation** — current implementation is slower than no speculation (serial token-by-token verification)
2. **Replace uniform compression with importance-aware hierarchical compression** — combine skeleton + heavy-hitters + TurboQuant
3. **Integrate DynamicHierarchicalCache into the main engine** — currently unused dead code

---

## 2. CURRENT STATE — CRITICAL GAPS ANALYSIS

### 2.1 Compression (Target: 50x, Current: 5-50x with quality loss)

| Component | File | Status | Problem |
|-----------|------|--------|---------|
| AttentionMatcher | `csa/compression/matcher.py` | **NAIVE** | Uniform sampling only — ignores token importance |
| DynamicHierarchicalCache | `csa/compression/dynamic_cache.py` | **ORPHANED** | Fully implemented but NOT used by engine |
| FP8Quantizer | `csa/compression/quantizer.py` | Working | Only quantizes skeleton, not integrated with hierarchical cache |
| TurboQuant (3-bit) | `csa/quantization/turboquant.py` | Working | Standalone, not wired into generation |

**Root Cause**: `DynamicHierarchicalCache` implements the correct two-tier architecture (skeleton + details with H2O eviction) but is never called by `CSAEngine`. The engine uses the naive `AttentionMatcher` instead.

### 2.2 Speculative Decoding (Target: 5-10x, Current: 2-3x or WORSE)

| Component | File | Status | Problem |
|-----------|------|--------|---------|
| SSDSpeculator | `csa/speculation/ssd.py` | **BROKEN** | Serial token-by-token verification — SLOWER than baseline |
| CUDA Stream "parallelism" | `ssd.py` | **FAKE** | Uses ThreadPoolExecutor (CPU threads) — not real GPU parallelism |
| Speculation cache | `ssd.py` | Inefficient | Keyed on `(num_accepted, rejection_pos)` — wrong abstraction |
| Draft model init | `ssd.py` | Fragile | Tries vLLM first, falls back to transformers — vLLM not needed |
| Acceptance math | `ssd.py` | **MISSING** | No probability comparison — just argmax equality check |
| Engine integration | `csa/core/engine.py` | Incomplete | `_full_generate()` runs speculate but doesn't use results properly |

**Why Current SSD is SLOWER than no speculation**:
```python
# CURRENT (BROKEN): Verifies one token at a time — THIS IS SERIAL!
for i, token in enumerate(speculated_tokens):
    input_tensor = torch.tensor([[token]], device=device)
    outputs = target_model(input_tensor, past_key_values=skeleton_kv)  # Forward per token!
    predicted = torch.argmax(outputs.logits[:, -1, :]).item()
    if predicted == token:
        accepted.append(token)
    else:
        break
```

**What proper speculative decoding looks like**:
```python
# CORRECT: Verify ALL K tokens in ONE forward pass
inputs = torch.cat([current_token] + speculated_tokens)  # K+1 tokens
outputs = target_model(inputs, past_key_values=skeleton_kv)  # SINGLE forward!
# Compare draft distribution vs target distribution for each position
# Accept contiguous prefix where distributions match
```

### 2.3 CompressedAttention Integration

| Component | File | Status | Problem |
|-----------|------|--------|---------|
| CompressedAttention | `csa/attention/compressed_attention.py` | Partial | Works with compressed KV but ignores quantization errors |
| AttentionPatcher | `csa/attention/patcher.py` | Working | Correctly patches models, but patched layers not fully utilized |
| Engine usage | `csa/core/engine.py` | Incomplete | Passes `past_key_values=skeleton_kv` but generation doesn't leverage patched attention |

---

## 3. MATHEMATICAL FOUNDATION

### 3.1 Speculative Decoding Acceptance Rate Math

For speculative decoding to achieve speedup, we need:

```
Expected Speedup = (1 + α × K) / (1 + β × K)

Where:
  α = acceptance rate (fraction of speculated tokens accepted)
  K = number of speculative tokens per round (speculate_k)
  β = normalized cost of (draft forward + verification) / target forward

For 5-10x speedup with K=5:
  Need α ≥ 0.80 (80% acceptance rate)
  Need β ≤ 0.15 (verification must be ~7x cheaper than full forward)
```

**Key Insight**: Acceptance rate α depends on draft/target model alignment:
- Draft = smaller version of target → α ≈ 0.70-0.90
- Draft = unrelated small model → α ≈ 0.30-0.50 (too low!)
- **Solution**: Use the compressed target model itself as the draft (self-speculation)

### 3.2 Self-Speculation (No Separate Draft Model)

Instead of a separate draft model, use the target model with:
- **Fewer layers** (e.g., use only first 6 layers of 12-layer model)
- **Higher compression** (more aggressive KV compression for draft)
- **Lower precision** (INT8/FP8 for draft forward)

This gives α ≈ 0.75-0.85 with NO extra model and minimal overhead (β ≈ 0.3).

### 3.3 Compression Ratio Math

```
Total Compression = Skeleton Compression × Detail Retention × Quantization Bits

Target: 50x total compression

Option A (Current naive):
  Uniform 10x + FP8 2x = 20x (insufficient)

Option B (Hierarchical):
  Skeleton 20x (uniform coverage) 
  × Detail retention 0.4 (keep 40% of tokens at full precision via H2O)
  × TurboQuant 3-bit (32-bit → 3-bit = 10.7x)
  = 20 × 0.4 × 10.7 ≈ 85x effective compression

Option C (Optimal):
  Skeleton 50x (aggressive uniform)
  × Detail retention 0.3 (keep 30% via importance sampling)
  × TurboQuant 3-bit = 50 × 0.3 × 10.7 ≈ 160x effective
  But only 30% detail retention may hurt quality — need tuning.

RECOMMENDED:
  Skeleton 25x + Detail 0.5 (H2O) + TurboQuant 3-bit = 25 × 0.5 × 10.7 ≈ 134x
  Then tune down to achieve 50x with quality:
  Skeleton 20x + Detail 0.6 + TurboQuant 3-bit = 20 × 0.6 × 10.7 ≈ 128x → tune
  Practical: Skeleton 30x + Detail 0.5 + TurboQuant 4-bit (8x) = 30 × 0.5 × 8 = 120x → tune
  
  Final config for 50x: Skeleton 25x + Detail 0.4 + TurboQuant 3-bit → prune to 50x
```

---

## 4. ARCHITECTURE REDESIGN

### 4.1 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        CSA ENGINE (csa/core/engine.py)                    │
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌─────────────────────────────┐  │
│  │   PREFILL    │───▶│  COMPRESS    │───▶│    QUANTIZE (TurboQuant)   │  │
│  │  (Target     │    │  (Dynamic    │    │    3-bit on skeleton +     │  │
│  │   Model)     │    │   Hierarchical│    │    details                │  │
│  └──────────────┘    │   Cache)     │    └──────────────┬────────────┘  │
│                      └──────┬───────┘                   │                │
│                             │                           │                │
│                      ┌──────▼───────┐                   │                │
│                      │ SKELETON     │                   │                │
│                      │ (uniform     │                   │                │
│                      │  25x)        │                   │                │
│                      └──────┬───────┘                   │                │
│                             │                           │                │
│                      ┌──────▼───────┐    ┌──────────────▼─────────────┐  │
│                      │ DETAIL       │    │  COMPRESSED KV CACHE       │  │
│                      │ (H2O heavy   │───▶│  (skeleton + details      │  │
│                      │  hitters,    │    │   combined, quantized)     │  │
│                      │  0.4 keep)   │    │                           │  │
│                      └──────────────┘    └──────────────┬────────────┘  │
│                                                          │                │
│  ┌───────────────────────────────────────────────────────▼────────────┐  │
│  │              GENERATION LOOP (with Self-Speculation)              │  │
│  │                                                                    │  │
│  │  ┌──────────────────┐    ┌──────────────────┐    ┌─────────────┐ │  │
│  │  │ DRAFT (self-sp.) │    │ VERIFY (target)  │    │  ACCEPT     │ │  │
│  │  │ Fewer layers or  │───▶│ One forward pass│───▶│  Contiguous │ │  │
│  │  │ higher comp.     │    │ for all K tokens│    │  prefix     │ │  │
│  │  └──────────────────┘    └──────────────────┘    └─────────────┘ │  │
│  │                                                                    │  │
│  │  ┌─────────────────────────────────────────────────────────────┐  │  │
│  │  │ BACKGROUND RECOVERY (non-blocking, residual correction)     │  │  │
│  │  │ Recovers fine details lost during compression               │  │  │
│  │  └─────────────────────────────────────────────────────────────┘  │  │
│  └────────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘

Data Flow:
  Prompt → Prefill → Compress (Hierarchical) → Quantize (3-bit) → Generate (Speculative)
```

### 4.2 Compression Pipeline Redesign (For 50x)

**Current** (engine.py `_simple_generate`):
```python
# NAIVE: Uniform compression + FP8
for layer_kv in full_kv:
    comp_kv = self.matcher.compress(layer_kv)  # Uniform sampling only!
    comp_kv = (self.quantizer.quantize(comp_kv[0]), ...)  # FP8 only
```

**Redesigned** (use DynamicHierarchicalCache):
```python
# 1. Initialize hierarchical cache from prefill
self.dynamic_cache = DynamicHierarchicalCache(
    skeleton_budget=total_tokens // 25,    # 25x skeleton compression
    detail_budget=128,                      # H2O: keep 128 heavy hitters
    recent_window=64,                       # Always keep last 64 tokens
    num_layers=num_layers
)
self.dynamic_cache.initialize(full_kv, prefill_scores)

# 2. During generation: update cache with new tokens
# (importance scores from CompressedAttention hooks)
self.dynamic_cache.update(layer_idx, new_kv, attention_scores)

# 3. Get combined cache for generation
kv_for_generation = self.dynamic_cache.get_all_caches()  # skeleton + details

# 4. Apply 3-bit TurboQuant to combined cache
quantized_kv = []
for k, v in kv_for_generation:
    qk = turbo_quant.quantize(k.view(-1, dim))
    qv = turbo_quant.quantize(v.view(-1, dim))
    quantized_kv.append((qk, qv))
```

### 4.3 SSD Speculation Redesign (For 5-10x Speedup)

**Current Problem**: Serial verification, one token per forward pass.

**Redesigned Self-Speculation Algorithm**:

```python
class SelfSpeculativeDecoder:
    """
    Self-speculation: uses target model with reduced computation for drafting.
    
    Key insight: Instead of a separate draft model, use the target model
    with fewer layers or higher compression for fast drafting.
    """
    
    def __init__(self, target_model, num_draft_layers=6, speculate_k=5):
        self.target_model = target_model
        self.num_draft_layers = num_draft_layers  # Use subset of layers
        self.speculate_k = speculate_k
        
        # Extract draft sub-model (first N layers)
        self.draft_layers = self._extract_draft_layers(num_draft_layers)
        
    def draft(self, input_ids, past_kv, speculate_k):
        """Generate K speculative tokens using draft sub-model (fast)."""
        # Use only first N layers for drafting — much faster
        draft_tokens = []
        current_input = input_ids
        
        for step in range(speculate_k):
            # Forward through draft layers only
            draft_logits = self._forward_draft(current_input, past_kv)
            next_token = torch.argmax(draft_logits[:, -1, :], dim=-1)
            draft_tokens.append(next_token.item())
            current_input = torch.cat([current_input, next_token.unsqueeze(0)], dim=1)
            
        return draft_tokens
    
    def verify(self, input_ids, draft_tokens, past_kv):
        """
        Verify ALL draft tokens in ONE target forward pass.
        
        This is the KEY to speedup — batch verification!
        """
        # Concatenate: current + all draft tokens
        verify_input = torch.cat([
            input_ids, 
            torch.tensor(draft_tokens, device=input_ids.device).unsqueeze(0)
        ], dim=1)
        
        # Single forward pass through FULL target model
        with torch.no_grad():
            outputs = self.target_model(verify_input, past_key_values=past_kv)
            target_logits = outputs.logits  # Shape: (1, K+1, vocab_size)
        
        # Accept contiguous prefix where draft matches target
        accepted = []
        for i, draft_token in enumerate(draft_tokens):
            target_token = torch.argmax(target_logits[:, i, :], dim=-1).item()
            if draft_token == target_token:
                accepted.append(draft_token)
            else:
                break  # Reject this and all subsequent
        
        return accepted
```

### 4.4 Acceptance Rate: The Probability Matching Approach

The equal-argmax check (`draft_token == target_token`) is too strict. Proper speculative decoding uses probability distribution matching:

```python
def verify_with_distribution_matching(self, target_logits, draft_logits, draft_tokens):
    """
    Accept draft tokens using distribution matching (not just argmax equality).
    
    For each position i:
      p_target = softmax(target_logits[i])
      p_draft = softmax(draft_logits[i])
      
      Accept if: p_draft(draft_token[i]) <= p_target(draft_token[i])
      (This ensures the draft doesn't assign higher probability than target)
      
      If rejected: sample from (p_target - p_draft)_+ and continue
    """
    accepted = []
    
    for i, token in enumerate(draft_tokens):
        p_target = F.softmax(target_logits[:, i, :], dim=-1)
        p_draft = F.softmax(draft_logits[:, i, :], dim=-1)
        
        # Probability of this token under each distribution
        prob_target = p_target[0, token].item()
        prob_draft = p_draft[0, token].item()
        
        if prob_draft <= prob_target * 1.1:  # Small margin for numerical stability
            accepted.append(token)
        else:
            # Rejected — optionally sample a new token from the difference
            break
    
    return accepted
```

---

## 5. IMPLEMENTATION PLAN (for Programmer)

### Phase 1: Fix Compression Pipeline (50x target) — PRIORITY 1

**Files to modify**:
- `csa/core/engine.py` — integrate `DynamicHierarchicalCache`
- `csa/compression/matcher.py` — add importance-based compression
- `csa/quantization/cache.py` — wire TurboQuant into generation

**Specific tasks**:
1. In `CSAEngine.__init__()`: Initialize `DynamicHierarchicalCache` instead of just `AttentionMatcher`
2. In `_simple_generate()` / `_full_generate()`: Use `dynamic_cache.initialize()` after prefill
3. Add importance scoring: use `AttentionScoreExtractor` hooks during prefill to get real attention scores
4. Replace uniform sampling in `AttentionMatcher` with attention-score-based selection
5. Apply `TurboQuantCache` (3-bit) to the combined skeleton+details output
6. Pass quantized cache to `CompressedAttention` for generation

**Expected result**: 50x compression with minimal quality loss.

### Phase 2: Fix SSD Speculation (5-10x target) — PRIORITY 2

**Files to modify**:
- `csa/speculation/ssd.py` — COMPLETE REWRITE
- `csa/core/engine.py` — fix `_full_generate()` to use batch verification

**Specific tasks**:
1. **REWRITE SSDSpeculator** with self-speculation:
   - Remove vLLM dependency
   - Implement `SelfSpeculativeDecoder` (fewer layers for draft)
   - Implement batch verification (ONE forward pass for ALL K tokens)
   - Implement distribution matching for acceptance (not just argmax equality)

2. **Fix `_full_generate()` in engine.py**:
   - Call draft generation (K tokens)
   - Single target forward pass on all K+1 tokens
   - Accept contiguous prefix
   - Append accepted tokens to output
   - Repeat until max_new_tokens

3. **Remove fake CUDA stream "parallelism"**:
   - ThreadPoolExecutor does NOT give GPU parallelism
   - Real GPU parallelism = batch verification in one forward pass
   - If async needed: use `torch.cuda.Stream` properly with `torch.cuda.synchronize()`

**Expected result**: 5-10x speedup (from current 2-3x or worse).

### Phase 3: Optimize CompressedAttention — PRIORITY 3

**Files to modify**:
- `csa/attention/compressed_attention.py` — handle quantized KV properly

**Specific tasks**:
1. Handle dequantization on-the-fly in attention computation
2. Use Flash Attention if available (`torch.nn.functional.scaled_dot_product_attention`)
3. Minimize host-device transfers
4. Cache attention patterns for reuse

### Phase 4: Background Recovery — PRIORITY 4

**Files to modify**:
- `csa/recovery/background.py` — implement actual residual recovery

**Specific tasks**:
1. Compute residual: `residual = full_kv - decompressed_kv`
2. Store residuals in quantized form
3. Apply residuals during generation for quality improvement
4. Ensure truly non-blocking (use CUDA streams properly)

---

## 6. SPECIFIC CODE CRITIQUES

### 6.1 `csa/speculation/ssd.py` — CRITICAL ISSUES

**Issue 1: Fake async "parallelism" (lines 182-203)**
```python
# CURRENT (WRONG):
def _speculate_cuda_streams(self, current_tokens, predicted_outcomes):
    def speculate_single(outcome, stream_idx):
        stream = self.cuda_streams[stream_idx % len(self.cuda_streams)]
        with torch.cuda.stream(stream):
            return outcome, self.speculate_with_cache(current_tokens, outcome)
    
    futures = []
    for i, outcome in enumerate(predicted_outcomes):
        future = self.executor.submit(speculate_single, outcome, i)  # CPU threads!
        futures.append(future)
```
**Problem**: `ThreadPoolExecutor` + `torch.cuda.stream()` does NOT give parallel GPU execution. Python threads release GIL for I/O but not for GPU computation.  
**Fix**: Remove ThreadPoolExecutor. Use batch verification instead.

**Issue 2: Serial verification (lines 242-272)**
```python
# CURRENT (SLOW):
for i, token in enumerate(speculated_tokens):
    input_tensor = torch.tensor([[token]], device=device)
    outputs = target_model(input_tensor, past_key_values=skeleton_kv)  # Forward per token!
```
**Problem**: One forward pass per token — this is slower than no speculation!  
**Fix**: One forward pass for all K tokens, then compare.

**Issue 3: Wrong acceptance criterion (line 266)**
```python
if predicted == token:  # Argmax equality only
```
**Problem**: Rejects when draft's argmax != target's argmax, even if both assign high probability.  
**Fix**: Use distribution matching (draft_prob <= target_prob for the token).

### 6.2 `csa/core/engine.py` — ISSUES

**Issue 1: DynamicHierarchicalCache not used**
The engine imports `DynamicHierarchicalCache` but never uses it. It uses `AttentionMatcher` (uniform sampling) instead.

**Issue 2: `_full_generate()` doesn't properly use speculation results**
```python
# Current: verifies but then generates with standard model.generate()
# instead of using accepted speculative tokens directly
```
**Fix**: After verification, append accepted tokens and continue from there.

**Issue 3: CompressedAttention not properly leveraged**
The engine patches the model with `CompressedAttention` but then passes `past_key_values=skeleton_kv` to standard `model.generate()`. The standard generation doesn't know how to use compressed KV properly.  
**Fix**: Custom generation loop that uses `CompressedAttention` directly.

### 6.3 `csa/compression/matcher.py` — ISSUES

**Issue**: Only uniform sampling, no importance-based compression.  
**Fix**: Add `method="importance"` that uses attention scores. Wire up `AttentionScoreExtractor` to get real scores during prefill.

---

## 7. PERFORMANCE BOTTLENECK ANALYSIS

### Current Profiling Results (from profiling.py + README):
```
Compress:    XXXms (once per prompt — acceptable)
Forward:     XXXms per token (MAIN BOTTLENECK)
Verify:      K × Forward per token (CURRENT — HORRIBLE)
Generate:    Linear in sequence length (need speculation)
```

### After Redesign:
```
Prefill:     O(L) — one-time cost (acceptable)
Draft:       O(L × N_draft / N_total) — fast (fewer layers)
Verify:      O(L) — ONE forward for all K tokens (BATCHED!)
Generate:    O(L / (α × K)) — α×K tokens per iteration (5-10x fewer iterations)
```

---

## 8. MEMORY BANDWIDTH OPTIMIZATION

### Techniques to implement:

1. **KV Cache in FP8/INT8**: Reduces memory bandwidth by 2-4x
2. **Flash Attention**: If input is long, use `F.scaled_dot_product_attention()` 
3. **CUDAGraph**: Capture the generation loop in a CUDA graph for fixed-length speculation
4. **Memory Pool**: Reuse tensors instead of allocating new ones (partially in `background.py`)
5. **Pinned Memory**: For CPU-GPU transfers if needed

### Current Memory Waste:
- `CompressedKVCache._decompressed_cache` stores decompressed layers — wastes memory
- `SSDSpeculator.speculation_cache` stores token lists — minimal impact but wrong abstraction
- `BackgroundRecovery.memory_pool` implemented but not used for actual recovery

---

## 9. TARGET METRICS (After Implementation)

| Metric | Current | Target | After Redesign (Estimated) |
|--------|---------|--------|----------------------------|
| KV Compression | 5-50x (variable quality) | 50x | 50x (hierarchical + 3-bit) |
| Inference Speedup | 2-3x (claimed, not verified) | 5-10x | 7-9x (self-speculation) |
| Acceptance Rate | ~30-50% (argmax only) | 75-85% | 80%+ (distribution matching) |
| Memory per token | ~8KB (FP16) | <0.5KB | ~0.3KB (3-bit quantized) |
| End-to-end latency | Baseline | 5-10x faster | 7-9x faster |

---

## 10. COORDINATION WITH PROGRAMMER

### Instructions for Programmer:

1. **DO NOT modify `ssd.py` incrementally** — it needs a complete rewrite. Create a new `SelfSpeculativeDecoder` class.

2. **Phase 1 first** (compression) — get 50x compression working before attempting speedup.

3. **Phase 2** (speculation) — use the self-speculation approach (no separate draft model).

4. **Test after each phase**:
   - Compression: verify 50x reduction with <5% quality loss
   - Speculation: verify 5-10x speedup on GPT-2 (easy test model)

5. **Key files you'll modify**:
   - `csa/core/engine.py` (main integration)
   - `csa/speculation/ssd.py` (COMPLETE REWRITE)
   - `csa/compression/matcher.py` (add importance sampling)

6. **Do NOT touch**:
   - `csa/attention/patcher.py` (working correctly)
   - `csa/quantization/turboquant.py` (working correctly)
   - `csa/compression/dynamic_cache.py` (just need to USE it, not modify it)

---

## 11. VERIFICATION PLAN

After Programmer completes implementation:

1. **Compression verification**:
   ```python
   from csa import CSAEngine
   engine = CSAEngine(target_model="gpt2", compression_ratio=50)
   # Check: skeleton_kv[0][0].shape[2] == original_length / 50
   ```

2. **Speedup verification**:
   ```python
   # Time with speculation vs without
   engine_sp = CSAEngine(target_model="gpt2", use_speculation=True)
   engine_no = CSAEngine(target_model="gpt2", use_speculation=False)
   # Measure tokens/second for both
   ```

3. **Acceptance rate verification**:
   ```python
   # During generation, log accepted/rejected tokens
   # Target: >80% acceptance rate
   ```

---

## 12. ARCHITECTURAL DECISIONS LOG

| Date | Decision | Rationale |
|------|----------|-----------|
| 2026-04-28 | Use self-speculation (not separate draft model) | Eliminates draft model overhead, better acceptance rates |
| 2026-04-28 | Batch verification (not serial) | Critical for 5-10x speedup — serial is slower |
| 2026-04-28 | Integrate DynamicHierarchicalCache | Already implemented, just needs wiring |
| 2026-04-28 | Remove vLLM dependency from SSD | Not needed, adds complexity, fragile imports |
| 2026-04-28 | Distribution matching for acceptance | Better than argmax equality (higher acceptance) |

---

**END OF DESIGN DOCUMENT**

Architect signature: Krishna (TheExploreEcho)  
Next action: Programmer implements Phase 1 + Phase 2  
Expected completion: Phase 1 (1 session), Phase 2 (1 session)
