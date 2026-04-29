# QA Acceptance Criteria - CS (Speculative Speculative Decoding) Framework

## Project Goals
- **50x KV Cache Compression** (measured reduction in memory footprint)
- **5-10x Inference Speedup** (measured tokens/second improvement vs baseline)
- **No Fine-Tuning Required** (works with pre-trained models out of the box)
- **Clean Python Framework** (minimal dependencies beyond torch + transformers)

---

## 1. KV Cache Compression (Target: 50x)

### Acceptance Criteria
- [ ] Compression ratio of 50x achieves measurable ~50x reduction in KV cache memory
- [ ] `AttentionMatcher` with `compression_ratio=50` produces correct shapes
- [ ] Compressed cache works with `CompressedAttention` without decompression
- [ ] FP8 quantization (`FP8Quantizer`) further reduces memory footprint
- [ ] `DynamicHierarchicalCache` provides adaptive compression
- [ ] Memory measurement: `torch.cuda.memory_allocated()` shows ~50x reduction

### Test Method
```python
# Measure KV cache memory with and without compression
# Baseline: standard KV cache for seq_len=2048, 32 heads, head_dim=128
# Compressed: AttentionMatcher(compression_ratio=50) applied
# Expected: ~50x reduction in memory usage
```

### Current State (from code review)
- `AttentionMatcher` in `csa/compression/matcher.py` supports compression_ratio up to 50
- Uniform sampling method implemented, importance sampling partially implemented
- `CompressedKVCache` and `EfficientCompressedCache` in `csa/compression/cache_wrapper.py`
- **Gap**: Need to verify actual memory reduction at 50x ratio with CompressedAttention

---

## 2. Inference Speedup (Target: 5-10x)

### Acceptance Criteria
- [ ] Baseline generation speed measured (tokens/second without CSA)
- [ ] CSA generation speed measured (tokens/second with compression + speculation)
- [ ] Speedup factor = CSA_speed / baseline_speed ≥ 5x
- [ ] Target range: 5-10x speedup
- [ ] `SSDSpeculator` integrated and functional with `CompressedAttention`
- [ ] CUDA streams utilized for parallel speculation (if GPU available)

### Test Method
```python
# Benchmark on same hardware (GPU preferred)
# Baseline: model.generate() with standard attention
# CSA: CSAEngine.generate() with compression + SSD speculation
# Measure: tokens/second over 100+ generated tokens
```

### Current State (from code review)
- `SSDSpeculator` exists in `csa/speculation/ssd.py` but integration "pending"
- `CSAEngine` has `_full_generate()` method for SSD mode but may not be fully wired
- `CompressedAttention` implemented in `csa/attention/compressed_attention.py`
- `AttentionPatcher` patches models in `csa/attention/patcher.py`
- **Gap**: SSD speculation not fully integrated with CompressedAttention pipeline

---

## 3. No Fine-Tuning Required

### Acceptance Criteria
- [ ] Works with pre-trained models from HuggingFace hub without modification
- [ ] Supported models: GPT-2, LLaMA, OPT (per README)
- [ ] No training/lora/adapter steps needed before inference
- [ ] `CSAEngine` initializes with any compatible model path

### Test Method
```python
from csa.core import CSAEngine
engine = CSAEngine(target_model_path="gpt2")  # No fine-tuning
result = engine.generate("Hello", max_new_tokens=50)
```

### Current State
- `CSAEngine` uses `AutoModelForCausalLM.from_pretrained()` - ✓ no fine-tuning
- `AttentionPatcher` supports GPT-2, LLaMA, OPT - ✓
- **Status**: Requirement appears met

---

## 4. Framework Quality

### Acceptance Criteria
- [ ] All existing 52 tests continue to pass
- [ ] New tests added for 50x compression verification
- [ ] New tests added for 5x+ speedup verification
- [ ] Code is well-documented with docstrings
- [ ] Notebooks run without errors (`notebooks/cs_framework_test.ipynb`)
- [ ] Benchmark notebook demonstrates speedup (`notebooks/cs_benchmark.ipynb`)

### Current State
- 52 tests passing (per README)
- Test files: `test_csa_comprehensive.py`, `test_attention.py`, etc.
- Notebooks exist but may need updates for new features

---

## 5. Notebook Requirements

### Required Notebooks
1. **`notebooks/cs_framework_test.ipynb`** - Main test notebook
   - Loads model with CSAEngine
   - Demonstrates 50x compression
   - Measures speedup
   - Runs end-to-end without errors

2. **`notebooks/cs_benchmark.ipynb`** - Benchmarking notebook
   - Compares baseline vs CSA speed
   - Plots compression vs speedup tradeoff
   - Reports final speedup factor

### Acceptance
- [ ] Both notebooks run successfully on GPU
- [ ] Output shows ≥50x compression
- [ ] Output shows ≥5x speedup
- [ ] No manual intervention required (run all cells)

---

## 6. Performance Baselines (for verification)

| Metric | Baseline (no CSA) | Target with CSA | Status |
|--------|-------------------|-----------------|--------|
| KV Cache Size (2048 seq) | ~1GB | ~20MB (50x less) | TODO |
| Generation Speed (tokens/s) | ~20 tok/s | ~100-200 tok/s | TODO |
| Compression Ratio | 1x | 50x | Current: up to 50x in code |
| Speedup Factor | 1x | 5-10x | Current: unverified |

---

## Review Checklist (updated as team delivers)

- [ ] Architect design document reviewed
- [ ] Programmer implementation reviewed
- [ ] Compression verified at 50x
- [ ] Speedup verified at 5-10x
- [ ] Notebooks created and tested
- [ ] All tests pass
- [ ] No fine-tuning required (verified)
- [ ] Framework ready for release

---

**Last Updated**: 2026-04-28
**QA Owner**: Product Owner/QA Agent
**Next Review**: After Programmer delivers initial implementation
