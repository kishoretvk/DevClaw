# PENDING TASKS - Production Readiness

**Date**: April 28, 2026
**Goal**: Make CS Framework production-ready with 50x compression + 5-10x speedup

---

## Task 1: Verify 5-10x Speedup on GPU
**Owner**: Developer-6 (Benchmark & Verification)
**Manager**: Manager Agent
**QA**: Product Owner/QA

### Acceptance Criteria (QA):
- [ ] Run `notebooks/cs_benchmark.ipynb` on GPU
- [ ] Measure speedup: standard GPT-2 vs CS Framework
- [ ] Speedup must be ≥5x (target 5-10x)
- [ ] Document results in `benchmarks/speedup_results.json`
- [ ] If speedup <5x, optimize and re-test

### Deliverable (Executable Code):
- `benchmarks/speedup_test.py` - executable benchmark script
- `benchmarks/speedup_results.json` - results with measured speedup

### QA Verification:
1. Run: `cd D:\git\DevClaw && python benchmarks/speedup_test.py`
2. Check output JSON has `"speedup": value >= 5`
3. Sign-off: QA writes `team_workspace/qa_speedup_signoff.md`

---

## Task 2: Test on LLaMA and OPT Models
**Owner**: Developer-5 (Debug Generation Pipeline)
**Manager**: Manager Agent
**QA**: Product Owner/QA

### Acceptance Criteria (QA):
- [ ] Test CSAEngine with LLaMA-2-7B (if GPU available)
- [ ] Test CSAEngine with OPT-1.3B (if GPU available)
- [ ] Verify compression works on all models (50x)
- [ ] Verify generation works on all models
- [ ] Document results in `benchmarks/multi_model_results.json`

### Deliverable (Executable Code):
- `tests/test_llama.py` - executable test for LLaMA
- `tests/test_opt.py` - executable test for OPT
- `benchmarks/multi_model_results.json` - results

### QA Verification:
1. Run: `cd D:\git\DevClaw && python tests/test_llama.py && python tests/test_opt.py`
2. Check all tests pass
3. Sign-off: QA writes `team_workspace/qa_multimodel_signoff.md`

---

## Task 3: Production Integration Test
**Owner**: Developer-4 (Fix Bugs)
**Manager**: Manager Agent
**QA**: Product Owner/QA

### Acceptance Criteria (QA):
- [ ] Create `tests/test_production.py` with end-to-end test:
  - Load GPT-2, compress 50x, generate 100 tokens
  - Measure actual speedup vs standard generation
  - Verify output quality (not garbage)
  - Measure memory usage before/after compression
- [ ] All tests pass
- [ ] Code coverage >80%

### Deliverable (Executable Code):
- `tests/test_production.py` - executable production test

### QA Verification:
1. Run: `cd D:\git\DevClaw && python tests/test_production.py`
2. Check exit code = 0 (all tests pass)
3. Check memory usage reduction ≥50x
4. Sign-off: QA writes `team_workspace/qa_production_signoff.md`

---

## Task 4: Manager Final Verification & Sign-off
**Owner**: Manager Agent
**QA**: Product Owner/QA (final review)

### Acceptance Criteria:
- [ ] Review ALL code changed:
  - `csa/core/engine.py`
  - `csa/speculation/ssd.py`
  - `csa/compression/dynamic_cache.py`
- [ ] Run generation test on CPU: PASSED
- [ ] Verify 50x compression: PASSED (honest_results.json)
- [ ] Verify speedup ≥5x: PENDING (need GPU)
- [ ] All tests pass
- [ ] No fine-tuning required: CONFIRMED

### Deliverable:
- `team_workspace/manager_signoff.md` - final sign-off document

### Final Sign-off (Product Owner/QA):
- [ ] 50x compression verified: YES
- [ ] 5-10x speedup verified: YES/NO (GPU needed)
- [ ] No fine-tuning: YES
- [ ] Notebooks created: YES
- [ ] Production ready: YES/NO

---

## Summary for User

| Task | Owner | QA Criteria | Status |
|------|-------|-----------------|--------|
| Task 1: Speedup verification | Dev-6 | speedup ≥5x | 🔄 Pending GPU |
| Task 2: LLaMA/OPT testing | Dev-5 | all models work | ❌ Not started |
| Task 3: Production test | Dev-4 | tests pass, coverage>80% | ❌ Not started |
| Task 4: Manager sign-off | Manager | all criteria met | 🔄 In progress |

**Blocker**: GPU environment timeout prevents Task 1 completion.

**Path to Production**:
1. Complete Tasks 1-3 (developers)
2. Manager verifies all code
3. QA signs off on each task
4. Push final commits to git
5. Release as v1.0

---

**Files to Create (Executable Code)**:
- `benchmarks/speedup_test.py`
- `tests/test_llama.py`
- `tests/test_opt.py`
- `tests/test_production.py`

**Files for QA Sign-off**:
- `team_workspace/qa_speedup_signoff.md`
- `team_workspace/qa_multimodel_signoff.md`
- `team_workspace/qa_production_signoff.md`
- `team_workspace/manager_signoff.md`
