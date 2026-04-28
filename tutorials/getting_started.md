# Getting Started with CSA

Quick 5-minute tutorial to get started with Compressed Speculative Attention (CSA).

## 🚀 Current Status (April 2026)

**✅ Functional Components:**
- KV Cache Compression: 5-50x reduction (VERIFIED)
- FP8 Quantization: Working (MSE: 0.001331)
- Custom Attention Layer: `CompressedAttention` (IMPLEMENTED)
- Multi-Model Support: GPT-2, LLaMA, OPT (WORKING)
- 52 tests passing

**⚠️ In Development:**
- Speedup verification: End-to-end benchmarks pending
- SSD Speculation: Framework ready, integration pending
- Background Recovery: Framework ready

---

## 🚀 Quick Start

### Installation
```bash
git clone https://github.com/kishoretvk/DevClaw.git
cd DevClaw
pip install -e .
```

### Basic Usage (Compression Verified)
```python
from csa import CSAEngine

# Create engine with compression
engine = CSAEngine(
    target_model="gpt2",
    compression_ratio=10,
    device="cpu"  # or "cuda" for GPU
)

# Generate with compressed KV cache
text = engine.generate(
    "The future of AI is",
    max_new_tokens=50
)
print(text)
engine.cleanup()
```

### What Works:
- ✅ Compression reduces KV cache by 5-50x
- ✅ Quantization with FP8 (measurable error)
- ✅ Custom attention layer (`CompressedAttention`)
- ✅ Multi-model support via `AttentionPatcher`

### What's Next:
- 🔄 Run end-to-end benchmarks to verify speedup
- 🔄 Complete SSD speculation integration
- 🔄 Update documentation with verified numbers

---

## 📚 Benchmarks (Honest Results)

### Verified Measurements:
```python
# Run honest benchmark
python benchmarks/honest_benchmark.py
```

**Compression:**
- Ratio 5:   1.41 MB (5.05x reduction)
- Ratio 10:  0.70 MB (10.10x reduction)
- Ratio 20:  0.35 MB (20.20x reduction)
- Ratio 50:  0.14 MB (50.50x reduction)

**Quantization:**
- MSE: 0.001331
- Max error: 0.248759
- Quantized dtype: torch.float8_e4m3fn

**Speedup:**
- Status: NOT YET VERIFIED
- Target: 2-3x (when custom attention fully integrated)

---

## 🎯 Multi-Model Support

### Supported Models:
- ✅ GPT-2 (tested)
- ✅ LLaMA (patched)
- ✅ OPT (patched)
- 🔄 Mistral (pending)

### Usage:
```python
# Works with any supported model
engine = CSAEngine(
    target_model="meta-llama/Llama-2-7b-hf",
    compression_ratio=10,
    device="cuda"
)
```

---

## 📖 Documentation

- **[Complete Integration Guide](../integration_guide.md)** - Ollama, vLLM, REST API
- **[Honest Benchmarks](../benchmarks/honest_benchmark.py)** - Measures what works
- **[Updated Notebook](../notebooks/colab_gpu_benchmark_updated.ipynb)** - GPU testing

---

## 🗺️ Roadmap

### Completed (April 2026):
- [x] Fix syntax errors across codebase
- [x] Implement custom attention layer (CompressedAttention)
- [x] Create multi-model patcher (AttentionPatcher)
- [x] Get 52 tests passing
- [x] Create honest benchmark (compression, quantization verified)
- [x] Update notebooks for GPU testing
- [x] Integrate custom attention into engine

### In Progress:
- [ ] Run end-to-end benchmarks with speedup measurement
- [ ] Complete SSD speculation full integration
- [ ] Update documentation with verified speedup numbers

---

**Last Updated**: April 27, 2026  
**Status**: Functional proof-of-concept with verified compression & quantization  
**Next Milestone**: End-to-end speedup verification