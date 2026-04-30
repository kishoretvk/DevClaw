# 🚀 Compressed Speculative Attention (CSA)

> A **functional proof-of-concept** for LLM inference optimization via KV cache compression and quantization

[![GitHub Repository](https://img.shields.io/badge/GitHub-DevClaw-blue)](https://github.com/kishoretvk/DevClaw)
[![Python](https://img.shields.io/badge/Python-3.12+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

**Status: Core components working (50x compression verified), 52/52 tests passing, GPU speedup verification pending.**

## 📊 **Current Status**

| Component | Status | Notes |
|-----------|--------|-------|
| **KV Compression Algorithm** | ✅ Verified | **50x reduction** (honest_results.json) |
| **FP8 Quantization** | ✅ Working | MSE: 0.001331 |
| **Custom Attention Layer** | ✅ Implemented | Model-agnostic, multi-model |
| **Multi-Model Support** | ✅ Working | GPT-2, LLaMA, OPT patching |
| **Self-Speculative Decoding** | ✅ Implemented | Batch verification ready |
| **DynamicHierarchicalCache** | ✅ Working | 50x compression integrated |
| **Generation with Compression** | ✅ Working | End-to-end CPU verified |
| **Production Tests** | ✅ Passing | 52/52 tests pass |
| **Speedup Verification** | ⏳ Pending GPU | Colab notebooks ready |

**What's Verified:**
- ✅ **KV cache compression: 50x reduction** (benchmarks/honest_results.json)
- ✅ FP8 quantization: Working with MSE 0.001331
- ✅ Custom attention: `CompressedAttention` layer implemented
- ✅ Multi-model: `AttentionPatcher` supports GPT-2, LLaMA, OPT
- ✅ Self-speculation: Rewritten with batch verification
- ✅ Production tests: 52/52 passing (2 skipped)
- ✅ Test notebooks: `cs_framework_test.ipynb`, `cs_benchmark.ipynb`
- ✅ Colab notebooks: `cs_framework_colab_gpu.ipynb`, `cs_benchmark_colab.ipynb`
- ✅ No fine-tuning: Pure Python framework

## 🎯 **Goals**

### Target:
- **5-10x speedup** via self-speculative decoding + compressed KV cache
- **Minimal memory**: 50x KV cache reduction (verified) + FP8 quantization
- **Training-free**: Works with any autoregressive model out of the box
- **Production ready**: Multi-model support, comprehensive benchmarks

### Current (April 29, 2026):
- ✅ **Compression**: **50x verified** (honest_results.json)
- ✅ **Framework**: Self-Speculative Decoding implemented
- ✅ **Tests**: 52/52 passing, production-ready CPU code
- ✅ **No fine-tuning**: Pure Python, works with pre-trained models
- ⏳ **Speedup**: Code ready, GPU verification via Colab pending

## 🚀 **Quick Start**

### Installation
```bash
git clone https://github.com/kishoretvk/DevClaw.git
cd DevClaw
pip install -e .
```

### Basic Usage
```python
from csa import CSAEngine

# Compression mode (verified working)
engine = CSAEngine(
    target_model="gpt2",
    compression_ratio=10,
    device="cpu"  # or "cuda" for GPU
)
text = engine.generate("The future of AI is", max_new_tokens=50)
print(text)
engine.cleanup()
```

### GPU Verification (Google Colab)
Open the Colab notebooks to verify 5-10x speedup on GPU:
- `notebooks/cs_framework_colab_gpu.ipynb` - Framework test on GPU
- `notebooks/cs_benchmark_colab.ipynb` - Comprehensive benchmarks

## 📚 **Benchmarks (Verified Results)**

### Compression:
```
COMPRESSION BENCHMARK:
  Ratio 5:   5.05x reduction (1.41 MB vs 7.10 MB)
  Ratio 10:  10.10x reduction (0.70 MB vs 7.10 MB)
  Ratio 20:  20.20x reduction (0.35 MB vs 7.10 MB)
  Ratio 50:  50.50x reduction (0.14 MB vs 7.10 MB)
```

### Quantization:
```
QUANTIZATION BENCHMARK:
  MSE: 0.001331
  Max error: 0.248759
  Quantized dtype: torch.float8_e4m3fn
```

### Tests:
```
52 passed, 2 skipped
- test_csa_comprehensive.py: 26 tests
- test_dynamic_cache.py: 8 tests
- test_multimodel.py: 10 tests
- test_production.py: 8 tests
```

## 📖 **Documentation & Tutorials**

- **[Getting Started](./tutorials/getting_started.md)** - 5-minute hands-on tutorial
- **[Integration Guide](./integration_guide.md)** - Ollama, vLLM, custom engines
- **[API Reference](./integration_guide.md#production-deployment)** - REST API docs
- **[Benchmarks](./benchmarks/honest_benchmark.py)** - Measurement tools
- **[Benchmark Results](./benchmarks/honest_results.json)** - Verified numbers
- **[Colab Notebook](./notebooks/cs_framework_colab_gpu.ipynb)** - GPU testing

## 🏗 **Architecture**

```
CSA Framework Architecture
══════════════════════════════════════

┌─────────────────────────────────────┐
│           CSA Engine                │
│     (Main Orchestration)            │
└─────────────┬───────────────────────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼──────┐  ┌─────▼──────┐
│ Attention   │  │TurboQuant  │
│ Matching    │  │ (FP8)      │
│ (Compress)  │  │ (Quantize) │
└──────┬──────┘  └─────┬──────┘
       │               │
       └───────┬───────┘
               │
       ┌───────┴───────┐
       │               │
┌──────▼──────┐  ┌─────▼──────┐
│    SSD      │  │   Dynamic   │
│  Engine     │  │   Cache     │
│(Speculate)  │  │ (Compress)  │
└─────────────┘  └─────────────┘
```

Data Flow: Prompt → Compress → Quantize → Speculate → Generate

## 🔧 **Key Features**

- 🚀 **Compression**: 5-50x KV cache reduction (VERIFIED)
- 💾 **Quantization**: FP8 with measurable error
- 🔌 **Multi-Model**: GPT-2, LLaMA, OPT support
- ⚡ **Speculative Decoding**: Self-speculative decoding implemented
- 🧪 **Tested**: 52/52 tests passing
- 🔧 **Modular**: All components are plug-and-play
- 🖥️ **CPU Ready**: Works on CPU, GPU speedup pending

## 🗺 **Roadmap**

### Completed (April 2026):
- [x] KV cache compression (50x verified)
- [x] FP8 quantization
- [x] Custom attention layer (CompressedAttention)
- [x] Multi-model patcher (AttentionPatcher)
- [x] Self-Speculative Decoding (SSD)
- [x] DynamicHierarchicalCache (50x compression)
- [x] 52/52 tests passing
- [x] Colab notebooks for GPU verification
- [x] Production-ready CPU code

### In Progress:
- [ ] GPU speedup verification (5-10x target)
- [ ] End-to-end benchmarks on GPU

## 🤝 **Contributing**

We welcome contributions! Please see our [contributing guidelines](./CONTRIBUTING.md).

## 📚 **Citation**

```bibtex
@misc{csa2026,
  title={Compressed Speculative Attention: A Training-Free Framework for LLM Inference Optimization},
  author={Krishna (TheExploreEcho)},
  year={2026},
  url={https://github.com/kishoretvk/DevClaw}
}
```

Based on draft v0.1 by Krishna (TheExploreEcho)

---

**Last Updated**: April 29, 2026  
**Status**: CPU complete (52/52 tests), GPU verification pending  
**Next Milestone**: 5-10x speedup verification on GPU (Colab)
