# 🚀 Compressed Speculative Attention (CSA)

> A **functional proof-of-concept** for LLM inference optimization via KV cache compression and quantization

[![GitHub Repository](https://img.shields.io/badge/GitHub-DevClaw-blue)](https://github.com/kishoretvk/DevClaw)
[![Python](https://img.shields.io/badge/Python-3.12+-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-red)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-yellow)](./LICENSE)

**⚠️ HONEST STATUS: Core components are working (compression, quantization, multi-model support), but **speedup claims require verification** through end-to-end benchmarking with the integrated custom attention layer.**

## 📊 **Current Status**

| Component | Status | Notes |
|-----------|--------|-------|
| **KV Compression Algorithm** | ✅ Working | 5-50x reduction verified |
| **FP8 Quantization** | ✅ Working | MSE: 0.001331, measurable error |
| **Custom Attention Layer** | ✅ Implemented | Model-agnostic, multi-model support |
| **Multi-Model Support** | ✅ Working | GPT-2, LLaMA, OPT patching |
| **Generation with Compression** | ✅ Integrated | CompressedAttention passes KV directly |
| **SSD Speculation** | ⚠️ Framework Ready | Structure complete, integration pending |
| **Background Recovery** | ⚠️ Framework Ready | Thread structure in place |
| **Speedup Verification** | ❌ NOT YET VERIFIED | Requires end-to-end benchmarks |

**What's Verified:**
- ✅ KV cache compression: 5-50x reduction (measured)
- ✅ FP8 quantization: Working with MSE 0.001331
- ✅ Custom attention: `CompressedAttention` layer implemented
- ✅ Multi-model: `AttentionPatcher` supports GPT-2, LLaMA, OPT
- ✅ 52 tests passing (was broken, now fixed)
- ✅ Device support: GPU/CPU selection working

**What's Still in Development:**
- 🔄 End-to-end speedup verification (target: 2-3x)
- 🔄 Full SSD speculation integration for speedup
- 🔄 Background recovery full implementation
- 🔄 Comprehensive benchmarks with real models

## 🎯 **Original Goal vs Current State**

### Target (When Complete):
- **2-3x speedup** via compressed KV cache + SSD speculation
- **Minimal memory**: 30-50x KV cache reduction + 5x quantization
- **Training-free**: Works with any autoregressive model
- **Production ready**: All components integrated and verified

### Current (April 2026):
- ✅ **Compression**: Working (5-50x KV reduction verified)
- ✅ **Quantization**: Working (FP8 with measurable error)
- ✅ **Custom Attention**: Implemented (`CompressedAttention`)
- ✅ **Multi-model**: Working (GPT-2, LLaMA, OPT)
- ⚠️ **Speedup**: Framework ready, **NOT yet verified**

## 🚀 **Quick Start**

### Installation
```bash
git clone https://github.com/kishoretvk/DevClaw.git
cd DevClaw
pip install -e .
```

### Basic Usage (Compression Verified)
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

### What Works Now:
- ✅ Compression reduces KV cache by 5-50x
- ✅ Quantization with FP8 (MSE: 0.001331)
- ✅ Custom attention layer (`CompressedAttention`)
- ✅ Multi-model support via `AttentionPatcher`

### What's Next:
- 🔄 Run end-to-end benchmarks to verify speedup
- 🔄 Complete SSD speculation integration
- 🔄 Update documentation with verified numbers

## 📚 **Benchmarks (Honest Results)**

### Verified Measurements:
```
COMPRESSION BENCHMARK:
  Ratio 5:   5.05x reduction (1.41 MB vs 7.10 MB)
  Ratio 10:  10.10x reduction (0.70 MB vs 7.10 MB)
  Ratio 20:  20.20x reduction (0.35 MB vs 7.10 MB)
  Ratio 50:  50.50x reduction (0.14 MB vs 7.10 MB)

QUANTIZATION BENCHMARK:
  MSE: 0.001331
  Max error: 0.248759
  Quantized dtype: torch.float8_e4m3fn
```

### Speedup Status:
- ❌ **NOT YET VERIFIED** - Custom attention integrated but end-to-end benchmarks pending
- Target: 2-3x speedup (when verified)

## 📖 **Documentation & Tutorials**

### Getting Started:
- **[📖 Complete Getting Started Guide](./tutorials/getting_started.md)** - 5-minute hands-on tutorial

### Integration Guides:
- **[📖 Complete Integration Guide](./integration_guide.md)** - Ollama, vLLM, custom engines
- **[📖 Setup Automation](./setup.py)** - Automated installation
- **[📖 API Reference](./integration_guide.md#production-deployment)** - REST API docs

### Benchmarks & Analysis:
- **[📊 Honest Benchmarks](./benchmarks/honest_benchmark.py)** - Measures what works
- **[📊 Benchmark Results](./benchmarks/honest_results.json)** - Verified numbers
- **[📊 Updated Notebook](./notebooks/colab_gpu_benchmark_updated.ipynb)** - GPU testing

## 🏗 **Architecture**

```
CSA Framework Architecture
══════════════════════════════════════

┌─────────────────────────────┐
│               CSA Engine                    │
│         (Main Orchestration)                │
└─────────────────┬─────────────────────┘
                   │
         ┌─────────┴─────────┐
         │                   │
┌───────▼───────┐   ┌───────▼───────┐
│ Attention     │   │   TurboQuant  │
│ Matching      │   │   (3-bit)     │
│ (Compress)    │   │   (Quantize)  │
└───────┬───────┘   └───────┬───────┘
         │                   │
         ┌─────────▼─────────┐
         │                   │
┌───────▼───────┐   ┌───────▼───────┐
│     SSD       │   │  Background    │
│   Engine      │   │  Recovery     │
│ (Speculate)   │   │  (Refine)     │
└───────────────────┘   └───────────────────┘

Data Flow: Prompt → Compress → Quantize → Speculate → Generate → Recover
```

## 🔧 **Key Features (Current State)**

- 🚀 **Compression**: 5-50x KV cache reduction (VERIFIED)
- 💾 **Quantization**: FP8 with measurable error (WORKING)
- 🔌 **Custom Attention**: `CompressedAttention` layer (IMPLEMENTED)
- 🔌 **Multi-Model**: GPT-2, LLaMA, OPT support (WORKING)
- ⚡ **Modular**: All components are plug-and-play
- 🔄 **Speedup Target**: 2-3x (framework ready, NOT VERIFIED)

## 🗺 **Roadmap**

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
- [ ] Background recovery implementation
- [ ] Update documentation with verified speedup numbers

## 🤝 **Contributing**

We welcome contributions! Please see our [contributing guidelines](./CONTRIBUTING.md) and feel free to:
- 🐛 Report bugs and issues
- 💡 Suggest new features
- 🔧 Submit pull requests
- 📖 Improve documentation

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

**Last Updated**: April 27, 2026  
**Status**: Functional proof-of-concept with verified compression & quantization  
**Next Milestone**: End-to-end speedup verification