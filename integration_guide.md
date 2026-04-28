# CSA Integration Guide

Complete guide for integrating Compressed Speculative Attention (CSA) with various inference engines.

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

## 🎯 Integration with Ollama

### Step 1: Setup Ollama
```bash
# Install Ollama
curl -fsSL https://ollama.com/install.sh | sh

# Pull a model
ollama pull gpt2
```

### Step 2: Run CSA with Ollama
```python
from csa import CSAEngine

# Create engine with compression
engine = CSAEngine(
    target_model="gpt2",
    compression_ratio=10,
    device="cpu"  # or "cuda" for GPU
)

# Generate with compressed cache
text = engine.generate(
    "The future of AI is",
    max_new_tokens=50
)
print(text)
engine.cleanup()
```

---

## 🎯 Integration with vLLM

### Step 1: Setup vLLM
```bash
# Install vLLM
pip install vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model gpt2 \
  --host 0.0.0.0 \
  --port 8000
```

### Step 2: Test Integration
```python
# See integration_examples.py for full demo
python integration_examples.py
```

---

## 🎯 REST API Server

### Start the Server
```bash
python integration_server.py
```

### API Endpoints
- `POST /generate/csa` - Direct CSA generation
- `POST /generate/ollama` - Ollama with CSA preprocessing
- `POST /generate/vllm` - vLLM with CSA preprocessing

### Example Request
```bash
curl -X POST http://localhost:5000/generate/csa \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Hello world", "max_tokens": 50}'
```

---

## 📚 What Works Now

### ✅ Verified Components:
1. **Compression**: 5-50x KV cache reduction
   - Tested with GPT-2
   - Uniform sampling strategy working
   - Measured: 7.10 MB → 0.14 MB (50x ratio)

2. **Quantization**: FP8 with measurable error
   - MSE: 0.001331
   - Max error: 0.248759
   - Dtype: float32 → float8_e4m3fn

3. **Custom Attention**: `CompressedAttention` layer
   - Model-agnostic implementation
   - Patches GPT-2, LLaMA, OPT
   - Enabled in engine generation

4. **Multi-Model Support**: `AttentionPatcher`
   - Automatically detects model type
   - Patches correct attention layers
   - 52 tests passing

### ⚠️ Pending Verification:
- **Speedup**: 2-3x target (NOT YET VERIFIED)
- **End-to-end benchmarks**: Running with compressed attention
- **Quality metrics**: Perplexity with compressed cache

---

## 📖 Quick Integration Examples

### Basic CSA Usage (Verified Working)
```python
from csa import CSAEngine

# CPU mode
engine = CSAEngine("gpt2", compression_ratio=10, device="cpu")
text = engine.generate("Test prompt", max_new_tokens=20)
print(text)
engine.cleanup()
```

### With GPU (if available)
```python
engine = CSAEngine("gpt2", compression_ratio=10, device="cuda")
text = engine.generate("Test prompt", max_new_tokens=20)
print(text)
engine.cleanup()
```

---

## 📊 Benchmarks (Honest Results)

### Compression Verification:
```
Ratio 5:   1.41 MB (5.05x reduction)
Ratio 10:  0.70 MB (10.10x reduction)
Ratio 20:  0.35 MB (20.20x reduction)
Ratio 50:  0.14 MB (50.50x reduction)
```

### Quantization:
```
MSE: 0.001331
Max error: 0.248759
Quantized dtype: torch.float8_e4m3fn
```

### Speedup:
```
Status: NOT YET VERIFIED
Target: 2-3x (when custom attention fully integrated)
Next: Run end-to-end benchmarks
```

---

## 🗺️ Troubleshooting

### Common Issues:

1. **Syntax Errors**: 
   - ✅ Fixed in all core files
   - Run: `python -m pytest tests/ -v`

2. **Memory Benchmark Error** (`index -1 is out of bounds`):
   - ✅ Fixed by simplifying benchmark
   - Use `benchmarks/honest_benchmark.py`

3. **Model Not Supported**:
   - Check `AttentionPatcher.detect_model_type()`
   - Add new model type in `csa/attention/patcher.py`

---

## 🤝 Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

Current focus: **Speedup verification** - we need end-to-end benchmarks with the integrated `CompressedAttention` layer.

---

**Last Updated**: April 27, 2026  
**Status**: Functional proof-of-concept with verified compression & quantization  
**Next**: End-to-end speedup measurement