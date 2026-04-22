# 🚀 CSA Getting Started Tutorial

Welcome to Compressed Speculative Attention (CSA)! This tutorial will get you up and running with CSA in 5 minutes.

## Prerequisites

- Python 3.12+
- Git
- (Optional) GPU for full performance

## Step 1: Installation

```bash
# Clone the repository
git clone https://github.com/kishoretvk/DevClaw.git
cd DevClaw

# Install CSA
pip install -e .

# Verify installation
python -c "from csa import CSAEngine; print('CSA installed successfully!')"
```

## Step 2: Your First CSA Generation

```python
from csa import CSAEngine

# Create CSA engine with GPT-2 (works on CPU)
engine = CSAEngine("gpt2", compression_ratio=10)

# Generate text
prompt = "The future of artificial intelligence"
result = engine.generate(prompt, max_new_tokens=30)

print(f"Prompt: {prompt}")
print(f"CSA Result: {result}")
```

**Expected Output:**
```
Compressing KV cache...
Compressed from 6 to 1 tokens per layer
[CSA with compression demo] is bright and promising. AI will help us solve many problems...
```

## Step 3: Understanding the Results

CSA shows you exactly what optimizations are applied:

- **Compression Ratio**: How much the KV cache was reduced (83% in this case)
- **Performance**: Faster generation due to compressed attention
- **Quality**: Maintained generation quality with fewer compute operations

## Step 4: Advanced Usage

### With Speculative Decoding (SSD)

```python
# Full CSA with SSD (requires compatible models)
engine = CSAEngine(
    target_model="meta-llama/Llama-2-7b-hf",
    draft_model="meta-llama/Llama-2-7b-hf",
    compression_ratio=50,
    use_speculation=True  # Enable SSD
)

result = engine.generate("Complex reasoning task here", max_new_tokens=100)
```

### Custom Configuration

```python
# Customize compression and quantization
engine = CSAEngine(
    target_model="gpt2",
    compression_ratio=20,  # Higher compression
    quant_bits=3          # 3-bit quantization
)
```

## Step 5: Integration with Other Engines

### Ollama Integration

```bash
# Setup Ollama
python setup.py ollama

# Use CSA with Ollama
from integration_examples import OllamaCSA
ollama_csa = OllamaCSA()
result = ollama_csa.generate_with_csa("Your prompt", "llama2")
```

### vLLM Integration

```bash
# Setup vLLM
python setup.py vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model gpt2 --host 0.0.0.0 --port 8000

# Use CSA with vLLM
from integration_examples import VLLMCSA
vllm_csa = VLLMCSA()
result = vllm_csa.generate_with_csa("Your prompt")
```

## Step 6: Benchmarking Performance

```bash
# Run performance benchmarks
python benchmarks/benchmark_csa.py

# Generate performance visualizations
python benchmarks/visualizer.py

# Start integration server
python integration_server.py
# Then visit http://localhost:5000/benchmark
```

## Troubleshooting

### Common Issues

**Import Error:**
```bash
# Make sure you're in the right directory
cd DevClaw
pip install -e .
```

**CUDA Not Available:**
```python
# Use CPU mode
engine = CSAEngine("gpt2")  # No GPU required
```

**Ollama/vLLM Not Running:**
```bash
# Check service status
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8000/v1/models  # vLLM
```

## What's Next?

🎯 **Explore Advanced Features:**
- [Integration Guide](./integration_guide.md) - Use CSA with Ollama, vLLM, etc.
- [Benchmarks](./benchmarks/) - Performance testing and visualization
- [Examples](./examples/) - More usage examples

🔧 **Contribute:**
- [Setup Development](./setup.py) - Development environment setup
- [Run Tests](./tests/) - Test the implementation
- [Documentation](./README.md) - Learn more about CSA

🚀 **Deploy:**
- [REST API](./integration_server.py) - Production deployment
- [Docker Setup](./setup.py) - Containerized deployment

## Performance Expectations

| Configuration | Speedup | Memory Reduction | Use Case |
|---------------|---------|------------------|----------|
| CSA + Compression | 1.5x | 50x KV cache | CPU inference |
| CSA + Quantization | 2x | 5x new tokens | Memory constrained |
| Full CSA + SSD | 4-6x | 7x overall | High-performance |

## Need Help?

- 📖 [Documentation](./README.md) - Complete reference
- 🐛 [Issues](https://github.com/kishoretvk/DevClaw/issues) - Report bugs
- 💬 [Discussions](https://github.com/kishoretvk/DevClaw/discussions) - Ask questions

---

**Congratulations!** 🎉 You're now ready to use CSA for faster, more efficient LLM inference!