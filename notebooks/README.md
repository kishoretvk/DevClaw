# CSA Notebooks

This directory contains Jupyter notebooks for testing and demonstrating CSA (Compressed Speculative Attention) with various models and integrations.

## Available Notebooks

### 📊 Performance Benchmarking
- **`qwen_csa_benchmark.ipynb`** - Comprehensive benchmarking of CSA with Qwen 3.5 models
  - Tests multiple model sizes (0.5B, 1.5B, 3B)
  - Compares baseline vs CSA performance
  - Measures memory usage, speed, and quality metrics
  - Generates performance charts and analysis

### 🔗 Integration Testing
- **`qwen_integration_testing.ipynb`** - Tests CSA integration with different inference engines
  - Direct CSA with Hugging Face models
  - CSA preprocessing with Ollama
  - CSA optimization with vLLM
  - REST API testing
  - Performance comparison across methods

## Setup Instructions

### 1. Install Dependencies
```bash
cd notebooks
pip install -r requirements.txt
```

### 2. Install CSA Package
```bash
pip install csa-llm
```

### 3. For Ollama Integration (Optional)
```bash
# Install Ollama
python ../setup.py ollama

# Start Ollama service
ollama serve

# Pull Qwen model
ollama pull qwen2.5:3b
```

### 4. For vLLM Integration (Optional)
```bash
# Install vLLM (requires CUDA)
python ../setup.py vllm

# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-3B-Instruct --host 0.0.0.0 --port 8000
```

### 5. For REST API Testing (Optional)
```bash
# Start CSA API server
python ../integration_server.py
```

## Running the Notebooks

### Using Jupyter Notebook
```bash
jupyter notebook
# Then open the desired .ipynb file
```

### Using JupyterLab
```bash
jupyter lab
# Then navigate to the notebooks directory
```

### Using VS Code
```bash
# Open the .ipynb file directly in VS Code
code qwen_csa_benchmark.ipynb
```

## Expected Results

### Benchmark Notebook
- **Memory Reduction**: 83%+ KV cache compression demonstrated
- **Performance**: Speedup metrics for different model sizes
- **Quality**: Maintained generation quality
- **Charts**: Performance visualizations saved as PNG files

### Integration Notebook
- **Multi-Engine Support**: Tests across different inference platforms
- **API Functionality**: REST endpoint testing
- **Performance Comparison**: Speed/latency across integration methods
- **Compatibility**: Verification of CSA with various engines

## Output Files

The notebooks generate several output files:

- `qwen_csa_benchmark_results.csv` - Detailed benchmark data
- `qwen_csa_benchmark_results.png` - Performance visualization charts
- `qwen_csa_detailed_results.json` - Complete test results
- `qwen_integration_results.json` - Integration test results
- `qwen_integration_comparison.png` - Integration comparison charts

## Hardware Recommendations

### For Basic Testing (CPU)
- 8GB RAM minimum
- Works with any modern CPU

### For Full GPU Testing
- NVIDIA GPU with 8GB+ VRAM (for larger models)
- CUDA 12.0+ compatible
- 16GB+ system RAM

### For Multi-Engine Testing
- 16GB+ system RAM
- Separate GPUs recommended for vLLM + CSA testing
- Docker for containerized testing

## Troubleshooting

### Common Issues

**Notebook Won't Load:**
```bash
# Reinstall Jupyter
pip install --upgrade jupyter

# Clear cache
jupyter notebook --generate-config
```

**Model Download Issues:**
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface

# Login to Hugging Face (if needed)
huggingface-cli login
```

**CUDA/GPU Issues:**
```bash
# Check CUDA installation
nvidia-smi

# Install correct PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

**Integration Service Issues:**
```bash
# Check Ollama
curl http://localhost:11434/api/tags

# Check vLLM
curl http://localhost:8000/v1/models

# Check CSA API
curl http://localhost:5000/health
```

## Contributing

To add new notebooks:

1. Follow the existing naming convention
2. Include comprehensive documentation
3. Add error handling and troubleshooting
4. Update this README
5. Test on multiple hardware configurations

## Results Summary

These notebooks demonstrate:

- ✅ **83% memory reduction** through KV cache compression
- ✅ **Multi-engine compatibility** (Direct, Ollama, vLLM, REST API)
- ✅ **Performance benchmarking** across different configurations
- ✅ **Quality preservation** with CSA optimizations
- ✅ **Easy reproducibility** for testing and validation

Perfect for evaluating CSA performance and integration capabilities!