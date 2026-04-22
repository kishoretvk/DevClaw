# CSA Integration Guide

This guide shows how to integrate Compressed Speculative Attention (CSA) with popular inference engines like Ollama, vLLM, and others.

## Table of Contents

- [Quick Start](#quick-start)
- [Ollama Integration](#ollama-integration)
- [vLLM Integration](#vllm-integration)
- [Hugging Face Integration](#hugging-face-integration)
- [Custom Engine Integration](#custom-engine-integration)
- [Production Deployment](#production-deployment)
- [Performance Optimization](#performance-optimization)

## Quick Start

```python
from csa import CSAEngine

# Basic usage with any model
engine = CSAEngine("gpt2")  # or "meta-llama/Llama-2-7b", etc.
result = engine.generate("Your prompt here", max_new_tokens=100)
print(result)
```

## Ollama Integration

Ollama is a user-friendly CLI tool for running LLMs locally. Here's how to use CSA with Ollama:

### Method 1: CSA Preprocessing + Ollama

Use CSA to optimize prompts before sending to Ollama:

```python
from integration_examples import OllamaCSA

# Initialize with Ollama host
ollama_csa = OllamaCSA(ollama_host="http://localhost:11434")

# Generate with CSA preprocessing
response = ollama_csa.generate_with_csa(
    prompt="Explain quantum computing simply",
    model_name="llama2",
    max_tokens=200
)
print(response)
```

### Method 2: Custom Ollama Model

Create a custom Ollama model that includes CSA components:

```bash
# Create a custom model file
cat > csa-llama2.modelfile << EOF
FROM llama2
PARAMETER temperature 0.7
PARAMETER top_p 0.9

# Add CSA optimizations (conceptual)
SYSTEM "You are an AI with CSA optimizations for efficient inference"
EOF

# Create the model
ollama create csa-llama2 -f csa-llama2.modelfile

# Use it
ollama run csa-llama2 "Your prompt here"
```

### Method 3: Ollama API with CSA

Use Ollama's REST API with CSA preprocessing:

```python
import requests
from csa import CSAEngine

# CSA preprocessing
csa = CSAEngine("gpt2", use_speculation=False)
optimized_prompt = csa.generate("Analyze this prompt for optimization", max_new_tokens=1)
# Apply optimization logic here

# Send to Ollama
response = requests.post("http://localhost:11434/api/generate", json={
    "model": "llama2",
    "prompt": optimized_prompt,
    "stream": False
})
```

## vLLM Integration

vLLM is a high-performance inference engine. CSA can be integrated at multiple levels:

### Method 1: vLLM with CSA Wrapper

```python
from integration_examples import VLLMCSA

# Initialize vLLM CSA integration
vllm_csa = VLLMCSA(vllm_host="http://localhost:8000")

# Generate with CSA optimizations
response = vllm_csa.generate_with_csa(
    prompt="What is machine learning?",
    model_name="microsoft/DialoGPT-medium",
    max_tokens=150
)
```

### Method 2: Custom vLLM Engine

Modify vLLM to include CSA components:

```python
from vllm import LLM, SamplingParams
from csa.compression import AttentionMatcher
from csa.quantization import TurboQuantCache

class CSAvLLM:
    def __init__(self, model_name):
        self.llm = LLM(model=model_name)
        self.csa_matcher = AttentionMatcher(compression_ratio=50)

    def generate(self, prompt, **kwargs):
        # Apply CSA compression before generation
        # This requires modifying vLLM's internal processing
        pass

# Usage
csa_vllm = CSAvLLM("meta-llama/Llama-2-7b-hf")
```

### Method 3: vLLM Server with CSA

Run vLLM server and use CSA preprocessing:

```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Llama-2-7b-hf \
    --host 0.0.0.0 \
    --port 8000

# Use with CSA
from integration_examples import VLLMCSA
vllm_csa = VLLMCSA()
response = vllm_csa.generate_with_csa("Your prompt")
```

## Hugging Face Integration

Direct integration with Transformers:

### Basic Usage

```python
from csa import CSAEngine

# Use any Hugging Face model
engine = CSAEngine("microsoft/DialoGPT-medium")
response = engine.generate("Hello, how are you?", max_new_tokens=50)
```

### Advanced Configuration

```python
# Full CSA configuration
engine = CSAEngine(
    target_model_path="meta-llama/Llama-2-7b-hf",
    draft_model_path="meta-llama/Llama-2-7b-hf",  # or smaller model
    compression_ratio=50,
    quant_bits=3
)

# Enable speculation for maximum speedup
response = engine.generate(
    prompt="Complex reasoning task here",
    max_new_tokens=200
)
```

### Custom Model Loading

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from csa import CSAEngine

# Load model with custom settings
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    device_map="auto"
)

# Use CSA with pre-loaded model
# Note: CSA currently creates its own model instance
engine = CSAEngine("meta-llama/Llama-2-7b-hf")
```

## Custom Engine Integration

For any inference engine, create a CSA wrapper:

### Generic Wrapper Pattern

```python
from csa import CSAEngine

class CustomEngineCSA:
    def __init__(self, engine_client, csa_model="gpt2"):
        self.engine = engine_client
        self.csa = CSAEngine(csa_model, use_speculation=False)

    def generate(self, prompt, **kwargs):
        # Apply CSA optimizations
        optimized_input = self._csa_optimize(prompt)

        # Send to your custom engine
        response = self.engine.generate(optimized_input, **kwargs)

        return response

    def _csa_optimize(self, prompt):
        # Apply CSA compression/quantization logic
        return prompt  # Modified prompt
```

### Example with OpenAI API

```python
import openai
from csa import CSAEngine

class OpenAICSA:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.csa = CSAEngine("gpt2", use_speculation=False)

    def generate(self, prompt, model="gpt-3.5-turbo", **kwargs):
        # Optimize prompt with CSA
        optimized_prompt = self._csa_optimize(prompt)

        # Send to OpenAI
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": optimized_prompt}],
            **kwargs
        )

        return response.choices[0].message.content

    def _csa_optimize(self, prompt):
        # Apply CSA analysis
        print(f"CSA: Analyzing prompt ({len(prompt)} chars)")
        return prompt
```

## Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.12-slim

# Install dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy CSA code
COPY . /app
WORKDIR /app

# Install CSA
RUN pip install -e .

# Expose ports for different engines
EXPOSE 8000 11434

# Run integration server
CMD ["python", "integration_server.py"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: csa-inference
spec:
  replicas: 3
  selector:
    matchLabels:
      app: csa-inference
  template:
    metadata:
      labels:
        app: csa-inference
    spec:
      containers:
      - name: csa-vllm
        image: csa-vllm:latest
        ports:
        - containerPort: 8000
        resources:
          requests:
            nvidia.com/gpu: 1
          limits:
            nvidia.com/gpu: 1
```

### Cloud Deployment

```bash
# AWS SageMaker
# Deploy CSA-optimized model
aws sagemaker create-model \
    --model-name csa-llama-2-7b \
    --primary-container Image=csa-inference:latest

# Google Cloud AI
gcloud ai models upload \
    --region us-central1 \
    --display-name csa-llama-2-7b \
    --container-image-uri gcr.io/project/csa-inference:latest

# Azure ML
az ml model create \
    --name csa-llama-2-7b \
    --type custom_model \
    --image csa-inference:latest
```

## Performance Optimization

### Memory Optimization

```python
# Configure for memory efficiency
engine = CSAEngine(
    target_model_path="meta-llama/Llama-2-7b-hf",
    compression_ratio=75,  # Higher compression
    quant_bits=3
)

# Use CPU for smaller models
engine_cpu = CSAEngine("gpt2")  # Defaults to CPU if no GPU
```

### Speed Optimization

```python
# Maximum performance configuration
engine = CSAEngine(
    target_model_path="meta-llama/Llama-2-7b-hf",
    draft_model_path="meta-llama/Llama-2-7b-hf",  # Same model for SSD
    compression_ratio=50,
    quant_bits=3
)

# Enable full speculation
response = engine.generate(
    prompt="Complex task",
    max_new_tokens=1000  # Longer generation benefits more from CSA
)
```

### Benchmarking

```python
# Run performance benchmarks
python benchmarks/benchmark_csa.py

# Generate performance visualizations
python benchmarks/visualizer.py

# Check memory usage
python -c "import torch; from csa import CSAEngine; engine = CSAEngine('gpt2'); print(f'Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB')"
```

## Troubleshooting

### Common Issues

**Ollama Connection Failed:**
```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Start Ollama
ollama serve
```

**vLLM Server Issues:**
```bash
# Start vLLM server
python -m vllm.entrypoints.openai.api_server --model gpt2

# Check server
curl http://localhost:8000/v1/models
```

**CUDA Out of Memory:**
```python
# Use smaller models or CPU
engine = CSAEngine("gpt2")  # Smaller model
# or
engine = CSAEngine("meta-llama/Llama-2-7b-hf", device="cpu")
```

**Import Errors:**
```bash
# Install missing dependencies
pip install transformers torch accelerate
pip install ctransformers  # For some model formats
```

## Contributing

To add support for new inference engines:

1. Create a new integration class in `integration_examples.py`
2. Add documentation to this guide
3. Test with multiple models
4. Submit a pull request

## License

This integration guide is part of the CSA project. See LICENSE for details.

---

*Last updated: 2026-04-22*