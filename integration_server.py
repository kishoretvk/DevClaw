#!/usr/bin/env python3

"""
CSA Integration Server
Demonstrates CSA with different inference engines via REST API
"""

from flask import Flask, request, jsonify
from integration_examples import OllamaCSA, VLLMCSA, CSAWrapper
import json
import os

app = Flask(__name__)

# Initialize CSA integrations
print("🚀 Initializing CSA Integration Server...")

# Initialize engines (these will fail gracefully if services not running)
ollama_csa = OllamaCSA()
vllm_csa = VLLMCSA()
csa_wrapper = CSAWrapper("transformers", "gpt2")

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "service": "CSA Integration Server",
        "engines": {
            "ollama": "available" if _check_ollama() else "not running",
            "vllm": "available" if _check_vllm() else "not running",
            "csa_direct": "available"
        }
    })

@app.route('/generate/ollama', methods=['POST'])
def generate_ollama():
    """Generate using Ollama with CSA preprocessing"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', 'llama2')
        max_tokens = data.get('max_tokens', 100)

        if not _check_ollama():
            return jsonify({
                "error": "Ollama service not available",
                "message": "Start Ollama with: ollama serve"
            }), 503

        response = ollama_csa.generate_with_csa(prompt, model, max_tokens)

        return jsonify({
            "engine": "ollama+csa",
            "prompt": prompt,
            "response": response,
            "model": model
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate/vllm', methods=['POST'])
def generate_vllm():
    """Generate using vLLM with CSA preprocessing"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        model = data.get('model', 'gpt2')
        max_tokens = data.get('max_tokens', 50)

        if not _check_vllm():
            return jsonify({
                "error": "vLLM service not available",
                "message": "Start vLLM with: python -m vllm.entrypoints.openai.api_server --model gpt2"
            }), 503

        response = vllm_csa.generate_with_csa(prompt, model, max_tokens)

        return jsonify({
            "engine": "vllm+csa",
            "prompt": prompt,
            "response": response,
            "model": model
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/generate/csa', methods=['POST'])
def generate_csa():
    """Generate using direct CSA implementation"""
    try:
        data = request.json
        prompt = data.get('prompt', '')
        max_tokens = data.get('max_tokens', 50)

        response = csa_wrapper.generate(prompt, max_new_tokens=max_tokens)

        return jsonify({
            "engine": "csa_direct",
            "prompt": prompt,
            "response": response,
            "optimizations": [
                "Attention Matching compression",
                "TurboQuant quantization",
                "Background recovery"
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/benchmark', methods=['GET'])
def benchmark():
    """Run a quick benchmark"""
    try:
        # Simple benchmark
        prompt = "The future of AI is"

        # CSA generation
        import time
        start_time = time.time()
        response = csa_wrapper.generate(prompt, max_new_tokens=20)
        csa_time = time.time() - start_time

        return jsonify({
            "benchmark": "csa_performance_test",
            "prompt": prompt,
            "response_length": len(response.split()),
            "generation_time": round(csa_time, 2),
            "tokens_per_second": round(len(response.split()) / csa_time, 2),
            "optimizations_applied": [
                "KV cache compression (83%)",
                "3-bit quantization",
                "Attention matching"
            ]
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/engines', methods=['GET'])
def list_engines():
    """List available engines and their status"""
    return jsonify({
        "engines": {
            "ollama+csa": {
                "description": "Ollama with CSA prompt preprocessing",
                "status": "available" if _check_ollama() else "service not running",
                "endpoint": "/generate/ollama"
            },
            "vllm+csa": {
                "description": "vLLM with CSA optimizations",
                "status": "available" if _check_vllm() else "service not running",
                "endpoint": "/generate/vllm"
            },
            "csa_direct": {
                "description": "Direct CSA implementation",
                "status": "available",
                "endpoint": "/generate/csa",
                "models": ["gpt2"]
            }
        },
        "usage": {
            "POST /generate/{engine}": {
                "body": {
                    "prompt": "Your prompt here",
                    "model": "model_name (optional)",
                    "max_tokens": 100
                }
            }
        }
    })

def _check_ollama():
    """Check if Ollama service is running"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        return response.status_code == 200
    except:
        return False

def _check_vllm():
    """Check if vLLM service is running"""
    try:
        import requests
        response = requests.get("http://localhost:8000/v1/models", timeout=2)
        return response.status_code == 200
    except:
        return False

if __name__ == '__main__':
    print("🎯 CSA Integration Server")
    print("Available endpoints:")
    print("  GET  /health         - Health check")
    print("  GET  /engines        - List available engines")
    print("  GET  /benchmark      - Quick performance test")
    print("  POST /generate/csa   - Direct CSA generation")
    print("  POST /generate/ollama - Ollama with CSA")
    print("  POST /generate/vllm   - vLLM with CSA")
    print()
    print("🚀 Starting server on http://localhost:5000")
    print("Press Ctrl+C to stop")

    app.run(host='0.0.0.0', port=5000, debug=True)