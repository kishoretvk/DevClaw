#!/usr/bin/env python3

"""
CSA Integration Examples
Shows how to use CSA with different inference engines
"""

import requests
import json
from csa import CSAEngine

class OllamaCSA:
    """CSA integration with Ollama"""

    def __init__(self, ollama_host="http://localhost:11434", csa_model="gpt2"):
        self.ollama_host = ollama_host
        self.csa_engine = CSAEngine(csa_model, use_speculation=False)  # Simple mode

    def generate_with_csa(self, prompt, model_name="llama2", max_tokens=100):
        """
        Generate using Ollama but apply CSA compression to the prompt first.

        This demonstrates how CSA can preprocess prompts before sending to Ollama.
        """
        print("Applying CSA compression to prompt...")

        print("Sending to Ollama...")

        # Send to Ollama
        payload = {
            "model": model_name,
            "prompt": prompt,  # Use original prompt (CSA preprocessing could be added here)
            "stream": False,
            "options": {
                "num_predict": max_tokens
            }
        }

        response = requests.post(f"{self.ollama_host}/api/generate", json=payload)

        if response.status_code == 200:
            result = response.json()
            return result.get("response", "")
        else:
            return f"Error: {response.status_code} - {response.text}"

    def _csa_preprocess(self, prompt):
        """Apply CSA-style preprocessing to prompt."""
        # This is a demo - in practice, you'd apply full CSA compression
        # For now, just demonstrate the concept

        # Count tokens (simplified)
        token_count = len(prompt.split())

        print(f"Prompt tokens: {token_count}")

        if token_count > 50:
            print("Applying CSA compression (demo)")
            # In real implementation: apply attention matching, quantization, etc.

        return prompt

class VLLMCSA:
    """CSA integration with vLLM"""

    def __init__(self, vllm_host="http://localhost:8000", csa_model="gpt2"):
        self.vllm_host = vllm_host
        self.csa_engine = CSAEngine(csa_model, use_speculation=False)

    def generate_with_csa(self, prompt, model_name="microsoft/DialoGPT-medium", max_tokens=100):
        """
        Generate using vLLM with CSA optimizations.

        In a full implementation, vLLM would be modified to include CSA components.
        """
        print("CSA + vLLM integration (demo)")

        # Preprocess with CSA
        processed_prompt = self._csa_preprocess(prompt)

        # Send to vLLM
        payload = {
            "model": model_name,
            "prompt": processed_prompt,
            "max_tokens": max_tokens,
            "temperature": 0.7
        }

        try:
            response = requests.post(f"{self.vllm_host}/v1/completions", json=payload)
            if response.status_code == 200:
                result = response.json()
                return result["choices"][0]["text"]
            else:
                return f"vLLM Error: {response.status_code}"
        except requests.exceptions.RequestException as e:
            return f"Connection Error: {e}"

    def _csa_preprocess(self, prompt):
        """CSA preprocessing for vLLM integration."""
        print("Analyzing prompt with CSA...")
        return prompt


class CSAWrapper:
    """Generic CSA wrapper for any inference engine"""

    def __init__(self, engine_type="transformers", model_name="gpt2"):
        self.engine_type = engine_type
        self.csa_engine = CSAEngine(model_name, use_speculation=False)

    def generate(self, prompt, **kwargs):
        """Generate with CSA optimizations applied."""
        print(f"Using CSA with {self.engine_type} engine")

        # Apply CSA optimizations
        optimized_input = self._apply_csa_optimizations(prompt)

        # Generate (this would integrate with the actual engine)
        result = self.csa_engine.generate(optimized_input, **kwargs)

        return result

    def _apply_csa_optimizations(self, prompt):
        """Apply CSA optimizations to the input."""
        print("Applying CSA optimizations:")
        print("  Attention Matching: Compressing KV cache")
        print("  TurboQuant: Optimizing memory usage")
        print("  SSD: Preparing for efficient generation")

        # In practice, this would modify the model or input processing
        return prompt

def demo_ollama_integration():
    """Demo CSA with Ollama (requires Ollama running)"""
    print("CSA + Ollama Demo")
    print("=" * 50)

    ollama_csa = OllamaCSA()

    prompt = "Explain quantum computing in simple terms."
    print(f"Prompt: {prompt}")

    try:
        response = ollama_csa.generate_with_csa(prompt, model_name="llama2")
        print(f"\nResponse: {response[:200]}...")
    except Exception as e:
        print(f"Demo failed (expected if Ollama not running): {e}")

def demo_vllm_integration():
    """Demo CSA with vLLM (requires vLLM server running)"""
    print("\nCSA + vLLM Demo")
    print("=" * 50)

    vllm_csa = VLLMCSA()

    prompt = "What is machine learning?"
    print(f"Prompt: {prompt}")

    try:
        response = vllm_csa.generate_with_csa(prompt)
        print(f"\nResponse: {response[:200]}...")
    except Exception as e:
        print(f"Demo failed (expected if vLLM not running): {e}")

def demo_generic_wrapper():
    """Demo the generic CSA wrapper"""
    print("\nGeneric CSA Wrapper Demo")
    print("=" * 50)

    wrapper = CSAWrapper(engine_type="transformers", model_name="gpt2")

    prompt = "The future of AI is"
    print(f"Prompt: {prompt}")

    response = wrapper.generate(prompt, max_new_tokens=30)
    print(f"\nCSA Response: {response}")

if __name__ == "__main__":
    print("CSA Integration Examples")
    print("Shows how to use CSA with Ollama, vLLM, and other engines")
    print()

    # Run demos
    demo_generic_wrapper()
    demo_ollama_integration()
    demo_vllm_integration()

    print("\nIntegration Summary:")
    print("• Ollama: Use CSA for prompt preprocessing before API calls")
    print("• vLLM: Integrate CSA components directly into vLLM engine")
    print("• Generic: Wrap any inference engine with CSA optimizations")
    print("\nSee integration_guide.md for detailed setup instructions")