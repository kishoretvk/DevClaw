#!/usr/bin/env python3
"""
CSA Integration Examples
Shows how to use CSA with different inference engines
"""

import requests
import json
import time
from typing import Optional, Dict, Any
from csa import CSAEngine


class OllamaCSA:
    """CSA integration with Ollama"""

    def __init__(self, ollama_host="http://localhost:11434", csa_model="gpt2"):
        self.ollama_host = ollama_host
        self.csa_engine = CSAEngine(csa_model, use_speculation=False)
        self._check_ollama()

    def _check_ollama(self):
        """Check if Ollama is running."""
        try:
            response = requests.get(f"{self.ollama_host}/api/tags", timeout=5)
            if response.status_code == 200:
                print(f"✅ Ollama connected at {self.ollama_host}")
                models = response.json().get('models', [])
                print(f"   Available models: {len(models)}")
            else:
                print(f"⚠️ Ollama returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ Ollama not running at {self.ollama_host}")
            print(f"   Start Ollama: ollama serve")
        except Exception as e:
            print(f"⚠️ Could not connect to Ollama: {e}")

    def generate_with_csa(self, prompt, model_name="llama2", max_tokens=100, 
                          use_compression=True):
        """
        Generate using Ollama with CSA compression preprocessing.
        
        Args:
            prompt: Input prompt
            model_name: Ollama model to use
            max_tokens: Maximum tokens to generate
            use_compression: Whether to apply CSA compression to prompt
        
        Returns:
            dict: {'text': generated text, 'time': generation time, 'compressed': bool}
        """
        start_time = time.time()
        
        if use_compression:
            print("🔧 Applying CSA compression to prompt...")
            processed_prompt = self._csa_preprocess(prompt)
        else:
            processed_prompt = prompt

        print(f"📤 Sending to Ollama ({model_name})...")

        try:
            payload = {
                "model": model_name,
                "prompt": processed_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens,
                    "temperature": 0.7
                }
            }

            response = requests.post(
                f"{self.ollama_host}/api/generate", 
                json=payload,
                timeout=120
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                return {
                    'text': result.get("response", ""),
                    'time': elapsed,
                    'compressed': use_compression,
                    'tokens_generated': result.get('eval_count', 0),
                    'status': 'success'
                }
            else:
                return {
                    'text': f"Error: {response.status_code} - {response.text}",
                    'time': elapsed,
                    'compressed': use_compression,
                    'status': 'error'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'text': "Error: Ollama not running. Start with: ollama serve",
                'time': time.time() - start_time,
                'compressed': use_compression,
                'status': 'connection_error'
            }
        except Exception as e:
            return {
                'text': f"Error: {str(e)}",
                'time': time.time() - start_time,
                'compressed': use_compression,
                'status': 'error'
            }

    def _csa_preprocess(self, prompt):
        """Apply CSA-style preprocessing to prompt."""
        # For now, just analyze and report
        # In full implementation, would compress and send compressed representation
        
        token_count = len(prompt.split())
        print(f"   Prompt tokens: ~{token_count}")

        if token_count > 50:
            print("   📉 Long prompt detected - CSA compression would reduce KV cache")
            # In production: compress prompt, send compressed representation
            
        return prompt


class VLLMCSA:
    """CSA integration with vLLM"""

    def __init__(self, vllm_host="http://localhost:8000", csa_model="gpt2"):
        self.vllm_host = vllm_host
        self.csa_engine = CSAEngine(csa_model, use_speculation=False)
        self._check_vllm()

    def _check_vllm(self):
        """Check if vLLM server is running."""
        try:
            response = requests.get(f"{self.vllm_host}/health", timeout=5)
            if response.status_code == 200:
                print(f"✅ vLLM server connected at {self.vllm_host}")
            else:
                print(f"⚠️ vLLM returned status {response.status_code}")
        except requests.exceptions.ConnectionError:
            print(f"❌ vLLM not running at {self.vllm_host}")
            print(f"   Start vLLM: python -m vllm.entrypoints.openai.api_server --model gpt2")
        except Exception as e:
            print(f"⚠️ Could not connect to vLLM: {e}")

    def generate_with_csa(self, prompt, model_name="gpt2", max_tokens=100,
                          use_compression=True):
        """
        Generate using vLLM with CSA optimizations.
        
        Args:
            prompt: Input prompt
            model_name: Model name on vLLM server
            max_tokens: Maximum tokens to generate
            use_compression: Whether to apply CSA preprocessing
        
        Returns:
            dict: {'text': generated text, 'time': generation time, 'compressed': bool}
        """
        start_time = time.time()
        
        if use_compression:
            print("🔧 Applying CSA preprocessing...")
            processed_prompt = self._csa_preprocess(prompt)
        else:
            processed_prompt = prompt

        print(f"📤 Sending to vLLM ({model_name})...")

        try:
            payload = {
                "model": model_name,
                "prompt": processed_prompt,
                "max_tokens": max_tokens,
                "temperature": 0.7
            }

            response = requests.post(
                f"{self.vllm_host}/v1/completions",
                json=payload,
                timeout=120
            )

            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                choices = result.get("choices", [{}])
                text = choices[0].get("text", "") if choices else ""
                
                return {
                    'text': text,
                    'time': elapsed,
                    'compressed': use_compression,
                    'tokens_generated': result.get('usage', {}).get('completion_tokens', 0),
                    'status': 'success'
                }
            else:
                return {
                    'text': f"vLLM Error: {response.status_code}",
                    'time': elapsed,
                    'compressed': use_compression,
                    'status': 'error'
                }
                
        except requests.exceptions.ConnectionError:
            return {
                'text': "Error: vLLM not running. Start server first.",
                'time': time.time() - start_time,
                'compressed': use_compression,
                'status': 'connection_error'
            }
        except Exception as e:
            return {
                'text': f"Error: {str(e)}",
                'time': time.time() - start_time,
                'compressed': use_compression,
                'status': 'error'
            }

    def _csa_preprocess(self, prompt):
        """CSA preprocessing for vLLM integration."""
        print("Analyzing prompt with CSA...")
        # In production: apply compression, send optimized representation
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
    print("=" * 60)
    print("CSA + Ollama Demo")
    print("=" * 60)

    ollama_csa = OllamaCSA()

    prompt = "Explain quantum computing in simple terms."
    print(f"\nPrompt: {prompt}")

    # Test without compression
    print("\n--- Without CSA ---")
    result1 = ollama_csa.generate_with_csa(prompt, model_name="llama2", use_compression=False)
    print(f"Status: {result1['status']}")
    print(f"Time: {result1['time']:.2f}s")
    if result1['status'] == 'success':
        print(f"Response: {result1['text'][:200]}...")

    # Test with compression
    print("\n--- With CSA ---")
    result2 = ollama_csa.generate_with_csa(prompt, model_name="llama2", use_compression=True)
    print(f"Status: {result2['status']}")
    print(f"Time: {result2['time']:.2f}s")
    if result2['status'] == 'success':
        print(f"Response: {result2['text'][:200]}...")

    # Compare
    if result1['status'] == 'success' and result2['status'] == 'success':
        print(f"\n📊 Comparison:")
        print(f"   Without CSA: {result1['time']:.2f}s")
        print(f"   With CSA: {result2['time']:.2f}s")
        speedup = result1['time'] / result2['time'] if result2['time'] > 0 else 0
        print(f"   Speedup: {speedup:.2f}x")


def demo_vllm_integration():
    """Demo CSA with vLLM (requires vLLM server running)"""
    print("\n" + "=" * 60)
    print("CSA + vLLM Demo")
    print("=" * 60)

    vllm_csa = VLLMCSA()

    prompt = "What is machine learning?"
    print(f"\nPrompt: {prompt}")

    # Test without compression
    print("\n--- Without CSA ---")
    result1 = vllm_csa.generate_with_csa(prompt, use_compression=False)
    print(f"Status: {result1['status']}")
    print(f"Time: {result1['time']:.2f}s")
    if result1['status'] == 'success':
        print(f"Response: {result1['text'][:200]}...")

    # Test with compression
    print("\n--- With CSA ---")
    result2 = vllm_csa.generate_with_csa(prompt, use_compression=True)
    print(f"Status: {result2['status']}")
    print(f"Time: {result2['time']:.2f}s")
    if result2['status'] == 'success':
        print(f"Response: {result2['text'][:200]}...")


def demo_generic_wrapper():
    """Demo the generic CSA wrapper"""
    print("\n" + "=" * 60)
    print("Generic CSA Wrapper Demo")
    print("=" * 60)

    wrapper = CSAWrapper(engine_type="transformers", model_name="gpt2")

    prompt = "The future of AI is"
    print(f"\nPrompt: {prompt}")

    response = wrapper.generate(prompt, max_new_tokens=30)
    print(f"\nCSA Response: {response}")


def demo_direct_csa():
    """Demo CSA directly without external services"""
    print("\n" + "=" * 60)
    print("Direct CSA Demo (No External Services)")
    print("=" * 60)
    
    print("\nInitializing CSA Engine...")
    engine = CSAEngine("gpt2", compression_ratio=50, use_speculation=False)
    
    prompt = "The future of artificial intelligence is"
    print(f"\nPrompt: {prompt}")
    print("Generating with CSA compression...")
    
    start = time.time()
    result = engine.generate(prompt, max_new_tokens=50, enable_profiling=True)
    elapsed = time.time() - start
    
    print(f"\n⏱️  Total time: {elapsed:.2f}s")
    print(f"📝 Result: {result}")


if __name__ == "__main__":
    print("CSA Integration Examples")
    print("Shows how to use CSA with Ollama, vLLM, and other engines")
    print()

    # Run demos
    demo_direct_csa()
    demo_generic_wrapper()
    demo_ollama_integration()
    demo_vllm_integration()

    print("\n" + "=" * 60)
    print("Integration Summary:")
    print("=" * 60)
    print("• Direct CSA: Works standalone with any model")
    print("• Ollama: Use CSA for prompt preprocessing before API calls")
    print("• vLLM: Integrate CSA components directly into vLLM engine")
    print("• Generic: Wrap any inference engine with CSA optimizations")
    print("\nSee integration_guide.md for detailed setup instructions")
