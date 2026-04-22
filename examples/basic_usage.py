#!/usr/bin/env python3

"""
Basic CSA usage example with GPT-2
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from csa.core import CSAEngine

def main():
    # Use GPT-2 for testing (no auth required)
    target_model = "gpt2"
    draft_model = "gpt2"  # Same for simplicity

    print("Initializing CSA Engine with GPT-2...")
    try:
        engine = CSAEngine(target_model, draft_model, compression_ratio=10)
        print("CSA Engine initialized successfully!")

        prompt = "The future of AI is"
        print(f"Generating with prompt: {prompt}")

        generated_text = engine.generate(prompt, max_new_tokens=20)
        print(f"Generated text: {generated_text}")

    except Exception as e:
        print(f"Error: {e}")
        print("Falling back to basic generation...")

        # Fallback to standard generation
        try:
            model = AutoModelForCausalLM.from_pretrained(target_model)
            tokenizer = AutoTokenizer.from_pretrained(target_model)
            tokenizer.pad_token = tokenizer.eos_token

            inputs = tokenizer(prompt, return_tensors="pt")
            with torch.no_grad():
                outputs = model.generate(**inputs, max_new_tokens=20, do_sample=True, temperature=0.7)
            generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
            print(f"Standard generation: {generated}")
        except Exception as e2:
            print(f"Fallback also failed: {e2}")

if __name__ == "__main__":
    main()