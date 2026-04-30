"""
Speedup Test for CS Framework
Verifies 5-10x speedup with compressed KV cache + self-speculative decoding
"""

import torch
import time
import json
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_standard_generation(model, input_ids, max_new_tokens=50):
    """Test standard generation speed."""
    start = time.time()
    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            use_cache=True
        )
    elapsed = time.time() - start
    tokens_per_sec = max_new_tokens / elapsed
    return elapsed, tokens_per_sec, output


def test_cs_generation(engine, prompt, max_new_tokens=50):
    """Test CS Framework generation speed."""
    start = time.time()
    text = engine.generate(prompt, max_new_tokens=max_new_tokens, enable_profiling=False)
    elapsed = time.time() - start
    tokens_per_sec = max_new_tokens / elapsed
    return elapsed, tokens_per_sec, text


def run_speedup_benchmark():
    """Run speedup benchmark comparing standard vs CS Framework."""
    print("=" * 60)
    print("CS Framework Speedup Benchmark")
    print("=" * 60)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Load model
    print("\nLoading GPT-2...")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Test prompts
    prompts = [
        "The future of artificial intelligence",
        "Machine learning is transforming",
        "Deep learning models can"
    ]

    results = {
        "device": device,
        "model": "gpt2",
        "tests": [],
        "summary": {}
    }

    total_standard_time = 0
    total_cs_time = 0
    speedups = []

    for i, prompt in enumerate(prompts):
        print(f"\nTest {i+1}/{len(prompts)}: {prompt[:30]}...")

        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
        max_new = 50

        # Standard generation
        print("  Standard generation...")
        std_time, std_tps, std_output = test_standard_generation(model, input_ids, max_new)
        print(f"    Time: {std_time:.3f}s, Speed: {std_tps:.2f} tok/s")

        # CS Framework generation
        print("  CS Framework generation...")
        try:
            from csa import CSAEngine
            engine = CSAEngine(
                target_model="gpt2",
                compression_ratio=50,
                use_speculation=True,
                device=device
            )
            cs_time, cs_tps, cs_text = test_cs_generation(engine, prompt, max_new)
            print(f"    Time: {cs_time:.3f}s, Speed: {cs_tps:.2f} tok/s")

            speedup = std_time / cs_time if cs_time > 0 else 0
            print(f"    Speedup: {speedup:.2f}x")

            engine.cleanup()

        except Exception as e:
            print(f"    Error: {e}")
            cs_time = std_time  # No speedup if failed
            speedup = 1.0

        test_result = {
            "prompt": prompt,
            "standard_time": std_time,
            "cs_time": cs_time,
            "standard_tps": std_tps,
            "cs_tps": cs_tps,
            "speedup": speedup
        }
        results["tests"].append(test_result)

        total_standard_time += std_time
        total_cs_time += cs_time
        speedups.append(speedup)

    # Summary
    avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
    results["summary"] = {
        "total_standard_time": total_standard_time,
        "total_cs_time": total_cs_time,
        "avg_speedup": avg_speedup,
        "min_speedup": min(speedups) if speedups else 1.0,
        "max_speedup": max(speedups) if speedups else 1.0,
        "target_met": avg_speedup >= 5.0
    }

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Average speedup: {avg_speedup:.2f}x")
    print(f"Target: 5-10x speedup")
    print(f"Target met: {'YES' if avg_speedup >= 5.0 else 'NO'}")
    print(f"Min speedup: {results['summary']['min_speedup']:.2f}x")
    print(f"Max speedup: {results['summary']['max_speedup']:.2f}x")

    # Save results
    output_file = "benchmarks/speedup_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_file}")

    return results


if __name__ == "__main__":
    results = run_speedup_benchmark()
    sys.exit(0 if results["summary"]["target_met"] else 1)
