"""
End-to-End CSA Benchmark Script.

Produces clean, reproducible "Baseline vs CSA" comparison showing:
- Tokens/sec
- Peak VRAM
- Speedup multiplier
- Compression ratio
- Output quality (word count, text preview)

Usage:
    python benchmarks/e2e_benchmark.py [--model gpt2] [--max-tokens 50] [--runs 3]
"""

import argparse
import json
import time
import torch
import numpy as np
from datetime import datetime


def get_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def get_peak_gpu_memory_mb():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / 1024 / 1024
    return 0


def run_baseline(model_name, prompt, max_tokens, num_runs=3):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    input_ids = tokenizer(prompt, return_tensors="pt").to(device)

    model_mem = get_gpu_memory_mb()

    # Warm-up
    with torch.no_grad():
        model.generate(input_ids.input_ids, max_new_tokens=5, do_sample=False,
                       pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()

    times, peak_mems, texts = [], [], []
    for _ in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        with torch.no_grad():
            output = model.generate(input_ids.input_ids, max_new_tokens=max_tokens,
                                    do_sample=False, pad_token_id=tokenizer.eos_token_id)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        times.append(elapsed)
        peak_mems.append(get_peak_gpu_memory_mb())
        texts.append(tokenizer.decode(output[0][input_ids.input_ids.shape[1]:], skip_special_tokens=True))

    del model, tokenizer
    torch.cuda.empty_cache()

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "model_memory_mb": model_mem,
        "peak_memory_mb": np.mean(peak_mems),
        "tokens_per_sec": max_tokens / np.mean(times),
        "texts": texts
    }


def run_csa(model_name, prompt, max_tokens, num_runs=3):
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from csa import CSAEngine

    engine = CSAEngine(
        model_name,
        compression_ratio=50,
        compression_frequency="once",
        skip_compression_threshold=512,
        use_speculation=False
    )

    model_mem = get_gpu_memory_mb()

    # Warm-up
    engine.generate(prompt[:50], max_new_tokens=5, enable_profiling=False)
    torch.cuda.synchronize()

    times, peak_mems, texts, comp_times = [], [], [], []
    for _ in range(num_runs):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

        start = time.time()
        text = engine.generate(prompt, max_new_tokens=max_tokens, enable_profiling=False)
        torch.cuda.synchronize()
        elapsed = time.time() - start

        times.append(elapsed)
        peak_mems.append(get_peak_gpu_memory_mb())
        texts.append(text)
        comp_times.append(getattr(engine, "compression_time", 0))

    engine.cleanup()
    del engine
    torch.cuda.empty_cache()

    return {
        "avg_time": np.mean(times),
        "std_time": np.std(times),
        "model_memory_mb": model_mem,
        "peak_memory_mb": np.mean(peak_mems),
        "compression_time": np.mean(comp_times),
        "tokens_per_sec": max_tokens / np.mean(times),
        "texts": texts
    }


def main():
    parser = argparse.ArgumentParser(description="CSA End-to-End Benchmark")
    parser.add_argument("--model", default="gpt2", help="Model name")
    parser.add_argument("--max-tokens", type=int, default=50, help="Max tokens to generate")
    parser.add_argument("--runs", type=int, default=3, help="Number of runs")
    args = parser.parse_args()

    prompt = "The future of artificial intelligence is " * 20  # ~200 tokens

    print(f"Model: {args.model}")
    print(f"Prompt length: {len(prompt)} chars")
    print(f"Max tokens: {args.max_tokens}")
    print(f"Runs: {args.runs}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print("=" * 60)

    # Baseline
    print("\n--- Baseline ---")
    baseline = run_baseline(args.model, prompt, args.max_tokens, args.runs)
    print(f"  Time: {baseline['avg_time']:.3f}s (+/- {baseline['std_time']:.3f}s)")
    print(f"  Speed: {baseline['tokens_per_sec']:.1f} tok/s")
    print(f"  Model memory: {baseline['model_memory_mb']:.0f} MB")
    print(f"  Peak memory: {baseline['peak_memory_mb']:.0f} MB")

    # CSA
    print("\n--- CSA ---")
    csa = run_csa(args.model, prompt, args.max_tokens, args.runs)
    print(f"  Time: {csa['avg_time']:.3f}s (+/- {csa['std_time']:.3f}s)")
    print(f"  Speed: {csa['tokens_per_sec']:.1f} tok/s")
    print(f"  Compression time: {csa['compression_time']:.3f}s")
    print(f"  Model memory: {csa['model_memory_mb']:.0f} MB")
    print(f"  Peak memory: {csa['peak_memory_mb']:.0f} MB")

    # Comparison
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    speedup = baseline["avg_time"] / csa["avg_time"] if csa["avg_time"] > 0 else 0
    time_change = (csa["avg_time"] - baseline["avg_time"]) / baseline["avg_time"] * 100

    print(f"  Speedup: {speedup:.2f}x")
    print(f"  Time change: {time_change:+.1f}%")
    print(f"  Baseline speed: {baseline['tokens_per_sec']:.1f} tok/s")
    print(f"  CSA speed: {csa['tokens_per_sec']:.1f} tok/s")

    # Output quality
    baseline_words = len(baseline["texts"][0].split())
    csa_words = len(csa["texts"][0].split())
    print(f"\n  Baseline output: {baseline_words} words")
    print(f"  CSA output: {csa_words} words")

    if csa_words == 0:
        print("  WARNING: CSA produced empty output!")
    elif csa_words < baseline_words * 0.5:
        print("  WARNING: CSA output significantly shorter than baseline")

    # Text preview
    print(f"\n  Baseline: {baseline['texts'][0][:100]}...")
    print(f"  CSA: {csa['texts'][0][:100]}...")

    # Export
    result = {
        "timestamp": datetime.now().isoformat(),
        "model": args.model,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
        "config": {"max_tokens": args.max_tokens, "runs": args.runs},
        "baseline": {k: v for k, v in baseline.items() if k != "texts"},
        "csa": {k: v for k, v in csa.items() if k != "texts"},
        "speedup": speedup,
        "time_change_pct": time_change
    }

    output_file = f"benchmarks/e2e_result_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main()
