"""
Performance benchmark for Multi-LoRA MoE BGMV CUDA kernels.

Compares:
  1. PyTorch reference (naive loop)
  2. CUDA BGMV MoE (our kernel)

Run:
    python -m pytest flashinfer/tests/moe/bench_bgmv_moe.py -v -s

Or standalone:
    python flashinfer/tests/moe/bench_bgmv_moe.py

Copyright (c) 2025 by FlashInfer team.
Licensed under the Apache License, Version 2.0.
"""

import time
from dataclasses import dataclass
from typing import Callable

import torch

from test_bgmv_moe import generate_test_data, reference_bgmv_moe


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark run."""
    name: str
    num_tokens: int
    hidden_size: int
    rank: int
    num_experts: int
    top_k: int
    num_loras: int
    num_slices: int
    dtype: torch.dtype


# Model configurations to benchmark
CONFIGS = [
    # Decode regime (1-32 tokens)
    BenchmarkConfig("Decode-1tok-Qwen3", 1, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-4tok-Qwen3", 4, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-Qwen3", 8, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-16tok-Qwen3", 16, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-32tok-Qwen3", 32, 2048, 16, 128, 2, 4, 1, torch.bfloat16),

    # Prefill regime (64-1024 tokens)
    BenchmarkConfig("Prefill-64tok-Qwen3", 64, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Prefill-128tok-Qwen3", 128, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Prefill-256tok-Qwen3", 256, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Prefill-512tok-Qwen3", 512, 2048, 16, 128, 2, 4, 1, torch.bfloat16),

    # Larger model (Nemotron-Super-120B style)
    BenchmarkConfig("Decode-1tok-Nemotron", 1, 4096, 32, 256, 4, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-Nemotron", 8, 4096, 32, 256, 4, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-32tok-Nemotron", 32, 4096, 32, 256, 4, 4, 1, torch.bfloat16),
    BenchmarkConfig("Prefill-128tok-Nemotron", 128, 4096, 32, 256, 4, 4, 1, torch.bfloat16),

    # Different ranks
    BenchmarkConfig("Decode-8tok-rank8", 8, 2048, 8, 64, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-rank32", 8, 2048, 32, 64, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-rank64", 8, 2048, 64, 64, 2, 4, 1, torch.bfloat16),
]


def benchmark_fn(fn: Callable, warmup: int = 10, repeat: int = 100) -> float:
    """Benchmark a function, return median time in microseconds."""
    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    # Timed runs
    times = []
    for _ in range(repeat):
        torch.cuda.synchronize()
        start = time.perf_counter_ns()
        fn()
        torch.cuda.synchronize()
        end = time.perf_counter_ns()
        times.append((end - start) / 1000.0)  # ns -> us

    times.sort()
    return times[len(times) // 2]  # median


def run_benchmark(config: BenchmarkConfig):
    """Run benchmark for a single configuration."""
    data = generate_test_data(
        config.num_tokens, config.hidden_size, config.rank,
        config.num_experts, config.top_k, config.num_loras,
        config.num_slices, config.dtype
    )

    results = {"config": config.name}

    # 1. PyTorch reference
    def ref_fn():
        return reference_bgmv_moe(
            data["x"], data["lora_a_weights"], data["lora_b_weights"],
            data["sorted_token_ids"], data["expert_ids"],
            data["lora_indices"], data["topk_weights"]
        )

    # Only benchmark reference for small configs (it's very slow for large ones)
    if config.num_tokens <= 32 and config.num_experts <= 128:
        ref_time = benchmark_fn(ref_fn, warmup=3, repeat=10)
        results["pytorch_ref_us"] = ref_time
    else:
        results["pytorch_ref_us"] = float("nan")

    # 2. CUDA BGMV MoE
    try:
        from flashinfer.fused_moe.bgmv_moe import bgmv_moe

        def cuda_fn():
            return bgmv_moe(
                data["x"], data["lora_a_weights"], data["lora_b_weights"],
                data["sorted_token_ids"], data["expert_ids"],
                data["lora_indices"], data["topk_weights"],
                config.num_experts
            )

        cuda_time = benchmark_fn(cuda_fn)
        results["cuda_bgmv_moe_us"] = cuda_time
    except ImportError:
        results["cuda_bgmv_moe_us"] = float("nan")

    # Compute speedup
    if results["pytorch_ref_us"] != float("nan") and results["cuda_bgmv_moe_us"] != float("nan"):
        results["speedup_vs_ref"] = results["pytorch_ref_us"] / results["cuda_bgmv_moe_us"]
    else:
        results["speedup_vs_ref"] = float("nan")

    return results


def main():
    """Run all benchmarks and print results."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\n{'='*80}")
    print(f"BGMV MoE Kernel Benchmark")
    print(f"Device: {device_name}")
    print(f"{'='*80}\n")

    # Header
    print(f"{'Config':<30} {'PyTorch Ref (μs)':>18} {'CUDA BGMV (μs)':>16} {'Speedup':>10}")
    print(f"{'-'*30} {'-'*18} {'-'*16} {'-'*10}")

    for config in CONFIGS:
        results = run_benchmark(config)

        ref_str = f"{results['pytorch_ref_us']:.1f}" if results['pytorch_ref_us'] == results['pytorch_ref_us'] else "N/A"
        cuda_str = f"{results['cuda_bgmv_moe_us']:.1f}" if results['cuda_bgmv_moe_us'] == results['cuda_bgmv_moe_us'] else "N/A"
        speedup_str = f"{results['speedup_vs_ref']:.1f}x" if results['speedup_vs_ref'] == results['speedup_vs_ref'] else "N/A"

        print(f"{results['config']:<30} {ref_str:>18} {cuda_str:>16} {speedup_str:>10}")

    print(f"\n{'='*80}")


if __name__ == "__main__":
    main()
