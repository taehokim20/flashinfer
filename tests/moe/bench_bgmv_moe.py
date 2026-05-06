"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import os

os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")

import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch

# Import test helpers (reference impl + data generation)
sys.path.insert(0, sys.path[0])  # ensure local imports work
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
    # Prefill regime (64-512 tokens)
    BenchmarkConfig("Prefill-64tok-Qwen3", 64, 2048, 16, 128, 2, 4, 1, torch.bfloat16),
    BenchmarkConfig(
        "Prefill-128tok-Qwen3", 128, 2048, 16, 128, 2, 4, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-256tok-Qwen3", 256, 2048, 16, 128, 2, 4, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-512tok-Qwen3", 512, 2048, 16, 128, 2, 4, 1, torch.bfloat16
    ),
    # Larger model (Nemotron-Super-120B style)
    BenchmarkConfig("Decode-1tok-Nemotron", 1, 4096, 32, 256, 4, 4, 1, torch.bfloat16),
    BenchmarkConfig("Decode-8tok-Nemotron", 8, 4096, 32, 256, 4, 4, 1, torch.bfloat16),
    BenchmarkConfig(
        "Decode-32tok-Nemotron", 32, 4096, 32, 256, 4, 4, 1, torch.bfloat16
    ),
    BenchmarkConfig(
        "Prefill-128tok-Nemotron", 128, 4096, 32, 256, 4, 4, 1, torch.bfloat16
    ),
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


# ============================================================
# Triton SGMV Wrapper
# ============================================================

_triton_available = None


def _check_triton_available():
    """Check if Triton is available for the baseline kernel."""
    global _triton_available
    if _triton_available is not None:
        return _triton_available
    try:
        from triton_moe_lora_baseline import triton_moe_lora  # noqa: F401

        _triton_available = True
    except (ImportError, ModuleNotFoundError):
        try:
            import triton  # noqa: F401

            # triton is available but our baseline file isn't importable
            # Try with explicit path
            import importlib.util
            import os

            spec_path = os.path.join(
                os.path.dirname(__file__), "triton_moe_lora_baseline.py"
            )
            if os.path.exists(spec_path):
                spec = importlib.util.spec_from_file_location(
                    "triton_moe_lora_baseline", spec_path
                )
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                _triton_available = True
            else:
                _triton_available = False
        except ImportError:
            _triton_available = False
    return _triton_available


def _get_triton_module():
    """Import the standalone Triton baseline module."""
    try:
        from triton_moe_lora_baseline import triton_moe_lora

        return triton_moe_lora
    except ImportError:
        import importlib.util
        import os

        spec_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "triton_moe_lora_baseline.py"
        )
        spec = importlib.util.spec_from_file_location(
            "triton_moe_lora_baseline", spec_path
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.triton_moe_lora


# ============================================================
# Benchmark Runner
# ============================================================


def run_benchmark(config: BenchmarkConfig):
    """Run benchmark for a single configuration."""
    data = generate_test_data(
        config.num_tokens,
        config.hidden_size,
        config.rank,
        config.num_experts,
        config.top_k,
        config.num_loras,
        config.num_slices,
        config.dtype,
    )

    results = {"config": config.name}

    # 1. PyTorch reference (only for small configs)
    if config.num_tokens <= 32 and config.num_experts <= 128:

        def ref_fn():
            return reference_bgmv_moe(
                data["x"],
                data["lora_a_weights"],
                data["lora_b_weights"],
                data["sorted_token_ids"],
                data["expert_ids"],
                data["lora_indices"],
                data["topk_weights"],
            )

        ref_time = benchmark_fn(ref_fn, warmup=3, repeat=10)
        results["pytorch_ref_us"] = ref_time
    else:
        results["pytorch_ref_us"] = float("nan")

    # 2. Triton MoE LoRA baseline (pre-allocated, fair comparison)
    if _check_triton_available():
        try:
            import triton
            from triton_moe_lora_baseline import (
                _moe_lora_shrink_kernel,
                _moe_lora_expand_kernel,
            )

            num_tokens = config.num_tokens
            num_pairs = data["num_pairs"]
            rank = config.rank
            hidden_dim = config.hidden_size
            num_experts = config.num_experts
            max_loras = config.num_loras
            num_slices = config.num_slices
            feat_out = config.hidden_size
            top_k = config.top_k
            dtype = config.dtype
            device = "cuda"

            # Pre-allocate all Triton buffers (same as CUDA path)
            t_a_ptrs = torch.tensor(
                [w.data_ptr() for w in data["lora_a_weights"]],
                device=device,
                dtype=torch.uint64,
            )
            t_b_ptrs = torch.tensor(
                [w.data_ptr() for w in data["lora_b_weights"]],
                device=device,
                dtype=torch.uint64,
            )
            t_shrink_out = torch.zeros(
                num_slices, num_pairs, rank, dtype=dtype, device=device
            )
            t_output = torch.zeros(
                num_tokens, feat_out * num_slices, dtype=torch.float32, device=device
            )

            # Use vLLM's default configs for fused_moe_lora
            BLOCK_M = 64
            BLOCK_N_S = min(64, max(16, rank))  # shrink N = rank
            BLOCK_K_S = max(16, min(32, hidden_dim))  # shrink K = hidden
            BLOCK_N_E = min(64, max(16, feat_out))  # expand N = feat_out
            BLOCK_K_E = max(16, min(32, rank))  # expand K = rank

            grid_shrink = (
                triton.cdiv(num_pairs, BLOCK_M) * triton.cdiv(rank, BLOCK_N_S),
                num_slices,
            )
            grid_expand = (
                triton.cdiv(num_pairs, BLOCK_M) * triton.cdiv(feat_out, BLOCK_N_E),
                num_slices,
            )

            # Warmup (triggers Triton JIT)
            _moe_lora_shrink_kernel[grid_shrink](
                data["x"],
                t_a_ptrs,
                t_shrink_out,
                data["expert_ids"],
                data["lora_indices"],
                N=rank,
                K=hidden_dim,
                num_pairs=num_pairs,
                top_k_num=top_k,
                num_experts=num_experts,
                max_loras=max_loras,
                stride_am=data["x"].stride(0),
                stride_ak=data["x"].stride(1),
                stride_bl=data["lora_a_weights"][0].stride(0),
                stride_be=data["lora_a_weights"][0].stride(1),
                stride_bk=data["lora_a_weights"][0].stride(3),
                stride_bn=data["lora_a_weights"][0].stride(2),
                stride_cm=t_shrink_out.stride(1),
                stride_cn=t_shrink_out.stride(2),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N_S,
                BLOCK_SIZE_K=BLOCK_K_S,
            )
            _moe_lora_expand_kernel[grid_expand](
                t_shrink_out,
                t_b_ptrs,
                t_output,
                data["expert_ids"],
                data["topk_weights"],
                data["lora_indices"],
                N=feat_out,
                K=rank,
                num_pairs=num_pairs,
                num_tokens=num_tokens,
                top_k_num=top_k,
                num_experts=num_experts,
                max_loras=max_loras,
                stride_am=t_shrink_out.stride(1),
                stride_ak=t_shrink_out.stride(2),
                stride_bl=data["lora_b_weights"][0].stride(0),
                stride_be=data["lora_b_weights"][0].stride(1),
                stride_bk=data["lora_b_weights"][0].stride(3),
                stride_bn=data["lora_b_weights"][0].stride(2),
                stride_cm=t_output.stride(0),
                stride_cn=t_output.stride(1),
                BLOCK_SIZE_M=BLOCK_M,
                BLOCK_SIZE_N=BLOCK_N_E,
                BLOCK_SIZE_K=BLOCK_K_E,
            )
            torch.cuda.synchronize()

            def triton_fn():
                t_shrink_out.zero_()
                t_output.zero_()
                _moe_lora_shrink_kernel[grid_shrink](
                    data["x"],
                    t_a_ptrs,
                    t_shrink_out,
                    data["expert_ids"],
                    data["lora_indices"],
                    N=rank,
                    K=hidden_dim,
                    num_pairs=num_pairs,
                    top_k_num=top_k,
                    num_experts=num_experts,
                    max_loras=max_loras,
                    stride_am=data["x"].stride(0),
                    stride_ak=data["x"].stride(1),
                    stride_bl=data["lora_a_weights"][0].stride(0),
                    stride_be=data["lora_a_weights"][0].stride(1),
                    stride_bk=data["lora_a_weights"][0].stride(3),
                    stride_bn=data["lora_a_weights"][0].stride(2),
                    stride_cm=t_shrink_out.stride(1),
                    stride_cn=t_shrink_out.stride(2),
                    BLOCK_SIZE_M=BLOCK_M,
                    BLOCK_SIZE_N=BLOCK_N_S,
                    BLOCK_SIZE_K=BLOCK_K_S,
                )
                _moe_lora_expand_kernel[grid_expand](
                    t_shrink_out,
                    t_b_ptrs,
                    t_output,
                    data["expert_ids"],
                    data["topk_weights"],
                    data["lora_indices"],
                    N=feat_out,
                    K=rank,
                    num_pairs=num_pairs,
                    num_tokens=num_tokens,
                    top_k_num=top_k,
                    num_experts=num_experts,
                    max_loras=max_loras,
                    stride_am=t_shrink_out.stride(1),
                    stride_ak=t_shrink_out.stride(2),
                    stride_bl=data["lora_b_weights"][0].stride(0),
                    stride_be=data["lora_b_weights"][0].stride(1),
                    stride_bk=data["lora_b_weights"][0].stride(3),
                    stride_bn=data["lora_b_weights"][0].stride(2),
                    stride_cm=t_output.stride(0),
                    stride_cn=t_output.stride(1),
                    BLOCK_SIZE_M=BLOCK_M,
                    BLOCK_SIZE_N=BLOCK_N_E,
                    BLOCK_SIZE_K=BLOCK_K_E,
                )

            triton_time = benchmark_fn(triton_fn)
            results["triton_sgmv_us"] = triton_time
        except Exception as e:
            results["triton_sgmv_us"] = float("nan")
            results["triton_error"] = str(e)[:80]
    else:
        results["triton_sgmv_us"] = float("nan")

    # 3. CUDA BGMV MoE (our kernel) — pre-allocated buffers, only time kernel calls
    try:
        from flashinfer.fused_moe.bgmv_moe import (
            bgmv_moe_shrink,
            bgmv_moe_expand,
            fill_w_ptr,
        )

        num_tokens = config.num_tokens
        num_pairs = data["num_pairs"]
        rank = config.rank
        num_experts = config.num_experts
        num_slices = config.num_slices
        feat_out = config.hidden_size
        dtype = config.dtype
        device = "cuda"

        # Pre-allocate all buffers (this is what vLLM's punica_gpu.py does)
        w_ptr_a = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
        lora_stride_a = fill_w_ptr(w_ptr_a, data["lora_a_weights"][0], num_experts, 0)

        w_ptr_b = torch.zeros(num_slices, num_experts, dtype=torch.int64, device=device)
        lora_stride_b = fill_w_ptr(w_ptr_b, data["lora_b_weights"][0], num_experts, 0)

        shrink_out = torch.zeros(
            num_slices, num_pairs, rank, dtype=dtype, device=device
        )

        slice_start_loc = torch.zeros(num_slices, dtype=torch.int64, device=device)
        output_slices = [feat_out] * num_slices

        y_accum = torch.zeros(
            num_tokens, feat_out * num_slices, dtype=torch.float32, device=device
        )

        # Warmup
        bgmv_moe_shrink(
            shrink_out,
            data["x"],
            w_ptr_a,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["lora_indices"],
            lora_stride_a,
        )
        bgmv_moe_expand(
            y_accum,
            shrink_out,
            w_ptr_b,
            data["sorted_token_ids"],
            data["expert_ids"],
            data["topk_weights"],
            data["lora_indices"],
            slice_start_loc,
            output_slices,
            lora_stride_b,
        )
        torch.cuda.synchronize()

        def cuda_fn():
            shrink_out.zero_()
            y_accum.zero_()
            bgmv_moe_shrink(
                shrink_out,
                data["x"],
                w_ptr_a,
                data["sorted_token_ids"],
                data["expert_ids"],
                data["lora_indices"],
                lora_stride_a,
            )
            bgmv_moe_expand(
                y_accum,
                shrink_out,
                w_ptr_b,
                data["sorted_token_ids"],
                data["expert_ids"],
                data["topk_weights"],
                data["lora_indices"],
                slice_start_loc,
                output_slices,
                lora_stride_b,
            )

        cuda_time = benchmark_fn(cuda_fn)
        results["cuda_bgmv_moe_us"] = cuda_time
    except ImportError:
        results["cuda_bgmv_moe_us"] = float("nan")

    # Compute speedups
    cuda_t = results["cuda_bgmv_moe_us"]
    triton_t = results["triton_sgmv_us"]
    ref_t = results["pytorch_ref_us"]

    # nan != nan is True, use this for nan check
    results["speedup_vs_triton"] = (
        triton_t / cuda_t if triton_t == triton_t and cuda_t == cuda_t else float("nan")
    )
    results["speedup_vs_ref"] = (
        ref_t / cuda_t if ref_t == ref_t and cuda_t == cuda_t else float("nan")
    )

    return results


def main():
    """Run all benchmarks and print results."""
    if not torch.cuda.is_available():
        print("CUDA not available, skipping benchmarks.")
        return

    device_name = torch.cuda.get_device_name(0)
    triton_status = (
        "available"
        if _check_triton_available()
        else "NOT available (pip install triton)"
    )

    print(f"\n{'=' * 100}")
    print("Multi-LoRA MoE BGMV Kernel Benchmark")
    print(f"Device: {device_name}")
    print(f"Triton SGMV: {triton_status}")
    print(f"{'=' * 100}\n")

    # Header
    print(
        f"{'Config':<28} {'Ref (μs)':>10} {'Triton (μs)':>12} {'CUDA (μs)':>10} "
        f"{'vs Triton':>10} {'vs Ref':>8}"
    )
    print(f"{'-' * 28} {'-' * 10} {'-' * 12} {'-' * 10} {'-' * 10} {'-' * 8}")

    for config in CONFIGS:
        results = run_benchmark(config)

        def fmt(v):
            return f"{v:.1f}" if v == v else "N/A"

        def fmt_speedup(v):
            return f"{v:.2f}x" if v == v else "N/A"

        print(
            f"{results['config']:<28} "
            f"{fmt(results['pytorch_ref_us']):>10} "
            f"{fmt(results['triton_sgmv_us']):>12} "
            f"{fmt(results['cuda_bgmv_moe_us']):>10} "
            f"{fmt_speedup(results['speedup_vs_triton']):>10} "
            f"{fmt_speedup(results['speedup_vs_ref']):>8}"
        )

    print(f"\n{'=' * 100}")
    print("\nNotes:")
    print("  - 'Ref' = PyTorch naive loop (only for small configs)")
    print("  - 'Triton' = Standalone Triton MoE LoRA kernel (similar to vLLM's SGMV)")
    print("  - 'CUDA' = Our BGMV MoE CUDA kernel (this PR)")
    print("  - 'vs Triton' = Triton time / CUDA time (>1 means CUDA is faster)")
    print("  - All times are median of 100 runs after 10 warmup iterations")


if __name__ == "__main__":
    main()
