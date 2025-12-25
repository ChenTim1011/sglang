"""
Benchmark for RVV INT8 GEMM kernel.

This script benchmarks the RVV optimized INT8 GEMM implementation
used for W8A8 quantization against a PyTorch Float16 baseline (simulating dequantized execution).

Usage:
    python benchmark/bench_rvv_gemm_int8.py

Note:
    - Packed format: weights are in [NB, K, BLOCK_N] layout for optimal memory access.
      This provides better cache utilization compared to unpacked [N, K] format.
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Callable

import torch

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

# CI environment detection
IS_CI = os.getenv("CI", "false").lower() == "true"

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


@dataclass
class BenchmarkConfig:
    M: int
    N: int
    K: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    pytorch_ms: float
    pytorch_std: float
    rvv_ms: float | None
    rvv_std: float | None
    speedup: float | None
    correct: bool | None


# Benchmark Configurations
TINYLLAMA_CONFIGS = [
    BenchmarkConfig(1, 2048, 2048, "TinyLlama QKV decode"),
    BenchmarkConfig(1, 5632, 2048, "TinyLlama FFN up decode"),
    BenchmarkConfig(1, 2048, 5632, "TinyLlama FFN down decode"),
    BenchmarkConfig(4, 2048, 2048, "TinyLlama QKV batch=4"),
    BenchmarkConfig(32, 2048, 2048, "TinyLlama QKV batch=32"),  # Prefill-like
]

SMALL_CONFIGS = [
    BenchmarkConfig(1, 64, 64, "Small 64x64"),
    BenchmarkConfig(4, 256, 256, "Small 256x256"),
]

ALL_CONFIGS = SMALL_CONFIGS if IS_CI else SMALL_CONFIGS + TINYLLAMA_CONFIGS


def benchmark_function(
    fn: Callable, warmup: int = 5, repeat: int = 20
) -> tuple[float, float]:
    # Warmup
    for _ in range(warmup):
        fn()

    # Timed runs
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append((end - start) * 1000)

    mean_ms = sum(times) / len(times)
    variance = sum((t - mean_ms) ** 2 for t in times) / len(times)
    std_ms = variance**0.5
    return mean_ms, std_ms


def run_benchmark(
    config: BenchmarkConfig, warmup: int = 5, repeat: int = 20
) -> BenchmarkResult:
    """
    Run benchmark for a single configuration.

    Uses packed weight format (is_packed=True) which matches production use case.

    Args:
        config: Benchmark configuration
        warmup: Number of warmup iterations
        repeat: Number of timed iterations

    Returns:
        BenchmarkResult with timing data
    """
    M, N, K = config.M, config.N, config.K

    # Inputs for INT8 Kernel
    mat1 = torch.randint(0, 100, (M, K), dtype=torch.uint8)
    mat2 = torch.randint(-50, 50, (N, K), dtype=torch.int8)
    scales1 = torch.rand(M, dtype=torch.float32)
    scales2 = torch.rand(N, dtype=torch.float32)
    bias = torch.randn(N, dtype=torch.float32)
    out_dtype = torch.float16

    # Inputs for PyTorch Baseline (Float16)
    mat1_f16 = mat1.to(torch.float16)
    mat2_f16 = mat2.to(torch.float16)
    bias_f16 = bias.to(torch.float16)

    # Benchmark PyTorch FP16 GEMM
    def pytorch_fn():
        # y = x @ w.T + b
        return torch.mm(mat1_f16, mat2_f16.t()) + bias_f16

    pytorch_ms, pytorch_std = benchmark_function(pytorch_fn, warmup, repeat)

    # Benchmark RVV INT8
    rvv_ms, rvv_std, speedup, correct = None, None, None, None

    if HAS_SGL_KERNEL and hasattr(torch.ops.sgl_kernel, "int8_scaled_mm_cpu"):
        try:
            # Packed format: [NB, K, BLOCK_N] layout for optimal memory access
            packed_mat2 = None
            use_packed_format = False

            if hasattr(torch.ops.sgl_kernel, "convert_weight_packed"):
                try:
                    # Pack the weight matrix for RVV optimized access pattern
                    packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
                    use_packed_format = True
                except Exception as e:
                    print(
                        f"Warning: Failed to pack weights: {e}, falling back to unpacked format"
                    )
                    use_packed_format = False

            def rvv_fn():
                if use_packed_format and packed_mat2 is not None:
                    # is_packed=True means weights are in packed format [NB, K, BLOCK_N]
                    return torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                        mat1, packed_mat2, scales1, scales2, bias, out_dtype, True
                    )
                else:
                    # is_packed=False means weights are in unpacked format [N, K] with strided access
                    return torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                        mat1, mat2, scales1, scales2, bias, out_dtype, False
                    )

            rvv_ms, rvv_std = benchmark_function(rvv_fn, warmup, repeat)

            # Simple correctness check
            # Note: Precision differences expected between INT8 and FP16
            rvv_out = rvv_fn()
            correct = rvv_out.shape == (M, N) and torch.all(torch.isfinite(rvv_out))

            if rvv_ms > 0:
                speedup = pytorch_ms / rvv_ms

        except Exception as e:
            print(f"RVV INT8 Error: {e}")

    return BenchmarkResult(
        config, pytorch_ms, pytorch_std, rvv_ms, rvv_std, speedup, correct
    )


def print_result(result: BenchmarkResult):
    c = result.config
    print(f"  {c.description:<30} [{c.M}x{c.K}] @ [{c.N}x{c.K}]")
    print(f"    PyTorch FP16: {result.pytorch_ms:8.3f} ms")
    if result.rvv_ms:
        status = "✓" if result.correct else "?"
        print(
            f"    RVV INT8:     {result.rvv_ms:8.3f} ms  [{status}] Speedup: {result.speedup:.2f}x"
        )
    else:
        print("    RVV INT8:     N/A")


def main():
    print("=" * 60)
    print("RVV INT8 GEMM Benchmark (vs PyTorch FP16)")
    print("=" * 60)
    print()

    results = []
    for config in ALL_CONFIGS:
        res = run_benchmark(config)
        results.append(res)
        print_result(res)
        print()


if __name__ == "__main__":
    main()
