"""Benchmark RVV vs torch_native FP16 GEMM."""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from _rvv_bench_utils import (
    IS_CI,
    print_benchmark_result,
)

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("sgl_kernel not found")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    M: int
    N: int
    K: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# Target shapes: hidden_size=1536, intermediate_size=8960.
# GEMM (M,K)@(N,K): Q proj (1,1536)@(1536,1536),
# FFN up+gate fused (1,1536)@(17920,1536), FFN down (1,8960)@(1536,8960).
STANDARD_CONFIGS = [
    BenchmarkConfig(1, 1536, 1536, "Q proj decode"),
    BenchmarkConfig(1, 17920, 1536, "FFN up decode"),
    BenchmarkConfig(1, 1536, 8960, "FFN down decode"),
    BenchmarkConfig(4, 1536, 1536, "Q proj batch=4"),
    BenchmarkConfig(8, 1536, 1536, "Q proj batch=8"),
    BenchmarkConfig(256, 1536, 1536, "Q proj prefill M=256"),
    BenchmarkConfig(256, 17920, 1536, "FFN up prefill M=256"),
    BenchmarkConfig(256, 1536, 8960, "FFN down prefill M=256"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 1536, 1536, "Q proj decode"),
    BenchmarkConfig(1, 17920, 1536, "FFN up decode"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    M, N, K = config.M, config.N, config.K

    x = torch.randn(M, K, dtype=torch.float16)
    weight = torch.randn(N, K, dtype=torch.float16)
    # RVV path: _rvv_process_weight_after_loading converts bias to FP32
    bias_fp32 = torch.randn(N, dtype=torch.float32)
    # torch_native path: bias stays in the same dtype as the inputs
    bias_fp16 = bias_fp32.half()

    if backend_name == "rvv":
        if not HAS_SGL_KERNEL:
            raise RuntimeError("sgl_kernel is not available")
        weight_packed = torch.ops.sgl_kernel.convert_weight_packed(weight)

        for _ in range(warmup):
            sgl_kernel.weight_packed_linear(x, weight_packed, bias_fp32, True)

        start = time.time()
        for _ in range(num_iterations):
            sgl_kernel.weight_packed_linear(x, weight_packed, bias_fp32, True)
        end = time.time()
    else:
        for _ in range(warmup):
            F.linear(x, weight, bias_fp16)

        start = time.time()
        for _ in range(num_iterations):
            F.linear(x, weight, bias_fp16)
        end = time.time()

    return (end - start) / num_iterations


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 3 if quick else 10
    warmup = 1 if quick else 3

    rvv_time = run_single_backend("rvv", config, iterations, warmup)
    torch_time = run_single_backend("torch_native", config, iterations, warmup)

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=torch_time / rvv_time,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    params = f"[{c.M:4}x{c.K:4}] @ [{c.N:4}x{c.K:4}] FP16"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVV GEMM Kernel (FP16)")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV GEMM Kernel Benchmark (FP16)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print("-" * 60)

    results = []
    for config in configs:
        try:
            result = run_benchmark(config, quick=args.quick)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")

    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print("=" * 60)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Total Benchmarks: {len(results)}")
        print("=" * 60)
    else:
        print("No successful benchmarks completed.")


if __name__ == "__main__":
    main()
