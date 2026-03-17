"""Benchmark RVV vs torch_native INT8 GEMM."""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
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


# Target shapes: hidden_size=1536, intermediate_size=8960
STANDARD_CONFIGS = [
    BenchmarkConfig(1, 1536, 1536, "Q proj decode"),
    BenchmarkConfig(1, 17920, 1536, "FFN up decode"),
    BenchmarkConfig(1, 1536, 8960, "FFN down decode"),
    BenchmarkConfig(4, 1536, 1536, "Q proj batch=4"),
    BenchmarkConfig(4, 17920, 1536, "FFN up batch=4"),
    BenchmarkConfig(4, 1536, 8960, "FFN down batch=4"),
    BenchmarkConfig(8, 1536, 1536, "Q proj batch=8"),
    BenchmarkConfig(8, 17920, 1536, "FFN up batch=8"),
    BenchmarkConfig(8, 1536, 8960, "FFN down batch=8"),
    BenchmarkConfig(16, 1536, 1536, "Q proj batch=16"),
    BenchmarkConfig(16, 17920, 1536, "FFN up batch=16"),
    BenchmarkConfig(16, 1536, 8960, "FFN down batch=16"),
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

    mat1 = torch.randint(0, 100, (M, K), dtype=torch.uint8)
    mat2 = torch.randint(-50, 50, (N, K), dtype=torch.int8)
    scales1 = torch.rand(M, dtype=torch.float32)
    scales2 = torch.rand(N, dtype=torch.float32)
    bias = torch.randn(N, dtype=torch.float32)
    out_dtype = torch.float16

    if backend_name == "rvv":
        if not HAS_SGL_KERNEL:
            raise RuntimeError("sgl_kernel is not available")

        use_packed_format = False
        packed_mat2 = None
        try:
            packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
            use_packed_format = True
        except (AttributeError, Exception):
            pass

        for _ in range(warmup):
            if use_packed_format and packed_mat2 is not None:
                torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                    mat1, packed_mat2, scales1, scales2, bias, out_dtype, True
                )
            else:
                torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                    mat1, mat2, scales1, scales2, bias, out_dtype, False
                )

        start = time.time()
        for _ in range(num_iterations):
            if use_packed_format and packed_mat2 is not None:
                torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                    mat1, packed_mat2, scales1, scales2, bias, out_dtype, True
                )
            else:
                torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                    mat1, mat2, scales1, scales2, bias, out_dtype, False
                )
        end = time.time()
    else:
        mat1_f16 = mat1.to(torch.float16)
        mat2_f16 = mat2.to(torch.float16)
        bias_f16 = bias.to(torch.float16)

        for _ in range(warmup):
            torch.mm(mat1_f16, mat2_f16.t()) + bias_f16

        start = time.time()
        for _ in range(num_iterations):
            torch.mm(mat1_f16, mat2_f16.t()) + bias_f16
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
    params = f"[{c.M:4}x{c.K:4}] @ [{c.N:4}x{c.K:4}] INT8"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVV INT8 GEMM Kernel")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV INT8 GEMM Kernel Benchmark")
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
