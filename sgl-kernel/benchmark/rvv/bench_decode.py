"""Benchmark RVV vs torch_native FP16 decode attention."""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
from _rvv_bench_utils import (
    IS_CI,
    create_decode_mock_forward_batch,
    create_decode_mock_layer,
    create_decode_mock_runner,
    print_benchmark_result,
)

try:
    from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("SGLang not found")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    num_requests: int
    num_heads: int
    head_dim: int
    seq_len: int
    description: str
    kv_dtype: torch.dtype = torch.float16


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float
    throughput_rvv: float


# Target attention shape: num_heads=12, head_dim=128
STANDARD_CONFIGS = [
    BenchmarkConfig(1, 12, 128, 128, "decode BS=1 seq=128"),
    BenchmarkConfig(4, 12, 128, 128, "decode BS=4 seq=128"),
    BenchmarkConfig(8, 12, 128, 128, "decode BS=8 seq=128"),
    BenchmarkConfig(1, 12, 128, 512, "decode BS=1 seq=512"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 12, 128, 32, "CI decode"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    v_head_dim = config.head_dim

    mock_runner = create_decode_mock_runner(
        config.num_heads, config.head_dim, v_head_dim, config.kv_dtype
    )
    mock_layer = create_decode_mock_layer(config.num_heads, config.head_dim, v_head_dim)

    if backend_name == "rvv":
        backend = RVVAttnBackend(mock_runner)
    elif backend_name == "torch_native":
        backend = TorchNativeAttnBackend(mock_runner)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    dtype = torch.float16
    q = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    k = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    v = torch.randn(config.num_requests, config.num_heads, v_head_dim, dtype=dtype)

    forward_batch = create_decode_mock_forward_batch(
        config.num_requests,
        config.num_heads,
        config.head_dim,
        v_head_dim,
        config.seq_len,
        config.kv_dtype,
    )

    backend.init_forward_metadata(forward_batch)

    for _ in range(warmup):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 3 if quick else 10
    warmup = 1 if quick else 3

    rvv_time = run_single_backend("rvv", config, iterations, warmup)
    torch_time = run_single_backend("torch_native", config, iterations, warmup)

    speedup = torch_time / rvv_time
    throughput = config.num_requests / rvv_time

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=speedup,
        throughput_rvv=throughput,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    params = f"BS={c.num_requests}, H={c.num_heads}, D={c.head_dim}"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
        throughput_rvv=result.throughput_rvv,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Decode Attention (FP16/BF16)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Decode Attention Benchmark (FP16)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print(f"Quick Mode: {args.quick}")
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
