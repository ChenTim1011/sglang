"""Benchmark RVV vs torch_native INT8 extend attention."""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
from _rvv_bench_utils import (
    IS_CI,
    ExtendMockForwardBatch,
    ExtendMockLayer,
    ExtendMockRunner,
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
    num_reqs: int
    seq_len: int
    extend_len: int
    num_heads: int
    head_dim: int
    mode: str  # "extend" or "prefill"
    description: str
    kv_dtype: torch.dtype = torch.int8


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# Target attention shape: num_heads=12, head_dim=128
INT8_CONFIGS = [
    BenchmarkConfig(1, 128, 128, 12, 128, "prefill", "INT8 prefill Seq=128"),
    BenchmarkConfig(1, 512, 512, 12, 128, "prefill", "INT8 prefill Seq=512"),
    BenchmarkConfig(1, 512, 128, 12, 128, "extend", "INT8 extend Seq=512 Ext=128"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 32, 8, 12, 128, "extend", "CI INT8 extend"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = INT8_CONFIGS


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    runner = ExtendMockRunner(config.num_heads, config.head_dim, config.kv_dtype)
    layer = ExtendMockLayer(config.num_heads, config.head_dim)

    if backend_name == "rvv":
        # RVVAttnBackend auto-detects INT8 from the KV buffer dtype; no subclass needed.
        backend = RVVAttnBackend(runner)
    else:
        backend = TorchNativeAttnBackend(runner)

    actual_extend_len = (
        config.seq_len if config.mode == "prefill" else config.extend_len
    )
    actual_seq_len = config.seq_len

    batch = ExtendMockForwardBatch(
        config.num_reqs,
        actual_seq_len,
        actual_extend_len,
        config.num_heads,
        config.head_dim,
        config.kv_dtype,
    )
    backend.init_forward_metadata(batch)

    total_extend_len = config.num_reqs * actual_extend_len
    dtype = torch.float16
    q = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)
    k = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)
    v = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)

    for _ in range(warmup):
        backend.forward_extend(q, k, v, layer, batch)

    start = time.time()
    for _ in range(num_iterations):
        backend.forward_extend(q, k, v, layer, batch)
    end = time.time()

    return (end - start) / num_iterations


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 3 if quick else 10
    warmup = 1 if quick else 3

    rvv_time = run_single_backend("rvv", config, iterations, warmup)

    if config.kv_dtype == torch.int8:
        config_fp16 = BenchmarkConfig(
            config.num_reqs,
            config.seq_len,
            config.extend_len,
            config.num_heads,
            config.head_dim,
            config.mode,
            config.description,
            torch.float16,
        )
        torch_time = run_single_backend("torch_native", config_fp16, iterations, warmup)
    else:
        torch_time = run_single_backend("torch_native", config, iterations, warmup)

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=torch_time / rvv_time,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    params = f"Mode={c.mode}, Seq={c.seq_len}, Ext={c.extend_len}, INT8"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Extend Attention (INT8)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Extend Attention Benchmark (INT8)")
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
