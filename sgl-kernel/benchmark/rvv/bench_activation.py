"""Benchmark RVV vs torch_native activation kernels."""

import platform
import sys
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from _rvv_bench_utils import (
    IS_CI,
    benchmark_function,
    print_benchmark_result,
    print_benchmark_summary,
)

try:
    import sgl_kernel  # noqa: F401

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("sgl_kernel not found")
    sys.exit(1)


@dataclass
class BenchmarkConfig:
    batch: int
    seq_len: int
    dim: int  # half-dimension (kernel output width)
    dtype: torch.dtype
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    op: str
    rvv_ms: float
    torch_ms: float
    speedup: float


# Intermediate size for this benchmark target.
# Kernel input shape uses 2 * dim on the last axis.
# fmt: off
STANDARD_CONFIGS = [
    BenchmarkConfig(1,      1,  8960, torch.float16,  "decode 1 tok FP16"),
    BenchmarkConfig(1,      1,  8960, torch.bfloat16, "decode 1 tok BF16"),
    BenchmarkConfig(1,    128,  8960, torch.float16,  "decode 128 tok FP16"),
    BenchmarkConfig(1,    128,  8960, torch.bfloat16, "decode 128 tok BF16"),
    BenchmarkConfig(4,     64,  8960, torch.float16,  "decode BS=4 FP16"),
    BenchmarkConfig(4,     64,  8960, torch.bfloat16, "decode BS=4 BF16"),
    BenchmarkConfig(1,   512,  8960, torch.float16,  "prefill 512 FP16"),
    BenchmarkConfig(1,   512,  8960, torch.bfloat16, "prefill 512 BF16"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 1, 8960, torch.float16,  "1 tok FP16"),
    BenchmarkConfig(1, 128, 8960, torch.bfloat16, "128 tok BF16"),
]
# fmt: on

ALL_CONFIGS = CI_CONFIGS if IS_CI else STANDARD_CONFIGS

WARMUP = 2 if IS_CI else 5
REPEAT = 5 if IS_CI else 20


def run_op(backend: str, op_name: str, config: BenchmarkConfig) -> float:
    """Run one (backend, op) combination and return mean latency in ms."""
    b, s, d = config.batch, config.seq_len, config.dim
    x = torch.randn(b, s, 2 * d, dtype=config.dtype)

    if backend == "rvv":
        if op_name == "silu":
            fn = lambda: torch.ops.sgl_kernel.silu_and_mul_cpu(x)  # noqa: E731
        elif op_name == "gelu_tanh":
            fn = lambda: torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)  # noqa: E731
        else:  # gelu_exact
            fn = lambda: torch.ops.sgl_kernel.gelu_and_mul_cpu(x)  # noqa: E731
    else:  # torch_native
        half = x[..., :d]
        gate = x[..., d:]
        if op_name == "silu":
            fn = lambda: F.silu(half) * gate  # noqa: E731
        elif op_name == "gelu_tanh":
            fn = lambda: F.gelu(half, approximate="tanh") * gate  # noqa: E731
        else:
            fn = lambda: F.gelu(half, approximate="none") * gate  # noqa: E731

    mean_ms, _ = benchmark_function(fn, warmup=WARMUP, repeat=REPEAT)
    return mean_ms


def run_benchmark(config: BenchmarkConfig, op_name: str) -> BenchmarkResult:
    rvv_ms = run_op("rvv", op_name, config)
    torch_ms = run_op("torch_native", op_name, config)
    return BenchmarkResult(
        config=config,
        op=op_name,
        rvv_ms=rvv_ms,
        torch_ms=torch_ms,
        speedup=torch_ms / rvv_ms,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    shape_str = f"[{c.batch}x{c.seq_len}x{2*c.dim}] {str(c.dtype).split('.')[-1]}"
    label = f"{result.op:<12} {c.description}"
    print_benchmark_result(
        label,
        shape_str,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    print("=" * 70)
    print("RVV Activation Kernel Benchmark (SiLU / GELU-tanh / GELU-exact)")
    print("=" * 70)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode:  {IS_CI}")
    print("-" * 70)

    results = []
    for op in ("silu", "gelu_tanh", "gelu_exact"):
        for config in ALL_CONFIGS:
            try:
                result = run_benchmark(config, op)
                results.append(result)
                print_result(result)
            except Exception as e:
                print(f"FAILED {op} {config.description}: {e}")

    print_benchmark_summary(results)


if __name__ == "__main__":
    main()
