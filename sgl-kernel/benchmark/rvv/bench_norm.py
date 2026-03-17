"""Benchmark RVV vs torch_native normalization kernels."""

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
    hidden_size: int
    dtype: torch.dtype
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    op: str
    rvv_ms: float
    torch_ms: float
    speedup: float


# fmt: off
STANDARD_CONFIGS = [
    BenchmarkConfig(1,    4096, torch.float16,  "B=1 H=4096 FP16"),
    BenchmarkConfig(1,    4096, torch.bfloat16, "B=1 H=4096 BF16"),
    BenchmarkConfig(128,  4096, torch.float16,  "B=128 H=4096 FP16"),
    BenchmarkConfig(128,  4096, torch.bfloat16, "B=128 H=4096 BF16"),
    BenchmarkConfig(256,  4096, torch.float16,  "B=256 H=4096 FP16"),
    BenchmarkConfig(256,  4096, torch.bfloat16, "B=256 H=4096 BF16"),
]

CI_CONFIGS = [
    BenchmarkConfig(1,   4096, torch.float16,  "B=1 H=4096 FP16"),
    BenchmarkConfig(128, 4096, torch.bfloat16, "B=128 H=4096 BF16"),
]
# fmt: on

ALL_CONFIGS = CI_CONFIGS if IS_CI else STANDARD_CONFIGS

WARMUP = 2 if IS_CI else 5
REPEAT = 5 if IS_CI else 20


# ── Torch-native reference implementations ──────────────────────────────


def _torch_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype) * weight


def _torch_fused_add_rmsnorm(x, residual, weight, eps):
    xf = x.float() + residual.float()
    residual_out = xf.to(x.dtype)
    var = xf.pow(2).mean(-1, keepdim=True)
    out = (xf * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return out, residual_out


def _torch_l2norm(x, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps)).to(x.dtype)


def _torch_gemma_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)


def _torch_gemma3_rmsnorm(x, weight, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)


def _torch_fused_rmsnorm_gated(x, weight, gate, eps):
    xf = x.float()
    var = xf.pow(2).mean(-1, keepdim=True)
    normed = (xf * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return (normed * F.silu(gate.float())).to(x.dtype)


def _torch_gemma_fused_add_rmsnorm(x, residual, weight, eps):
    xf = x.float() + residual.float()
    residual_out = xf.to(x.dtype)
    var = xf.pow(2).mean(-1, keepdim=True)
    out = (xf * torch.rsqrt(var + eps) * (1.0 + weight.float())).to(x.dtype)
    return out, residual_out


def _torch_layernorm(x, weight, eps):
    xf = x.float()
    var, mean = torch.var_mean(xf, dim=-1, keepdim=True, correction=0)
    return ((xf - mean) * torch.rsqrt(var + eps)).to(x.dtype) * weight


def _torch_fused_add_layernorm(x, residual, weight, eps):
    xf = x.float() + residual.float()
    residual_out = xf.to(x.dtype)
    var, mean = torch.var_mean(xf, dim=-1, keepdim=True, correction=0)
    out = ((xf - mean) * torch.rsqrt(var + eps)).to(x.dtype) * weight
    return out, residual_out


# ── Benchmark runner ─────────────────────────────────────────────────────

ALL_OPS = [
    "rmsnorm",
    "fused_add_rmsnorm",
    "l2norm",
    "layernorm",
    "gemma_rmsnorm",
    "gemma3_rmsnorm",
    "fused_rmsnorm_gated",
    "gemma_fused_add_rmsnorm",
    "fused_add_layernorm",
]


def run_op(backend: str, op_name: str, config: BenchmarkConfig) -> float:
    """Run one (backend, op) combination and return mean latency in ms."""
    b, h = config.batch, config.hidden_size
    eps = 1e-6
    x = torch.randn(b, h, dtype=config.dtype)
    weight = torch.randn(h, dtype=config.dtype)

    if op_name in (
        "fused_add_rmsnorm",
        "gemma_fused_add_rmsnorm",
        "fused_add_layernorm",
    ):
        residual = torch.randn(b, h, dtype=config.dtype)

    if op_name == "fused_rmsnorm_gated":
        gate = torch.randn(b, h, dtype=config.dtype)

    if backend == "rvv":
        if op_name == "rmsnorm":
            fn = lambda: torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, eps)
        elif op_name == "fused_add_rmsnorm":
            fn = lambda: torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(
                x.clone(), residual.clone(), weight, eps
            )
        elif op_name == "l2norm":
            fn = lambda: torch.ops.sgl_kernel.l2norm_cpu(x, eps)
        elif op_name == "layernorm":
            fn = lambda: torch.ops.sgl_kernel.layernorm_cpu(x.clone(), weight, eps)
        elif op_name == "gemma_rmsnorm":
            fn = lambda: torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
        elif op_name == "gemma3_rmsnorm":
            fn = lambda: torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, weight, eps)
        elif op_name == "fused_rmsnorm_gated":
            fn = lambda: torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(
                x, weight, gate, eps
            )
        elif op_name == "gemma_fused_add_rmsnorm":
            fn = lambda: torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(
                x.clone(), residual.clone(), weight, eps
            )
        elif op_name == "fused_add_layernorm":
            fn = lambda: torch.ops.sgl_kernel.fused_add_layernorm_cpu(
                x.clone(), residual.clone(), weight, eps
            )
    else:  # torch_native
        if op_name == "rmsnorm":
            fn = lambda: _torch_rmsnorm(x, weight, eps)
        elif op_name == "fused_add_rmsnorm":
            fn = lambda: _torch_fused_add_rmsnorm(x, residual, weight, eps)
        elif op_name == "l2norm":
            fn = lambda: _torch_l2norm(x, eps)
        elif op_name == "layernorm":
            fn = lambda: _torch_layernorm(x, weight, eps)
        elif op_name == "gemma_rmsnorm":
            fn = lambda: _torch_gemma_rmsnorm(x, weight, eps)
        elif op_name == "gemma3_rmsnorm":
            fn = lambda: _torch_gemma3_rmsnorm(x, weight, eps)
        elif op_name == "fused_rmsnorm_gated":
            fn = lambda: _torch_fused_rmsnorm_gated(x, weight, gate, eps)
        elif op_name == "gemma_fused_add_rmsnorm":
            fn = lambda: _torch_gemma_fused_add_rmsnorm(x, residual, weight, eps)
        elif op_name == "fused_add_layernorm":
            fn = lambda: _torch_fused_add_layernorm(x, residual, weight, eps)

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
    shape_str = f"[{c.batch}x{c.hidden_size}] {str(c.dtype).split('.')[-1]}"
    label = f"{result.op:<28} {c.description}"
    print_benchmark_result(
        label,
        shape_str,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    print("=" * 70)
    print("RVV Norm Kernel Benchmark")
    print("=" * 70)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode:  {IS_CI}")
    print("-" * 70)

    results = []
    for op in ALL_OPS:
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
