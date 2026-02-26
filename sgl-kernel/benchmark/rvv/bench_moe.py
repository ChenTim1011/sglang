# Benchmark RVV vs torch_native FP16 MoE

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from utils import (
    IS_CI,
    print_benchmark_result,
    print_benchmark_summary,
)

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("sgl_kernel not found")
    sys.exit(1)


# CPU Quant Method from SGLang
class CPUQuantMethod:
    UNQUANT = 0


@dataclass
class BenchmarkConfig:
    m: int  # num tokens
    n: int  # intermediate size
    k: int  # hidden size
    e: int  # num experts
    topk: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


STANDARD_CONFIGS = [
    BenchmarkConfig(1, 64, 64, 4, 2, "Small Decode (BS=1)"),
    BenchmarkConfig(4, 128, 128, 8, 2, "Small Batch (BS=4)"),
    BenchmarkConfig(4, 2048, 5632, 8, 2, "FFN-like MoE (BS=4)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 64, 64, 4, 2, "Small Decode (BS=1)"),
    BenchmarkConfig(4, 128, 128, 8, 2, "Small Batch (BS=4)"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


def torch_naive_fused_moe(a, w1, w2, score, topk, renormalize=False):
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)
    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)
    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            w1_ = w1[i]
            w2_ = w2[i]

            x = a[mask].float() @ w1_.t().float()
            x = F.silu(x[:, : w1_.shape[0] // 2]) * x[:, w1_.shape[0] // 2 :]

            out[mask] = (x @ w2_.t().float()).to(a.dtype)

    out = out.to(a.dtype)
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    m, n, k, e, topk = config.m, config.n, config.k, config.e, config.topk

    dtype = torch.float16
    a = torch.randn((m, k), dtype=dtype) / 10
    w1 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
    w2 = torch.randn((e, k, n), dtype=dtype) / 10
    score = torch.randn((m, e), dtype=dtype)

    if backend_name == "rvv":
        if not HAS_SGL_KERNEL:
            raise RuntimeError("sgl_kernel is not available")

        topk_weights, topk_ids = torch.ops.sgl_kernel.grouped_topk_cpu(
            a, score, topk, False, 1, 1, 0, None, None
        )
        packed_w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
        packed_w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)

        for _ in range(warmup):
            torch.ops.sgl_kernel.fused_experts_cpu(
                a,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                True,
                CPUQuantMethod.UNQUANT,
                None,
                None,
                None,
                None,
                None,
                True,
            )

        start = time.time()
        for _ in range(num_iterations):
            torch.ops.sgl_kernel.fused_experts_cpu(
                a,
                packed_w1,
                packed_w2,
                topk_weights,
                topk_ids,
                True,
                CPUQuantMethod.UNQUANT,
                None,
                None,
                None,
                None,
                None,
                True,
            )
        end = time.time()
    else:
        for _ in range(warmup):
            torch_naive_fused_moe(a, w1, w2, score, topk, False)

        start = time.time()
        for _ in range(num_iterations):
            torch_naive_fused_moe(a, w1, w2, score, topk, False)
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
    params = f"Tok={c.m:<4} Exp={c.e} N={c.n:<4} K={c.k:<4} Topk={c.topk} FP16"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Fused MoE Kernel (FP16)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Fused MoE Kernel Benchmark (FP16)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print("-" * 60)
    print()

    results = []
    for config in configs:
        try:
            result = run_benchmark(config, quick=args.quick)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")

    print_benchmark_summary(results)


if __name__ == "__main__":
    main()
