"""
Benchmark script for RVV prefill cache kernel.

This script benchmarks the RVV optimized prefill cache implementation
against a naive PyTorch implementation.

Usage:
    python3 bench_rvv_prefill_attention.py [--quick]
"""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import torch

# ============================================================================
# Configuration & Environment Setup
# ============================================================================

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("sgl_kernel not found")
    sys.exit(1)

# ============================================================================
# Benchmark Data Classes
# ============================================================================


@dataclass
class BenchmarkConfig:
    num_reqs: int
    seq_len: int
    extend_len: int
    num_heads: int
    head_dim: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# ============================================================================
# Benchmark Configurations
# ============================================================================

STANDARD_CONFIGS = [
    BenchmarkConfig(1, 128, 128, 8, 64, "Prefill (Seq=128)"),
    BenchmarkConfig(1, 512, 512, 8, 64, "Prefill (Seq=512)"),
    BenchmarkConfig(4, 128, 128, 8, 64, "Batch Prefill (B=4, Seq=128)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 32, 32, 4, 32, "CI Prefill"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS

# ============================================================================
# Naive Implementation
# ============================================================================


def naive_prefill_cache(
    k_new,  # [total_new_tokens, num_heads, head_dim]
    v_new,  # [total_new_tokens, num_heads, head_dim_v]
    k_buffer,  # [max_total_tokens, num_heads, head_dim]
    v_buffer,  # [max_total_tokens, num_heads, head_dim_v]
    req_to_token,  # [num_seqs, max_context_len]
    req_pool_indices,  # [num_seqs]
    seq_lens,  # [num_seqs] - total sequence lengths
    extend_start_loc,  # [num_seqs] - start locations in k_new
    extend_seq_lens,  # [num_seqs] - number of new tokens
):
    num_seqs = seq_lens.shape[0]

    for i in range(num_seqs):
        req_idx = req_pool_indices[i].item()
        seq_len = seq_lens[i].item()
        new_len = extend_seq_lens[i].item()
        start_loc = extend_start_loc[i].item()
        prefix_len = seq_len - new_len

        if new_len <= 0:
            continue

        # Get new tokens for this request
        k_src = k_new[start_loc : start_loc + new_len]
        v_src = v_new[start_loc : start_loc + new_len]

        # Write to buffer (Vectorized version)
        token_indices = req_to_token[req_idx, prefix_len : prefix_len + new_len].long()
        k_buffer.index_copy_(0, token_indices, k_src)
        v_buffer.index_copy_(0, token_indices, v_src)


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_single_impl(impl_name, config, num_iterations=20, warmup=5):
    # Setup data
    torch.manual_seed(42)
    dtype = torch.float16
    head_dim_v = config.head_dim

    max_context_len = config.seq_len + 16
    max_total_tokens = config.num_reqs * max_context_len

    k_buffer = torch.zeros(
        (max_total_tokens, config.num_heads, config.head_dim), dtype=dtype
    )
    v_buffer = torch.zeros(
        (max_total_tokens, config.num_heads, head_dim_v), dtype=dtype
    )

    req_to_token = torch.zeros((config.num_reqs, max_context_len), dtype=torch.int32)
    req_pool_indices = torch.arange(config.num_reqs, dtype=torch.int64)

    seq_lens_list = []
    extend_seq_lens_list = []
    extend_start_loc_list = []

    current_loc = 0
    used_tokens = 0

    for i in range(config.num_reqs):
        s_len = config.seq_len
        e_len = config.extend_len

        seq_lens_list.append(s_len)
        extend_seq_lens_list.append(e_len)
        extend_start_loc_list.append(current_loc)
        current_loc += e_len

        indices = torch.arange(used_tokens, used_tokens + s_len, dtype=torch.int32)
        used_tokens += s_len
        req_to_token[i, :s_len] = indices

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64)
    extend_seq_lens = torch.tensor(extend_seq_lens_list, dtype=torch.int64)
    extend_start_loc = torch.tensor(extend_start_loc_list, dtype=torch.int64)

    total_new_tokens = current_loc
    k_new = torch.randn(
        (total_new_tokens, config.num_heads, config.head_dim), dtype=dtype
    )
    v_new = torch.randn((total_new_tokens, config.num_heads, head_dim_v), dtype=dtype)

    # Warmup
    for _ in range(warmup):
        if impl_name == "rvv":
            torch.ops.sgl_kernel.prefill_cache_cpu(
                k_new,
                v_new,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_start_loc,
                extend_seq_lens,
            )
        else:
            naive_prefill_cache(
                k_new,
                v_new,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_start_loc,
                extend_seq_lens,
            )

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        if impl_name == "rvv":
            torch.ops.sgl_kernel.prefill_cache_cpu(
                k_new,
                v_new,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_start_loc,
                extend_seq_lens,
            )
        else:
            naive_prefill_cache(
                k_new,
                v_new,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_start_loc,
                extend_seq_lens,
            )
    end = time.time()

    return (end - start) / num_iterations


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 10 if quick else 100
    warmup = 2 if quick else 10

    rvv_time = run_single_impl("rvv", config, iterations, warmup)
    torch_time = run_single_impl("torch", config, iterations, warmup)

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=torch_time / rvv_time,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    print(f"  {c.description:<30} | Seq={c.seq_len}, Ext={c.extend_len}")
    print(f"    RVV:   {result.rvv_ms:8.3f} ms")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVV Prefill Cache")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Prefill Cache Benchmark")
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


if __name__ == "__main__":
    main()
