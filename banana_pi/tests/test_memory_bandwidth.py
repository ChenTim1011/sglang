"""
Memory Bandwidth vs. Throughput Analysis for INT8 vs FP16 KV Cache

This test measures:
1. Memory bandwidth usage (estimated from memory access patterns)
2. Decoding throughput (tokens/second)
3. Correlation between bandwidth reduction and throughput improvement

Usage:
    python test_memory_bandwidth.py
    python test_memory_bandwidth.py --seq-len 2048 --batch-size 8
"""

import argparse
import os
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# Import test utilities (required, no fallback)
from test_utils import (
    StatisticalResult,
    check_system_state,
    generate_fair_kv_buffers,
    measure_with_statistics,
)

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError as e:
    print(f"Error: Failed to import sgl_kernel: {e}")
    HAS_SGL_KERNEL = False
    sys.exit(1)


@dataclass
class BandwidthResult:
    """Memory bandwidth measurement result"""

    dtype: str
    seq_len: int
    batch_size: int
    num_heads: int
    head_dim: int

    # Memory metrics
    kv_cache_size_bytes: int
    estimated_bandwidth_gb_s: float

    # Performance metrics
    latency_ms: float
    throughput_tokens_s: float

    # Efficiency
    bandwidth_per_token_gb: float


def estimate_kv_cache_size(
    seq_len: int, batch_size: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> int:
    """Estimate KV cache size in bytes."""
    bytes_per_element = 1 if dtype == torch.int8 else 2  # INT8=1, FP16/BF16=2
    # K: [batch * seq_len, num_heads, head_dim]
    # V: [batch * seq_len, num_heads, head_dim]
    total_tokens = batch_size * seq_len
    k_size = total_tokens * num_heads * head_dim * bytes_per_element
    v_size = total_tokens * num_heads * head_dim * bytes_per_element
    return k_size + v_size


def estimate_bandwidth_per_decode(
    seq_len: int, num_heads: int, head_dim: int, dtype: torch.dtype
) -> float:
    """Estimate memory bandwidth per decode step (GB)."""
    bytes_per_element = 1 if dtype == torch.int8 else 2

    # Per decode step, we read:
    # - K: [seq_len, num_heads, head_dim] from cache
    # - V: [seq_len, num_heads, head_dim] from cache
    # - Q: [1, num_heads, head_dim] (new token)
    # - Output: [1, num_heads, head_dim] (write)

    k_read = seq_len * num_heads * head_dim * bytes_per_element
    v_read = seq_len * num_heads * head_dim * bytes_per_element
    q_read = 1 * num_heads * head_dim * 2  # Q is always FP16/BF16
    output_write = 1 * num_heads * head_dim * 2  # Output is FP16/BF16

    total_bytes = k_read + v_read + q_read + output_write
    return total_bytes / (1024**3)  # Convert to GB


def measure_decode_throughput(
    num_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
    warmup: int = 10,
    seed: int = 42,
) -> Tuple[float, float]:
    """Measure decode latency and throughput."""
    device = "cpu"
    q_dtype = torch.float16  # Query is always FP16

    # Use same seed for fair comparison
    torch.manual_seed(seed)

    # Prepare inputs
    q = torch.randn(batch_size, num_heads, head_dim, dtype=q_dtype, device=device)

    max_tokens = batch_size * seq_len

    # Generate fair KV buffers (FP16 first, then quantize for INT8)
    k_buffer, v_buffer, k_scale, v_scale = generate_fair_kv_buffers(
        max_tokens, num_heads, head_dim, dtype, device, seed
    )

    # Metadata
    req_to_token = torch.arange(max_tokens, dtype=torch.int64, device=device).reshape(
        batch_size, seq_len
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    # Outputs
    output = torch.zeros(batch_size, num_heads, head_dim, dtype=q_dtype, device=device)
    attn_logits = torch.zeros(
        batch_size, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
    )

    dummy_key = torch.zeros(
        batch_size, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_value = torch.zeros(
        batch_size, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_loc = torch.zeros(batch_size, dtype=torch.int64, device=device)

    # Warmup
    for _ in range(warmup):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
            )

    # Measure
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
                k_buffer,
                v_buffer,
                output,
                dummy_key,
                dummy_value,
                dummy_loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                1.0,
                0.0,
            )
    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_latency_ms = (total_time / num_iterations) * 1000
    throughput_tokens_s = (batch_size * num_iterations) / total_time

    return avg_latency_ms, throughput_tokens_s


def measure_bandwidth_and_throughput(
    seq_len: int,
    batch_size: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
) -> BandwidthResult:
    """Measure both bandwidth and throughput."""
    dtype_str = "INT8" if dtype == torch.int8 else "FP16"

    # Estimate KV cache size
    kv_cache_size = estimate_kv_cache_size(
        seq_len, batch_size, num_heads, head_dim, dtype
    )

    # Estimate bandwidth per decode
    bandwidth_per_token = estimate_bandwidth_per_decode(
        seq_len, num_heads, head_dim, dtype
    )

    # Measure throughput - use same seed for fair comparison
    latency_ms, throughput_tokens_s = measure_decode_throughput(
        num_heads, head_dim, seq_len, batch_size, dtype, num_iterations, seed=42
    )

    # Estimate total bandwidth (GB/s)
    # Bandwidth = bytes_per_token * tokens_per_second / (1024^3)
    estimated_bandwidth_gb_s = bandwidth_per_token * throughput_tokens_s

    return BandwidthResult(
        dtype=dtype_str,
        seq_len=seq_len,
        batch_size=batch_size,
        num_heads=num_heads,
        head_dim=head_dim,
        kv_cache_size_bytes=kv_cache_size,
        estimated_bandwidth_gb_s=estimated_bandwidth_gb_s,
        latency_ms=latency_ms,
        throughput_tokens_s=throughput_tokens_s,
        bandwidth_per_token_gb=bandwidth_per_token,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Memory Bandwidth vs Throughput Analysis"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--num-iterations", type=int, default=100, help="Number of iterations"
    )
    parser.add_argument(
        "--seq-lens",
        type=int,
        nargs="+",
        default=None,
        help="Multiple sequence lengths to test",
    )
    args = parser.parse_args()

    if not HAS_SGL_KERNEL:
        print("Error: sgl_kernel not available")
        return

    print("=" * 80)
    print("Memory Bandwidth vs. Throughput Analysis")
    print("=" * 80)

    # Check system state
    check_system_state()

    seq_lens_to_test = args.seq_lens if args.seq_lens else [args.seq_len]

    results_fp16: List[BandwidthResult] = []
    results_int8: List[BandwidthResult] = []

    for seq_len in seq_lens_to_test:
        print(f"\n{'='*80}")
        print(
            f"Testing SeqLen={seq_len}, BatchSize={args.batch_size}, Heads={args.num_heads}, HeadDim={args.head_dim}"
        )
        print(f"{'='*80}")

        # FP16
        print("\n--- FP16 Baseline ---")
        result_fp16 = measure_bandwidth_and_throughput(
            seq_len,
            args.batch_size,
            args.num_heads,
            args.head_dim,
            torch.float16,
            args.num_iterations,
        )
        results_fp16.append(result_fp16)

        print(f"KV Cache Size: {result_fp16.kv_cache_size_bytes / (1024**2):.2f} MB")
        print(
            f"Bandwidth per Token: {result_fp16.bandwidth_per_token_gb * 1024:.2f} MB"
        )
        print(f"Latency: {result_fp16.latency_ms:.3f} ms")
        print(f"Throughput: {result_fp16.throughput_tokens_s:.2f} tokens/s")
        print(f"Estimated Bandwidth: {result_fp16.estimated_bandwidth_gb_s:.2f} GB/s")

        # INT8
        print("\n--- INT8 Quantized ---")
        result_int8 = measure_bandwidth_and_throughput(
            seq_len,
            args.batch_size,
            args.num_heads,
            args.head_dim,
            torch.int8,
            args.num_iterations,
        )
        results_int8.append(result_int8)

        print(f"KV Cache Size: {result_int8.kv_cache_size_bytes / (1024**2):.2f} MB")
        print(
            f"Bandwidth per Token: {result_int8.bandwidth_per_token_gb * 1024:.2f} MB"
        )
        print(f"Latency: {result_int8.latency_ms:.3f} ms")
        print(f"Throughput: {result_int8.throughput_tokens_s:.2f} tokens/s")
        print(f"Estimated Bandwidth: {result_int8.estimated_bandwidth_gb_s:.2f} GB/s")

        # Comparison
        print("\n--- Comparison ---")
        cache_size_reduction = (
            1 - result_int8.kv_cache_size_bytes / result_fp16.kv_cache_size_bytes
        ) * 100
        bandwidth_reduction = (
            1 - result_int8.bandwidth_per_token_gb / result_fp16.bandwidth_per_token_gb
        ) * 100
        throughput_speedup = (
            result_int8.throughput_tokens_s / result_fp16.throughput_tokens_s
        )
        latency_speedup = result_fp16.latency_ms / result_int8.latency_ms

        print(f"Cache Size Reduction: {cache_size_reduction:.1f}%")
        print(f"Bandwidth Reduction: {bandwidth_reduction:.1f}%")
        print(f"Throughput Speedup: {throughput_speedup:.2f}x")
        print(f"Latency Speedup: {latency_speedup:.2f}x")

    # Summary table
    print(f"\n{'='*80}")
    print("Summary Table")
    print(f"{'='*80}")
    print(
        f"{'SeqLen':<8} | {'Type':<6} | {'Cache(MB)':<12} | {'BW/Token(MB)':<15} | {'Latency(ms)':<12} | {'Throughput(t/s)':<15}"
    )
    print("-" * 80)

    for i, seq_len in enumerate(seq_lens_to_test):
        r_fp16 = results_fp16[i]
        r_int8 = results_int8[i]
        print(
            f"{seq_len:<8} | {'FP16':<6} | {r_fp16.kv_cache_size_bytes/(1024**2):<12.2f} | {r_fp16.bandwidth_per_token_gb*1024:<15.2f} | {r_fp16.latency_ms:<12.3f} | {r_fp16.throughput_tokens_s:<15.2f}"
        )
        print(
            f"{seq_len:<8} | {'INT8':<6} | {r_int8.kv_cache_size_bytes/(1024**2):<12.2f} | {r_int8.bandwidth_per_token_gb*1024:<15.2f} | {r_int8.latency_ms:<12.3f} | {r_int8.throughput_tokens_s:<15.2f}"
        )
        print("-" * 80)


if __name__ == "__main__":
    main()
