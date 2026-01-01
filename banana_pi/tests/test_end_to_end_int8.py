"""
End-to-End Test for INT8 KV Cache with Real Model Workloads

This test performs end-to-end validation of INT8 KV cache:
1. Prefill (Extend) phase with INT8 KV cache
2. Decode phase with INT8 KV cache
3. Accuracy comparison with FP16 baseline
4. Performance comparison

Usage:
    python test_end_to_end_int8.py
    python test_end_to_end_int8.py --num-requests 4 --seq-len 256
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
    compare_tensors_fair,
    generate_fair_kv_buffers,
    measure_with_statistics,
    print_statistics,
)

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError as e:
    print(f"Error: Failed to import sgl_kernel: {e}")
    HAS_SGL_KERNEL = False
    sys.exit(1)


@dataclass
class E2EResult:
    """End-to-end test result"""

    phase: str  # "prefill" or "decode"
    dtype: str  # "FP16" or "INT8"
    num_requests: int
    seq_len: int
    num_heads: int
    head_dim: int

    # Accuracy metrics
    output_cosine_sim: float = 0.0
    output_mse: float = 0.0
    output_max_err: float = 0.0

    # Performance metrics (SGLang-style)
    latency_ms: float = 0.0  # End-to-end latency
    ttft_ms: float = (
        0.0  # Time to First Token (for prefill: = latency, for decode: first token latency)
    )
    itl_ms: float = 0.0  # Inter-Token Latency (average, for decode phase)
    throughput_tokens_s: float = 0.0  # Token throughput
    throughput_requests_s: float = 0.0  # Request throughput
    tpot_ms: float = 0.0  # Token Processing Time after First Token


def test_prefill_phase(
    num_requests: int,
    seq_len: int,
    extend_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_iterations: int = 10,
    warmup: int = 20,
    seed: int = 42,
) -> Tuple[E2EResult, torch.Tensor]:
    """Test Prefill (Extend) phase."""
    device = "cpu"
    q_dtype = torch.float16
    dtype_str = "INT8" if dtype == torch.int8 else "FP16"

    print(f"\n=== Testing Prefill Phase ({dtype_str}) ===")
    print(f"Config: Requests={num_requests}, SeqLen={seq_len}, ExtendLen={extend_len}")
    print(f"        Heads={num_heads}, HeadDim={head_dim}")

    # Use same seed for reproducibility
    torch.manual_seed(seed)

    prefix_len = seq_len - extend_len
    total_extend_len = num_requests * extend_len

    # Prepare inputs
    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )

    max_total_tokens = num_requests * seq_len
    max_context_len = seq_len

    # Generate fair KV buffers (FP16 first, then quantize for INT8)
    k_buffer, v_buffer, k_scale, v_scale = generate_fair_kv_buffers(
        max_total_tokens, num_heads, head_dim, dtype, device, seed
    )

    # Metadata
    seq_lens = torch.full((num_requests,), seq_len, dtype=torch.int64, device=device)
    extend_seq_lens = torch.full(
        (num_requests,), extend_len, dtype=torch.int64, device=device
    )
    extend_start_loc = (
        torch.arange(num_requests, dtype=torch.int64, device=device) * extend_len
    )
    req_pool_indices = torch.arange(num_requests, dtype=torch.int64, device=device)

    req_to_token = torch.zeros(
        num_requests, max_context_len, dtype=torch.int64, device=device
    )
    for i in range(num_requests):
        start_idx = i * seq_len
        req_to_token[i, :seq_len] = torch.arange(
            start_idx, start_idx + seq_len, dtype=torch.int64
        )

    # Output
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim, dtype=q_dtype, device=device
    )
    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0
    max_len_extend = extend_len

    # Warmup
    for _ in range(warmup):
        if dtype == torch.int8:
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
                k_scale,
                v_scale,
            )
        else:
            torch.ops.sgl_kernel.extend_attention_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                extend_seq_lens,
                extend_start_loc,
                max_len_extend,
                sm_scale,
                logit_cap,
            )

        # Measure with statistics
        def measure_one_iteration():
            start = time.perf_counter()
            if dtype == torch.int8:
                torch.ops.sgl_kernel.extend_attention_int8_cpu(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    extend_seq_lens,
                    extend_start_loc,
                    max_len_extend,
                    sm_scale,
                    logit_cap,
                    k_scale,
                    v_scale,
                )
            else:
                torch.ops.sgl_kernel.extend_attention_cpu(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    extend_seq_lens,
                    extend_start_loc,
                    max_len_extend,
                    sm_scale,
                    logit_cap,
                )
            return (time.perf_counter() - start) * 1000

        # Run multiple times for statistics
        stats = measure_with_statistics(measure_one_iteration, num_runs=num_iterations)
        latency_ms = stats.mean
        throughput_tokens_s = (
            (total_extend_len * 1000) / latency_ms if latency_ms > 0 else 0
        )
        throughput_requests_s = (
            (num_requests * 1000) / latency_ms if latency_ms > 0 else 0
        )
    else:
        # Fallback to simple measurement
        start_time = time.perf_counter()
        for _ in range(num_iterations):
            if dtype == torch.int8:
                torch.ops.sgl_kernel.extend_attention_int8_cpu(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    extend_seq_lens,
                    extend_start_loc,
                    max_len_extend,
                    sm_scale,
                    logit_cap,
                    k_scale,
                    v_scale,
                )
            else:
                torch.ops.sgl_kernel.extend_attention_cpu(
                    q_extend,
                    k_extend,
                    v_extend,
                    o_extend,
                    k_buffer,
                    v_buffer,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    extend_seq_lens,
                    extend_start_loc,
                    max_len_extend,
                    sm_scale,
                    logit_cap,
                )
        end_time = time.perf_counter()

        total_time = end_time - start_time
        latency_ms = (total_time / num_iterations) * 1000
        throughput_tokens_s = (total_extend_len * num_iterations) / total_time
        throughput_requests_s = (num_requests * num_iterations) / total_time

    # For Prefill phase: TTFT = latency (entire prefill is the "first token")
    # ITL and TPOT don't apply to prefill
    result = E2EResult(
        phase="prefill",
        dtype=dtype_str,
        num_requests=num_requests,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        latency_ms=latency_ms,
        ttft_ms=latency_ms,  # Prefill: TTFT = total latency
        itl_ms=0.0,  # Not applicable for prefill
        throughput_tokens_s=throughput_tokens_s,
        throughput_requests_s=throughput_requests_s,
        tpot_ms=0.0,  # Not applicable for prefill
    )

    return result, o_extend.clone()


def test_decode_phase(
    num_requests: int,
    seq_len: int,
    num_heads: int,
    head_dim: int,
    dtype: torch.dtype,
    num_iterations: int = 100,
    warmup: int = 10,
    seed: int = 42,
) -> Tuple[E2EResult, torch.Tensor]:
    """Test Decode phase and return output for accuracy comparison."""
    device = "cpu"
    q_dtype = torch.float16
    dtype_str = "INT8" if dtype == torch.int8 else "FP16"

    print(f"\n=== Testing Decode Phase ({dtype_str}) ===")
    print(f"Config: Requests={num_requests}, SeqLen={seq_len}")
    print(f"        Heads={num_heads}, HeadDim={head_dim}")

    # Use same seed for reproducibility and fair comparison
    torch.manual_seed(seed)

    # Prepare inputs - use same query for both FP16 and INT8
    q = torch.randn(num_requests, num_heads, head_dim, dtype=q_dtype, device=device)

    max_tokens = num_requests * seq_len

    # Generate fair KV buffers (FP16 first, then quantize for INT8)
    k_buffer, v_buffer, k_scale, v_scale = generate_fair_kv_buffers(
        max_tokens, num_heads, head_dim, dtype, device, seed
    )

    # Metadata
    req_to_token = torch.arange(max_tokens, dtype=torch.int64, device=device).reshape(
        num_requests, seq_len
    )
    req_pool_indices = torch.arange(num_requests, dtype=torch.int64, device=device)
    seq_lens = torch.full((num_requests,), seq_len, dtype=torch.int64, device=device)

    # Outputs
    output = torch.zeros(
        num_requests, num_heads, head_dim, dtype=q_dtype, device=device
    )
    attn_logits = torch.zeros(
        num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
    )

    dummy_key = torch.zeros(
        num_requests, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_value = torch.zeros(
        num_requests, num_heads, head_dim, dtype=q_dtype, device=device
    )
    dummy_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

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
                sm_scale,
                logit_cap,
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
                sm_scale,
                logit_cap,
            )

    # Measure decode phase with proper metrics
    # We measure multiple sequential decode calls to get accurate ITL and TPOT
    # Simulate a decode sequence: first token (TTFT) + subsequent tokens (ITL/TPOT)
    num_decode_tokens = 5  # Simulate 5 tokens to measure ITL and TPOT

    # Measure first token (TTFT)
    def measure_first_token():
        start = time.perf_counter()
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
                sm_scale,
                logit_cap,
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
                sm_scale,
                logit_cap,
            )
        return (time.perf_counter() - start) * 1000

    # Measure subsequent tokens (for ITL/TPOT)
    def measure_subsequent_tokens():
        start = time.perf_counter()
        for _ in range(num_decode_tokens - 1):  # Skip first token
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
                    sm_scale,
                    logit_cap,
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
                    sm_scale,
                    logit_cap,
                )
        return (time.perf_counter() - start) * 1000

    # Measure statistics
    stats_ttft = measure_with_statistics(measure_first_token, num_runs=num_iterations)
    stats_subsequent = measure_with_statistics(
        measure_subsequent_tokens, num_runs=num_iterations
    )

    ttft_ms = stats_ttft.mean
    subsequent_time_ms = stats_subsequent.mean
    itl_ms = (
        subsequent_time_ms / (num_decode_tokens - 1) if num_decode_tokens > 1 else 0.0
    )
    tpot_ms = itl_ms  # TPOT is the same as ITL for decode phase

    # Total latency = TTFT + subsequent tokens
    latency_ms = ttft_ms + subsequent_time_ms

    # Throughput: tokens per second
    total_tokens = num_requests * num_decode_tokens
    throughput_tokens_s = (total_tokens * 1000) / latency_ms if latency_ms > 0 else 0
    throughput_requests_s = (num_requests * 1000) / latency_ms if latency_ms > 0 else 0

    # Decode phase metrics:
    # - TTFT: Time to first token (first decode call latency)
    # - ITL: Inter-token latency (average time between tokens)
    # - TPOT: Time per token after first (same as ITL for decode phase)
    result = E2EResult(
        phase="decode",
        dtype=dtype_str,
        num_requests=num_requests,
        seq_len=seq_len,
        num_heads=num_heads,
        head_dim=head_dim,
        latency_ms=latency_ms,
        ttft_ms=ttft_ms,  # First token latency
        itl_ms=itl_ms,  # Average inter-token latency
        throughput_tokens_s=throughput_tokens_s,
        throughput_requests_s=throughput_requests_s,
        tpot_ms=tpot_ms,  # Time per token after first (same as ITL)
    )

    return result, output.clone()


def compare_outputs(
    output_fp16: torch.Tensor, output_int8: torch.Tensor
) -> Dict[str, float]:
    """Compare FP16 and INT8 outputs."""
    return compare_tensors_fair(output_fp16, output_int8)


def main():
    parser = argparse.ArgumentParser(description="End-to-End INT8 KV Cache Test")
    parser.add_argument(
        "--num-requests", type=int, default=4, help="Number of requests"
    )
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length")
    parser.add_argument(
        "--extend-len", type=int, default=64, help="Extend length for prefill"
    )
    parser.add_argument(
        "--num-heads", type=int, default=32, help="Number of attention heads"
    )
    parser.add_argument("--head-dim", type=int, default=64, help="Head dimension")
    parser.add_argument(
        "--num-iterations", type=int, default=50, help="Number of iterations"
    )
    args = parser.parse_args()

    if not HAS_SGL_KERNEL:
        print("Error: sgl_kernel not available")
        return

    print("=" * 80)
    print("End-to-End INT8 KV Cache Test")
    print("=" * 80)

    # Check system state
    check_system_state()

    # Test Prefill Phase - use same seed for fair comparison
    result_prefill_fp16, output_prefill_fp16 = test_prefill_phase(
        args.num_requests,
        args.seq_len,
        args.extend_len,
        args.num_heads,
        args.head_dim,
        torch.float16,
        args.num_iterations,
        seed=42,
    )

    result_prefill_int8, output_prefill_int8 = test_prefill_phase(
        args.num_requests,
        args.seq_len,
        args.extend_len,
        args.num_heads,
        args.head_dim,
        torch.int8,
        args.num_iterations,
        seed=42,
    )

    # Compare Prefill outputs
    prefill_comparison = compare_outputs(output_prefill_fp16, output_prefill_int8)
    result_prefill_int8.output_cosine_sim = prefill_comparison["cosine_similarity"]
    result_prefill_int8.output_mse = prefill_comparison["mse"]
    result_prefill_int8.output_max_err = prefill_comparison["max_error"]

    # Test Decode Phase - use same seed for fair comparison
    result_decode_fp16, output_fp16 = test_decode_phase(
        args.num_requests,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        torch.float16,
        args.num_iterations,
        seed=42,
    )

    result_decode_int8, output_int8 = test_decode_phase(
        args.num_requests,
        args.seq_len,
        args.num_heads,
        args.head_dim,
        torch.int8,
        args.num_iterations,
        seed=42,
    )

    # Compare outputs
    decode_comparison = compare_outputs(output_fp16, output_int8)
    result_decode_int8.output_cosine_sim = decode_comparison["cosine_similarity"]
    result_decode_int8.output_mse = decode_comparison["mse"]
    result_decode_int8.output_max_err = decode_comparison["max_error"]

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)

    print("\n--- Prefill Phase (SGLang-style metrics) ---")
    print(f"FP16:")
    print(f"  Latency (TTFT): {result_prefill_fp16.latency_ms:.3f} ms")
    print(f"  Token Throughput: {result_prefill_fp16.throughput_tokens_s:.2f} tokens/s")
    print(
        f"  Request Throughput: {result_prefill_fp16.throughput_requests_s:.2f} req/s"
    )
    print(f"INT8:")
    print(f"  Latency (TTFT): {result_prefill_int8.latency_ms:.3f} ms")
    print(f"  Token Throughput: {result_prefill_int8.throughput_tokens_s:.2f} tokens/s")
    print(
        f"  Request Throughput: {result_prefill_int8.throughput_requests_s:.2f} req/s"
    )
    print(
        f"Speedup: {result_prefill_fp16.latency_ms / result_prefill_int8.latency_ms:.2f}x"
    )
    print(
        f"Accuracy: Cosine Sim={prefill_comparison['cosine_similarity']:.6f}, MSE={prefill_comparison['mse']:.6f}, MaxErr={prefill_comparison['max_error']:.6f}"
    )

    print("\n--- Decode Phase (SGLang-style metrics) ---")
    print(f"FP16:")
    print(f"  Latency: {result_decode_fp16.latency_ms:.3f} ms")
    print(f"  TTFT: {result_decode_fp16.ttft_ms:.3f} ms")
    print(f"  ITL: {result_decode_fp16.itl_ms:.3f} ms")
    print(f"  TPOT: {result_decode_fp16.tpot_ms:.3f} ms")
    print(f"  Token Throughput: {result_decode_fp16.throughput_tokens_s:.2f} tokens/s")
    print(f"  Request Throughput: {result_decode_fp16.throughput_requests_s:.2f} req/s")
    print(f"INT8:")
    print(f"  Latency: {result_decode_int8.latency_ms:.3f} ms")
    print(f"  TTFT: {result_decode_int8.ttft_ms:.3f} ms")
    print(f"  ITL: {result_decode_int8.itl_ms:.3f} ms")
    print(f"  TPOT: {result_decode_int8.tpot_ms:.3f} ms")
    print(f"  Token Throughput: {result_decode_int8.throughput_tokens_s:.2f} tokens/s")
    print(f"  Request Throughput: {result_decode_int8.throughput_requests_s:.2f} req/s")
    print(
        f"Speedup: {result_decode_fp16.latency_ms / result_decode_int8.latency_ms:.2f}x"
    )
    print(
        f"Accuracy: Cosine Sim={decode_comparison['cosine_similarity']:.6f}, MSE={decode_comparison['mse']:.6f}, MaxErr={decode_comparison['max_error']:.6f}"
    )

    # Print statistics
    print("\n--- Statistical Analysis ---")
    print("Note: Statistics are computed from multiple runs for better reliability")

    # Pass criteria
    prefill_accuracy_pass = prefill_comparison["cosine_similarity"] > 0.99
    decode_accuracy_pass = decode_comparison["cosine_similarity"] > 0.99

    if prefill_accuracy_pass and decode_accuracy_pass:
        print("\n✅ End-to-End Test PASSED")
    else:
        print("\n⚠️  Accuracy check failed:")
        if not prefill_accuracy_pass:
            print(
                f"  - Prefill: Cosine similarity {prefill_comparison['cosine_similarity']:.6f} < 0.99"
            )
        if not decode_accuracy_pass:
            print(
                f"  - Decode: Cosine similarity {decode_comparison['cosine_similarity']:.6f} < 0.99"
            )


if __name__ == "__main__":
    main()
