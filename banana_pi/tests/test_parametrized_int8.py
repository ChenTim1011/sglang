"""
Parametrized Test for INT8 KV Cache

This test runs multiple configurations to ensure INT8 quantization
works correctly across different parameter combinations.

Usage:
    python test_parametrized_int8.py
    python test_parametrized_int8.py --quick  # Run fewer configurations
"""

import argparse
import os
import sys
from itertools import product
from typing import Dict, List, Tuple

import torch

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError as e:
    print(f"Error: Failed to import sgl_kernel: {e}")
    HAS_SGL_KERNEL = False
    sys.exit(1)

# Import test utilities (required, no fallback)
from test_utils import (
    check_system_state,
    compare_tensors_fair,
    generate_fair_kv_buffers,
)


def test_decode_configuration(
    num_heads: int, head_dim: int, seq_len: int, batch_size: int, seed: int = 42
) -> Dict[str, float]:
    """Test decode attention for a specific configuration."""
    device = "cpu"
    dtype = torch.float16

    torch.manual_seed(seed)

    # Prepare inputs
    q = torch.randn(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    max_tokens = batch_size * seq_len

    # Generate fair KV buffers
    k_buffer_fp16, v_buffer_fp16, _, _ = generate_fair_kv_buffers(
        max_tokens, num_heads, head_dim, torch.float16, device, seed
    )
    k_buffer_int8, v_buffer_int8, k_scale, v_scale = generate_fair_kv_buffers(
        max_tokens, num_heads, head_dim, torch.int8, device, seed
    )

    # Metadata
    req_to_token = torch.arange(max_tokens, dtype=torch.int64, device=device).reshape(
        batch_size, seq_len
    )
    req_pool_indices = torch.arange(batch_size, dtype=torch.int64, device=device)
    seq_lens = torch.full((batch_size,), seq_len, dtype=torch.int64, device=device)

    # Outputs
    output_fp16 = torch.zeros(
        batch_size, num_heads, head_dim, dtype=dtype, device=device
    )
    output_int8 = torch.zeros(
        batch_size, num_heads, head_dim, dtype=dtype, device=device
    )
    attn_logits = torch.zeros(
        batch_size, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
    )

    dummy_key = torch.zeros(batch_size, num_heads, head_dim, dtype=dtype, device=device)
    dummy_value = torch.zeros(
        batch_size, num_heads, head_dim, dtype=dtype, device=device
    )
    dummy_loc = torch.zeros(batch_size, dtype=torch.int64, device=device)

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    # Run FP16
    torch.ops.sgl_kernel.decode_attention_cpu(
        q,
        k_buffer_fp16,
        v_buffer_fp16,
        output_fp16,
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

    # Run INT8
    attn_logits.zero_()
    torch.ops.sgl_kernel.decode_attention_int8_cpu(
        q,
        k_buffer_int8,
        v_buffer_int8,
        output_int8,
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

    # Compare outputs
    comparison = compare_tensors_fair(output_fp16, output_int8)

    return comparison


def main():
    parser = argparse.ArgumentParser(description="Parametrized INT8 KV Cache Test")
    parser.add_argument("--quick", action="store_true", help="Run fewer configurations")
    args = parser.parse_args()

    if not HAS_SGL_KERNEL:
        print("Error: sgl_kernel not available")
        return

    print("=" * 80)
    print("Parametrized INT8 KV Cache Test")
    print("=" * 80)

    # Check system state
    check_system_state()

    # Define test configurations
    if args.quick:
        # Quick test: fewer configurations
        seq_lens = [32, 128, 512]
        batch_sizes = [1, 4]
        head_dims = [32, 64]
        num_heads_list = [16, 32]
    else:
        # Full test: comprehensive configurations
        seq_lens = [32, 64, 128, 256, 512, 1024]
        batch_sizes = [1, 2, 4, 8]
        head_dims = [32, 64, 128]
        num_heads_list = [8, 16, 32]

    # Generate all combinations
    configurations = list(product(seq_lens, batch_sizes, head_dims, num_heads_list))

    print(f"\nTesting {len(configurations)} configurations...")
    print(f"SeqLens: {seq_lens}")
    print(f"BatchSizes: {batch_sizes}")
    print(f"HeadDims: {head_dims}")
    print(f"NumHeads: {num_heads_list}")
    print()

    results = []
    passed = 0
    failed = 0

    for idx, (seq_len, batch_size, head_dim, num_heads) in enumerate(configurations, 1):
        try:
            comparison = test_decode_configuration(
                num_heads=num_heads,
                head_dim=head_dim,
                seq_len=seq_len,
                batch_size=batch_size,
            )

            cos_sim = comparison["cosine_similarity"]
            passed_test = cos_sim > 0.99

            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(
                f"[{idx:3d}/{len(configurations)}] {status} | "
                f"SeqLen={seq_len:4d}, BS={batch_size:2d}, HeadDim={head_dim:3d}, Heads={num_heads:2d} | "
                f"CosSim={cos_sim:.6f}"
            )

            if passed_test:
                passed += 1
            else:
                failed += 1

            results.append(
                {
                    "seq_len": seq_len,
                    "batch_size": batch_size,
                    "head_dim": head_dim,
                    "num_heads": num_heads,
                    "cosine_similarity": cos_sim,
                    "mse": comparison["mse"],
                    "max_error": comparison["max_error"],
                    "passed": passed_test,
                }
            )
        except Exception as e:
            print(
                f"[{idx:3d}/{len(configurations)}] ❌ ERROR | "
                f"SeqLen={seq_len:4d}, BS={batch_size:2d}, HeadDim={head_dim:3d}, Heads={num_heads:2d} | "
                f"Error: {str(e)[:50]}"
            )
            failed += 1

    # Summary
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Total Configurations: {len(configurations)}")
    print(f"Passed: {passed} ({passed/len(configurations)*100:.1f}%)")
    print(f"Failed: {failed} ({failed/len(configurations)*100:.1f}%)")

    if failed > 0:
        print("\nFailed Configurations:")
        for r in results:
            if not r["passed"]:
                print(
                    f"  SeqLen={r['seq_len']}, BS={r['batch_size']}, "
                    f"HeadDim={r['head_dim']}, Heads={r['num_heads']}: "
                    f"CosSim={r['cosine_similarity']:.6f}"
                )

    # Overall result
    if failed == 0:
        print("\n✅ All configurations PASSED")
    else:
        print(f"\n⚠️  {failed} configuration(s) FAILED")


if __name__ == "__main__":
    main()
