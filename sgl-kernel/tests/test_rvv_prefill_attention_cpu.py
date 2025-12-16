"""
Unit Test for prefill_cache_cpu kernel (RVV optimized).

This test verifies that the CPU prefill cache kernel correctly writes
new K/V tokens into the paged KV cache buffers.
Designed to run on RISC-V hardware platforms (e.g., Banana Pi).

Usage on RISC-V hardware:
    cd ~/.local_riscv_env/workspace/sglang/sgl-kernel
    pytest tests/test_rvv_prefill_attention_cpu.py -v
"""

import os
import sys

import pytest
import torch

# Add parent directory to path to import sgl_kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


def _has_prefill_cache_cpu() -> bool:
    """Check if prefill_cache_cpu op is available."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "prefill_cache_cpu")
    except (AttributeError, RuntimeError):
        return False


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
    """
    Naive reference implementation of prefill cache using PyTorch.
    """
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

        # Write to buffer
        for j in range(new_len):
            token_idx = req_to_token[req_idx, prefix_len + j].item()
            k_buffer[token_idx] = k_src[j]
            v_buffer[token_idx] = v_src[j]


@pytest.mark.skipif(
    not _has_prefill_cache_cpu(), reason="prefill_cache_cpu not available"
)
@pytest.mark.parametrize("num_heads", [4, 8, 16])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [32, 128, 256])
@pytest.mark.parametrize("extend_len", [16, 32, 128])
@pytest.mark.parametrize("num_requests", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_prefill_cache_cpu(
    num_heads, head_dim, seq_len, extend_len, num_requests, dtype
):
    """Test prefill_cache_cpu with various configurations and data types."""

    actual_seq_len = max(seq_len, extend_len)

    torch.manual_seed(42)
    head_dim_v = head_dim

    max_context_len = actual_seq_len + 16
    max_total_tokens = num_requests * max_context_len

    # Create buffers
    k_buffer = torch.zeros((max_total_tokens, num_heads, head_dim), dtype=dtype)
    v_buffer = torch.zeros((max_total_tokens, num_heads, head_dim_v), dtype=dtype)

    # Create request mapping
    req_to_token = torch.zeros((num_requests, max_context_len), dtype=torch.int32)
    req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    # Setup requests
    seq_lens_list = []
    extend_seq_lens_list = []
    extend_start_loc_list = []

    current_loc = 0
    used_tokens = 0

    for i in range(num_requests):
        s_len = actual_seq_len
        e_len = extend_len

        seq_lens_list.append(s_len)
        extend_seq_lens_list.append(e_len)
        extend_start_loc_list.append(current_loc)
        current_loc += e_len

        # Assign tokens (sequential for simplicity)
        indices = torch.arange(used_tokens, used_tokens + s_len, dtype=torch.int32)
        used_tokens += s_len

        req_to_token[i, :s_len] = indices

    seq_lens = torch.tensor(seq_lens_list, dtype=torch.int64)
    extend_seq_lens = torch.tensor(extend_seq_lens_list, dtype=torch.int64)
    extend_start_loc = torch.tensor(extend_start_loc_list, dtype=torch.int64)

    total_new_tokens = current_loc

    # Create new K/V data
    k_new = torch.randn((total_new_tokens, num_heads, head_dim), dtype=dtype)
    v_new = torch.randn((total_new_tokens, num_heads, head_dim_v), dtype=dtype)

    # Run naive implementation (on a copy)
    k_buffer_ref = k_buffer.clone()
    v_buffer_ref = v_buffer.clone()
    naive_prefill_cache(
        k_new,
        v_new,
        k_buffer_ref,
        v_buffer_ref,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_start_loc,
        extend_seq_lens,
    )

    # Run kernel
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

    # Compare
    atol = 1e-3
    rtol = 1e-3

    torch.testing.assert_close(k_buffer, k_buffer_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(v_buffer, v_buffer_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
