"""
Unit Test for extend_attention_cpu kernel (RVV optimized, FP16/BF16).

Usage:
    pytest tests/test_rvv_extend.py -v
"""

import os
import platform
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


def _has_extend_attention_cpu() -> bool:
    """Check if extend_attention_cpu op is available."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "extend_attention_cpu")
    except (AttributeError, RuntimeError):
        return False


def naive_extend_attention(
    q_extend,  # [total_extend_len, num_heads, head_dim]
    k_extend,  # [total_extend_len, num_heads, head_dim]
    v_extend,  # [total_extend_len, num_heads, head_dim_v]
    k_buffer,  # [max_total_tokens, num_heads, head_dim]
    v_buffer,  # [max_total_tokens, num_heads, head_dim_v]
    req_to_token,  # [num_seqs, max_context_len]
    req_pool_indices,  # [num_seqs]
    seq_lens,  # [num_seqs] - total sequence lengths
    extend_seq_lens,  # [num_seqs] - extend lengths for each request
    extend_start_loc,  # [num_seqs] - start locations in q_extend
    sm_scale,
    logit_cap,
    k_scale=1.0,
    v_scale=1.0,
):
    """
    Naive reference implementation of extend attention using PyTorch.
    """
    num_seqs = seq_lens.shape[0]
    num_heads = q_extend.shape[1]
    head_dim = q_extend.shape[2]
    head_dim_v = v_extend.shape[2]

    o_extend = torch.zeros(
        q_extend.shape[0],
        num_heads,
        head_dim_v,
        dtype=q_extend.dtype,
        device=q_extend.device,
    )

    for i in range(num_seqs):
        seq_len = seq_lens[i].item()
        extend_len = extend_seq_lens[i].item()
        prefix_len = seq_len - extend_len
        start_loc = extend_start_loc[i].item()
        req_idx = req_pool_indices[i].item()

        # Get query for this request
        q = q_extend[start_loc : start_loc + extend_len]

        # Construct full key/value for this request
        # Prefix part from buffer
        k_prefix = []
        v_prefix = []
        for j in range(prefix_len):
            token_idx = req_to_token[req_idx, j].item()

            k_val = k_buffer[token_idx]
            v_val = v_buffer[token_idx]

            # Dequantize if needed
            if k_val.dtype == torch.int8:
                k_val = k_val.to(torch.float32) * k_scale
            if v_val.dtype == torch.int8:
                v_val = v_val.to(torch.float32) * v_scale

            k_prefix.append(k_val)
            v_prefix.append(v_val)

        if prefix_len > 0:
            k_prefix = torch.stack(k_prefix).to(q.dtype)
            v_prefix = torch.stack(v_prefix).to(q.dtype)
        else:
            k_prefix = torch.empty(
                0, num_heads, head_dim, dtype=q.dtype, device=q.device
            )
            v_prefix = torch.empty(
                0, num_heads, head_dim_v, dtype=q.dtype, device=q.device
            )

        # Extend part
        k_ext = k_extend[start_loc : start_loc + extend_len]
        v_ext = v_extend[start_loc : start_loc + extend_len]

        # Concatenate: prefix (from cache) + extend (new)
        k = torch.cat([k_prefix, k_ext], dim=0)
        v = torch.cat([v_prefix, v_ext], dim=0)

        # Compute attention
        q_t = q.transpose(0, 1)
        k_t = k.transpose(0, 1)
        v_t = v.transpose(0, 1)

        scores = torch.bmm(q_t, k_t.transpose(1, 2)) * sm_scale

        # Causal mask
        mask = torch.ones_like(scores) * float("-inf")
        for r in range(extend_len):
            valid_len = prefix_len + r + 1
            mask[:, r, :valid_len] = 0

        scores = scores + mask

        if logit_cap > 0:
            scores = logit_cap * torch.tanh(scores / logit_cap)

        probs = torch.softmax(scores, dim=-1)

        output = torch.bmm(probs, v_t)
        output = output.transpose(0, 1)

        o_extend[start_loc : start_loc + extend_len] = output

    return o_extend


@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_cpu not available"
)
@pytest.mark.parametrize("num_heads", [4, 8, 16, 32])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 128, 256])
@pytest.mark.parametrize("extend_len", [16, 32, 128])
@pytest.mark.parametrize("num_requests", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_extend_attention_cpu(
    num_heads, head_dim, seq_len, extend_len, num_requests, dtype
):
    """Test extend_attention_cpu with various configurations."""
    if extend_len > seq_len:
        pytest.skip("extend_len cannot be larger than seq_len")

    device = "cpu"

    head_dim_v = head_dim
    max_context_len = seq_len + 16
    max_total_tokens = num_requests * max_context_len

    # Setup requests
    seq_lens = torch.tensor([seq_len] * num_requests, dtype=torch.int64, device=device)
    extend_seq_lens = torch.tensor(
        [extend_len] * num_requests, dtype=torch.int64, device=device
    )

    # Calculate start locations
    extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
    current_loc = 0
    for i in range(num_requests):
        extend_start_loc[i] = current_loc
        current_loc += extend_len

    req_pool_indices = torch.arange(num_requests, dtype=torch.int64, device=device)
    total_extend_len = extend_len * num_requests

    # Create tensors
    q_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    k_extend = torch.randn(
        total_extend_len, num_heads, head_dim, dtype=dtype, device=device
    )
    v_extend = torch.randn(
        total_extend_len, num_heads, head_dim_v, dtype=dtype, device=device
    )
    o_extend = torch.zeros(
        total_extend_len, num_heads, head_dim_v, dtype=dtype, device=device
    )

    # KV cache buffers
    k_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim, dtype=dtype, device=device
    )
    v_buffer = torch.randn(
        max_total_tokens, num_heads, head_dim_v, dtype=dtype, device=device
    )

    # Request to token mapping
    req_to_token = torch.zeros(
        num_requests, max_context_len, dtype=torch.int64, device=device
    )

    # Initialize mapping for the prefix part
    prefix_len = seq_len - extend_len
    if prefix_len > 0:
        for i in range(num_requests):
            start_idx = i * max_context_len
            req_to_token[i, :prefix_len] = torch.arange(
                start_idx, start_idx + prefix_len, dtype=torch.int64
            )

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0
    max_len_extend = (
        extend_len  # Assuming max_len_extend is the max of extend_len in the batch
    )

    # Run Kernel
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

    # Run Reference
    ref_output = naive_extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
    )

    # Compare
    torch.testing.assert_close(o_extend, ref_output, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
