"""
Test for decode_attention_cpu kernel (RVV optimized, INT8).

Usage:
    python3 tests/test_rvv_decode_int8.py -v
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
    print("Warning: sgl_kernel not available, skipping tests")


def is_riscv_platform():
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


def is_decode_attention_int8_available():
    """Check if decode_attention_int8_cpu is available in sgl_kernel."""
    if not HAS_SGL_KERNEL:
        return False
    return hasattr(torch.ops.sgl_kernel, "decode_attention_int8_cpu")


def naive_attention_decode(
    query,  # [num_requests, num_heads, head_dim]
    k_buffer,  # [max_tokens, num_heads, head_dim]
    v_buffer,  # [max_tokens, num_heads, head_dim]
    req_to_token,  # [num_requests, max_seq_len]
    req_pool_indices,  # [num_requests]
    seq_lens,  # [num_requests]
    sm_scale,
    logit_cap,
    k_scale=1.0,
    v_scale=1.0,
):
    """
    Naive reference implementation of attention decode using PyTorch.
    Assumes the new token is already in k_buffer/v_buffer.
    """
    num_requests = query.shape[0]
    num_heads = query.shape[1]
    head_dim = query.shape[2]
    head_dim_v = v_buffer.shape[2]

    output = torch.zeros(
        num_requests, num_heads, head_dim_v, dtype=query.dtype, device=query.device
    )

    for req_idx in range(num_requests):
        seq_len = seq_lens[req_idx].item()
        pool_idx = req_pool_indices[req_idx].item()

        # Get query for this request
        q = query[req_idx]  # [num_heads, head_dim]

        # Gather keys and values from buffer
        # Indices for this request: req_to_token[req_idx, :seq_len]
        token_indices = req_to_token[req_idx, :seq_len]

        k = k_buffer[token_indices]  # [seq_len, num_heads, head_dim]
        v = v_buffer[token_indices]  # [seq_len, num_heads, head_dim_v]

        # Transpose to [num_heads, seq_len, head_dim]
        k = k.transpose(0, 1)
        v = v.transpose(0, 1)

        # Dequantize if needed
        if k.dtype == torch.int8:
            k = k.to(torch.float32) * k_scale
        if v.dtype == torch.int8:
            v = v.to(torch.float32) * v_scale

        # Ensure q is in appropriate type for computation
        if k.dtype != q.dtype:
            k = k.to(q.dtype)
        if v.dtype != q.dtype:
            v = v.to(q.dtype)

        # Attention scores: Q @ K^T
        # q: [num_heads, head_dim] -> [num_heads, 1, head_dim]
        # k: [num_heads, seq_len, head_dim] -> [num_heads, head_dim, seq_len] (transpose last two)
        scores = torch.bmm(q.unsqueeze(1), k.transpose(1, 2))  # [num_heads, 1, seq_len]
        scores = scores.squeeze(1)  # [num_heads, seq_len]

        scores *= sm_scale

        if logit_cap > 0:
            scores = torch.clamp(scores, min=-logit_cap, max=logit_cap)

        probs = torch.softmax(scores, dim=-1)  # [num_heads, seq_len]

        # Output: probs @ V
        # probs: [num_heads, 1, seq_len]
        # v: [num_heads, seq_len, head_dim_v]
        out = torch.bmm(probs.unsqueeze(1), v).squeeze(1)  # [num_heads, head_dim_v]
        output[req_idx] = out

    return output


@pytest.mark.skipif(
    not is_decode_attention_int8_available(),
    reason="decode_attention_int8_cpu not available",
)
@pytest.mark.parametrize("num_heads", [4, 8, 16])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [32, 128, 256])
@pytest.mark.parametrize("num_requests", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_decode_attention_cpu_int8(num_heads, head_dim, seq_len, num_requests, dtype):
    """Test decode_attention_cpu with INT8 KV cache."""
    device = "cpu"
    head_dim_v = head_dim
    max_seq_len = seq_len + 16

    query = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)

    # Quantization params
    k_scale = 0.01
    v_scale = 0.01

    # Create random float data then quantize
    max_tokens = num_requests * max_seq_len
    k_buffer_float = (
        torch.randn(max_tokens, num_heads, head_dim, dtype=torch.float32, device=device)
        * 5.0
    )
    v_buffer_float = (
        torch.randn(
            max_tokens, num_heads, head_dim_v, dtype=torch.float32, device=device
        )
        * 5.0
    )

    k_buffer_int8 = (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_buffer_int8 = (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)

    # Metadata
    req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.long, device=device
    )
    req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
    seq_lens = torch.tensor([seq_len] * num_requests, dtype=torch.long, device=device)

    for i in range(num_requests):
        start_idx = i * max_seq_len
        req_to_token[i, :seq_len] = torch.arange(
            start_idx, start_idx + seq_len, dtype=torch.long
        )

    output = torch.zeros(
        num_requests, num_heads, head_dim_v, dtype=dtype, device=device
    )
    attn_logits = torch.zeros(
        num_requests, num_heads, 1, head_dim_v + 1, dtype=torch.float32, device=device
    )

    # New key/value inputs (should be int8)
    key_float = torch.randn(
        num_requests, num_heads, head_dim, dtype=torch.float32, device=device
    )
    value_float = torch.randn(
        num_requests, num_heads, head_dim_v, dtype=torch.float32, device=device
    )

    key_int8 = (key_float / k_scale).round().clamp(-128, 127).to(torch.int8)
    value_int8 = (value_float / v_scale).round().clamp(-128, 127).to(torch.int8)

    loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
    for i in range(num_requests):
        loc[i] = req_to_token[i, seq_len - 1]

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    # Run INT8 Kernel
    torch.ops.sgl_kernel.decode_attention_int8_cpu(
        query,
        k_buffer_int8,
        v_buffer_int8,
        output,
        key_int8,
        value_int8,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
        k_scale,
        v_scale,
    )

    # Reference (simulated int8 dequantization)
    k_buffer_ref = k_buffer_int8.clone()
    v_buffer_ref = v_buffer_int8.clone()
    for i in range(num_requests):
        l = loc[i].item()
        k_buffer_ref[l] = key_int8[i]
        v_buffer_ref[l] = value_int8[i]

    ref_output = naive_attention_decode(
        query,
        k_buffer_ref,
        v_buffer_ref,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
        k_scale,
        v_scale,
    )

    # Tolerance: Int8 has quantization error, so we use looser tolerance
    torch.testing.assert_close(output, ref_output, atol=1e-1, rtol=1e-1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
