"""
Test for decode_attention_cpu kernel (RVV optimized).

This test verifies that the CPU attention decode kernel produces correct results.
Designed to run on RISC-V hardware platforms (e.g., Banana Pi).

Usage on RISC-V hardware:
    cd ~/.local_riscv_env/workspace/sglang/sgl-kernel
    pytest tests/test_rvv_decode_attention_cpu.py -v
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


def is_decode_attention_available():
    """Check if decode_attention_cpu is available in sgl_kernel."""
    if not HAS_SGL_KERNEL:
        return False
    return hasattr(torch.ops.sgl_kernel, "decode_attention_cpu")


def naive_attention_decode(
    query,  # [num_requests, num_heads, head_dim]
    k_buffer,  # [max_tokens, num_heads, head_dim]
    v_buffer,  # [max_tokens, num_heads, head_dim]
    req_to_token,  # [num_requests, max_seq_len]
    req_pool_indices,  # [num_requests]
    seq_lens,  # [num_requests]
    sm_scale,
    logit_cap,
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
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("num_heads", [4, 8, 16, 32])
@pytest.mark.parametrize("head_dim", [32, 64, 96, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 128, 256])
@pytest.mark.parametrize("num_requests", [1, 2, 4, 8])  # Added batch size testing
def test_decode_attention_cpu(num_heads, head_dim, seq_len, num_requests):
    """Test decode_attention_cpu with various configurations."""
    device = "cpu"
    dtype = torch.float16  # RVV kernel typically uses FP16

    head_dim_v = head_dim  # Simplified for testing
    max_seq_len = seq_len + 16  # Buffer slightly larger

    # Create inputs
    query = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)

    # Buffer setup
    max_tokens = num_requests * max_seq_len
    k_buffer = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype, device=device)
    v_buffer = torch.randn(
        max_tokens, num_heads, head_dim_v, dtype=dtype, device=device
    )

    # Metadata
    req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.long, device=device
    )
    req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
    seq_lens = torch.tensor([seq_len] * num_requests, dtype=torch.long, device=device)

    # Initialize req_to_token with random valid indices
    for i in range(num_requests):
        # Assign tokens for this request to a specific block in the buffer
        start_idx = i * max_seq_len
        req_to_token[i, :seq_len] = torch.arange(
            start_idx, start_idx + seq_len, dtype=torch.long
        )

    # Output tensors
    output = torch.zeros(
        num_requests, num_heads, head_dim_v, dtype=dtype, device=device
    )
    attn_logits = torch.zeros(
        num_requests, num_heads, 1, head_dim_v + 1, dtype=torch.float32, device=device
    )

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0
    num_kv_splits = 1

    # Run Kernel
    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        req_to_token,
        req_pool_indices,
        seq_lens,
        attn_logits,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )

    # Run Reference
    ref_output = naive_attention_decode(
        query,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )

    # Compare
    # Relaxed tolerance for FP16
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)


@pytest.mark.skipif(
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
def test_decode_attention_cpu_numerical_stability():
    """Test decode_attention_cpu with extreme values."""
    device = "cpu"
    dtype = torch.float16

    num_requests = 1
    num_heads = 4
    head_dim = 64
    seq_len = 32
    max_seq_len = 64

    # Large values to test overflow handling (if any) or at least finite output
    query = (
        torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device) * 5.0
    )
    k_buffer = (
        torch.randn(max_seq_len, num_heads, head_dim, dtype=dtype, device=device) * 5.0
    )
    v_buffer = (
        torch.randn(max_seq_len, num_heads, head_dim, dtype=dtype, device=device) * 5.0
    )

    req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.long, device=device
    )
    req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)

    req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
    seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

    output = torch.zeros(num_requests, num_heads, head_dim, dtype=dtype, device=device)
    attn_logits = torch.zeros(
        num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
    )

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0
    num_kv_splits = 1

    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        req_to_token,
        req_pool_indices,
        seq_lens,
        attn_logits,
        num_kv_splits,
        sm_scale,
        logit_cap,
    )

    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
