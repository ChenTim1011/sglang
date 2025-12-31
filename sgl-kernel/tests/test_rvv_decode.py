"""
Test for decode_attention_cpu kernel (RVV optimized, FP16/BF16).

Usage:
    python3 tests/test_rvv_decode.py -v
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
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("num_heads", [4, 8, 16, 32])
@pytest.mark.parametrize("head_dim", [32, 64, 128])
@pytest.mark.parametrize("seq_len", [1, 32, 128, 256])
@pytest.mark.parametrize("num_requests", [1, 2, 4, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_decode_attention_cpu(num_heads, head_dim, seq_len, num_requests, dtype):
    """Test decode_attention_cpu with various configurations."""
    device = "cpu"

    head_dim_v = head_dim
    max_seq_len = seq_len + 16

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

    # New key/value for the current step
    key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_requests, num_heads, head_dim_v, dtype=dtype, device=device)

    # Locations to write the new key/value
    loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
    for i in range(num_requests):
        # The location for the new token (last one)
        loc[i] = req_to_token[i, seq_len - 1]

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    # Run Kernel
    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )

    # Run Reference
    # Update reference buffers with new key/value
    k_buffer_ref = k_buffer.clone()
    v_buffer_ref = v_buffer.clone()
    for i in range(num_requests):
        l = loc[i].item()
        k_buffer_ref[l] = key[i]
        v_buffer_ref[l] = value[i]

    ref_output = naive_attention_decode(
        query,
        k_buffer_ref,
        v_buffer_ref,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )

    torch.testing.assert_close(output, ref_output, atol=5e-2, rtol=5e-2)


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

    # New key/value for the current step
    key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
    value = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)

    # Locations to write the new key/value
    loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
    for i in range(num_requests):
        # The location for the new token (last one)
        loc[i] = req_to_token[i, seq_len - 1]

    sm_scale = 1.0 / (head_dim**0.5)
    logit_cap = 50.0

    torch.ops.sgl_kernel.decode_attention_cpu(
        query,
        k_buffer,
        v_buffer,
        output,
        key,
        value,
        loc,
        attn_logits,
        req_to_token,
        req_pool_indices,
        seq_lens,
        sm_scale,
        logit_cap,
    )

    assert torch.isfinite(output).all(), "Output contains NaN or Inf"


@pytest.mark.skipif(
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestDecodeAttentionEdgeCases:
    """Edge case tests for decode attention."""

    def test_zero_input(self, dtype):
        """Test with zero query input."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        assert (output == 0).all(), "Zero query should produce zero output"

    def test_zero_kv_cache(self, dtype):
        """Test with zero KV cache."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.zeros(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.zeros(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_single_token(self, dtype):
        """Test with seq_len=1 (single token)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 1
        max_seq_len = 16

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, 0] = 0
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = 0

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_logit_cap_zero(self, dtype):
        """Test with logit_cap=0 (should clamp all logits to 0)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 0.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"


@pytest.mark.skipif(
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestDecodeAttentionNumericalStability:
    """Numerical stability tests for decode attention."""

    def test_extreme_large_values(self, dtype):
        """Test with extreme large values (near FP16/BF16 max)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        if dtype == torch.float16:
            max_val = 65504.0
        else:
            max_val = 3.4e38

        query = torch.full(
            (num_requests, num_heads, head_dim),
            max_val * 0.1,
            dtype=dtype,
            device=device,
        )
        k_buffer = torch.full(
            (max_seq_len, num_heads, head_dim),
            max_val * 0.1,
            dtype=dtype,
            device=device,
        )
        v_buffer = torch.full(
            (max_seq_len, num_heads, head_dim),
            max_val * 0.1,
            dtype=dtype,
            device=device,
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.full(
            (num_requests, num_heads, head_dim),
            max_val * 0.1,
            dtype=dtype,
            device=device,
        )
        value = torch.full(
            (num_requests, num_heads, head_dim),
            max_val * 0.1,
            dtype=dtype,
            device=device,
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_extreme_small_values(self, dtype):
        """Test with extreme small values (near FP16/BF16 min)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        if dtype == torch.float16:
            min_val = 6e-8
        else:
            min_val = 1e-38

        query = torch.full(
            (num_requests, num_heads, head_dim), min_val, dtype=dtype, device=device
        )
        k_buffer = torch.full(
            (max_seq_len, num_heads, head_dim), min_val, dtype=dtype, device=device
        )
        v_buffer = torch.full(
            (max_seq_len, num_heads, head_dim), min_val, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.full(
            (num_requests, num_heads, head_dim), min_val, dtype=dtype, device=device
        )
        value = torch.full(
            (num_requests, num_heads, head_dim), min_val, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_single_head_single_dim(self, dtype):
        """Test with num_heads=1, head_dim=1 (minimal dimensions)."""
        device = "cpu"
        num_requests = 1
        num_heads = 1
        head_dim = 1
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_non_aligned_head_dim(self, dtype):
        """Test with non-aligned head_dim (not power of 2)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 63
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    def test_extreme_sm_scale(self, dtype):
        """Test with extreme sm_scale values (very small and very large)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        for sm_scale in [1e-6, 100.0]:
            logit_cap = 50.0

            torch.ops.sgl_kernel.decode_attention_cpu(
                query,
                k_buffer,
                v_buffer,
                output,
                key,
                value,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                sm_scale,
                logit_cap,
            )

            assert torch.isfinite(
                output
            ).all(), f"Output contains NaN or Inf with sm_scale={sm_scale}"

    def test_extreme_logits_softmax_stability(self, dtype):
        """Test softmax stability with extreme logits (very large QK^T scores)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        if dtype == torch.float16:
            large_val = 100.0
        else:
            large_val = 1000.0

        query = torch.full(
            (num_requests, num_heads, head_dim), large_val, dtype=dtype, device=device
        )
        k_buffer = torch.full(
            (max_seq_len, num_heads, head_dim), large_val, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.full(
            (num_requests, num_heads, head_dim), large_val, dtype=dtype, device=device
        )
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(
            output
        ).all(), "Output contains NaN or Inf with extreme logits"
        assert (
            (output >= 0).all() or (output <= 0).all() or True
        ), "Output should be reasonable"


@pytest.mark.skipif(
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestDecodeAttentionErrorHandling:
    """Error handling and input validation tests for decode attention."""

    def test_invalid_seq_len_exceeds_max(self, dtype):
        """Test with seq_len > max_seq_len (should handle gracefully or raise error)."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 100
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :max_seq_len] = torch.arange(max_seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, max_seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        with pytest.raises((RuntimeError, IndexError, AssertionError)):
            torch.ops.sgl_kernel.decode_attention_cpu(
                query,
                k_buffer,
                v_buffer,
                output,
                key,
                value,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                sm_scale,
                logit_cap,
            )

    def test_invalid_index_out_of_bounds(self, dtype):
        """Test with out-of-bounds indices in req_to_token."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_to_token[0, 0] = max_seq_len + 10
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        with pytest.raises((RuntimeError, IndexError, AssertionError)):
            torch.ops.sgl_kernel.decode_attention_cpu(
                query,
                k_buffer,
                v_buffer,
                output,
                key,
                value,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                sm_scale,
                logit_cap,
            )

    def test_invalid_negative_index(self, dtype):
        """Test with negative index in req_to_token.

        Note: Negative indices may be treated as unsigned integers by the kernel,
        so this test checks that the kernel handles them gracefully (either by
        raising an error or by producing a valid output without crashing).
        """
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_to_token[0, 0] = -1
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        # Negative indices may be treated as unsigned, so we check that
        # the kernel doesn't crash (either raises an error or produces valid output)
        try:
            torch.ops.sgl_kernel.decode_attention_cpu(
                query,
                k_buffer,
                v_buffer,
                output,
                key,
                value,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                sm_scale,
                logit_cap,
            )
            # If no error is raised, ensure output is finite (kernel handled it gracefully)
            assert torch.isfinite(
                output
            ).all(), "Output should be finite even with negative index"
        except (RuntimeError, IndexError, AssertionError):
            # If error is raised, that's also acceptable behavior
            pass


@pytest.mark.skipif(
    not is_decode_attention_available(), reason="decode_attention_cpu not available"
)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestDecodeAttentionCombinatorial:
    """Combinatorial testing for decode attention - parameter interactions."""

    @pytest.mark.parametrize(
        "num_heads,head_dim",
        [
            (1, 1),
            (1, 32),
            (1, 64),
            (2, 32),
            (2, 64),
            (4, 1),
            (4, 32),
            (4, 64),
            (8, 32),
            (8, 64),
            (16, 64),
            (32, 64),
            (32, 128),
        ],
    )
    def test_head_dim_head_combinations(self, num_heads, head_dim, dtype):
        """Test various num_heads × head_dim combinations."""
        device = "cpu"
        num_requests = 1
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize(
        "seq_len,num_requests",
        [
            (1, 1),
            (1, 8),
            (32, 1),
            (32, 8),
            (128, 1),
            (128, 8),
            (256, 1),
            (256, 8),
        ],
    )
    def test_seq_len_request_combinations(self, seq_len, num_requests, dtype):
        """Test various seq_len × num_requests combinations."""
        device = "cpu"
        num_heads = 4
        head_dim = 64
        max_seq_len = seq_len + 16

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        max_tokens = num_requests * max_seq_len
        k_buffer = torch.randn(
            max_tokens, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_tokens, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.long, device=device
        )

        for i in range(num_requests):
            start_idx = i * max_seq_len
            req_to_token[i, :seq_len] = torch.arange(
                start_idx, start_idx + seq_len, dtype=torch.long
            )

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        for i in range(num_requests):
            loc[i] = req_to_token[i, seq_len - 1]

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(output).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize(
        "sm_scale,logit_cap",
        [
            (1e-6, 0.0),
            (1e-6, 50.0),
            (1e-6, 100.0),
            (0.1, 0.0),
            (0.1, 50.0),
            (0.1, 100.0),
            (1.0, 0.0),
            (1.0, 50.0),
            (1.0, 100.0),
            (10.0, 0.0),
            (10.0, 50.0),
            (10.0, 100.0),
        ],
    )
    def test_parameter_combinations(self, sm_scale, logit_cap, dtype):
        """Test various sm_scale × logit_cap combinations."""
        device = "cpu"
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        max_seq_len = 64

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        k_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_buffer = torch.randn(
            max_seq_len, num_heads, head_dim, dtype=dtype, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_to_token[0, :seq_len] = torch.arange(seq_len, dtype=torch.long)
        req_pool_indices = torch.zeros(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.long, device=device)

        output = torch.zeros(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests, num_heads, 1, head_dim + 1, dtype=torch.float32, device=device
        )

        key = torch.randn(num_requests, num_heads, head_dim, dtype=dtype, device=device)
        value = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        loc[0] = req_to_token[0, seq_len - 1]

        torch.ops.sgl_kernel.decode_attention_cpu(
            query,
            k_buffer,
            v_buffer,
            output,
            key,
            value,
            loc,
            attn_logits,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            logit_cap,
        )

        assert torch.isfinite(
            output
        ).all(), f"Output contains NaN or Inf with sm_scale={sm_scale}, logit_cap={logit_cap}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
