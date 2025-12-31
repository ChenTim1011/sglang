"""
Unit Test for extend_attention_cpu kernel (RVV optimized, INT8).

Usage:
    python3 tests/test_rvv_extend_int8.py -v
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
    """Check if extend_attention_int8_cpu op is available."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "extend_attention_int8_cpu")
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
@pytest.mark.parametrize("num_heads", [4, 8, 16])
@pytest.mark.parametrize("head_dim", [32, 64])
@pytest.mark.parametrize("seq_len_extend_len", [(32, 16), (32, 32), (128, 32)])
@pytest.mark.parametrize("num_requests", [1, 2, 4])
@pytest.mark.parametrize("dtype", [torch.float16])
def test_extend_attention_cpu_int8(
    num_heads, head_dim, seq_len_extend_len, num_requests, dtype
):
    """Test extend_attention_cpu with INT8 KV cache (Prefix part)."""
    seq_len, extend_len = seq_len_extend_len
    if extend_len > seq_len:
        pytest.skip("extend_len cannot be larger than seq_len")

    device = "cpu"
    head_dim_v = head_dim
    max_context_len = seq_len + 16
    max_total_tokens = num_requests * max_context_len

    # Quantization params
    k_scale = 0.01
    v_scale = 0.01

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
    # Query and Extend K/V are still in float/bf16
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

    # KV cache buffers (INT8)
    # Generate float data first, then quantize
    k_buffer_float = (
        torch.randn(
            max_total_tokens, num_heads, head_dim, dtype=torch.float32, device=device
        )
        * 5.0
    )
    v_buffer_float = (
        torch.randn(
            max_total_tokens, num_heads, head_dim_v, dtype=torch.float32, device=device
        )
        * 5.0
    )

    k_buffer_int8 = (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
    v_buffer_int8 = (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)

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
    max_len_extend = extend_len

    # Run Kernel (INT8)
    # Note: k_buffer and v_buffer are INT8, k_extend and v_extend are FP16/BF16
    torch.ops.sgl_kernel.extend_attention_int8_cpu(
        q_extend,
        k_extend,
        v_extend,
        o_extend,
        k_buffer_int8,
        v_buffer_int8,
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

    # Run Reference
    ref_output = naive_extend_attention(
        q_extend,
        k_extend,
        v_extend,
        k_buffer_int8,
        v_buffer_int8,
        req_to_token,
        req_pool_indices,
        seq_lens,
        extend_seq_lens,
        extend_start_loc,
        sm_scale,
        logit_cap,
        k_scale,
        v_scale,
    )

    # Compare (looser tolerance for INT8)
    torch.testing.assert_close(o_extend, ref_output, atol=1e-1, rtol=1e-1)


@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_int8_cpu not available"
)
class TestExtendAttentionInt8EdgeCases:
    """Edge case tests for INT8 extend attention."""

    def test_quantization_range_boundaries(self):
        """Test with quantization range boundaries (-128, 127, 0)."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 16
        max_context_len = seq_len + 16

        k_scale = 0.01
        v_scale = 0.01

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )
        v_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )

        k_buffer_int8[0, 0, 0] = -128
        k_buffer_int8[0, 0, 1] = 127
        k_buffer_int8[0, 0, 2] = 0
        v_buffer_int8[0, 0, 0] = -128
        v_buffer_int8[0, 0, 1] = 127
        v_buffer_int8[0, 0, 2] = 0

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        req_to_token[0, : seq_len - extend_len] = torch.arange(
            seq_len - extend_len, dtype=torch.int64
        )

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int8,
            v_buffer_int8,
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

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

    def test_extreme_small_scale(self):
        """Test with extremely small scale values."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 16
        max_context_len = seq_len + 16

        k_scale = 1e-6
        v_scale = 1e-6

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )
        v_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )

        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        req_to_token[0, : seq_len - extend_len] = torch.arange(
            seq_len - extend_len, dtype=torch.int64
        )

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int8,
            v_buffer_int8,
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

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"


@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_int8_cpu not available"
)
class TestExtendAttentionInt8NumericalStability:
    """Numerical stability tests for INT8 extend attention."""

    def test_no_nan_inf(self):
        """Test that output never contains NaN or Inf."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 16
        max_context_len = seq_len + 16

        k_scale = 0.01
        v_scale = 0.01

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_float = (
            torch.randn(
                max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
            )
            * 5.0
        )
        v_buffer_float = (
            torch.randn(
                max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
            )
            * 5.0
        )

        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        req_to_token[0, : seq_len - extend_len] = torch.arange(
            seq_len - extend_len, dtype=torch.int64
        )

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int8,
            v_buffer_int8,
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

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"


@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_int8_cpu not available"
)
class TestExtendAttentionInt8ErrorHandling:
    """Error handling and input validation tests for INT8 extend attention."""

    def test_invalid_extend_len_exceeds_seq_len(self):
        """Test with extend_len > seq_len (should handle gracefully or raise error)."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 64
        max_context_len = seq_len + 16

        k_scale = 0.01
        v_scale = 0.01

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )
        v_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        with pytest.raises((RuntimeError, IndexError, AssertionError)):
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer_int8,
                v_buffer_int8,
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

    def test_invalid_scale_zero(self):
        """Test with zero scale (should handle gracefully or raise error)."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 16
        max_context_len = seq_len + 16

        k_scale = 0.0
        v_scale = 0.0

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )
        v_buffer_int8 = torch.zeros(
            max_context_len, num_heads, head_dim, dtype=torch.int8, device=device
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        prefix_len = seq_len - extend_len
        if prefix_len > 0:
            req_to_token[0, :prefix_len] = torch.arange(prefix_len, dtype=torch.int64)

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        try:
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend,
                v_extend,
                o_extend,
                k_buffer_int8,
                v_buffer_int8,
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
            assert torch.isfinite(
                o_extend
            ).all(), "Output should be finite even with zero scale"
        except (RuntimeError, ValueError, ZeroDivisionError):
            pass


@pytest.mark.skipif(
    not _has_extend_attention_cpu(), reason="extend_attention_int8_cpu not available"
)
class TestExtendAttentionInt8Combinatorial:
    """Combinatorial testing for INT8 extend attention - parameter interactions."""

    @pytest.mark.parametrize(
        "seq_len,extend_len",
        [
            (16, 1),
            (16, 8),
            (16, 16),
            (32, 1),
            (32, 16),
            (32, 32),
            (64, 1),
            (64, 32),
            (64, 64),
            (128, 32),
            (128, 64),
        ],
    )
    def test_seq_len_extend_len_combinations(self, seq_len, extend_len):
        """Test various seq_len × extend_len combinations."""
        if extend_len > seq_len:
            pytest.skip("extend_len cannot be larger than seq_len")

        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        max_context_len = seq_len + 16

        k_scale = 0.01
        v_scale = 0.01

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )
        v_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )

        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        prefix_len = seq_len - extend_len
        if prefix_len > 0:
            req_to_token[0, :prefix_len] = torch.arange(prefix_len, dtype=torch.int64)

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int8,
            v_buffer_int8,
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

        assert torch.isfinite(o_extend).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize(
        "k_scale,v_scale",
        [
            (1e-6, 1e-6),
            (1e-6, 0.01),
            (0.01, 1e-6),
            (0.01, 0.01),
            (0.01, 100.0),
            (100.0, 0.01),
            (100.0, 100.0),
        ],
    )
    def test_scale_combinations(self, k_scale, v_scale):
        """Test various k_scale × v_scale combinations."""
        device = "cpu"
        dtype = torch.float16
        num_requests = 1
        num_heads = 4
        head_dim = 64
        seq_len = 32
        extend_len = 16
        max_context_len = seq_len + 16

        q_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            extend_len, num_heads, head_dim, dtype=dtype, device=device
        )

        k_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )
        v_buffer_float = torch.randn(
            max_context_len, num_heads, head_dim, dtype=torch.float32, device=device
        )

        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )
        prefix_len = seq_len - extend_len
        if prefix_len > 0:
            req_to_token[0, :prefix_len] = torch.arange(prefix_len, dtype=torch.int64)

        req_pool_indices = torch.zeros(num_requests, dtype=torch.int64, device=device)
        seq_lens = torch.tensor([seq_len], dtype=torch.int64, device=device)
        extend_seq_lens = torch.tensor([extend_len], dtype=torch.int64, device=device)
        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)

        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 50.0
        max_len_extend = extend_len

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer_int8,
            v_buffer_int8,
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

        assert torch.isfinite(
            o_extend
        ).all(), f"Output contains NaN or Inf with k_scale={k_scale}, v_scale={v_scale}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
