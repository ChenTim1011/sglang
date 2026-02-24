"""
Unit Test for extend_attention_cpu kernel (RVV optimized, INT8).

Usage:
    python3 test_rvv_extend_int8.py -v
"""

import importlib.util
import unittest

# Import sgl_kernel to register operators
import sgl_kernel  # noqa: F401
import torch
from torch.nn.functional import scaled_dot_product_attention


def has_op(op_name: str) -> bool:
    """Check if a specific operator is available in sgl_kernel."""
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, op_name)
    except (AttributeError, RuntimeError):
        return False


def naive_attention_extend(
    q_extend: torch.Tensor,
    k_extend: torch.Tensor,
    v_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    extend_start_loc: torch.Tensor,
    sm_scale: float,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    enable_gqa: bool = False,
) -> torch.Tensor:
    """Reference implementation of extend attention using PyTorch SDPA."""
    num_seqs = seq_lens.shape[0]
    num_heads = q_extend.shape[1]
    head_dim_v = v_extend.shape[2]

    query = q_extend.movedim(0, 1)
    output = torch.zeros(
        (q_extend.shape[0], q_extend.shape[1], head_dim_v),
        dtype=q_extend.dtype,
        device=q_extend.device,
    )

    start_q = 0
    for seq_idx in range(num_seqs):
        seq_len = seq_lens[seq_idx].item()
        extend_len = extend_seq_lens[seq_idx].item()
        prefix_len = seq_len - extend_len
        start_loc = extend_start_loc[seq_idx].item()

        end_q = start_q + extend_len
        per_req_query = query[:, start_q:end_q, :]

        per_req_query_padded = torch.empty(
            (per_req_query.shape[0], seq_len, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )
        per_req_query_padded[:, prefix_len:, :] = per_req_query

        req_pool_idx = req_pool_indices[seq_idx].item()
        per_req_tokens = req_to_token[req_pool_idx, :seq_len]
        per_req_key = k_buffer[per_req_tokens].movedim(0, 1)
        per_req_value = v_buffer[per_req_tokens].movedim(0, 1)

        if per_req_key.dtype == torch.int8:
            per_req_key = per_req_key.to(torch.float32) * k_scale
        if per_req_value.dtype == torch.int8:
            per_req_value = per_req_value.to(torch.float32) * v_scale

        if per_req_key.dtype != per_req_query.dtype:
            per_req_key = per_req_key.to(per_req_query.dtype)
        if per_req_value.dtype != per_req_query.dtype:
            per_req_value = per_req_value.to(per_req_query.dtype)

        per_req_out_padded = scaled_dot_product_attention(
            per_req_query_padded.unsqueeze(0),
            per_req_key.unsqueeze(0),
            per_req_value.unsqueeze(0),
            enable_gqa=enable_gqa,
            scale=sm_scale,
            is_causal=True,
        )
        per_req_out_padded = per_req_out_padded.squeeze(0).movedim(1, 0)

        output[start_q:end_q, :, :] = per_req_out_padded[prefix_len:, :, :]
        start_q = end_q

    return output


class TestExtendAttentionInt8CPU(unittest.TestCase):
    """Test extend_attention_int8_cpu kernel correctness"""

    @classmethod
    def setUpClass(cls):
        """Setup once before all tests"""
        cls.device = "cpu"
        import sgl_kernel  # noqa: F401

        if not has_op("extend_attention_int8_cpu"):
            raise unittest.SkipTest("extend_attention_int8_cpu not available")

    def _run_extend_int8_test(
        self,
        num_heads,
        head_dim,
        seq_len,
        extend_len,
        num_requests,
        dtype,
        num_heads_kv=None,
        head_dim_v=None,
        mla=False,
        logit_cap=0.0,
        k_scale=0.01,
        v_scale=0.01,
    ):
        """Test extend_attention_int8_cpu with various configurations."""
        device = self.device

        if num_heads_kv is None:
            num_heads_kv = num_heads
        if head_dim_v is None:
            head_dim_v = head_dim

        enable_gqa = num_heads != num_heads_kv
        H_BUF = 1 if mla else num_heads_kv

        max_context_len = seq_len + 16
        max_total_tokens = num_requests * max_context_len

        # Setup requests
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.int64, device=device
        )
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

        # Create tensors (Query and Extend K/V are in float/bf16)
        q_extend = torch.randn(
            total_extend_len, num_heads, head_dim, dtype=dtype, device=device
        )
        k_extend = torch.randn(
            total_extend_len, num_heads_kv, head_dim, dtype=dtype, device=device
        )
        v_extend = torch.randn(
            total_extend_len, num_heads_kv, head_dim_v, dtype=dtype, device=device
        )
        o_extend = torch.zeros(
            total_extend_len, num_heads, head_dim_v, dtype=dtype, device=device
        )

        # KV cache buffers (INT8) - Generate float data first, then quantize
        k_buffer_float = (
            torch.randn(
                max_total_tokens,
                H_BUF,
                head_dim,
                dtype=torch.float32,
                device=device,
            )
            * 5.0
        )
        v_buffer_float = (
            torch.randn(
                max_total_tokens,
                H_BUF,
                head_dim_v,
                dtype=torch.float32,
                device=device,
            )
            * 5.0
        )

        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
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
        max_len_extend = extend_len

        # Run INT8 Kernel
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
        ref_output = naive_attention_extend(
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
            enable_gqa=enable_gqa,
        )

        # Tolerance: Int8 has larger quantization error
        torch.testing.assert_close(o_extend, ref_output, atol=1.5, rtol=0.5)

    # Test cases
    def test_extend_int8_batching(self):
        """INT8 extend attention test with batching (Verifies offsets/strides)"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(4, 64, 64, 16, 4, dtype)

    def test_extend_int8_gqa(self):
        """INT8 GQA extend test: LLaMA-style H_Q=32, H_KV=8"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(
                num_heads=32,
                head_dim=64,
                seq_len=128,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
                num_heads_kv=8,
            )

    def test_extend_int8_mla(self):
        """INT8 MLA extend test: DeepSeek-style num_heads_kv=1 and shared buffer"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(
                num_heads=16,
                head_dim=192,
                seq_len=128,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
                num_heads_kv=1,
                head_dim_v=128,
                mla=True,
            )

    def test_extend_int8_logit_cap(self):
        """INT8 logit_cap extend functionality test"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(
                num_heads=8,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=2,
                dtype=dtype,
                logit_cap=50.0,
            )

    def test_extend_int8_odd_dimensions(self):
        """INT8 extend with odd dimensions"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(
                num_heads=4,
                head_dim=32,
                seq_len=50,
                extend_len=20,
                num_requests=2,
                dtype=dtype,
            )

    def test_extend_int8_scale(self):
        """INT8 extend test with larger quantization scale"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8_test(
                num_heads=4,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=2,
                dtype=dtype,
                k_scale=0.05,
                v_scale=0.05,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
