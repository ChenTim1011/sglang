"""Unit tests for RVV extend attention kernel (INT8 KV cache).

Uses INT8 cached-prefix buffers plus float extend tokens. Most cases use the
shared INT8 extend tolerance; a few coarse-scale or per-token-scale cases keep
their looser thresholds local to the test that needs them.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_extend_int8 -v
"""

import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision


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
    k_scale_buf: torch.Tensor = None,  # [max_tokens, num_kv_heads] per-token scales
    v_scale_buf: torch.Tensor = None,  # [max_tokens, num_kv_heads] per-token scales
) -> torch.Tensor:
    """Reference implementation of extend attention using PyTorch SDPA.

    Mirrors the kernel's two-stage behavior:
      Stage 1 (prefix): dequantize cached INT8 keys/values.
      Stage 2 (extend): read new keys/values directly from k_extend/v_extend.
    """
    num_seqs = seq_lens.shape[0]
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

        # Stage 2 reads the new FP extend tokens directly.
        per_req_key_ext = k_extend[start_loc : start_loc + extend_len].movedim(0, 1)
        per_req_val_ext = v_extend[start_loc : start_loc + extend_len].movedim(0, 1)

        if prefix_len > 0:
            # Stage 1 reads cached INT8 prefix tokens.
            per_req_tokens_prefix = req_to_token[req_pool_idx, :prefix_len]
            per_req_key_pre = k_buffer[per_req_tokens_prefix].movedim(0, 1)
            per_req_val_pre = v_buffer[per_req_tokens_prefix].movedim(0, 1)

            if per_req_key_pre.dtype == torch.int8:
                if k_scale_buf is not None:
                    scales_k = k_scale_buf[per_req_tokens_prefix].unsqueeze(-1)
                    per_req_key_pre = (
                        per_req_key_pre.movedim(0, 1).to(torch.float32) * scales_k
                    ).movedim(1, 0)
                else:
                    per_req_key_pre = per_req_key_pre.to(torch.float32) * k_scale
            if per_req_val_pre.dtype == torch.int8:
                if v_scale_buf is not None:
                    scales_v = v_scale_buf[per_req_tokens_prefix].unsqueeze(-1)
                    per_req_val_pre = (
                        per_req_val_pre.movedim(0, 1).to(torch.float32) * scales_v
                    ).movedim(1, 0)
                else:
                    per_req_val_pre = per_req_val_pre.to(torch.float32) * v_scale

            per_req_key_pre = per_req_key_pre.to(per_req_query.dtype)
            per_req_val_pre = per_req_val_pre.to(per_req_query.dtype)

            per_req_key = torch.cat([per_req_key_pre, per_req_key_ext], dim=1)
            per_req_value = torch.cat([per_req_val_pre, per_req_val_ext], dim=1)
        else:
            per_req_key = per_req_key_ext
            per_req_value = per_req_val_ext

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


class TestRVVExtendInt8(CustomTestCase):
    """Test suite for RVV extend attention with INT8 KV cache."""

    @classmethod
    def setUpClass(cls):
        """Set shared test device once for the class."""
        cls.device = "cpu"

        if not has_sgl_kernel_op("extend_attention_int8_cpu"):
            raise unittest.SkipTest("extend_attention_int8_cpu not available")

    def run_case_extend_int8_attention(
        self,
        num_heads,
        head_dim,
        seq_len,
        extend_len,
        num_requests,
        num_heads_kv=None,
        head_dim_v=None,
        mla=False,
        logit_cap=0.0,
        k_scale=0.01,
        v_scale=0.01,
        dtype=torch.float16,
        atol_override=None,
    ):
        """Run one INT8 extend case and compare against the reference path."""
        device = self.device

        if num_heads_kv is None:
            num_heads_kv = num_heads
        if head_dim_v is None:
            head_dim_v = head_dim

        enable_gqa = num_heads != num_heads_kv
        H_BUF = 1 if mla else num_heads_kv

        max_context_len = seq_len + 16
        max_total_tokens = num_requests * max_context_len

        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.int64, device=device
        )
        extend_seq_lens = torch.tensor(
            [extend_len] * num_requests, dtype=torch.int64, device=device
        )

        extend_start_loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        current_loc = 0
        for i in range(num_requests):
            extend_start_loc[i] = current_loc
            current_loc += extend_len

        req_pool_indices = torch.arange(num_requests, dtype=torch.int64, device=device)
        total_extend_len = extend_len * num_requests

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

        # Build cached INT8 prefix buffers from float source tensors.
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
        if mla:
            # MLA stores V as a view into the shared KV buffer.
            v_buffer_int8 = k_buffer_int8.narrow(2, 0, head_dim_v)
        else:
            v_buffer_int8 = (
                (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
            )

        req_to_token = torch.zeros(
            num_requests, max_context_len, dtype=torch.int64, device=device
        )

        prefix_len = seq_len - extend_len
        if prefix_len > 0:
            for i in range(num_requests):
                start_idx = i * max_context_len
                req_to_token[i, :prefix_len] = torch.arange(
                    start_idx, start_idx + prefix_len, dtype=torch.int64
                )

        sm_scale = 1.0 / (head_dim**0.5)
        max_len_extend = extend_len

        # Pre-fill cached-token scales for the static-scale path.
        k_scale_buf = torch.full(
            (max_total_tokens, H_BUF), k_scale, dtype=torch.float32
        )
        v_scale_buf = torch.full(
            (max_total_tokens, H_BUF), v_scale, dtype=torch.float32
        )

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
            k_scale_buf,
            v_scale_buf,
            k_scale,
            v_scale,
        )

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

        atol = rtol = (
            atol_override
            if atol_override is not None
            else precision["attention_extend_int8"][dtype]
        )
        torch.testing.assert_close(o_extend, ref_output, atol=atol, rtol=rtol)

    def test_case_extend_int8_batching(self):
        """Case: batching layout validates offsets and strides."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=4,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=4,
                dtype=dtype,
            )

    def test_case_extend_int8_gqa(self):
        """Case: GQA extend layout with H_Q=32 and H_KV=8."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=32,
                head_dim=64,
                seq_len=128,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
                num_heads_kv=8,
            )

    def test_case_extend_int8_mla(self):
        """Case: MLA extend layout with shared KV storage."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
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

    def test_case_extend_int8_logit_cap(self):
        """Case: logit-cap path in INT8 extend."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=8,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=2,
                dtype=dtype,
                logit_cap=50.0,
            )

    def test_case_extend_int8_odd_dimensions(self):
        """Case: odd dimension extend layout."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=4,
                head_dim=32,
                seq_len=50,
                extend_len=20,
                num_requests=2,
                dtype=dtype,
            )

    def test_case_extend_int8_scale(self):
        """Case: extend with larger quantization scales.

        This is intentionally looser than the common static-scale path because
        coarse scalar quantization injects materially more INT8 error before the
        attention reduction starts.
        """
        large_scale_tol = 3.5e-1
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=4,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=2,
                dtype=dtype,
                k_scale=0.05,
                v_scale=0.05,
                atol_override=large_scale_tol,
            )

    def test_case_extend_int8_no_prefix(self):
        """INT8 extend-only test with no cached prefix (pure Stage 2 path)."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_extend_int8_attention(
                num_heads=4,
                head_dim=64,
                seq_len=32,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
            )

    def test_case_extend_int8_per_token_varying_scale(self):
        """Per-token dynamic scale for the cached prefix path."""
        torch.manual_seed(11)
        num_heads = 8
        num_heads_kv = 2
        head_dim = 64
        head_dim_v = 64
        seq_len = 64
        extend_len = 16
        prefix_len = seq_len - extend_len
        num_requests = 2
        sm_scale = 1.0 / (head_dim**0.5)
        max_context_len = seq_len + 16
        max_total_tokens = num_requests * max_context_len

        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                q_extend = torch.randn(
                    extend_len * num_requests, num_heads, head_dim, dtype=dtype
                )
                k_extend = torch.randn(
                    extend_len * num_requests, num_heads_kv, head_dim, dtype=dtype
                )
                v_extend = torch.randn(
                    extend_len * num_requests, num_heads_kv, head_dim_v, dtype=dtype
                )
                o_extend = torch.zeros(
                    extend_len * num_requests, num_heads, head_dim_v, dtype=dtype
                )

                k_buffer_float = (
                    torch.randn(max_total_tokens, num_heads_kv, head_dim) * 5.0
                )
                v_buffer_float = (
                    torch.randn(max_total_tokens, num_heads_kv, head_dim_v) * 5.0
                )

                # Vary scales per cached token to exercise token-indexed lookup.
                k_scale_buf = torch.rand(max_total_tokens, num_heads_kv) * 0.099 + 0.001
                v_scale_buf = torch.rand(max_total_tokens, num_heads_kv) * 0.099 + 0.001

                k_buffer_int8 = (
                    (k_buffer_float / k_scale_buf.unsqueeze(-1))
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                )
                v_buffer_int8 = (
                    (v_buffer_float / v_scale_buf.unsqueeze(-1))
                    .round()
                    .clamp(-128, 127)
                    .to(torch.int8)
                )

                req_to_token = torch.zeros(
                    num_requests, max_context_len, dtype=torch.int64
                )
                for i in range(num_requests):
                    start_idx = i * max_context_len
                    req_to_token[i, :prefix_len] = torch.arange(
                        start_idx, start_idx + prefix_len, dtype=torch.int64
                    )

                seq_lens = torch.full((num_requests,), seq_len, dtype=torch.int64)
                extend_seq_lens = torch.full(
                    (num_requests,), extend_len, dtype=torch.int64
                )
                extend_start_loc = torch.tensor(
                    [i * extend_len for i in range(num_requests)], dtype=torch.int64
                )
                req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

                dummy_scale = 0.01

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
                    extend_len,
                    sm_scale,
                    0.0,
                    k_scale_buf,
                    v_scale_buf,
                    dummy_scale,
                    dummy_scale,
                )

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
                    enable_gqa=True,
                    k_scale_buf=k_scale_buf,
                    v_scale_buf=v_scale_buf,
                )

                atol = rtol = 3.5e-1
                torch.testing.assert_close(o_extend, ref_output, atol=atol, rtol=rtol)

    def test_case_extend_int8_static_unit_scale_is_written_literally(self):
        """A scalar scale of 1.0 must be stored literally in per-token scale buffers."""
        dtype = torch.float16
        num_heads = 2
        num_heads_kv = 2
        head_dim = 32
        extend_len = 1
        num_requests = 1
        sm_scale = 1.0 / (head_dim**0.5)

        q_extend = torch.randn(extend_len, num_heads, head_dim, dtype=dtype)
        k_extend = torch.randn(extend_len, num_heads_kv, head_dim, dtype=dtype)
        v_extend = torch.randn(extend_len, num_heads_kv, head_dim, dtype=dtype)
        o_extend = torch.zeros(extend_len, num_heads, head_dim, dtype=dtype)
        k_buffer = torch.zeros(8, num_heads_kv, head_dim, dtype=torch.int8)
        v_buffer = torch.zeros(8, num_heads_kv, head_dim, dtype=torch.int8)
        req_to_token = torch.tensor([[3, 0, 0, 0]], dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        extend_seq_lens = torch.tensor([1], dtype=torch.long)
        extend_start_loc = torch.tensor([0], dtype=torch.long)
        k_scale_buf = torch.zeros(8, num_heads_kv, dtype=torch.float32)
        v_scale_buf = torch.zeros(8, num_heads_kv, dtype=torch.float32)

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
            extend_len,
            sm_scale,
            0.0,
            k_scale_buf,
            v_scale_buf,
            1.0,
            1.0,
        )

        torch.testing.assert_close(k_scale_buf[3], torch.ones(num_heads_kv))
        torch.testing.assert_close(v_scale_buf[3], torch.ones(num_heads_kv))

    def test_case_extend_int8_rejects_bad_scale_buffer_shape(self):
        dtype = torch.float16
        q_extend = torch.randn(1, 2, 32, dtype=dtype)
        k_extend = torch.randn(1, 2, 32, dtype=dtype)
        v_extend = torch.randn(1, 2, 32, dtype=dtype)
        o_extend = torch.zeros(1, 2, 32, dtype=dtype)
        k_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        v_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        req_to_token = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        extend_seq_lens = torch.tensor([1], dtype=torch.long)
        extend_start_loc = torch.tensor([0], dtype=torch.long)
        bad_k_scale_buf = torch.ones(7, 2, dtype=torch.float32)
        v_scale_buf = torch.ones(8, 2, dtype=torch.float32)

        with self.assertRaises(RuntimeError):
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
                1,
                1.0,
                0.0,
                bad_k_scale_buf,
                v_scale_buf,
                0.01,
                0.01,
            )

    def test_case_extend_int8_rejects_invalid_dynamic_scale_inputs(self):
        dtype = torch.float16
        q_extend = torch.randn(1, 2, 32, dtype=dtype)
        k_extend = torch.randn(1, 2, 32, dtype=dtype)
        v_extend = torch.randn(1, 2, 32, dtype=dtype)
        o_extend = torch.zeros(1, 2, 32, dtype=dtype)
        k_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        v_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        req_to_token = torch.tensor([[0, 0, 0, 0]], dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        extend_seq_lens = torch.tensor([1], dtype=torch.long)
        extend_start_loc = torch.tensor([0], dtype=torch.long)
        k_scale_buf = torch.ones(8, 2, dtype=torch.float32)
        v_scale_buf = torch.ones(8, 2, dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, "sm_scale must be finite and positive"
        ):
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
                1,
                float("nan"),
                0.0,
                k_scale_buf,
                v_scale_buf,
                0.01,
                0.01,
            )

        with self.assertRaisesRegex(
            RuntimeError,
            "k_scale must be finite and positive, or NaN for dynamic quantization",
        ):
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
                1,
                1.0,
                0.0,
                k_scale_buf,
                v_scale_buf,
                float("inf"),
                0.01,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
