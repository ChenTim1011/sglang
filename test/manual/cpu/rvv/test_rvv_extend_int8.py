"""Unit tests for RVV extend attention kernel (INT8 KV cache).

Covers INT8 cached-prefix + float extend tokens; coarse-scale and per-token
cases keep their looser thresholds local to the test that needs them.
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

    def _run_extend_int8(
        self,
        num_heads,
        head_dim,
        seq_len,
        extend_len,
        num_requests,
        num_heads_kv=None,
        head_dim_v=None,
        logit_cap=0.0,
        k_scale=0.01,
        v_scale=0.01,
        dtype=torch.float16,
        atol_override=None,
    ):
        device = self.device

        if num_heads_kv is None:
            num_heads_kv = num_heads
        if head_dim_v is None:
            head_dim_v = head_dim

        enable_gqa = num_heads != num_heads_kv

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
                num_heads_kv,
                head_dim,
                dtype=torch.float32,
                device=device,
            )
            * 5.0
        )
        v_buffer_float = (
            torch.randn(
                max_total_tokens,
                num_heads_kv,
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
            (max_total_tokens, num_heads_kv), k_scale, dtype=torch.float32
        )
        v_scale_buf = torch.full(
            (max_total_tokens, num_heads_kv), v_scale, dtype=torch.float32
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

    def test_batching(self):
        """Multi-request batch verifies per-request offset and stride calculations."""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
                num_heads=4,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=4,
                dtype=dtype,
            )

    def test_gqa(self):
        """GQA (H_Q=32, H_KV=8) verifies ratio broadcasting with INT8 prefix dequantization."""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
                num_heads=32,
                head_dim=64,
                seq_len=128,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
                num_heads_kv=8,
            )

    def test_logit_cap(self):
        """logit_cap + INT8 prefix: tanh-cap must apply after integer dequantization."""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
                num_heads=8,
                head_dim=64,
                seq_len=64,
                extend_len=16,
                num_requests=2,
                dtype=dtype,
                logit_cap=50.0,
            )

    def test_odd_dims(self):
        """Non-power-of-2 dimensions exercise vector tail handling in both stages."""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
                num_heads=4,
                head_dim=32,
                seq_len=50,
                extend_len=20,
                num_requests=2,
                dtype=dtype,
            )

    def test_large_scale(self):
        """Coarse static scales (0.05) inject more INT8 error; uses a looser tolerance."""
        large_scale_tol = 3.5e-1
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
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

    def test_no_prefix(self):
        """No cached prefix means Stage 1 is skipped; Stage 2 must still produce correct output."""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_extend_int8(
                num_heads=4,
                head_dim=64,
                seq_len=32,
                extend_len=32,
                num_requests=2,
                dtype=dtype,
            )

    def test_per_token_varying_scale(self):
        """Per-token scales on the cached prefix: each token has a distinct dequantization scale."""
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

    def test_unit_scale_written_literally(self):
        """scale=1.0 must be stored literally in scale buffers, not treated as a sentinel."""
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

    def test_rejects_bad_scale_shape(self):
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

    def test_rejects_invalid_dynamic_inputs(self):
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


def _compute_dynamic_int8(fp_tensor: torch.Tensor):
    """Match C++ quantize_row_int8_symmetric_infer_scale.

    scale = max(abs(row)) / 127.0  (eps=1e-8, computed in FP32).

    Returns:
        q_int8: torch.int8  same shape as fp_tensor
        scale:  torch.float32  shape = fp_tensor.shape[:-1]
    """
    eps = 1e-8
    fp32 = fp_tensor.float()
    max_abs = fp32.abs().amax(dim=-1)
    scale = max_abs.clamp(min=eps) / 127.0
    q = (fp32 / scale.unsqueeze(-1)).round().clamp(-128, 127).to(torch.int8)
    return q, scale


class TestRVVExtendInt8NanScale(CustomTestCase):
    """Tests for the dynamic-quantization (NaN scale) path in extend INT8 attention.

    When k_scale / v_scale = NaN the extend kernel:
      1. Expects FP16/BF16 k_extend/v_extend inputs.
      2. Computes per-head scale  max(abs(row)) / 127.0  for each extend token.
      3. Writes computed scales into k_scale_buf at the corresponding token slots.
      4. Stage 2 attention still uses the FP k_extend/v_extend directly (the
         quantized data is only for future decode steps).
      5. Stage 1 (prefix) reads the existing INT8 buffer with per-token scales.

    Wrong scales stored during prefill propagate to all subsequent decode steps,
    which is a likely source of the observed 15-point GSM8k regression.
    """

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        if not has_sgl_kernel_op("extend_attention_int8_cpu"):
            raise unittest.SkipTest("extend_attention_int8_cpu not available")

    def _run_extend_nan_scale(
        self,
        num_heads: int,
        num_heads_kv: int,
        head_dim: int,
        seq_len: int,
        extend_len: int,
        num_requests: int,
        dtype=torch.bfloat16,
        logit_cap: float = 0.0,
        atol_override=None,
    ):
        """Run one NaN-scale extend case.

        The cached INT8 prefix buffer has per-token scales already stored
        (from a prior extend/decode step).  k_extend/v_extend are FP so the
        kernel quantizes them with dynamically inferred scales and writes
        those scales into k_scale_buf.

        Because Stage 2 attention uses the FP extend tensors directly the
        attention output must match the plain-FP reference, *and* the written
        scale buffer entries must match the expected dynamic scales.
        """
        device = self.device
        enable_gqa = num_heads != num_heads_kv
        prefix_len = seq_len - extend_len
        max_context_len = seq_len + 16
        max_total_tokens = num_requests * max_context_len
        sm_scale = 1.0 / (head_dim**0.5)
        total_extend_tokens = extend_len * num_requests

        torch.manual_seed(31)

        q_extend = torch.randn(total_extend_tokens, num_heads, head_dim, dtype=dtype)
        k_extend = (
            torch.randn(total_extend_tokens, num_heads_kv, head_dim, dtype=dtype) * 3.0
        )
        v_extend = (
            torch.randn(total_extend_tokens, num_heads_kv, head_dim, dtype=dtype) * 3.0
        )
        o_extend = torch.zeros(total_extend_tokens, num_heads, head_dim, dtype=dtype)

        # Pre-cached INT8 prefix buffer with random per-token scales.
        k_cached_fp = torch.randn(max_total_tokens, num_heads_kv, head_dim) * 3.0
        v_cached_fp = torch.randn(max_total_tokens, num_heads_kv, head_dim) * 3.0

        k_scale_buf = torch.rand(max_total_tokens, num_heads_kv) * 0.09 + 0.01
        v_scale_buf = torch.rand(max_total_tokens, num_heads_kv) * 0.09 + 0.01

        k_buffer_int8 = (
            (k_cached_fp / k_scale_buf.unsqueeze(-1))
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )
        v_buffer_int8 = (
            (v_cached_fp / v_scale_buf.unsqueeze(-1))
            .round()
            .clamp(-128, 127)
            .to(torch.int8)
        )

        # Token mapping: req_to_token[req_pool_idx, seq_pos] = physical token slot.
        # Both prefix AND extend positions must be filled; the kernel uses
        # req_to_token[prefix_len + t] to find where to write each extend token's
        # INT8 data and scale.  Leaving extend positions as 0 causes all extend
        # tokens to overwrite slot 0 instead of their assigned slots.
        req_to_token = torch.zeros(num_requests, max_context_len, dtype=torch.long)

        # Slots for the extend tokens (written by the kernel into k_buffer)
        extend_token_slots = []
        for i in range(num_requests):
            base = i * max_context_len + prefix_len
            extend_token_slots.append(torch.arange(base, base + extend_len))
        extend_token_slots = torch.stack(extend_token_slots)  # [B, extend_len]

        for i in range(num_requests):
            if prefix_len > 0:
                start = i * max_context_len
                req_to_token[i, :prefix_len] = torch.arange(start, start + prefix_len)
            req_to_token[i, prefix_len : prefix_len + extend_len] = extend_token_slots[
                i
            ]

        req_pool_indices = torch.arange(num_requests, dtype=torch.long)
        seq_lens = torch.full((num_requests,), seq_len, dtype=torch.long)
        extend_seq_lens = torch.full((num_requests,), extend_len, dtype=torch.long)
        extend_start_loc = torch.arange(num_requests, dtype=torch.long) * extend_len

        k_scale_buf_kernel = k_scale_buf.clone()
        v_scale_buf_kernel = v_scale_buf.clone()

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
            logit_cap,
            k_scale_buf_kernel,
            v_scale_buf_kernel,
            float("nan"),
            float("nan"),
        )

        # Verify scale buffer entries for every extend token.
        for req_i in range(num_requests):
            for tok_i in range(extend_len):
                src_offset = req_i * extend_len + tok_i
                k_tok = k_extend[src_offset]  # [num_heads_kv, head_dim]
                v_tok = v_extend[src_offset]
                _, expected_k_scale = _compute_dynamic_int8(k_tok)  # [num_heads_kv]
                _, expected_v_scale = _compute_dynamic_int8(v_tok)

                slot = extend_token_slots[req_i, tok_i].item()
                for h in range(num_heads_kv):
                    self.assertAlmostEqual(
                        k_scale_buf_kernel[slot, h].item(),
                        expected_k_scale[h].item(),
                        places=4,
                        msg=f"k_scale_buf mismatch req={req_i} tok={tok_i} head={h}",
                    )
                    self.assertAlmostEqual(
                        v_scale_buf_kernel[slot, h].item(),
                        expected_v_scale[h].item(),
                        places=4,
                        msg=f"v_scale_buf mismatch req={req_i} tok={tok_i} head={h}",
                    )

        # Stage 2 uses FP extend directly, so output must match the plain-FP reference.
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
            enable_gqa=enable_gqa,
            k_scale_buf=k_scale_buf,  # original prefix scales (not the new extend slots)
            v_scale_buf=v_scale_buf,
        )

        atol = rtol = (
            atol_override
            if atol_override is not None
            else precision["attention_extend_int8"][dtype]
        )
        torch.testing.assert_close(o_extend, ref_output, atol=atol, rtol=rtol)

    def test_nan_scale_basic(self):
        """NaN scale: basic extend with prefix (Stage 1 + Stage 2)."""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self._run_extend_nan_scale(
                    num_heads=8,
                    num_heads_kv=8,
                    head_dim=64,
                    seq_len=64,
                    extend_len=16,
                    num_requests=2,
                    dtype=dtype,
                )

    def test_nan_scale_no_prefix(self):
        """NaN scale: pure extend (no cached prefix, Stage 2 only)."""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self._run_extend_nan_scale(
                    num_heads=8,
                    num_heads_kv=8,
                    head_dim=64,
                    seq_len=32,  # seq_len == extend_len → prefix_len = 0
                    extend_len=32,
                    num_requests=2,
                    dtype=dtype,
                )

    def test_nan_scale_gqa(self):
        """NaN scale: GQA extend (H_Q=32, H_KV=8)."""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self._run_extend_nan_scale(
                    num_heads=32,
                    num_heads_kv=8,
                    head_dim=64,
                    seq_len=128,
                    extend_len=32,
                    num_requests=2,
                    dtype=dtype,
                )

    def test_nan_scale_qwen25_1p5b_like(self):
        """NaN scale: Qwen2.5-1.5B-Instruct configuration.

        Simulates the prefill step that precedes the decode steps which showed
        a 15-point GSM8k accuracy regression.  Wrong scales stored here would
        cause every subsequent decode step to use incorrect dequantization.
        Config: 12 Q heads, 2 KV heads, head_dim=128, BF16.
        """
        for extend_len in [32, 128, 512]:
            with self.subTest(extend_len=extend_len):
                self._run_extend_nan_scale(
                    num_heads=12,
                    num_heads_kv=2,
                    head_dim=128,
                    seq_len=extend_len + 64,
                    extend_len=extend_len,
                    num_requests=2,
                    dtype=torch.bfloat16,
                )

    def test_nan_scale_first_token(self):
        """NaN scale: extend with extend_len=1 (single new token, no prefix)."""
        self._run_extend_nan_scale(
            num_heads=12,
            num_heads_kv=2,
            head_dim=128,
            seq_len=1,
            extend_len=1,
            num_requests=4,
            dtype=torch.bfloat16,
        )

    def test_nan_scale_rejects_int8_extend_input(self):
        """NaN scale must raise an error when k_extend/v_extend are INT8."""
        dtype = torch.float16
        q_extend = torch.randn(1, 4, 32, dtype=dtype)
        k_extend_int8 = torch.randint(-8, 8, (1, 4, 32), dtype=torch.int8)
        v_extend_int8 = torch.randint(-8, 8, (1, 4, 32), dtype=torch.int8)
        o_extend = torch.zeros(1, 4, 32, dtype=dtype)
        k_buffer = torch.zeros(8, 4, 32, dtype=torch.int8)
        v_buffer = torch.zeros(8, 4, 32, dtype=torch.int8)
        req_to_token = torch.zeros(1, 4, dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        extend_seq_lens = torch.tensor([1], dtype=torch.long)
        extend_start_loc = torch.tensor([0], dtype=torch.long)
        k_scale_buf = torch.ones(8, 4, dtype=torch.float32)
        v_scale_buf = torch.ones(8, 4, dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError,
            "dynamic quantization scales require floating k_extend/v_extend inputs",
        ):
            torch.ops.sgl_kernel.extend_attention_int8_cpu(
                q_extend,
                k_extend_int8,
                v_extend_int8,
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
                float("nan"),
                float("nan"),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
