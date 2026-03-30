"""Unit tests for RVV decode attention kernel (INT8 KV cache).

Uses INT8 KV buffers plus scalar or per-token dequantization scales. Most
cases use the shared INT8 decode tolerance; a few coarse-scale cases override
it locally where quantization noise is intentionally higher.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_decode_int8 -v
"""

import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision


def naive_attention_decode(
    query: torch.Tensor,  # [num_requests, num_heads, head_dim]
    k_buffer: torch.Tensor,  # [max_tokens, num_heads, head_dim]
    v_buffer: torch.Tensor,  # [max_tokens, num_heads, head_dim_v]
    req_to_token: torch.Tensor,  # [num_requests, max_seq_len]
    req_pool_indices: torch.Tensor,  # [num_requests]
    seq_lens: torch.Tensor,  # [num_requests]
    sm_scale: float,
    logit_cap: float = 0.0,
    k_scale: float = 1.0,
    v_scale: float = 1.0,
    enable_gqa: bool = False,
    k_scale_buf: torch.Tensor = None,  # [max_tokens, num_kv_heads] per-token scales
    v_scale_buf: torch.Tensor = None,  # [max_tokens, num_kv_heads] per-token scales
) -> torch.Tensor:
    """Reference implementation of decode attention using PyTorch SDPA.

    When k_scale_buf / v_scale_buf are provided each cached token is dequantized
    with its own per-token per-head scale; otherwise the scalar k_scale / v_scale
    is applied uniformly (backward-compatible default).
    """
    num_requests = query.shape[0]
    num_heads = query.shape[1]
    head_dim_v = v_buffer.shape[2]

    query_transposed = query.movedim(0, 1)
    output = torch.zeros(
        num_requests, num_heads, head_dim_v, dtype=query.dtype, device=query.device
    )

    start_q = 0
    for seq_idx in range(num_requests):
        seq_len_kv = seq_lens[seq_idx].item()
        end_q = start_q + 1

        per_req_query = query_transposed[:, start_q:end_q, :]

        req_pool_idx = req_pool_indices[seq_idx].item()
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_buffer[per_req_tokens].movedim(0, 1)
        per_req_value = v_buffer[per_req_tokens].movedim(0, 1)

        if per_req_key.dtype == torch.int8:
            if k_scale_buf is not None:
                # Broadcast cached per-token scales over head_dim.
                scales_k = k_scale_buf[per_req_tokens].unsqueeze(-1)
                per_req_key = per_req_key.movedim(0, 1).to(torch.float32) * scales_k
                per_req_key = per_req_key.movedim(1, 0)
            else:
                per_req_key = per_req_key.to(torch.float32) * k_scale
        if per_req_value.dtype == torch.int8:
            if v_scale_buf is not None:
                scales_v = v_scale_buf[per_req_tokens].unsqueeze(-1)
                per_req_value = per_req_value.movedim(0, 1).to(torch.float32) * scales_v
                per_req_value = per_req_value.movedim(1, 0)
            else:
                per_req_value = per_req_value.to(torch.float32) * v_scale

        if per_req_key.dtype != per_req_query.dtype:
            per_req_key = per_req_key.to(per_req_query.dtype)
        if per_req_value.dtype != per_req_query.dtype:
            per_req_value = per_req_value.to(per_req_query.dtype)

        per_req_out = scaled_dot_product_attention(
            per_req_query.unsqueeze(0),
            per_req_key.unsqueeze(0),
            per_req_value.unsqueeze(0),
            enable_gqa=enable_gqa,
            scale=sm_scale,
            is_causal=False,
        )
        per_req_out = per_req_out.squeeze(0).movedim(1, 0)

        output[start_q:end_q, :, :] = per_req_out
        start_q = end_q

    return output


class TestRVVDecodeInt8(CustomTestCase):
    """Test suite for RVV decode attention with INT8 KV cache."""

    @classmethod
    def setUpClass(cls):
        cls.device = "cpu"
        try:
            import sgl_kernel  # noqa: F401
        except ImportError as e:
            raise unittest.SkipTest(f"sgl_kernel import failed: {e}") from e

        if not has_sgl_kernel_op("decode_attention_int8_cpu"):
            raise unittest.SkipTest("decode_attention_int8_cpu not available")

    def run_case_decode_int8_attention(
        self,
        num_heads,
        head_dim,
        seq_len,
        num_requests,
        num_heads_kv=None,
        k_scale=0.01,
        v_scale=0.01,
        dtype=torch.float16,
        logit_cap=0.0,
        atol_override=None,
    ):
        """Run one INT8 decode case and compare against the reference path."""
        device = self.device
        if num_heads_kv is None:
            num_heads_kv = num_heads
        enable_gqa = num_heads != num_heads_kv
        head_dim_v = head_dim
        max_seq_len = seq_len + 16

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )

        max_tokens = num_requests * max_seq_len

        k_buffer_float = (
            torch.randn(
                max_tokens, num_heads_kv, head_dim, dtype=torch.float32, device=device
            )
            * 5.0
        )

        v_buffer_float = (
            torch.randn(
                max_tokens, num_heads_kv, head_dim_v, dtype=torch.float32, device=device
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
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.long, device=device
        )

        # Fragment the mapping so cache access is not accidentally contiguous.
        req_to_token = (
            torch.randperm(max_tokens, device=device)
            .reshape(num_requests, max_seq_len)
            .to(torch.long)
        )

        key_float = torch.randn(
            num_requests, num_heads_kv, head_dim, dtype=torch.float32, device=device
        )
        value_float = torch.randn(
            num_requests, num_heads_kv, head_dim_v, dtype=torch.float32, device=device
        )

        key_int8 = (key_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        value_int8 = (value_float / v_scale).round().clamp(-128, 127).to(torch.int8)

        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        for i in range(num_requests):
            loc[i] = req_to_token[i, seq_len - 1]

        output = torch.zeros(
            num_requests, num_heads, head_dim_v, dtype=dtype, device=device
        )
        attn_logits = torch.zeros(
            num_requests,
            num_heads,
            1,
            head_dim_v + 1,
            dtype=torch.float32,
            device=device,
        )

        sm_scale = 1.0 / (head_dim**0.5)

        # Pre-fill cached-token scales for the static-scale path.
        k_scale_buf = torch.full(
            (max_tokens, num_heads_kv), k_scale, dtype=torch.float32
        )
        v_scale_buf = torch.full(
            (max_tokens, num_heads_kv), v_scale, dtype=torch.float32
        )

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
            k_scale_buf,
            v_scale_buf,
            k_scale,
            v_scale,
        )

        # Mirror the kernel-side cache update in the reference buffers.
        k_buffer_ref = k_buffer_int8.clone()
        v_buffer_ref = v_buffer_int8.clone()

        for i in range(num_requests):
            loc_idx = loc[i].item()
            k_buffer_ref[loc_idx] = key_int8[i]
            v_buffer_ref[loc_idx] = value_int8[i]

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
            enable_gqa=enable_gqa,
        )

        atol = rtol = (
            atol_override
            if atol_override is not None
            else precision["attention_decode_int8"][dtype]
        )
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

    def test_case_decode_int8_gqa(self):
        """Case: GQA decode across representative INT8 configurations."""
        configs = [
            (32, 8, 64, 63, 4),
            (32, 8, 64, 129, 4),
            (32, 8, 64, 256, 4),
            (16, 2, 64, 128, 2),
            (12, 3, 64, 128, 2),
        ]
        for num_heads, num_heads_kv, head_dim, seq_len, num_requests in configs:
            for dtype in [torch.float16, torch.bfloat16]:
                with self.subTest(
                    num_heads=num_heads,
                    num_heads_kv=num_heads_kv,
                    head_dim=head_dim,
                    seq_len=seq_len,
                    dtype=dtype,
                ):
                    self.run_case_decode_int8_attention(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        seq_len=seq_len,
                        num_requests=num_requests,
                        dtype=dtype,
                        num_heads_kv=num_heads_kv,
                    )

    def test_case_decode_int8_small_scale(self):
        """Case: decode with small quantization scales."""
        configs = [
            (8, 64, 63, 2, None),
            (8, 64, 129, 2, None),
            (8, 64, 128, 2, None),
            (16, 64, 128, 2, 4),
        ]
        for num_heads, head_dim, seq_len, num_requests, num_heads_kv in configs:
            for dtype in [torch.float16, torch.bfloat16]:
                with self.subTest(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_heads_kv=num_heads_kv,
                    dtype=dtype,
                ):
                    self.run_case_decode_int8_attention(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        seq_len=seq_len,
                        num_requests=num_requests,
                        dtype=dtype,
                        num_heads_kv=num_heads_kv,
                        k_scale=0.001,
                        v_scale=0.001,
                    )

    def test_case_decode_int8_large_scale(self):
        """Case: decode with large quantization scales.

        This is intentionally looser than the common INT8 decode path because
        coarse static scales amplify quantization error before softmax.
        """
        large_scale_tol = 3e-1
        configs = [
            (16, 128, 129, 4, None),
            (16, 128, 256, 4, None),
            (32, 128, 256, 4, 8),
        ]
        for num_heads, head_dim, seq_len, num_requests, num_heads_kv in configs:
            for dtype in [torch.float16, torch.bfloat16]:
                with self.subTest(
                    num_heads=num_heads,
                    head_dim=head_dim,
                    num_heads_kv=num_heads_kv,
                    dtype=dtype,
                ):
                    self.run_case_decode_int8_attention(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        seq_len=seq_len,
                        num_requests=num_requests,
                        dtype=dtype,
                        num_heads_kv=num_heads_kv,
                        k_scale=0.1,
                        v_scale=0.1,
                        atol_override=large_scale_tol,
                    )

    def test_case_decode_int8_odd_dimensions(self):
        """Case: odd dimension decode layout."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_decode_int8_attention(
                num_heads=17, head_dim=127, seq_len=255, num_requests=3, dtype=dtype
            )

    def test_case_decode_int8_minimal(self):
        """Case: minimal decode configuration."""
        for dtype in [torch.float16, torch.bfloat16]:
            self.run_case_decode_int8_attention(
                num_heads=1, head_dim=32, seq_len=1, num_requests=1, dtype=dtype
            )

    def test_case_decode_int8_per_layer_scale_isolation(self):
        """Case: per-layer scale isolation preserves decode correctness."""
        num_heads = 8
        num_heads_kv = 2
        head_dim = 64
        head_dim_v = head_dim
        seq_len = 32
        num_requests = 2
        dtype = torch.bfloat16
        scale_A = 0.01
        scale_B = 0.50

        max_seq_len = seq_len + 16
        max_tokens = num_requests * max_seq_len
        sm_scale = 1.0 / (head_dim**0.5)

        torch.manual_seed(42)
        query = torch.randn(num_requests, num_heads, head_dim, dtype=dtype)
        k_float = torch.randn(max_tokens, num_heads_kv, head_dim) * 5.0
        v_float = torch.randn(max_tokens, num_heads_kv, head_dim) * 5.0
        k_int8 = (k_float / scale_A).round().clamp(-128, 127).to(torch.int8)
        v_int8 = (v_float / scale_A).round().clamp(-128, 127).to(torch.int8)

        req_to_token = torch.zeros(num_requests, max_seq_len, dtype=torch.long)
        req_pool_indices = torch.arange(num_requests, dtype=torch.long)
        seq_lens = torch.full((num_requests,), seq_len, dtype=torch.long)
        for i in range(num_requests):
            start = i * max_seq_len
            req_to_token[i, :seq_len] = torch.arange(start, start + seq_len)

        key_new = torch.randn(num_requests, num_heads_kv, head_dim)
        val_new = torch.randn(num_requests, num_heads_kv, head_dim)
        key_new_int8 = (key_new / scale_A).round().clamp(-128, 127).to(torch.int8)
        val_new_int8 = (val_new / scale_A).round().clamp(-128, 127).to(torch.int8)
        loc = torch.tensor(
            [req_to_token[i, seq_len - 1].item() for i in range(num_requests)],
            dtype=torch.int64,
        )

        attn_logits = torch.zeros(num_requests, num_heads, 1, head_dim + 1)

        k_ref = k_int8.clone()
        v_ref = v_int8.clone()
        for i in range(num_requests):
            k_ref[loc[i]] = key_new_int8[i]
            v_ref[loc[i]] = val_new_int8[i]
        ref_output = naive_attention_decode(
            query,
            k_ref,
            v_ref,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            k_scale=scale_A,
            v_scale=scale_A,
            enable_gqa=True,
        )

        def _run(scale_buf_val: float) -> torch.Tensor:
            out = torch.zeros(num_requests, num_heads, head_dim_v, dtype=dtype)
            k_scale_buf = torch.full((max_tokens, num_heads_kv), scale_buf_val)
            v_scale_buf = torch.full((max_tokens, num_heads_kv), scale_buf_val)
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
                query,
                k_int8,
                v_int8,
                out,
                key_new_int8,
                val_new_int8,
                loc,
                attn_logits,
                req_to_token,
                req_pool_indices,
                seq_lens,
                sm_scale,
                0.0,
                k_scale_buf,
                v_scale_buf,
                scale_buf_val,
                scale_buf_val,
            )
            return out

        atol = rtol = precision["attention_decode_int8"][dtype]

        out_wrong = _run(scale_B)
        self.assertFalse(
            torch.allclose(out_wrong, ref_output, atol=atol, rtol=rtol),
            "Using wrong scale (scale_B) should produce incorrect output — "
            "if this assertion fails, scale_B is too close to scale_A.",
        )

        out_correct = _run(scale_A)
        torch.testing.assert_close(out_correct, ref_output, atol=atol, rtol=rtol)

    def test_case_decode_int8_logit_cap(self):
        """INT8 decode with logit_cap > 0 must exercise the tanh-softcap branch."""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.run_case_decode_int8_attention(
                    num_heads=8,
                    head_dim=64,
                    seq_len=64,
                    num_requests=2,
                    dtype=dtype,
                    logit_cap=30.0,
                )
                self.run_case_decode_int8_attention(
                    num_heads=8,
                    num_heads_kv=2,
                    head_dim=64,
                    seq_len=64,
                    num_requests=2,
                    dtype=dtype,
                    logit_cap=30.0,
                )

    def test_case_decode_int8_mqa(self):
        """INT8 decode with MQA (num_kv_heads=1) exercises the H_KV=1 reduction path."""
        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                self.run_case_decode_int8_attention(
                    num_heads=16,
                    num_heads_kv=1,
                    head_dim=64,
                    seq_len=128,
                    num_requests=4,
                    dtype=dtype,
                )

    def test_case_decode_int8_per_token_varying_scale(self):
        """Per-token dynamic scale: each cached token has a distinct dequantization scale."""
        torch.manual_seed(7)
        num_heads = 8
        num_heads_kv = 2
        head_dim = 64
        seq_len = 64
        num_requests = 2
        new_scale = 0.01
        sm_scale = 1.0 / (head_dim**0.5)
        max_seq_len = seq_len + 16
        max_tokens = num_requests * max_seq_len

        for dtype in [torch.float16, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                query = torch.randn(num_requests, num_heads, head_dim, dtype=dtype)
                k_buffer_float = torch.randn(max_tokens, num_heads_kv, head_dim) * 5.0
                v_buffer_float = torch.randn(max_tokens, num_heads_kv, head_dim) * 5.0

                # Vary scales per cached token to exercise token-indexed lookup.
                k_scale_buf = torch.rand(max_tokens, num_heads_kv) * 0.099 + 0.001
                v_scale_buf = torch.rand(max_tokens, num_heads_kv) * 0.099 + 0.001

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

                req_to_token = (
                    torch.randperm(max_tokens)
                    .reshape(num_requests, max_seq_len)
                    .to(torch.long)
                )
                req_pool_indices = torch.arange(num_requests, dtype=torch.long)
                seq_lens = torch.full((num_requests,), seq_len, dtype=torch.long)

                key_float = torch.randn(num_requests, num_heads_kv, head_dim)
                value_float = torch.randn(num_requests, num_heads_kv, head_dim)
                key_int8 = (
                    (key_float / new_scale).round().clamp(-128, 127).to(torch.int8)
                )
                value_int8 = (
                    (value_float / new_scale).round().clamp(-128, 127).to(torch.int8)
                )

                loc = torch.tensor(
                    [req_to_token[i, seq_len - 1].item() for i in range(num_requests)],
                    dtype=torch.long,
                )

                output = torch.zeros(num_requests, num_heads, head_dim, dtype=dtype)
                attn_logits = torch.zeros(num_requests, num_heads, 1, head_dim + 1)

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
                    0.0,
                    k_scale_buf,
                    v_scale_buf,
                    new_scale,
                    new_scale,
                )

                # Mirror the kernel-side writes in the reference state.
                k_ref = k_buffer_int8.clone()
                v_ref = v_buffer_int8.clone()
                k_scale_buf_ref = k_scale_buf.clone()
                v_scale_buf_ref = v_scale_buf.clone()
                for i in range(num_requests):
                    t = loc[i].item()
                    k_ref[t] = key_int8[i]
                    v_ref[t] = value_int8[i]
                    k_scale_buf_ref[t] = new_scale
                    v_scale_buf_ref[t] = new_scale

                ref_output = naive_attention_decode(
                    query,
                    k_ref,
                    v_ref,
                    req_to_token,
                    req_pool_indices,
                    seq_lens,
                    sm_scale,
                    enable_gqa=True,
                    k_scale_buf=k_scale_buf_ref,
                    v_scale_buf=v_scale_buf_ref,
                )

                atol = rtol = precision["attention_decode_int8"][dtype]
                torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

    def test_case_decode_int8_static_unit_scale_is_literal(self):
        """A scalar scale of 1.0 must remain a literal static scale, not a dynamic sentinel."""
        dtype = torch.float16
        num_heads = 4
        num_heads_kv = 4
        head_dim = 32
        seq_len = 4
        num_requests = 1
        sm_scale = 1.0 / (head_dim**0.5)
        max_seq_len = 8
        max_tokens = 8

        query = torch.randn(num_requests, num_heads, head_dim, dtype=dtype)
        k_buffer_int8 = torch.randint(
            -4, 5, (max_tokens, num_heads_kv, head_dim), dtype=torch.int8
        )
        v_buffer_int8 = torch.randint(
            -4, 5, (max_tokens, num_heads_kv, head_dim), dtype=torch.int8
        )
        key_int8 = torch.randint(
            -4, 5, (num_requests, num_heads_kv, head_dim), dtype=torch.int8
        )
        value_int8 = torch.randint(
            -4, 5, (num_requests, num_heads_kv, head_dim), dtype=torch.int8
        )
        output = torch.zeros(num_requests, num_heads, head_dim, dtype=dtype)
        attn_logits = torch.zeros(num_requests, num_heads, 1, head_dim + 1)
        req_to_token = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([seq_len], dtype=torch.long)
        loc = torch.tensor([seq_len - 1], dtype=torch.long)
        k_scale_buf = torch.ones(max_tokens, num_heads_kv, dtype=torch.float32)
        v_scale_buf = torch.ones(max_tokens, num_heads_kv, dtype=torch.float32)

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
            0.0,
            k_scale_buf,
            v_scale_buf,
            1.0,
            1.0,
        )

        k_ref = k_buffer_int8.clone()
        v_ref = v_buffer_int8.clone()
        k_ref[loc[0].item()] = key_int8[0]
        v_ref[loc[0].item()] = value_int8[0]
        ref_output = naive_attention_decode(
            query,
            k_ref,
            v_ref,
            req_to_token,
            req_pool_indices,
            seq_lens,
            sm_scale,
            k_scale=1.0,
            v_scale=1.0,
        )
        torch.testing.assert_close(
            output,
            ref_output,
            atol=precision["attention_decode_int8"][dtype],
            rtol=precision["attention_decode_int8"][dtype],
        )

    def test_case_decode_int8_rejects_mixed_kv_dtypes(self):
        dtype = torch.float16
        query = torch.randn(1, 2, 32, dtype=dtype)
        k_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        v_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        output = torch.zeros(1, 2, 32, dtype=dtype)
        key = torch.randint(-8, 8, (1, 2, 32), dtype=torch.int8)
        value = torch.randn(1, 2, 32, dtype=dtype)
        loc = torch.tensor([0], dtype=torch.long)
        attn_logits = torch.zeros(1, 2, 1, 33)
        req_to_token = torch.zeros(1, 4, dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        k_scale_buf = torch.ones(8, 2, dtype=torch.float32)
        v_scale_buf = torch.ones(8, 2, dtype=torch.float32)

        with self.assertRaisesRegex(
            RuntimeError, "expect key and value to have the same dtype"
        ):
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
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
                1.0,
                0.0,
                k_scale_buf,
                v_scale_buf,
                0.01,
                0.01,
            )

    def test_case_decode_int8_rejects_bad_scale_buffer_shape(self):
        dtype = torch.float16
        query = torch.randn(1, 2, 32, dtype=dtype)
        k_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        v_buffer = torch.zeros(8, 2, 32, dtype=torch.int8)
        output = torch.zeros(1, 2, 32, dtype=dtype)
        key = torch.randint(-8, 8, (1, 2, 32), dtype=torch.int8)
        value = torch.randint(-8, 8, (1, 2, 32), dtype=torch.int8)
        loc = torch.tensor([0], dtype=torch.long)
        attn_logits = torch.zeros(1, 2, 1, 33)
        req_to_token = torch.zeros(1, 4, dtype=torch.long)
        req_pool_indices = torch.tensor([0], dtype=torch.long)
        seq_lens = torch.tensor([1], dtype=torch.long)
        bad_k_scale_buf = torch.ones(7, 2, dtype=torch.float32)
        v_scale_buf = torch.ones(8, 2, dtype=torch.float32)

        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.decode_attention_int8_cpu(
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
                1.0,
                0.0,
                bad_k_scale_buf,
                v_scale_buf,
                0.01,
                0.01,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
