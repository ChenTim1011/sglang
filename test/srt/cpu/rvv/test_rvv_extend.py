"""Unit tests for RVV extend attention kernel (BF16/FP16)."""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision, run_sdpa_forward_extend

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("extend_attention_cpu"),
    "extend_attention_cpu not available (non-RISC-V build)",
)
class TestRVVExtendBase(CustomTestCase):
    """Shared fixtures and helpers for RVV extend attention tests."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _run_extend(
        self,
        B,
        N_CTX,
        H_Q,
        H_KV,
        D,
        DV,
        logit_cap=0.0,
        force_prefix_zero=False,
        dtype=torch.bfloat16,
    ):
        gen = torch.Generator(device=self.device)
        gen.manual_seed(1234)
        _hi = max(N_CTX // 2, 2)  # randint requires hi > lo=1; guard N_CTX=1
        if force_prefix_zero:
            b_seq_len_prefix = torch.zeros(B, dtype=torch.int32)
            b_seq_len_extend = torch.randint(
                1, _hi, (B,), dtype=torch.int32, generator=gen
            )
        else:
            b_seq_len_prefix = torch.randint(
                1, _hi, (B,), dtype=torch.int32, generator=gen
            )
            b_seq_len_extend = torch.randint(
                1, _hi, (B,), dtype=torch.int32, generator=gen
            )
        b_seq_len = b_seq_len_prefix + b_seq_len_extend
        max_len_in_batch = torch.max(b_seq_len, 0)[0].item()

        b_req_idx = torch.arange(B, dtype=torch.int32)
        req_to_tokens = torch.empty((B, max_len_in_batch), dtype=torch.int32)
        b_start_loc = torch.zeros((B,), dtype=torch.int32)
        b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
        b_start_loc_extend = torch.zeros((B,), dtype=torch.int32)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

        for i in range(B):
            req_to_tokens[i, : b_seq_len[i]] = torch.arange(
                b_start_loc[i], b_start_loc[i] + b_seq_len[i]
            )

        total_token_num = torch.sum(b_seq_len).item()
        extend_token_num = torch.sum(b_seq_len_extend).item()

        k_buffer = torch.randn((total_token_num, H_KV, D), dtype=dtype, generator=gen)
        v_buffer = torch.randn((total_token_num, H_KV, DV), dtype=dtype, generator=gen)

        k_extend = torch.empty((extend_token_num, H_KV, D), dtype=dtype)
        v_extend = torch.empty((extend_token_num, H_KV, DV), dtype=dtype)
        q_extend = torch.empty((extend_token_num, H_Q, D), dtype=dtype)

        for i in range(B):
            extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
            extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
            extend_start = b_start_loc_extend[i]
            extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
            k_extend[extend_start:extend_end] = k_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            v_extend[extend_start:extend_end] = v_buffer[
                extend_start_in_buffer:extend_end_in_buffer
            ]
            q_extend[extend_start:extend_end] = torch.randn(
                (b_seq_len_extend[i], H_Q, D), dtype=dtype, generator=gen
            )

        # Feed non-contiguous views to match the kernel path used in production.
        q_extend_k = q_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_extend_k = k_extend.transpose(0, 1).contiguous().transpose(0, 1)
        v_extend_k = v_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_buffer_k = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer_k = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)

        b_seq_len_extend = b_seq_len - b_seq_len_prefix
        b_start_loc_extend = torch.zeros_like(b_seq_len)
        b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
        max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()

        sm_scale = 1.0 / (D**0.5)

        b_req_idx = b_req_idx.to(torch.int64)
        b_seq_len = b_seq_len.to(torch.int64)

        enable_gqa = H_Q != H_KV
        o_ref = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
        run_sdpa_forward_extend(
            q_extend,
            o_ref,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_prefix,
            b_seq_len_extend,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
            causal=True,
        )

        o_extend = torch.empty((extend_token_num, H_Q, DV), dtype=dtype)
        torch.ops.sgl_kernel.extend_attention_cpu(
            q_extend_k,
            k_extend_k,
            v_extend_k,
            o_extend,
            k_buffer_k,
            v_buffer_k,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            sm_scale,
            logit_cap,
        )

        atol = rtol = (
            precision["attention_decode_logit_cap"].get(
                dtype, precision["attention_extend"][dtype]
            )
            if logit_cap > 0.0
            else precision["attention_extend"][dtype]
        )
        torch.testing.assert_close(o_ref, o_extend, atol=atol, rtol=rtol)


class TestRVVExtendMHA(TestRVVExtendBase):
    """MHA path tests for RVV extend attention."""

    def test_mha(self):
        """Varied shapes cover head-count and prefix/extend length boundaries."""
        configs = [
            (2, 128, 16, 16, 128, 96, 0.0),
            (2, 64, 8, 8, 32, 32, 0.0),
            (4, 64, 8, 8, 128, 96, 0.0),
            (2, 128, 16, 16, 128, 96, 50.0),
            (1, 256, 4, 4, 64, 64, 0.0),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV, logit_cap in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B,
                    N_CTX=N_CTX,
                    H_Q=H_Q,
                    H_KV=H_KV,
                    D=D,
                    DV=DV,
                    logit_cap=logit_cap,
                    dtype=dtype,
                ):
                    self._run_extend(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        dtype=dtype,
                        logit_cap=logit_cap,
                    )

    def test_mha_zero_prefix(self):
        """Zero prefix ensures Stage 1 (cached KV) is skipped entirely."""
        configs = [
            (2, 128, 16, 16, 128, 96, 0.0),
            (1, 64, 8, 8, 64, 64, 0.0),
            (4, 64, 8, 8, 128, 96, 0.0),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV, logit_cap in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self._run_extend(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        dtype=dtype,
                        logit_cap=logit_cap,
                        force_prefix_zero=True,
                    )

    def test_mha_single_token(self):
        """N_CTX=1 hits the single-row BLOCK_M boundary; tiling must not OOB."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype, force_prefix_zero=False):
                self._run_extend(
                    B=2,
                    N_CTX=1,
                    H_Q=8,
                    H_KV=8,
                    D=64,
                    DV=64,
                    dtype=dtype,
                )
            with self.subTest(dtype=dtype, force_prefix_zero=True):
                self._run_extend(
                    B=2,
                    N_CTX=1,
                    H_Q=8,
                    H_KV=8,
                    D=64,
                    DV=64,
                    dtype=dtype,
                    force_prefix_zero=True,
                )


class TestRVVExtendGQA(TestRVVExtendBase):
    """GQA path tests for RVV extend attention."""

    def test_gqa(self):
        """GQA shapes verify H_Q/H_KV ratio broadcasting is correct in both stages."""
        configs = [
            (2, 128, 32, 8, 128, 96),
            (4, 128, 24, 4, 64, 64),
            (1, 256, 16, 2, 128, 128),
            (2, 64, 18, 3, 32, 32),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self._run_extend(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        dtype=dtype,
                    )

    def test_gqa_logit_cap(self):
        """GQA + logit_cap: tanh-softcap must interact correctly with GQA broadcasting."""
        configs = [
            (2, 128, 32, 8, 128, 96),
            (1, 256, 16, 2, 128, 128),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(B=B, H_Q=H_Q, H_KV=H_KV, dtype=dtype):
                    self._run_extend(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        logit_cap=50.0,
                        dtype=dtype,
                    )

    def test_gqa_zero_prefix(self):
        """GQA with no prefix: Stage 1 (cached KV) is skipped, only Stage 2 runs."""
        configs = [
            (2, 128, 32, 8, 128, 96),
            (1, 256, 16, 2, 128, 128),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(B=B, H_Q=H_Q, H_KV=H_KV, dtype=dtype):
                    self._run_extend(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        dtype=dtype,
                        force_prefix_zero=True,
                    )


if __name__ == "__main__":
    unittest.main()
