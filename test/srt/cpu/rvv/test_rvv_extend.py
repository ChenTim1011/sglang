"""Unit tests for RVV extend attention kernel (BF16/FP16).

- MHA (Multi-Head Attention): num_heads == num_heads_kv
- GQA (Grouped Query Attention): num_heads != num_heads_kv
- MLA (Multi-Latent Attention): shared KV buffer with num_heads_kv=1
- Various configurations: small, medium, large prompts
- Different head dimensions and extend lengths

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_extend -v
"""

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

    def run_case_extend_attention(
        self,
        B,
        N_CTX,
        H_Q,
        H_KV,
        D,
        DV,
        mla=False,
        logit_cap=0.0,
        force_prefix_zero=False,
        dtype=torch.bfloat16,
    ):
        _hi = max(N_CTX // 2, 2)  # randint requires hi > lo=1; guard N_CTX=1
        if force_prefix_zero:
            b_seq_len_prefix = torch.zeros(B, dtype=torch.int32)
            b_seq_len_extend = torch.randint(1, _hi, (B,), dtype=torch.int32)
        else:
            b_seq_len_prefix = torch.randint(1, _hi, (B,), dtype=torch.int32)
            b_seq_len_extend = torch.randint(1, _hi, (B,), dtype=torch.int32)
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

        H_BUF = 1 if mla else H_KV
        k_buffer = torch.randn((total_token_num, H_BUF, D), dtype=dtype)
        v_buffer = torch.randn((total_token_num, H_BUF, DV), dtype=dtype)

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
                (b_seq_len_extend[i], H_Q, D), dtype=dtype
            )

        # q_extend, k_extend, v_extend, k_buffer and v_buffer supports non-contiguous tensors
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

        # handle index type
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
            precision["logit_cap"][torch.float16]
            if (logit_cap > 0.0 and dtype == torch.float16)
            else precision["default"][dtype]
        )
        torch.testing.assert_close(o_ref, o_extend, atol=atol, rtol=rtol)


class TestRVVExtendMHA(TestRVVExtendBase):
    """Test suite for RVV extend attention MHA path."""

    def test_case_mha_cases(self):
        """Case: MHA extend across representative shapes and dtypes."""
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV, logit_cap)
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
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=False,
                        dtype=dtype,
                        logit_cap=logit_cap,
                    )

    def test_case_mha_extend_only(self):
        """Case: pure extend stage with zero prefix lengths."""
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
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=False,
                        dtype=dtype,
                        logit_cap=logit_cap,
                        force_prefix_zero=True,
                    )

    def test_case_mha_single_token_extend(self):
        """Case: N_CTX=1 — single token extend exercises the M=1 boundary of BLOCK_M=32.

        BLOCK_M=32 means m_count=min(32,M); at M=1 only acc0 is active and
        acc1-acc3 must be correctly guarded. Tests both with prefix (Stage 1+2)
        and without (Stage 2 only).
        """
        for dtype in self.dtypes:
            # With prefix: both Stage 1 (Q@K_prefix) and Stage 2 (Q@K_extend) run with M=1
            with self.subTest(dtype=dtype, force_prefix_zero=False):
                self.run_case_extend_attention(
                    B=2,
                    N_CTX=1,
                    H_Q=8,
                    H_KV=8,
                    D=64,
                    DV=64,
                    mla=False,
                    dtype=dtype,
                )
            # Without prefix: Stage 2 only with M=1
            with self.subTest(dtype=dtype, force_prefix_zero=True):
                self.run_case_extend_attention(
                    B=2,
                    N_CTX=1,
                    H_Q=8,
                    H_KV=8,
                    D=64,
                    DV=64,
                    mla=False,
                    dtype=dtype,
                    force_prefix_zero=True,
                )


class TestRVVExtendGQA(TestRVVExtendBase):
    """Test suite for RVV extend attention GQA path."""

    def test_case_gqa_cases(self):
        """Case: GQA extend across representative shapes and dtypes."""
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV)
        configs = [
            (2, 128, 32, 8, 128, 96),  # GQA (4:1)
            (4, 128, 24, 4, 64, 64),  # 6:1 ratio
            (1, 256, 16, 2, 128, 128),  # 8:1 ratio
            (2, 64, 18, 3, 32, 32),  # Non-power-of-2 ratio
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=False,
                        dtype=dtype,
                    )

    def test_case_gqa_zero_prefix(self):
        """GQA extend with no prefix tokens (Stage 2 only path)."""
        configs = [
            (2, 128, 32, 8, 128, 96),
            (1, 256, 16, 2, 128, 128),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(B=B, H_Q=H_Q, H_KV=H_KV, dtype=dtype):
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=False,
                        dtype=dtype,
                        force_prefix_zero=True,
                    )


class TestRVVExtendMLA(TestRVVExtendBase):
    """Test suite for RVV extend attention MLA path."""

    def test_case_mla_cases(self):
        """Case: MLA extend across representative shapes and dtypes."""
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV)
        configs = [
            (2, 128, 22, 1, 192, 128),
            (4, 256, 16, 1, 192, 128),
            (1, 128, 17, 1, 192, 128),  # Odd number of heads
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=True,
                        dtype=dtype,
                    )

    def test_case_mla_zero_prefix(self):
        """MLA extend with no prefix tokens (Stage 2 only path)."""
        configs = [
            (2, 128, 22, 1, 192, 128),
            (1, 128, 17, 1, 192, 128),
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(B=B, H_Q=H_Q, dtype=dtype):
                    self.run_case_extend_attention(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=True,
                        dtype=dtype,
                        force_prefix_zero=True,
                    )

    def test_case_mla_with_logit_cap(self):
        """MLA extend with logit_cap > 0: kernel applies tanh softcapping on shared KV path."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self.run_case_extend_attention(
                    B=2,
                    N_CTX=128,
                    H_Q=22,
                    H_KV=1,
                    D=192,
                    DV=128,
                    mla=True,
                    dtype=dtype,
                    logit_cap=50.0,
                )


if __name__ == "__main__":
    unittest.main()
