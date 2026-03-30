"""Unit tests for RVV decode attention kernel (FP16/BF16).

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_decode -v
"""

import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision, run_sdpa_forward_decode

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeBase(CustomTestCase):
    """Shared fixtures and helpers for RVV decode attention tests."""

    def setUp(self):
        super().setUp()
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def run_case_decode_attention_grouped(
        self,
        B,
        H_Q,
        H_KV,
        D,
        D_V,
        seq_len=1024,
        dtype=torch.float16,
        logit_cap=0.0,
        num_kv_splits=2,
    ):
        device = self.device
        seed = 1234
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        enable_gqa = H_Q != H_KV

        q = torch.randn(B, H_Q, D, dtype=dtype, device=device, generator=gen)
        k_buffer = torch.randn(
            total_tokens, H_KV, D, dtype=dtype, device=device, generator=gen
        )
        v_buffer = torch.randn(
            total_tokens, H_KV, D_V, dtype=dtype, device=device, generator=gen
        )

        key = torch.randn(B, H_KV, D, dtype=dtype, device=device, generator=gen)
        value = torch.randn(B, H_KV, D_V, dtype=dtype, device=device, generator=gen)
        loc = torch.randint(
            0, total_tokens, (B,), dtype=torch.int64, device=device, generator=gen
        )

        k_buffer[loc] = key
        v_buffer[loc] = value

        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        # Fragment the token map so cache access is not accidentally contiguous.
        random_indices = torch.randperm(total_tokens, device=device, generator=gen)
        req_to_token = random_indices.reshape(B, seq_len).to(torch.int32)

        b_req_idx = torch.arange(B, device=device).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device=device,
        )

        # Feed non-contiguous views to match the kernel contract.
        k_buffer_k = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer_k = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        q_k = q.transpose(0, 1).contiguous().transpose(0, 1)
        key_k = key.transpose(0, 1).contiguous().transpose(0, 1)
        value_k = value.transpose(0, 1).contiguous().transpose(0, 1)

        torch.ops.sgl_kernel.decode_attention_cpu(
            q_k,
            k_buffer_k,
            v_buffer_k,
            o,
            key_k,
            value_k,
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            logit_cap,
        )

        run_sdpa_forward_decode(
            q,
            o_grouped,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=enable_gqa,
            logit_cap=logit_cap,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        prec = (
            precision["attention_decode_logit_cap"].get(
                q.dtype, precision["attention_decode"][q.dtype]
            )
            if logit_cap > 0.0
            else precision["attention_decode"][q.dtype]
        )
        # BF16 tanh softcapping is slightly noisier in the polynomial path.
        cos_sim_threshold = (
            0.98 if (logit_cap > 0.0 and q.dtype == torch.bfloat16) else 0.99
        )
        self.assertGreater(cos_sim.item(), cos_sim_threshold)
        torch.testing.assert_close(o, o_grouped, atol=prec, rtol=prec)


class TestRVVDecodeMHA(TestRVVDecodeBase):
    """Test suite for RVV decode attention MHA path."""

    def test_case_mha_cases(self):
        """Case: MHA decode across representative shapes and dtypes."""
        configs = [
            (1, 1, 1, 64, 64, 128),
            (2, 8, 8, 128, 128, 63),
            (2, 8, 8, 128, 128, 65),
            (2, 8, 8, 128, 128, 129),
            (2, 8, 8, 128, 128, 256),
            (4, 16, 16, 64, 64, 512),
            (2, 17, 17, 127, 127, 128),
            (8, 8, 8, 128, 128, 128),
            # Asymmetric Q/V head dim.
            (2, 8, 8, 128, 64, 128),
            # Non-power-of-2 head dims exercise vector tails.
            (2, 8, 8, 33, 55, 64),
            (2, 8, 8, 80, 80, 64),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self.run_case_decode_attention_grouped(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype
                    )


class TestRVVDecodeGQA(TestRVVDecodeBase):
    """Test suite for RVV decode attention GQA and MQA paths."""

    def test_case_gqa_mqa_cases(self):
        """Case: GQA and MQA decode across representative shapes and dtypes."""
        configs = [
            (2, 32, 8, 128, 128, 63),
            (2, 32, 8, 128, 128, 129),
            (2, 32, 8, 128, 128, 256),
            (4, 16, 1, 128, 128, 128),
            (2, 18, 3, 128, 128, 128),
            (2, 21, 7, 128, 128, 128),
            (1, 12, 3, 64, 64, 512),
            # Asymmetric Q/V head dim with GQA.
            (2, 32, 8, 128, 64, 128),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self.run_case_decode_attention_grouped(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype
                    )


class TestRVVDecodeLogitCap(TestRVVDecodeBase):
    """Test suite for RVV decode attention logit_cap (tanh softcapping) path."""

    def test_case_decode_logit_cap(self):
        """FP decode with logit_cap > 0 must exercise the tanh-cap branch."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self.run_case_decode_attention_grouped(
                    2, 8, 8, 128, 128, seq_len=64, logit_cap=30.0, dtype=dtype
                )
                self.run_case_decode_attention_grouped(
                    2, 8, 2, 128, 128, seq_len=64, logit_cap=30.0, dtype=dtype
                )

    def test_case_decode_seq_len_one(self):
        """FP decode with a single KV token exercises the kv-split boundary."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self.run_case_decode_attention_grouped(
                    1, 1, 1, 64, 64, seq_len=1, dtype=dtype
                )


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeKvSplits2(TestRVVDecodeBase):
    """Verify the production-default 2-split path for both MHA and GQA."""

    def test_case_kv_splits_2(self):
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self.run_case_decode_attention_grouped(
                    2, 8, 8, 128, 128, seq_len=64, dtype=dtype, num_kv_splits=2
                )
                self.run_case_decode_attention_grouped(
                    2, 8, 2, 128, 128, seq_len=64, dtype=dtype, num_kv_splits=2
                )


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeValidation(CustomTestCase):
    """Validation-only decode tests that should run once."""

    def test_case_decode_rejects_mixed_fp_dtypes(self):
        q = torch.randn(1, 2, 32, dtype=torch.float16)
        k_buffer = torch.randn(8, 2, 32, dtype=torch.float16)
        v_buffer = torch.randn(8, 2, 32, dtype=torch.float16)
        output = torch.zeros(1, 2, 32, dtype=torch.float16)
        key = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        value = torch.randn(1, 2, 32, dtype=torch.bfloat16)
        loc = torch.tensor([0], dtype=torch.int64)
        attn_logits = torch.empty((1, 2, 1, 33), dtype=torch.float32)
        req_to_token = torch.zeros((1, 4), dtype=torch.int32)
        req_pool_indices = torch.tensor([0], dtype=torch.int64)
        seq_lens = torch.tensor([1], dtype=torch.int64)

        with self.assertRaisesRegex(
            RuntimeError,
            "expect query, key, value, k_buffer, and v_buffer to have the same dtype",
        ):
            torch.ops.sgl_kernel.decode_attention_cpu(
                q,
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
            )


if __name__ == "__main__":
    unittest.main()
