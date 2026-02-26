"""
This file tests the RVV-optimized extend (prefill) attention kernel.
Extend attention is used during the initial prompt processing phase.

Features tested:
- MHA (Multi-Head Attention): num_heads == num_heads_kv
- GQA (Grouped Query Attention): num_heads != num_heads_kv
- MLA (Multi-Latent Attention): shared KV buffer with num_heads_kv=1
- Various configurations: small, medium, large prompts
- Different head dimensions and extend lengths

Usage:
    python3 test_rvv_extend.py -v                      # Run all tests
"""

import importlib.util
import unittest

import torch

from .utils import precision, run_sdpa_forward_extend

try:
    import sgl_kernel  # noqa: F401
except ImportError:
    pass

torch.manual_seed(1234)


def _has_extend_attention_cpu() -> bool:
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "extend_attention_cpu")
    except (AttributeError, RuntimeError):
        return False


@unittest.skipUnless(
    _has_extend_attention_cpu(), "extend_attention_cpu not available (non-RISC-V build)"
)
class RVVExtendTestBase(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _test_extend_attention_once(
        self,
        B,
        N_CTX,
        H_Q,
        H_KV,
        D,
        DV,
        mla=False,
        dtype=torch.bfloat16,
        logit_cap=0.0,
        force_prefix_zero=False,
    ):
        if force_prefix_zero:
            b_seq_len_prefix = torch.zeros(B, dtype=torch.int32)
            b_seq_len_extend = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
        else:
            b_seq_len_prefix = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
            b_seq_len_extend = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
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

        atol = rtol = precision[dtype]
        if logit_cap > 0.0 and dtype == torch.float16:
            atol = rtol = 1e-2
        torch.testing.assert_close(o_ref, o_extend, atol=atol, rtol=rtol)


class TestExtendAttentionMHA(RVVExtendTestBase):
    """
    Test RVV extend attention for MHA (Multi-Head Attention) path.
    """

    def test_mha_configs(self):
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV, logit_cap)
        configs = [
            (2, 128, 16, 16, 128, 96, 0.0),  # Basic MHA
            (2, 64, 8, 8, 32, 32, 0.0),  # Odd dimensions tail handling
            (4, 64, 8, 8, 128, 96, 0.0),  # Stress test: large batch
            (2, 128, 16, 16, 128, 96, 50.0),  # Logit cap > 0
            (1, 256, 4, 4, 64, 64, 0.0),  # Longer prompt context
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
                    self._test_extend_attention_once(
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

    def test_mha_extend_only(self):
        """Pure Stage 2 path: prefix_len=0 for all requests (no cached prefix tokens)."""
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
                    self._test_extend_attention_once(
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


class TestExtendAttentionGQA(RVVExtendTestBase):
    """
    Test RVV extend attention for GQA (Grouped Query Attention) path.
    """

    def test_gqa_configs(self):
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV)
        configs = [
            (2, 128, 32, 8, 128, 96),  # LLaMA-style GQA (4:1)
            (4, 128, 24, 4, 64, 64),  # 6:1 ratio
            (1, 256, 16, 2, 128, 128),  # 8:1 ratio
            (2, 64, 18, 3, 32, 32),  # Non-power-of-2 ratio
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self._test_extend_attention_once(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=False,
                        dtype=dtype,
                    )


class TestExtendAttentionMLA(RVVExtendTestBase):
    """
    Test RVV extend attention for MLA (Multi-Latent Attention) path - DeepSeek architecture.
    """

    def test_mla_configs(self):
        """Exhaustive (carpet-search) test over multiple MLA configs."""
        # Config format: (B, N_CTX, H_Q, H_KV, D, DV)
        configs = [
            (2, 128, 22, 1, 192, 128),  # DeepSeek-style typical config
            (4, 256, 16, 1, 192, 128),  # Larger batch/seq
            (1, 128, 17, 1, 192, 128),  # Odd number of heads
        ]
        for B, N_CTX, H_Q, H_KV, D, DV in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, N_CTX=N_CTX, H_Q=H_Q, H_KV=H_KV, D=D, DV=DV, dtype=dtype
                ):
                    self._test_extend_attention_once(
                        B=B,
                        N_CTX=N_CTX,
                        H_Q=H_Q,
                        H_KV=H_KV,
                        D=D,
                        DV=DV,
                        mla=True,
                        dtype=dtype,
                    )


if __name__ == "__main__":
    unittest.main()
