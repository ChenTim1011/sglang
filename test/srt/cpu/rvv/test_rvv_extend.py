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

import sys
import unittest

import torch

# Add parent directory to path to import test utilities
sys.path.insert(0, "/workspace/sglang/test/srt/cpu")

# Import existing test class to reuse reference implementation
try:
    from test_extend import TestExtendAttention
except ImportError:
    # Fallback for local development
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from test_extend import TestExtendAttention

torch.manual_seed(1234)


class TestRVVExtendAttention(TestExtendAttention):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _test_extend_attention_once(
        self, B, N_CTX, H_Q, H_KV, D, DV, mla=False, dtype=torch.bfloat16, logit_cap=0.0
    ):
        b_seq_len_prefix = torch.randint(1, N_CTX // 2, (B,), dtype=torch.int32)
        if mla:
            b_seq_len_prefix.zero_()
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
        q_extend = q_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_extend = k_extend.transpose(0, 1).contiguous().transpose(0, 1)
        v_extend = v_extend.transpose(0, 1).contiguous().transpose(0, 1)
        k_buffer = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)

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
        self._run_sdpa_forward_extend(
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
            q_extend,
            k_extend,
            v_extend,
            o_extend,
            k_buffer,
            v_buffer,
            req_to_tokens,
            b_req_idx,
            b_seq_len,
            b_seq_len_extend,
            b_start_loc_extend,
            max_len_extend,
            sm_scale,
            logit_cap,
        )

        torch.testing.assert_close(o_ref, o_extend, atol=1e-2, rtol=1e-2)


class TestExtendAttentionMHA(TestRVVExtendAttention):
    """
    Test RVV extend attention for MHA (Multi-Head Attention) path.
    """

    def test_extend_basic(self):
        """Basic MHA extend test"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=2, N_CTX=512, H_Q=16, H_KV=16, D=128, DV=96, dtype=dtype, mla=False
            )

    def test_extend_logit_cap(self):
        """Edge case: logit_cap > 0 to test clamping logic"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=2,
                N_CTX=512,
                H_Q=16,
                H_KV=16,
                D=128,
                DV=96,
                dtype=dtype,
                logit_cap=50.0,
                mla=False,
            )

    def test_extend_odd_dimensions(self):
        """Edge case: odd dimensions (tests tail handling in RVV loops)"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=2, N_CTX=256, H_Q=8, H_KV=8, D=32, DV=32, dtype=dtype, mla=False
            )

    def test_extend_large_batch(self):
        """Stress test: large batch size"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=5, N_CTX=128, H_Q=8, H_KV=8, D=128, DV=96, dtype=dtype, mla=False
            )


class TestExtendAttentionGQA(TestRVVExtendAttention):
    """
    Test RVV extend attention for GQA (Grouped Query Attention) path.
    """

    def test_gqa_llama_style(self):
        """Basic GQA test: LLaMA-style (32:8 head ratio)"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=2, N_CTX=512, H_Q=32, H_KV=8, D=128, DV=96, dtype=dtype, mla=False
            )


class TestExtendAttentionMLA(TestRVVExtendAttention):
    """
    Test RVV extend attention for MLA (Multi-Latent Attention) path - DeepSeek architecture.
    """

    def test_mla_deepseek_style(self):
        """Basic MLA test: DeepSeek-style configuration"""
        for dtype in self.dtypes:
            self._test_extend_attention_once(
                B=2, N_CTX=512, H_Q=22, H_KV=1, D=192, DV=128, dtype=dtype, mla=True
            )


if __name__ == "__main__":
    unittest.main()
