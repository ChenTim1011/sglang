"""
Test for decode_attention_cpu kernel (RVV optimized, FP16/BF16).

This file tests all three attention paths implemented in rvv/decode.cpp:
1. MHA (Multi-Head Attention): num_heads == num_heads_kv
2. GQA/MQA (Grouped/Multi-Query Attention): num_heads != num_heads_kv
3. MLA (Multi-Latent Attention): shared KV buffer, num_heads_kv=1, head_size=head_size_v+64

Usage:
    python3 test_rvv_decode.py -v                         # Run all tests
"""

import importlib.util
import unittest

import torch

from .utils import precision, run_sdpa_forward_decode

try:
    import sgl_kernel  # noqa: F401
except ImportError:
    pass

torch.manual_seed(1234)


def _has_decode_attention_cpu() -> bool:
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "decode_attention_cpu")
    except (AttributeError, RuntimeError):
        return False


@unittest.skipUnless(
    _has_decode_attention_cpu(), "decode_attention_cpu not available (non-RISC-V build)"
)
class RVVDecodeTestBase(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _test_grouped_decode_attention_once(
        self, B, H_Q, H_KV, D, D_V, seq_len=1024, dtype=torch.float16, is_mla=False
    ):
        device = self.device
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        logit_cap = 0.0
        num_kv_splits = 8
        enable_gqa = H_Q != H_KV

        # q represents the new token being generated, one per batch
        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)

        H_BUF = 1 if is_mla else H_KV
        # k_buffer and v_buffer represent all previous tokens
        k_buffer = torch.randn(total_tokens, H_BUF, D, dtype=dtype, device=device)
        if is_mla:
            v_buffer = k_buffer.narrow(2, 0, D_V)
        else:
            v_buffer = torch.randn(total_tokens, H_BUF, D_V, dtype=dtype, device=device)

        key = torch.randn(B, H_KV, D, dtype=dtype)
        if is_mla:
            value = key.narrow(2, 0, D_V)
            loc = torch.randperm(total_tokens)[:B].to(torch.int64)
        else:
            value = torch.randn(B, H_KV, D_V, dtype=dtype)
            loc = torch.randint(0, total_tokens, (B,)).to(torch.int64)

        if not is_mla:
            # For non-MLA, immediately set the kv cache for the new token
            k_buffer[loc] = key
            v_buffer[loc] = value

        k_buffer2 = k_buffer.clone() if is_mla else k_buffer
        if is_mla:
            v_buffer2 = k_buffer2.narrow(2, 0, D_V)
        else:
            v_buffer2 = v_buffer

        # o will have the same shape as q
        o = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_grouped = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        req_to_token = (
            torch.arange(total_tokens, device=device)
            .reshape(B, seq_len)
            .to(torch.int32)
        )
        b_req_idx = torch.arange(B, device=device).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1),
            dtype=torch.float32,
            device=device,
        )

        # Ensure buffers support non-contiguous tensor layouts
        k_buffer_k = k_buffer2.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer_k = v_buffer2.transpose(0, 1).contiguous().transpose(0, 1)
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
            key=key if is_mla else None,
            loc=loc if is_mla else None,
        )

        cos_sim = torch.nn.functional.cosine_similarity(
            o.flatten(), o_grouped.flatten(), dim=0
        )
        # Use precision tolerances mapped by dtype
        atol = rtol = precision[q.dtype]
        self.assertGreater(cos_sim.item(), 0.99)
        torch.testing.assert_close(o, o_grouped, atol=atol, rtol=rtol)

        if is_mla:
            torch.testing.assert_close(k_buffer, k_buffer2, atol=atol, rtol=rtol)
            torch.testing.assert_close(v_buffer, v_buffer2, atol=atol, rtol=rtol)


class TestDecodeAttentionMHA(RVVDecodeTestBase):
    """
    Test RVV decode attention for MHA (Multi-Head Attention) path.
    """

    def test_mha_configs(self):
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (1, 1, 1, 64, 64, 128),  # Minimal
            (2, 8, 8, 128, 128, 256),  # Typical small
            (4, 16, 16, 64, 64, 512),  # Med batch, med seq
            (2, 17, 17, 127, 127, 128),  # Odd dimensions handling
            (8, 8, 8, 128, 128, 128),  # Larger batch
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self._test_grouped_decode_attention_once(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=False
                    )


class TestDecodeAttentionGQA(RVVDecodeTestBase):
    """
    Test RVV decode attention for GQA/MQA (Grouped Query Attention) path.
    GQA: num_heads != num_heads_kv (e.g., LLaMA with H_Q=32, H_KV=8)
    MQA: num_heads_kv=1
    """

    def test_gqa_mqa_configs(self):
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (2, 32, 8, 128, 128, 256),  # LLaMA-style GQA (4:1)
            (4, 16, 1, 128, 128, 128),  # MQA style (16:1)
            (2, 18, 3, 128, 128, 128),  # 6x ratio, non-power-of-2
            (2, 21, 7, 128, 128, 128),  # 3x ratio, odd head counts
            (1, 12, 3, 64, 64, 512),  # 4x ratio, small odd KV heads
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self._test_grouped_decode_attention_once(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=False
                    )


class TestDecodeAttentionMLA(RVVDecodeTestBase):
    """
    Test RVV decode attention for MLA (Multi-Latent Attention) path.
    """

    def test_mla_configs(self):
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (2, 16, 1, 192, 128, 256),  # DeepSeek-style typical config
            (4, 8, 1, 192, 128, 512),  # Larger batch/seq
            (1, 17, 1, 192, 128, 128),  # Odd number of heads
            (2, 16, 1, 191, 127, 128),  # Odd head dimensions
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self._test_grouped_decode_attention_once(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=True
                    )


if __name__ == "__main__":
    unittest.main()
