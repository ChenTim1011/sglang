"""Unit tests for RVV decode attention kernel (FP16/BF16).

1. MHA (Multi-Head Attention): num_heads == num_heads_kv
2. GQA/MQA (Grouped/Multi-Query Attention): num_heads != num_heads_kv
3. MLA (Multi-Latent Attention): shared KV buffer, num_heads_kv=1, head_size=head_size_v+64

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_decode -v
"""

import importlib.util
import unittest

import torch

from .rvv_utils import precision, run_sdpa_forward_decode

try:
    import sgl_kernel  # noqa: F401
except ImportError:
    pass

torch.manual_seed(1234)


def has_sgl_kernel_decode_attention() -> bool:
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        torch.ops.sgl_kernel.decode_attention_cpu
        return True
    except (AttributeError, RuntimeError):
        return False


@unittest.skipUnless(
    has_sgl_kernel_decode_attention(),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeBase(unittest.TestCase):
    """Shared fixtures and helpers for RVV decode attention tests."""

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def run_case_decode_attention_grouped(
        self, B, H_Q, H_KV, D, D_V, seq_len=1024, is_mla=False, dtype=torch.float16
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

        # Simulate true Paged Attention by fragmenting the memory pool indices
        random_indices = torch.randperm(total_tokens, device=device)
        req_to_token = random_indices.reshape(B, seq_len).to(torch.int32)

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
        prec = precision[q.dtype]
        self.assertGreater(cos_sim.item(), 0.99)
        torch.testing.assert_close(o, o_grouped, atol=prec, rtol=prec)

        if is_mla:
            torch.testing.assert_close(k_buffer, k_buffer2, atol=prec, rtol=prec)
            torch.testing.assert_close(v_buffer, v_buffer2, atol=prec, rtol=prec)


class TestRVVDecodeMHA(TestRVVDecodeBase):
    """Test suite for RVV decode attention MHA path."""

    def test_case_mha_cases(self):
        """Case: MHA decode across representative shapes and dtypes."""
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (1, 1, 1, 64, 64, 128),
            (2, 8, 8, 128, 128, 63),
            (2, 8, 8, 128, 128, 65),
            (2, 8, 8, 128, 128, 129),
            (2, 8, 8, 128, 128, 256),
            (4, 16, 16, 64, 64, 512),
            (2, 17, 17, 127, 127, 128),
            (8, 8, 8, 128, 128, 128),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self.run_case_decode_attention_grouped(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=False
                    )


class TestRVVDecodeGQA(TestRVVDecodeBase):
    """Test suite for RVV decode attention GQA and MQA paths."""

    def test_case_gqa_mqa_cases(self):
        """Case: GQA and MQA decode across representative shapes and dtypes."""
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (2, 32, 8, 128, 128, 63),
            (2, 32, 8, 128, 128, 129),
            (2, 32, 8, 128, 128, 256),
            (4, 16, 1, 128, 128, 128),
            (2, 18, 3, 128, 128, 128),
            (2, 21, 7, 128, 128, 128),
            (1, 12, 3, 64, 64, 512),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self.run_case_decode_attention_grouped(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=False
                    )


class TestRVVDecodeMLA(TestRVVDecodeBase):
    """Test suite for RVV decode attention MLA path."""

    def test_case_mla_cases(self):
        """Case: MLA decode across representative shapes and dtypes."""
        # Config format: (B, H_Q, H_KV, D, D_V, seq_len)
        configs = [
            (2, 16, 1, 192, 128, 63),
            (2, 16, 1, 192, 128, 129),
            (2, 16, 1, 192, 128, 256),
            (4, 8, 1, 192, 128, 512),
            (1, 17, 1, 192, 128, 128),
            (2, 16, 1, 191, 127, 128),
            (1, 22, 1, 576, 512, 128),
            (1, 22, 1, 576, 512, 888),
        ]
        for B, H_Q, H_KV, D, D_V, seq_len in configs:
            for dtype in self.dtypes:
                with self.subTest(
                    B=B, H_Q=H_Q, H_KV=H_KV, D=D, D_V=D_V, seq_len=seq_len, dtype=dtype
                ):
                    self.run_case_decode_attention_grouped(
                        B, H_Q, H_KV, D, D_V, seq_len=seq_len, dtype=dtype, is_mla=True
                    )


if __name__ == "__main__":
    unittest.main()
