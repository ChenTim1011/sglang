"""Correctness tests for the RVV decode attention kernel.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_attention_backend -v
"""

import unittest

import torch

from .rvv_utils import has_sgl_kernel_op, precision, run_sdpa_forward_decode

try:
    import sgl_kernel  # noqa: F401
except ImportError:
    pass

torch.manual_seed(42)


@unittest.skipUnless(
    has_sgl_kernel_op("decode_attention_cpu"),
    "decode_attention_cpu not available (non-RISC-V build)",
)
class TestRVVDecodeKernelSanity(unittest.TestCase):
    """Correctness tests for the RVV decode_attention_cpu kernel.

    Verifies that the C++ kernel output matches PyTorch SDPA reference.
    This complements TestRVVAttentionBackendBenchmark which only checks
    throughput (min_throughput=0.0 gives zero correctness protection).
    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def _run_decode_case(self, B, H_Q, H_KV, D, D_V, seq_len, dtype):
        device = self.device
        total_tokens = B * seq_len
        sm_scale = 1.0 / (D**0.5)
        num_kv_splits = 8

        q = torch.randn(B, H_Q, D, dtype=dtype, device=device)
        k_buffer = torch.randn(total_tokens, H_KV, D, dtype=dtype, device=device)
        v_buffer = torch.randn(total_tokens, H_KV, D_V, dtype=dtype, device=device)

        key = torch.randn(B, H_KV, D, dtype=dtype)
        value = torch.randn(B, H_KV, D_V, dtype=dtype)
        loc = torch.randint(0, total_tokens, (B,)).to(torch.int64)
        k_buffer[loc] = key
        v_buffer[loc] = value

        o_rvv = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)
        o_ref = torch.zeros(B, H_Q, D_V, dtype=dtype, device=device)

        random_indices = torch.randperm(total_tokens, device=device)
        req_to_token = random_indices.reshape(B, seq_len).to(torch.int32)
        b_req_idx = torch.arange(B, device=device).to(torch.int64)
        b_seq_len = torch.full((B,), seq_len, device=device).to(torch.int64)

        attn_logits = torch.empty(
            (B, H_Q, num_kv_splits, D_V + 1), dtype=torch.float32, device=device
        )

        k_buffer_k = k_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        v_buffer_k = v_buffer.transpose(0, 1).contiguous().transpose(0, 1)
        q_k = q.transpose(0, 1).contiguous().transpose(0, 1)
        key_k = key.transpose(0, 1).contiguous().transpose(0, 1)
        value_k = value.transpose(0, 1).contiguous().transpose(0, 1)

        torch.ops.sgl_kernel.decode_attention_cpu(
            q_k,
            k_buffer_k,
            v_buffer_k,
            o_rvv,
            key_k,
            value_k,
            loc,
            attn_logits,
            req_to_token,
            b_req_idx,
            b_seq_len,
            sm_scale,
            0.0,
        )

        run_sdpa_forward_decode(
            q,
            o_ref,
            k_buffer,
            v_buffer,
            req_to_token,
            b_req_idx,
            b_seq_len,
            scaling=sm_scale,
            enable_gqa=(H_Q != H_KV),
        )

        prec = precision["default"][dtype]
        torch.testing.assert_close(o_rvv, o_ref, atol=prec, rtol=prec)

    def test_decode_mha_sanity(self):
        """MHA decode: kernel output matches PyTorch SDPA reference."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._run_decode_case(
                    B=2, H_Q=8, H_KV=8, D=64, D_V=64, seq_len=64, dtype=dtype
                )

    def test_decode_gqa_sanity(self):
        """GQA decode: kernel output matches PyTorch SDPA reference."""
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                self._run_decode_case(
                    B=2, H_Q=16, H_KV=4, D=64, D_V=64, seq_len=64, dtype=dtype
                )


if __name__ == "__main__":
    unittest.main()
