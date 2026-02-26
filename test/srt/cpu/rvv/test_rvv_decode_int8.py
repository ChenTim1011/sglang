"""
This file tests INT8-quantized decode attention for RVV backend.
INT8 quantization reduces memory bandwidth and can improve performance on CPU.

Key differences from FP16/BF16 tests:
- Uses int8 buffers with scale factors (k_scale, v_scale)
- Higher tolerance due to quantization error (atol=1e-1, rtol=1e-1)
- Tests include quantization and dequantization in reference path

Usage:
    python3 test_rvv_decode_int8.py -v
"""

import importlib.util
import unittest

import torch
from torch.nn.functional import scaled_dot_product_attention

try:
    from .utils import int8_decode_precision
except ImportError:
    from test.srt.cpu.rvv.utils import int8_decode_precision


def has_op(op_name: str) -> bool:
    """Check if a specific operator is available in sgl_kernel."""
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, op_name)
    except (AttributeError, RuntimeError):
        return False


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
) -> torch.Tensor:
    """Reference implementation of decode attention using PyTorch SDPA."""
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
            per_req_key = per_req_key.to(torch.float32) * k_scale
        if per_req_value.dtype == torch.int8:
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


class TestDecodeAttentionInt8CPU(unittest.TestCase):
    """
    Test INT8-quantized decode attention for RVV backend.

    """

    @classmethod
    def setUpClass(cls):
        """Setup once before all tests"""
        cls.device = "cpu"
        # Import sgl_kernel to ensure it's loaded
        import sgl_kernel  # noqa: F401

        if not has_op("decode_attention_int8_cpu"):
            raise unittest.SkipTest("decode_attention_int8_cpu not available")

    def _run_decode_int8_test(
        self, num_heads, head_dim, seq_len, num_requests, dtype, num_heads_kv=None
    ):
        """
        Test decode_attention_int8_cpu with various configurations.

        Args:
            num_heads: Number of query heads (H_Q)
            head_dim: Dimension of each head
            seq_len: Sequence length for existing tokens
            num_requests: Number of requests (batch size)
            dtype: Query/output dtype (torch.float16 or torch.bfloat16)
            num_heads_kv: Number of KV heads (H_KV); None defaults to num_heads (MHA).
                          Set num_heads_kv < num_heads to exercise the GQA path.

        This test:
        1. Creates random FP32 data and quantizes to INT8 with scale factors
        2. Runs RVV INT8 kernel
        3. Runs reference implementation (naive_attention_decode with INT8 support)
        4. Compares results with relaxed tolerance (INT8 has quantization error)
        """
        device = self.device
        if num_heads_kv is None:
            num_heads_kv = num_heads
        enable_gqa = num_heads != num_heads_kv
        head_dim_v = head_dim
        max_seq_len = seq_len + 16

        # Quantization parameters
        k_scale = 0.01
        v_scale = 0.01

        # Query in FP16/BF16 (not quantized in decode attention)
        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )

        # Create KV cache: Float32 -> INT8 quantization
        max_tokens = num_requests * max_seq_len

        k_buffer_float = (
            torch.randn(
                max_tokens, num_heads_kv, head_dim, dtype=torch.float32, device=device
            )
            * 5.0
        )  # Scale up for better quantization range

        v_buffer_float = (
            torch.randn(
                max_tokens, num_heads_kv, head_dim_v, dtype=torch.float32, device=device
            )
            * 5.0
        )

        # Quantize: x_int8 = round(x_float / scale).clamp(-128, 127)
        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        # Metadata: request-to-token mapping and sequence lengths
        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.long, device=device
        )

        # Initialize request-to-token mapping (each request has contiguous tokens)
        for i in range(num_requests):
            start_idx = i * max_seq_len
            req_to_token[i, :seq_len] = torch.arange(
                start_idx, start_idx + seq_len, dtype=torch.long
            )

        # New key/value for current decode step (FP32 -> INT8)
        key_float = torch.randn(
            num_requests, num_heads_kv, head_dim, dtype=torch.float32, device=device
        )
        value_float = torch.randn(
            num_requests, num_heads_kv, head_dim_v, dtype=torch.float32, device=device
        )

        key_int8 = (key_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        value_int8 = (value_float / v_scale).round().clamp(-128, 127).to(torch.int8)

        # Locations to write new key/value (last position of each sequence)
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        for i in range(num_requests):
            loc[i] = req_to_token[i, seq_len - 1]

        # Output buffers
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

        # Attention parameters
        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 0.0

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
            k_scale,
            v_scale,
        )

        # Clone buffers and update with new key/value for reference
        k_buffer_ref = k_buffer_int8.clone()
        v_buffer_ref = v_buffer_int8.clone()

        for i in range(num_requests):
            l = loc[i].item()
            k_buffer_ref[l] = key_int8[i]
            v_buffer_ref[l] = value_int8[i]

        # naive_attention_decode handles INT8 dequantization internally
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

        atol = rtol = int8_decode_precision[dtype]
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

    def _run_decode_int8_mla_test(
        self,
        head_dim=192,
        head_dim_v=128,
        seq_len=128,
        num_requests=2,
        dtype=torch.bfloat16,
    ):
        """
        Test decode_attention_int8_cpu specifically for MLA structure.
        In MLA, v_buffer is a memory view inside k_buffer.
        H_KV is typically 1.
        """
        device = self.device
        num_heads = 16  # H_Q
        max_seq_len = seq_len + 16

        k_scale = 0.01
        v_scale = 0.01

        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )

        # MLA kv structure [max_tokens, 1, head_dim]
        max_tokens = num_requests * max_seq_len
        k_buffer_float = (
            torch.randn(max_tokens, 1, head_dim, dtype=torch.float32, device=device)
            * 5.0
        )
        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = k_buffer_int8.narrow(2, 0, head_dim_v)

        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.long, device=device
        )

        for i in range(num_requests):
            start_idx = i * max_seq_len
            req_to_token[i, :seq_len] = torch.arange(
                start_idx, start_idx + seq_len, dtype=torch.long
            )

        key_float = torch.randn(
            num_requests, 1, head_dim, dtype=torch.float32, device=device
        )
        key_int8 = (key_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        value_int8 = key_int8.narrow(2, 0, head_dim_v)

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
        logit_cap = 0.0

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
            k_scale,
            v_scale,
        )

        k_buffer_ref = k_buffer_int8.clone()
        v_buffer_ref = k_buffer_ref.narrow(2, 0, head_dim_v)

        for i in range(num_requests):
            l = loc[i].item()
            k_buffer_ref[l] = key_int8[i]

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
        )

        atol = rtol = int8_decode_precision[dtype]
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)

    def test_decode_int8_gqa(self):
        """INT8 GQA tests: exercises decode_attention_grouped_kernel_impl<scalar_t, int8_t>."""
        configs = [
            # (num_heads, num_heads_kv, head_dim, seq_len, num_requests)
            (32, 8, 64, 256, 4),  # LLaMA-style 4:1 ratio
            (16, 2, 64, 128, 2),  # 8:1 ratio
            (12, 3, 64, 128, 2),  # 4:1 ratio, non-power-of-2 heads
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
                    self._run_decode_int8_test(
                        num_heads=num_heads,
                        head_dim=head_dim,
                        seq_len=seq_len,
                        num_requests=num_requests,
                        dtype=dtype,
                        num_heads_kv=num_heads_kv,
                    )

    def test_decode_int8_small_scale(self):
        """INT8 test with small scale (higher quantization precision)"""
        # Modify _run_decode_int8_test to accept custom scales
        device = self.device
        head_dim = 64
        num_heads = 8
        seq_len = 128
        num_requests = 2

        # Small scale = less quantization error
        k_scale = 0.001
        v_scale = 0.001

        for dtype in [torch.float16, torch.bfloat16]:
            self._run_decode_int8_with_custom_scale(
                num_heads, head_dim, seq_len, num_requests, dtype, k_scale, v_scale
            )

    def test_decode_int8_large_scale(self):
        """INT8 test with large scale (more quantization error)"""
        device = self.device
        head_dim = 128
        num_heads = 16
        seq_len = 256
        num_requests = 4

        # Large scale = more quantization error, tests robustness
        k_scale = 0.1
        v_scale = 0.1

        for dtype in [torch.float16, torch.bfloat16]:
            self._run_decode_int8_with_custom_scale(
                num_heads, head_dim, seq_len, num_requests, dtype, k_scale, v_scale
            )

    def test_decode_int8_odd_dimensions(self):
        """Edge case: INT8 with odd dimensions"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_decode_int8_test(
                num_heads=17, head_dim=127, seq_len=255, num_requests=3, dtype=dtype
            )

    def test_decode_int8_minimal(self):
        """Edge case: INT8 minimal configuration"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_decode_int8_test(
                num_heads=1, head_dim=32, seq_len=1, num_requests=1, dtype=dtype
            )

    def test_decode_int8_mla(self):
        """MLA specific INT8 configuration: v_buffer is inside k_buffer"""
        for dtype in [torch.float16, torch.bfloat16]:
            self._run_decode_int8_mla_test(
                head_dim=192, head_dim_v=128, seq_len=128, num_requests=2, dtype=dtype
            )

    def _run_decode_int8_with_custom_scale(
        self, num_heads, head_dim, seq_len, num_requests, dtype, k_scale, v_scale
    ):
        """
        Helper method to run INT8 test with custom scale factors.
        This is a copy of _run_decode_int8_test with parameterized scales.
        """
        device = self.device
        head_dim_v = head_dim
        max_seq_len = seq_len + 16

        # Query in FP16/BF16 (not quantized in decode attention)
        query = torch.randn(
            num_requests, num_heads, head_dim, dtype=dtype, device=device
        )

        # Create KV cache: Float32 -> INT8 quantization
        max_tokens = num_requests * max_seq_len

        k_buffer_float = (
            torch.randn(
                max_tokens, num_heads, head_dim, dtype=torch.float32, device=device
            )
            * 5.0
        )

        v_buffer_float = (
            torch.randn(
                max_tokens, num_heads, head_dim_v, dtype=torch.float32, device=device
            )
            * 5.0
        )

        # Quantize with custom scales
        k_buffer_int8 = (
            (k_buffer_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        )
        v_buffer_int8 = (
            (v_buffer_float / v_scale).round().clamp(-128, 127).to(torch.int8)
        )

        # Metadata
        req_to_token = torch.zeros(
            num_requests, max_seq_len, dtype=torch.long, device=device
        )
        req_pool_indices = torch.arange(num_requests, dtype=torch.long, device=device)
        seq_lens = torch.tensor(
            [seq_len] * num_requests, dtype=torch.long, device=device
        )

        for i in range(num_requests):
            start_idx = i * max_seq_len
            req_to_token[i, :seq_len] = torch.arange(
                start_idx, start_idx + seq_len, dtype=torch.long
            )

        # New key/value for current decode step
        key_float = torch.randn(
            num_requests, num_heads, head_dim, dtype=torch.float32, device=device
        )
        value_float = torch.randn(
            num_requests, num_heads, head_dim_v, dtype=torch.float32, device=device
        )

        key_int8 = (key_float / k_scale).round().clamp(-128, 127).to(torch.int8)
        value_int8 = (value_float / v_scale).round().clamp(-128, 127).to(torch.int8)

        # Locations to write new key/value
        loc = torch.zeros(num_requests, dtype=torch.int64, device=device)
        for i in range(num_requests):
            loc[i] = req_to_token[i, seq_len - 1]

        # Output buffers
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

        # Attention parameters
        sm_scale = 1.0 / (head_dim**0.5)
        logit_cap = 0.0  # Disable logit cap: reference doesn't implement it

        # Run RVV INT8 Kernel
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
            k_scale,
            v_scale,
        )

        # Reference
        k_buffer_ref = k_buffer_int8.clone()
        v_buffer_ref = v_buffer_int8.clone()

        for i in range(num_requests):
            l = loc[i].item()
            k_buffer_ref[l] = key_int8[i]
            v_buffer_ref[l] = value_int8[i]

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
        )

        atol = rtol = int8_decode_precision[dtype]
        torch.testing.assert_close(output, ref_output, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main(verbosity=2)
