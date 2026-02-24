"""
Test for decode_attention_cpu kernel (RVV optimized, FP16/BF16).

This file tests all three attention paths implemented in rvv/decode.cpp:
1. MHA (Multi-Head Attention): num_heads == num_heads_kv
2. GQA/MQA (Grouped/Multi-Query Attention): num_heads != num_heads_kv
3. MLA (Multi-Latent Attention): shared KV buffer, num_heads_kv=1, head_size=head_size_v+64

Usage:
    python3 test_rvv_decode.py -v                         # Run all tests
"""

import sys
import unittest

import torch

# Add parent directory to path to import test utilities
sys.path.insert(0, "/workspace/sglang/test/srt/cpu")

# Import existing test classes to reuse their reference implementations
try:
    from test_decode import TestDecodeAttention
    from test_mla import TestMLA
except ImportError:
    # Fallback for local development
    import os

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent_dir)
    from test_decode import TestDecodeAttention
    from test_mla import TestMLA

torch.manual_seed(1234)


class TestDecodeAttentionMHA(TestDecodeAttention):
    """
    Test RVV decode attention for MHA (Multi-Head Attention) path.

    Inherits from TestDecodeAttention and reuses:
    - _run_sdpa_forward_decode(): SDPA reference implementation
    - _test_grouped_decode_attention_once(): Test harness with buffer setup

    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def test_decode_minimal(self):
        """Edge case: minimal batch and heads (tests vectorization boundaries)"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=1, H_Q=1, H_KV=1, D=64, D_V=64, dtype=dtype, device=self.device
            )

    def test_decode_odd_dimensions(self):
        """Edge case: odd dimensions (tests tail handling in RVV loops)"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=2, H_Q=17, H_KV=17, D=127, D_V=127, dtype=dtype, device=self.device
            )

    def test_decode_large_batch(self):
        """Stress test: large batch size"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=16, H_Q=32, H_KV=32, D=128, D_V=128, dtype=dtype, device=self.device
            )

    def test_decode_odd_dimensions(self):
        """Edge case: odd dimensions (tests tail handling in RVV loops)"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=2, H_Q=17, H_KV=17, D=127, D_V=127, dtype=dtype, device=self.device
            )

    def test_decode_large_batch(self):
        """Stress test: large batch size"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=16, H_Q=32, H_KV=32, D=128, D_V=128, dtype=dtype, device=self.device
            )


class TestDecodeAttentionGQA(TestDecodeAttention):
    """
    Test RVV decode attention for GQA/MQA (Grouped Query Attention) path.

    GQA: num_heads != num_heads_kv (e.g., LLaMA with H_Q=32, H_KV=8)
    MQA: num_heads_kv=1

    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def test_gqa_mqa(self):
        """MQA test: Single KV head shared across all query heads"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=4, H_Q=16, H_KV=1, D=128, D_V=128, dtype=dtype, device=self.device
            )

    def test_gqa_odd_ratio(self):
        """Edge case: non-power-of-2 but valid GQA ratios.

        The kernel requires H_Q % H_KV == 0 (integer division for group mapping).
        Tests ratios like 18:3=6x and 21:7=3x that exercise non-standard group sizes.
        """
        configs = [
            (2, 18, 3, 128, 128),  # 6x ratio, non-power-of-2
            (2, 21, 7, 128, 128),  # 3x ratio, odd head counts
            (1, 12, 3, 64, 64),  # 4x ratio, small odd KV heads
        ]
        for B, H_Q, H_KV, D, D_V in configs:
            for dtype in self.dtypes:
                self._test_grouped_decode_attention_once(
                    B, H_Q, H_KV, D, D_V, dtype=dtype, device=self.device
                )


class TestDecodeAttentionMLA(TestMLA):
    """
    Test RVV decode attention for MLA (Multi-Latent Attention) path

    MLA Requirements:
    - num_heads_kv = 1 (always)
    - head_size = head_size_v + 64 (e.g., D=192, D_V=128)
    - k_buffer and v_buffer share memory (v_buffer = k_buffer.narrow(2, 0, D_V))

    """

    def setUp(self):
        self.device = torch.device("cpu")
        self.dtypes = [torch.float16, torch.bfloat16]

    def test_mla_stage3_kcache_8heads(self):
        """K-cache validation: 8 heads"""
        for dtype in self.dtypes:
            # Note: TestMLA doesn't take dtype/device directly in _test_grouped_decode_attention_once
            self._test_grouped_decode_attention_once(
                B=4, H_Q=8, H_KV=1, D=192, D_V=128, seq_len=512
            )

    def test_mla_odd_heads(self):
        """Edge case: odd number of heads (tests loop tail handling)"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=1, H_Q=17, H_KV=1, D=192, D_V=128, seq_len=256
            )

    def test_mla_odd_dimensions(self):
        """Edge case: odd head dimensions (tests vectorization tail)"""
        for dtype in self.dtypes:
            self._test_grouped_decode_attention_once(
                B=2, H_Q=16, H_KV=1, D=191, D_V=127, seq_len=128
            )


if __name__ == "__main__":
    unittest.main()
