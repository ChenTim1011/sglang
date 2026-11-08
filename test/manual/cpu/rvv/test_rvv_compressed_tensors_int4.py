"""Unit tests for the RVV compressed-tensors INT4 policy."""

import unittest
from unittest.mock import patch

import torch

import sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_rvv as rvv_wna16
from sglang.srt.layers.quantization.utils import pack_cols
from sglang.test.test_utils import CustomTestCase

from .test_rvv_gemm import pack_signed_int4_rows


class TestRVVCompressedTensorsInt4(CustomTestCase):
    def test_uint4b8_pack_converts_to_signed_int4_bytes(self):
        rows, cols = 3, 16
        signed_q = torch.tensor(
            [
                [-8, -7, -1, 0, 1, 7, -2, 3, 4, -5, 6, -6, 2, -3, 5, -4],
                [7, 6, 5, 4, 3, 2, 1, 0, -1, -2, -3, -4, -5, -6, -7, -8],
                [0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6, -7, 7, -8],
            ],
            dtype=torch.int32,
        )
        uint4b8 = (signed_q + 8) & 0xF
        compressed_packed = pack_cols(uint4b8, num_bits=4, size_k=rows, size_n=cols)

        rvv_packed = rvv_wna16.compressed_uint4b8_to_rvv_signed_int4_bytes(
            compressed_packed, input_size_per_partition=cols
        )

        expected = pack_signed_int4_rows(signed_q.to(torch.int8))
        torch.testing.assert_close(rvv_packed, expected)

    def test_uint4b8_pack_applies_activation_order_permutation(self):
        rows, cols = 2, 16
        signed_q = torch.arange(rows * cols, dtype=torch.int32).reshape(rows, cols)
        signed_q = ((signed_q % 16) - 8).contiguous()
        uint4b8 = (signed_q + 8) & 0xF
        compressed_packed = pack_cols(uint4b8, num_bits=4, size_k=rows, size_n=cols)

        g_idx = torch.tensor([1, 0, 3, 2, 1, 0, 3, 2, 0, 1, 2, 3, 0, 1, 2, 3])
        input_permutation = torch.argsort(g_idx)
        rvv_packed = rvv_wna16.compressed_uint4b8_to_rvv_signed_int4_bytes(
            compressed_packed,
            input_size_per_partition=cols,
            input_permutation=input_permutation,
        )

        expected = pack_signed_int4_rows(
            signed_q[:, input_permutation].contiguous().to(torch.int8)
        )
        torch.testing.assert_close(rvv_packed, expected)

    def test_w4a8_dynamic_linear_is_the_rvv_int4_policy(self):
        class FakeScheme:
            kernel_config = None
            rvv_group_size = 128

        layer = torch.nn.Module()
        layer.use_riscv_rvv_int4_w4a8_dynamic_linear_backend = True
        layer.rvv_g_idx_sort_indices = None
        layer._rvv_int4_w_q = torch.empty(1, 64, dtype=torch.uint8)
        layer._rvv_int4_w_s = torch.empty(1, 1, dtype=torch.float32)
        layer._rvv_int4_w4a8_dynamic_w_q = torch.empty(1, 1, 65, dtype=torch.uint8)
        layer._rvv_int4_w4a8_dynamic_w_s = torch.empty(1, 1, 64, dtype=torch.float32)
        x_m1 = torch.randn(1, 128, dtype=torch.bfloat16)
        x_m2 = torch.randn(2, 128, dtype=torch.bfloat16)
        w4a8_dynamic_out_m1 = torch.randn(1, 64, dtype=torch.bfloat16)
        w4a8_dynamic_out_m2 = torch.randn(2, 64, dtype=torch.bfloat16)

        class FakeOps:
            def __init__(self):
                self.w4a8_dynamic_calls = 0

            def weight_w4a8_dynamic_linear(self, *args):
                self.w4a8_dynamic_calls += 1
                return (
                    w4a8_dynamic_out_m1 if args[0].size(0) == 1 else w4a8_dynamic_out_m2
                )

        fake_ops = FakeOps()
        with patch.object(torch.ops, "sgl_kernel", fake_ops):
            out_m1 = rvv_wna16.apply_weights_rvv_wna16(FakeScheme, layer, x_m1, None)
            out_m2 = rvv_wna16.apply_weights_rvv_wna16(FakeScheme, layer, x_m2, None)

        self.assertIs(out_m1, w4a8_dynamic_out_m1)
        self.assertIs(out_m2, w4a8_dynamic_out_m2)
        self.assertEqual(fake_ops.w4a8_dynamic_calls, 2)


if __name__ == "__main__":
    unittest.main()
