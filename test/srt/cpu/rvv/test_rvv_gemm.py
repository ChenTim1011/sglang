"""Unit Tests for RVV GEMM kernel.

Tests BF16 and INT8 GEMM operations on RISC-V Vector architecture.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_gemm -v
"""

import itertools
import unittest

import sgl_kernel  # noqa: F401
import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import (
    gemm_precision,
    native_w8a8_per_token_matmul,
    precision,
    sigmoid_mul_precision,
)

torch.manual_seed(1234)


class TestRVVGemm(CustomTestCase):
    """Test suite for RVV GEMM kernels."""

    M = [1, 2, 3, 4, 5, 101]
    N = [16, 32, 48, 64, 80]
    K = [100, 32 * 16]
    has_bias = [False, True]
    dtypes = [torch.float16, torch.bfloat16]

    # Specific tests for FMA path (N size > 64)
    N_fma = [100, 300]

    M_int8 = [2, 128]
    N_int8 = [32 * 12]
    K_int8 = [32 * 17]

    def run_case_gemm_bf16(self, M, N, K, has_bias, dtype):
        """Run one BF16 GEMM case and compare packed and unpacked paths."""
        mat1 = torch.randn(M, K, dtype=dtype)
        mat2 = torch.randn(N, K, dtype=dtype)

        ref = torch.matmul(mat1.float(), mat2.float().t())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            if dtype == torch.bfloat16:
                ref.add_(bias.bfloat16())
            else:
                ref.add_(bias.half())

        if dtype == torch.bfloat16:
            ref = ref.bfloat16()
        else:
            ref = ref.half()

        out = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, mat2, bias if has_bias else None, False
        )

        packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
        out2 = torch.ops.sgl_kernel.weight_packed_linear(
            mat1, packed_mat2, bias if has_bias else None, True
        )

        atol = rtol = gemm_precision[dtype]

        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out2, atol=atol, rtol=rtol)

    def test_case_gemm_bf16_matrix(self):
        """Case: BF16 and FP16 GEMM across shape and bias matrix."""
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.has_bias,
            self.dtypes,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                dtype=params[4],
            ):
                self.run_case_gemm_bf16(*params)

    def test_case_gemm_bf16_fma_path(self):
        """Case: GEMM FMA boundary where N exceeds one vector-length chunk."""
        for params in itertools.product(
            self.M,
            self.N_fma,
            self.K,
            self.has_bias,
            self.dtypes,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                dtype=params[4],
            ):
                self.run_case_gemm_bf16(*params)

    def run_case_gemm_int8(self, M, N, K, has_bias, dtype):
        """Run one INT8 GEMM case against the native reference implementation."""
        A = torch.randn((M, K), dtype=dtype) / 10

        # The Python reference uses round() while C++ RVV uses hardware vfcvt_x_f
        # (round-to-nearest-even), which can differ by ±1 for edge values.
        Aq, As = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = (torch.rand(N) * factor_for_scale).to(torch.float32)  # Ensure float32

        bias = torch.randn(N) if has_bias else None

        # Reference: uses RVV-aligned calculation with unpacked weights
        ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

        atol = rtol = precision[ref_out.dtype]

        # For RVV, both int8_scaled_mm_cpu and int8_scaled_mm_with_quant require packed weights
        Bq_packed = torch.ops.sgl_kernel.convert_weight_packed(Bq)

        # Test int8_scaled_mm_cpu with packed weights
        out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            Aq,
            Bq_packed,
            As,
            Bs,
            bias if has_bias else None,
            dtype,
            True,  # is_vnni=True for packed
        )

        # Test the fused version (quantizes A on-the-fly, uses packed weights)
        fused_out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A,
            Bq_packed,
            Bs,
            bias if has_bias else None,
            dtype,
            True,  # is_vnni=True for packed
        )

        # Validate mathematical correctness against the reference implementation
        torch.testing.assert_close(ref_out, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_out, fused_out, atol=atol, rtol=rtol)

    def test_case_gemm_int8_matrix(self):
        """Case: INT8 GEMM across shape, bias, and dtype matrix."""
        for params in itertools.product(
            self.M_int8,
            self.N_int8,
            self.K_int8,
            self.has_bias,
            self.dtypes,
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                dtype=params[4],
            ):
                self.run_case_gemm_int8(*params)

    def run_case_gemm_bf16_small_oc(self, M, N, K, has_bias, use_post_sigmul, dtype):
        """Run one small-output-channel GEMM case with optional sigmoid fusion."""
        use_post_sigmul = use_post_sigmul and N == 1
        mat1 = torch.randn(M, K, dtype=dtype)
        mat2 = torch.randn(N, K, dtype=dtype)

        ref = torch.nn.functional.linear(mat1.float(), mat2.float())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias)

        if use_post_sigmul:
            # fused_linear_sigmoid_mul requires post_mul_mat.size(1) to be divisible by 32
            # The output shape is [M, N] = [M, 1], so we need post_mul_mat to be [M, post_mul_size]
            # where post_mul_size is divisible by 32
            # For this test, we use a simple size that's divisible by 32
            post_mul_size = 32  # Use a fixed size that's divisible by 32
            mat_mul = torch.randn(M, post_mul_size, dtype=dtype)

            # pass the pre-packed weight and set is_vnni=True
            out = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,  # is_vnni (for RISC-V, this means packed)
                mat_mul,
            )
            # Output shape is [M, post_mul_size].
            # sigmoid(ref) broadcasts [M, 1] × mat_mul [M, post_mul_size] → [M, post_mul_size].
            # Compare ALL columns.
            ref = (torch.nn.functional.sigmoid(ref) * mat_mul.float()).to(dtype)
            atol = rtol = sigmoid_mul_precision[dtype]
        else:
            ref = ref.to(dtype)

            out = torch.ops.sgl_kernel.weight_packed_linear(
                mat1,
                torch.ops.sgl_kernel.convert_weight_packed(mat2),
                bias if has_bias else None,
                True,
            )
            atol = rtol = gemm_precision[dtype]

        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_case_gemm_bf16_small_oc(self):
        """Case: small output-channel GEMM with optional sigmoid fusion."""
        for params in itertools.product(
            [1, 8, 32, 1024], [12, 1], self.K, self.has_bias, [False, True], self.dtypes
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                use_post_sigmul=params[4],
                dtype=params[5],
            ):
                self.run_case_gemm_bf16_small_oc(*params)


if __name__ == "__main__":
    unittest.main()
