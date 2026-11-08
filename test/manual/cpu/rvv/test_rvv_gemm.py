"""Unit tests for RVV GEMM kernels."""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, helper_non_contiguous, precision

torch.manual_seed(1234)


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
    """RVV CPU W8A8 reference: activation is uint8, weight is int8."""
    A = A.to(torch.int32) - 128
    B = B.to(torch.int32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    M = A.numel() // A.shape[-1]
    K = A.shape[-1]
    origin_C_shape = A.shape[:-1] + (B.shape[0],)
    A = A.reshape(M, K)

    C = torch.matmul(A, B.transpose(0, 1))

    As = As.reshape(M, 1).to(torch.float32)
    C = As * C.to(torch.float32) * Bs.view(1, -1).to(torch.float32)

    if bias is not None:
        C.add_(bias.view(1, -1).to(torch.float32))

    return C.reshape(origin_C_shape).to(output_dtype)


def _has_rvv_bf16_gemm_ops() -> bool:
    return has_sgl_kernel_op("weight_packed_linear") and has_sgl_kernel_op(
        "convert_weight_packed"
    )


def _has_rvv_int8_gemm_ops() -> bool:
    return (
        has_sgl_kernel_op("convert_weight_packed")
        and has_sgl_kernel_op("per_token_quant_int8_cpu")
        and has_sgl_kernel_op("int8_scaled_mm_cpu")
        and has_sgl_kernel_op("int8_scaled_mm_with_quant")
    )


def _has_rvv_w4a8_dynamic_linear_ops() -> bool:
    return has_sgl_kernel_op("weight_w4a8_dynamic_linear") and has_sgl_kernel_op(
        "convert_weight_w4a8_dynamic_packed"
    )


def pack_signed_int4_rows(weight_int4: torch.Tensor) -> torch.Tensor:
    """Pack [N, K] signed int4 weights into [N, K/2] uint8 with low/high nibbles."""
    assert weight_int4.dim() == 2
    assert weight_int4.shape[1] % 2 == 0
    assert weight_int4.dtype == torch.int8
    lo = torch.bitwise_and(weight_int4[:, 0::2].to(torch.int16), 0x0F).to(torch.uint8)
    hi = torch.bitwise_and(weight_int4[:, 1::2].to(torch.int16), 0x0F).to(torch.uint8)
    return lo | (hi << 4)


def dequantize_grouped_int4_weight(
    packed_weight: torch.Tensor, scales: torch.Tensor, group_size: int
) -> torch.Tensor:
    """Dequantize packed signed int4 weight to float32 [N, K]."""
    N, packed_k = packed_weight.shape
    K = packed_k * 2
    unpacked = torch.empty((N, K), dtype=torch.float32)
    for n in range(N):
        for k in range(K):
            packed = int(packed_weight[n, k // 2].item())
            nibble = (packed >> 4) & 0x0F if (k & 1) else packed & 0x0F
            q = nibble if nibble < 8 else nibble - 16
            unpacked[n, k] = q * float(scales[n, k // group_size].item())
    return unpacked


def native_w4a8_dynamic_linear(
    x: torch.Tensor,
    weight_q: torch.Tensor,
    scales: torch.Tensor,
    bias: torch.Tensor | None,
    group_size: int,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Reference for RVV W4A8 dynamic linear/prefill probe."""
    assert weight_q.dtype == torch.int8
    assert weight_q.dim() == 2
    N, K = weight_q.shape
    assert K % group_size == 0

    rows = []
    for m in range(x.shape[0]):
        x_f = x[m].reshape(K).float()
        out = torch.zeros(N, dtype=torch.float32)
        for g in range(K // group_size):
            k_begin = g * group_size
            k_end = k_begin + group_size
            x_group = x_f[k_begin:k_end]
            scale = max(float(x_group.abs().max().item()) / 127.0, 1.0e-8)
            x_q = torch.clamp(torch.round(x_group / scale), -127, 127).to(torch.int32)
            w_q = weight_q[:, k_begin:k_end].to(torch.int32)
            dot = torch.matmul(w_q, x_q)
            out += dot.float() * (scale * scales[:, g].float())
        if bias is not None:
            out += bias.float()
        rows.append(out)
    return torch.stack(rows, dim=0).to(output_dtype)


class TestRVVGemm(CustomTestCase):
    M = [1, 2, 3, 4, 5, 101]
    N = [16, 32, 48, 64, 80]
    K = [100, 32 * 16]
    has_bias = [False, True]
    dtypes = [torch.float16, torch.bfloat16]

    # Cover the FMA path once N exceeds a single vector-length chunk.
    N_fma = [100, 300]

    M_int8 = [1, 2, 3, 4, 5, 8, 128]  # Exercises the TILE_M=4 short-row path.
    N_int8 = [16, 32 * 12, 80]  # Covers small-N, large-N, and tail handling.
    K_int8 = [32 * 17]
    N_w4a8_dynamic = [64, 81, 896]
    K_w4a8_dynamic = [128, 1536, 2048]

    def _run_bf16(self, M, N, K, has_bias, dtype):
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

        atol = rtol = precision["linear_gemm"][dtype]

        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out2, atol=atol, rtol=rtol)

    @unittest.skipUnless(_has_rvv_bf16_gemm_ops(), "RVV BF16 GEMM ops not available")
    def test_bf16(self):
        """Shape + bias matrix covers M-tail, N-chunk, and K-stride combinations."""
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
                self._run_bf16(*params)

    @unittest.skipUnless(_has_rvv_bf16_gemm_ops(), "RVV BF16 GEMM ops not available")
    def test_bf16_fma_path(self):
        """Large N forces the multi-chunk FMA loop; exercises accumulation across chunks."""
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
                self._run_bf16(*params)

    def _run_int8(self, M, N, K, has_bias, dtype):
        A = torch.randn((M, K), dtype=dtype) / 10

        Aq, As = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
        self.assertEqual(Aq.dtype, torch.uint8)
        self.assertEqual(As.dtype, torch.float32)
        self.assertEqual(tuple(As.shape), (M,))

        factor_for_scale = 1e-2
        int8_max = 127
        int8_min = -128

        B = (torch.rand((N, K), dtype=torch.float32) - 0.5) * 2
        Bq = (B * int8_max).clamp(min=int8_min, max=int8_max).to(torch.int8)
        Bs = (torch.rand(N) * factor_for_scale).to(torch.float32)

        bias = torch.randn(N) if has_bias else None

        # Match the unpacked RVV math path first, then compare packed variants.
        ref_out = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, bias, dtype)

        atol = rtol = precision["pointwise_default"][ref_out.dtype]

        Bq_packed = torch.ops.sgl_kernel.convert_weight_packed(Bq)

        out_unpacked = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            Aq,
            Bq,
            As,
            Bs,
            bias if has_bias else None,
            dtype,
            False,
        )

        out_packed = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            Aq,
            Bq_packed,
            As,
            Bs,
            bias if has_bias else None,
            dtype,
            True,
        )

        fused_out_unpacked = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A,
            Bq,
            Bs,
            bias if has_bias else None,
            dtype,
            False,
        )

        fused_out_packed = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            A,
            Bq_packed,
            Bs,
            bias if has_bias else None,
            dtype,
            True,
        )

        torch.testing.assert_close(ref_out, out_unpacked, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_out, out_packed, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_out, fused_out_unpacked, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_out, fused_out_packed, atol=atol, rtol=rtol)

    @unittest.skipUnless(_has_rvv_int8_gemm_ops(), "RVV INT8 GEMM ops not available")
    def test_int8(self):
        """TILE_M=4 short-row path and scale dequantization across the full shape matrix."""
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
                self._run_int8(*params)

    def _run_bf16_small_oc(self, M, N, K, has_bias, dtype):
        mat1 = torch.randn(M, K, dtype=dtype)
        mat2 = torch.randn(N, K, dtype=dtype)

        ref = torch.nn.functional.linear(mat1.float(), mat2.float())
        if has_bias:
            bias = torch.randn(N, dtype=torch.float32)
            ref.add_(bias)

        ref = ref.to(dtype)
        out = torch.ops.sgl_kernel.weight_packed_linear(
            mat1,
            torch.ops.sgl_kernel.convert_weight_packed(mat2),
            bias if has_bias else None,
            True,
        )
        atol = rtol = precision["linear_gemm"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @unittest.skipUnless(_has_rvv_bf16_gemm_ops(), "RVV BF16 GEMM ops not available")
    def test_bf16_small_oc(self):
        """N=1,12 exercises the scalar/sub-chunk output path that bypasses the main VL loop."""
        for params in itertools.product(
            [1, 8, 32, 1024], [12, 1], self.K, self.has_bias, self.dtypes
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                dtype=params[4],
            ):
                self._run_bf16_small_oc(*params)

    @unittest.skipUnless(_has_rvv_bf16_gemm_ops(), "RVV BF16 GEMM ops not available")
    def test_bf16_non_contiguous(self):
        """Non-contiguous activation must be handled without silent stride corruption."""
        M, N, K = 8, 64, 512
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                mat2 = torch.randn(N, K, dtype=dtype)
                mat1 = helper_non_contiguous(torch.randn(M, K, dtype=dtype))
                self.assertFalse(mat1.is_contiguous())

                ref = torch.matmul(mat1.float(), mat2.float().t()).to(dtype)
                packed_mat2 = torch.ops.sgl_kernel.convert_weight_packed(mat2)
                out = torch.ops.sgl_kernel.weight_packed_linear(
                    mat1, packed_mat2, None, True
                )
                atol = rtol = precision["linear_gemm"][dtype]
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @unittest.skipUnless(_has_rvv_int8_gemm_ops(), "RVV INT8 GEMM ops not available")
    def test_int8_non_contiguous(self):
        """Non-contiguous INT8 activation must not corrupt quantization or GEMM output."""
        M, N, K = 5, 32, 32 * 17
        for dtype in self.dtypes:
            with self.subTest(dtype=dtype):
                A = helper_non_contiguous(torch.randn(M, K, dtype=dtype) / 10)
                self.assertFalse(A.is_contiguous())
                Aq, As = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
                self.assertEqual(Aq.dtype, torch.uint8)

                Bq = (
                    (torch.rand(N, K, dtype=torch.float32) - 0.5)
                    .mul(127)
                    .clamp(-128, 127)
                    .to(torch.int8)
                )
                Bs = (torch.rand(N) * 1e-2).to(torch.float32)
                Bq_packed = torch.ops.sgl_kernel.convert_weight_packed(Bq)

                ref = native_w8a8_per_token_matmul(Aq, Bq, As, Bs, None, dtype)
                out_direct = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                    Aq, Bq, As, Bs, None, dtype, False
                )
                out = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                    A, Bq_packed, Bs, None, dtype, True
                )
                atol = rtol = precision["pointwise_default"][dtype]
                torch.testing.assert_close(ref, out_direct, atol=atol, rtol=rtol)
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    @unittest.skipUnless(_has_rvv_int8_gemm_ops(), "RVV INT8 GEMM ops not available")
    def test_int8_rejects_k_mismatch(self):
        """K-dim mismatch must raise rather than silently reading out-of-bounds."""
        dtype = torch.float16
        A = torch.randn(2, 64, dtype=dtype) / 10
        Aq, As = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
        bad_Bq = torch.randint(-8, 8, (32, 63), dtype=torch.int8)
        Bs = torch.rand(32, dtype=torch.float32) * 0.01

        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                Aq,
                bad_Bq,
                As,
                Bs,
                None,
                dtype,
                False,
            )

        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                A,
                bad_Bq,
                Bs,
                None,
                dtype,
                False,
            )

    @unittest.skipUnless(_has_rvv_int8_gemm_ops(), "RVV INT8 GEMM ops not available")
    def test_int8_rejects_packed_mismatch(self):
        """Packed weight NB/scale count mismatch must raise instead of producing garbage."""
        dtype = torch.float16
        K = 64
        A = torch.randn(2, K, dtype=dtype) / 10
        Aq, As = torch.ops.sgl_kernel.per_token_quant_int8_cpu(A)
        packed_Bq = torch.randint(-8, 8, (1, 32 * (K + 4)), dtype=torch.int8)
        Bs = torch.rand(80, dtype=torch.float32) * 0.01

        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                Aq,
                packed_Bq,
                As,
                Bs,
                None,
                dtype,
                True,
            )

        with self.assertRaises(RuntimeError):
            torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
                A,
                packed_Bq,
                Bs,
                None,
                dtype,
                True,
            )

    def _run_w4a8_dynamic_linear(self, M, N, K, has_bias, dtype, group_size):
        if K % group_size != 0:
            self.skipTest(f"K={K} is not divisible by group_size={group_size}")
        x = (torch.randn(M, K, dtype=dtype) / 8).contiguous()
        weight_q = torch.randint(-8, 8, (N, K), dtype=torch.int8)
        packed_weight = pack_signed_int4_rows(weight_q)
        scales = (torch.rand(N, K // group_size, dtype=torch.float32) * 0.1) + 0.01
        bias = torch.randn(N, dtype=torch.float32) if has_bias else None

        ref = native_w4a8_dynamic_linear(x, weight_q, scales, bias, group_size, dtype)

        out_direct = torch.ops.sgl_kernel.weight_w4a8_dynamic_linear(
            x,
            packed_weight,
            scales,
            bias if has_bias else None,
            None,
            group_size,
            False,
        )
        packed_weight2, packed_scales2 = (
            torch.ops.sgl_kernel.convert_weight_w4a8_dynamic_packed(
                packed_weight,
                scales,
                group_size,
            )
        )
        out_packed = torch.ops.sgl_kernel.weight_w4a8_dynamic_linear(
            x,
            packed_weight2,
            packed_scales2,
            bias if has_bias else None,
            None,
            group_size,
            True,
        )
        input_permutation = torch.randperm(K, dtype=torch.int64)
        ref_permuted = native_w4a8_dynamic_linear(
            x.index_select(-1, input_permutation),
            weight_q[:, input_permutation],
            scales,
            bias,
            group_size,
            dtype,
        )
        packed_weight_permuted, packed_scales_permuted = (
            torch.ops.sgl_kernel.convert_weight_w4a8_dynamic_packed(
                pack_signed_int4_rows(weight_q[:, input_permutation].contiguous()),
                scales,
                group_size,
            )
        )
        out_permuted = torch.ops.sgl_kernel.weight_w4a8_dynamic_linear(
            x,
            packed_weight_permuted,
            packed_scales_permuted,
            bias if has_bias else None,
            input_permutation,
            group_size,
            True,
        )

        atol = rtol = precision["linear_gemm"][dtype]
        torch.testing.assert_close(ref, out_direct, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref, out_packed, atol=atol, rtol=rtol)
        torch.testing.assert_close(ref_permuted, out_permuted, atol=atol, rtol=rtol)

    @unittest.skipUnless(
        _has_rvv_w4a8_dynamic_linear_ops(), "RVV W4A8 dynamic linear ops not available"
    )
    def test_w4a8_dynamic_linear(self):
        """W4A8 dynamic path should match quantized reference."""
        for params in itertools.product(
            [1, 4, 8],
            self.N_w4a8_dynamic,
            self.K_w4a8_dynamic,
            self.has_bias,
            self.dtypes,
            [128, 256],
        ):
            with self.subTest(
                M=params[0],
                N=params[1],
                K=params[2],
                has_bias=params[3],
                dtype=params[4],
                group_size=params[5],
            ):
                self._run_w4a8_dynamic_linear(*params)


if __name__ == "__main__":
    unittest.main()
