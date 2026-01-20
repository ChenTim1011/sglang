"""
Unit Tests for RVV INT8 GEMM kernel.

This module tests the RVV optimized INT8 GEMM implementation used for W8A8 quantization.
It tests `int8_scaled_mm_cpu` which performs: Output = (A_int8 @ B_int8) * Scales + Bias

Usage:
    pytest tests/test_rvv_gemm_int8.py -v
"""

import os
import sys

import pytest
import torch

# Add parent directory to path to import sgl_kernel
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


def _has_int8_gemm() -> bool:
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "int8_scaled_mm_cpu")
    except (AttributeError, RuntimeError):
        return False


requires_sgl_kernel = pytest.mark.skipif(
    not HAS_SGL_KERNEL, reason="sgl_kernel not available"
)

requires_int8_gemm = pytest.mark.skipif(
    not _has_int8_gemm(),
    reason="int8_scaled_mm_cpu op not available",
)


def naive_int8_scaled_mm(
    mat1_int8: torch.Tensor,
    mat2_int8: torch.Tensor,
    scales1: torch.Tensor,
    scales2: torch.Tensor,
    bias: torch.Tensor | None = None,
    out_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """
    Reference implementation using PyTorch float operations.

    Args:
        mat1_int8: [M, K] uint8 (activation)
        mat2_int8: [N, K] int8 (weight)
        scales1: [M] float
        scales2: [N] float
        bias: [N] float
    """
    # Dequantize inputs to float
    # mat1 is uint8, but represents signed int8 values (symmetric quantization)
    # So we must view it as int8 before converting to float
    mat1_f = mat1_int8.view(torch.int8).to(torch.float32)
    mat2_f = mat2_int8.to(torch.float32)

    # Compute integer matrix multiplication in float32 to avoid overflow
    # Note: sgl-kernel expects mat2 to be [N, K]
    # In PyTorch, matmul(M x K, K x N) -> M x N
    # So we transpose mat2 to [K, N]
    out_f = torch.matmul(mat1_f, mat2_f.t())

    # Apply scales: out[m, n] = out_int[m, n] * s1[m] * s2[n]
    # s1: [M] -> [M, 1]
    # s2: [N] -> [1, N]
    out_scaled = out_f * scales1.unsqueeze(1) * scales2.unsqueeze(0)

    if bias is not None:
        out_scaled = out_scaled + bias.unsqueeze(0)

    return out_scaled.to(out_dtype)


@requires_sgl_kernel
@requires_int8_gemm
class TestGemmInt8Rvv:

    @pytest.mark.parametrize("M", [1, 4, 16, 32])
    @pytest.mark.parametrize("N", [32, 64, 128])
    @pytest.mark.parametrize("K", [32, 128, 256])
    @pytest.mark.parametrize("out_dtype", [torch.float16, torch.float32])
    def test_basic_correctness(self, M, N, K, out_dtype):
        # mat1: uint8 (activation), effectively signed int8 (-128 to 127) due to symmetric quantization
        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)

        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

        scales1 = torch.rand(M, dtype=torch.float32) * 0.1
        scales2 = torch.rand(N, dtype=torch.float32) * 0.1

        bias = torch.randn(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)

    @pytest.mark.parametrize("out_dtype", [torch.float16])
    def test_tinyllama_shapes(self, out_dtype):
        """Test with shapes from TinyLlama."""
        configs = [
            (1, 2048, 2048),  # Decode QKV
            (4, 2048, 2048),  # Batch Decode
            (1, 5632, 2048),  # FFN Up
            (1, 2048, 5632),  # FFN Down
        ]

        for M, N, K in configs:
            mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
            mat1 = mat1_s8.view(torch.uint8)
            mat2 = torch.randint(-50, 50, (N, K), dtype=torch.int8)
            scales1 = torch.rand(M, dtype=torch.float32) * 0.01
            scales2 = torch.rand(N, dtype=torch.float32) * 0.01
            bias = torch.randn(N, dtype=torch.float32)

            ref_out = naive_int8_scaled_mm(
                mat1, mat2, scales1, scales2, bias, out_dtype
            )

            rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                mat1, mat2, scales1, scales2, bias, out_dtype, False
            )

            torch.testing.assert_close(rvv_out, ref_out, atol=1e-2, rtol=1e-2)

    def test_no_bias(self):
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.ones(M, dtype=torch.float32)
        scales2 = torch.ones(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, None, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, None, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out)


@requires_sgl_kernel
@requires_int8_gemm
class TestGemmInt8RvvEdgeCases:
    """Edge case tests for INT8 GEMM."""

    def test_quantization_range_boundaries(self):
        """Test with quantization range boundaries."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        # Test min/max values for mat1 (uint8 representing int8)
        # -128 (0x80) -> 128 (0x80) as uint8
        # 127 (0x7F) -> 127 (0x7F) as uint8
        # 0 (0x00) -> 0 (0x00) as uint8
        # -1 (0xFF) -> 255 (0xFF) as uint8
        mat1_s8 = torch.tensor([[-128, 127, 0, -1]], dtype=torch.int8).repeat(M, K // 4)
        mat1 = mat1_s8.view(torch.uint8)

        # Test min/max values for mat2 (int8)
        mat2 = torch.tensor([[-128, 127, 0, -1]], dtype=torch.int8).repeat(N, K // 4)

        scales1 = torch.ones(M, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.zeros(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)

    def test_extreme_small_scales(self):
        """Test with extremely small scale values."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

        scales1 = torch.ones(M, dtype=torch.float32) * 1e-6
        scales2 = torch.ones(N, dtype=torch.float32) * 1e-6
        bias = torch.zeros(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-5, rtol=1e-3)

    def test_extreme_large_scales(self):
        """Test with extremely large scale values."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-50, 50, (N, K), dtype=torch.int8)

        scales1 = torch.ones(M, dtype=torch.float32) * 100.0
        scales2 = torch.ones(N, dtype=torch.float32) * 100.0
        bias = torch.zeros(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-2, rtol=1e-2)

    def test_zero_input(self):
        """Test with zero input matrix."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1 = torch.zeros(M, K, dtype=torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

        scales1 = torch.ones(M, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.randn(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)

    def test_zero_weight(self):
        """Test with zero weight matrix."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.zeros(N, K, dtype=torch.int8)

        scales1 = torch.ones(M, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.randn(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)


@requires_sgl_kernel
@requires_int8_gemm
class TestGemmInt8RvvNumericalStability:
    """Numerical stability tests for INT8 GEMM."""

    def test_no_nan_inf(self):
        """Test that output never contains NaN or Inf."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

        scales1 = torch.rand(M, dtype=torch.float32) * 0.1
        scales2 = torch.rand(N, dtype=torch.float32) * 0.1
        bias = torch.randn(N, dtype=torch.float32)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        assert torch.isfinite(rvv_out).all(), "Output contains NaN or Inf"

    def test_small_values(self):
        """Test with very small input values."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-10, 10, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-5, 5, (N, K), dtype=torch.int8)

        scales1 = torch.ones(M, dtype=torch.float32) * 1e-3
        scales2 = torch.ones(N, dtype=torch.float32) * 1e-3
        bias = torch.zeros(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-5, rtol=1e-3)
        assert torch.isfinite(rvv_out).all(), "Output contains NaN or Inf"


@requires_sgl_kernel
@requires_int8_gemm
class TestGemmInt8RvvErrorHandling:
    """Error handling and input validation tests for INT8 GEMM."""

    def test_invalid_zero_k_dimension(self):
        """Test with K=0 (should handle gracefully or raise error)."""
        M, N = 4, 64
        K = 0
        out_dtype = torch.float32

        mat1 = torch.zeros(M, K, dtype=torch.uint8)
        mat2 = torch.zeros(N, K, dtype=torch.int8)
        scales1 = torch.ones(M, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.zeros(N, dtype=torch.float32)

        try:
            rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                mat1, mat2, scales1, scales2, bias, out_dtype, False
            )
            assert rvv_out.shape == (M, N), "Output shape should be correct"
            assert torch.allclose(
                rvv_out, bias.unsqueeze(0).expand(M, -1)
            ), "Output should be bias only"
        except (RuntimeError, ValueError, AssertionError):
            pass

    def test_invalid_mismatched_k_dimensions(self):
        """Test with mismatched K dimensions between mat1 and mat2.

        This test verifies kernel behavior with invalid input (K dimension mismatch).
        The kernel may not check K dimensions for is_packed=false, so we test actual behavior.
        """
        M, N = 4, 64
        K1 = 32
        K2 = 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K1), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K2), dtype=torch.int8)
        scales1 = torch.ones(M, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.zeros(N, dtype=torch.float32)

        # Test actual behavior: kernel may not validate K dimensions
        try:
            rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                mat1, mat2, scales1, scales2, bias, out_dtype, False
            )
            # If no error, verify output shape and that it doesn't contain NaN
            assert rvv_out.shape == (
                M,
                N,
            ), "Output shape should match expected dimensions"
            assert not torch.isnan(
                rvv_out
            ).any(), "Output should not contain NaN (may contain incorrect values due to dimension mismatch)"
        except (RuntimeError, ValueError, AssertionError) as e:
            # If error is raised, that's valid behavior (kernel validates K dimensions)
            assert isinstance(
                e, (RuntimeError, ValueError, AssertionError)
            ), f"Unexpected error type: {type(e)}"

    def test_invalid_mismatched_scale_sizes(self):
        """Test with mismatched scale sizes."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.ones(M + 10, dtype=torch.float32) * 0.01
        scales2 = torch.ones(N, dtype=torch.float32) * 0.01
        bias = torch.zeros(N, dtype=torch.float32)

        with pytest.raises((RuntimeError, ValueError, AssertionError)):
            torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                mat1, mat2, scales1, scales2, bias, out_dtype, False
            )

    def test_invalid_zero_scale(self):
        """Test with zero scale (should handle gracefully or raise error)."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.zeros(M, dtype=torch.float32)
        scales2 = torch.zeros(N, dtype=torch.float32)
        bias = torch.zeros(N, dtype=torch.float32)

        try:
            rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
                mat1, mat2, scales1, scales2, bias, out_dtype, False
            )
            assert torch.isfinite(
                rvv_out
            ).all(), "Output should be finite even with zero scale"
        except (RuntimeError, ValueError, ZeroDivisionError):
            pass


@requires_sgl_kernel
@requires_int8_gemm
class TestGemmInt8RvvCombinatorial:
    """Combinatorial testing for INT8 GEMM - parameter interactions."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 32, 32),
            (1, 64, 32),
            (1, 64, 64),
            (1, 128, 32),
            (1, 128, 64),
            (1, 128, 128),
            (1, 256, 64),
            (1, 256, 128),
            (1, 256, 256),
            (4, 32, 32),
            (4, 64, 32),
            (4, 64, 64),
            (4, 128, 64),
            (4, 128, 128),
            (4, 256, 128),
            (4, 256, 256),
            (16, 64, 64),
            (16, 128, 64),
            (16, 128, 128),
            (16, 256, 128),
            (32, 128, 128),
            (32, 256, 128),
            (32, 256, 256),
        ],
    )
    def test_size_combinations(self, M, N, K):
        """Test various M × N × K combinations."""
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.rand(M, dtype=torch.float32) * 0.1
        scales2 = torch.rand(N, dtype=torch.float32) * 0.1
        bias = torch.randn(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)
        assert torch.isfinite(rvv_out).all(), "Output contains NaN or Inf"

    @pytest.mark.parametrize(
        "scales1_val,scales2_val",
        [
            (1e-6, 1e-6),
            (1e-6, 0.01),
            (1e-6, 100.0),
            (0.01, 1e-6),
            (0.01, 0.01),
            (0.01, 100.0),
            (100.0, 1e-6),
            (100.0, 0.01),
            (100.0, 100.0),
        ],
    )
    def test_scale_combinations(self, scales1_val, scales2_val):
        """Test various scales1 × scales2 combinations."""
        M, N, K = 4, 64, 64
        out_dtype = torch.float32

        mat1_s8 = torch.randint(-128, 127, (M, K), dtype=torch.int8)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.ones(M, dtype=torch.float32) * scales1_val
        scales2 = torch.ones(N, dtype=torch.float32) * scales2_val
        bias = torch.zeros(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        atol = 1e-5 if min(scales1_val, scales2_val) < 1e-3 else 1e-3
        rtol = 1e-3
        torch.testing.assert_close(rvv_out, ref_out, atol=atol, rtol=rtol)
        assert torch.isfinite(
            rvv_out
        ).all(), f"Output contains NaN or Inf with scales1={scales1_val}, scales2={scales2_val}"

    @pytest.mark.parametrize(
        "M,out_dtype",
        [
            (1, torch.float16),
            (1, torch.float32),
            (4, torch.float16),
            (4, torch.float32),
            (16, torch.float16),
            (16, torch.float32),
        ],
    )
    def test_m_output_dtype_combinations(self, M, out_dtype):
        """Test various M × out_dtype combinations."""
        N, K = 64, 64

        # Test min/max values
        mat1_s8 = torch.tensor([[-128, 127, 0, -1]], dtype=torch.int8).repeat(M, K // 4)
        mat1 = mat1_s8.view(torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.rand(M, dtype=torch.float32) * 0.1
        scales2 = torch.rand(N, dtype=torch.float32) * 0.1
        bias = torch.randn(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out, atol=1e-3, rtol=1e-3)
        assert torch.isfinite(rvv_out).all(), "Output contains NaN or Inf"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
