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
    # mat1 is uint8, but PyTorch doesn't support uint8 matmul well, convert to float directly
    # Note: sgl-kernel implementation treats mat1 as uint8
    mat1_f = mat1_int8.to(torch.float32)

    # mat2 is int8
    mat2_f = mat2_int8.to(torch.float32)

    # Compute integer matrix multiplication in float32 to avoid overflow
    # mat1: [M, K], mat2: [N, K] -> mat2.T: [K, N]
    # Out_int = mat1 @ mat2.T
    out_f = torch.matmul(mat1_f, mat2_f.t())

    # Apply scales: out[m, n] = out_int[m, n] * s1[m] * s2[n]
    # s1: [M] -> [M, 1]
    # s2: [N] -> [1, N]
    out_scaled = out_f * scales1.unsqueeze(1) * scales2.unsqueeze(0)

    # Add bias
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
        # Activation (uint8)
        mat1 = torch.randint(0, 255, (M, K), dtype=torch.uint8)

        # Weight (int8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)

        # Scales
        scales1 = torch.rand(M, dtype=torch.float32) * 0.1
        scales2 = torch.rand(N, dtype=torch.float32) * 0.1

        # Bias
        bias = torch.randn(N, dtype=torch.float32)

        # Reference
        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, bias, out_dtype)

        # RVV Kernel
        # Signature: int8_scaled_mm_cpu(mat1, mat2, scales1, scales2, bias, out_dtype, is_vnni)
        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, bias, out_dtype, False
        )

        # Allow some tolerance for float operations order
        # But since logic is simple (int mul + float scale), should be quite close
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
            mat1 = torch.randint(
                0, 100, (M, K), dtype=torch.uint8
            )  # Keep values smaller to avoid huge accumulators
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

        mat1 = torch.randint(0, 255, (M, K), dtype=torch.uint8)
        mat2 = torch.randint(-128, 127, (N, K), dtype=torch.int8)
        scales1 = torch.ones(M, dtype=torch.float32)
        scales2 = torch.ones(N, dtype=torch.float32)

        ref_out = naive_int8_scaled_mm(mat1, mat2, scales1, scales2, None, out_dtype)

        rvv_out = torch.ops.sgl_kernel.int8_scaled_mm_cpu(
            mat1, mat2, scales1, scales2, None, out_dtype, False
        )

        torch.testing.assert_close(rvv_out, ref_out)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
