"""
Unit Tests for RVV GEMM kernel (gemm_rvv.cpp).

This module contains pytest-based tests for the RVV optimized GEMM
implementation used in Linear layers (QKV projection, FFN).

Tests are designed to:
1. Run fast (< 30 seconds total)
2. Cover correctness across various matrix sizes
3. Test edge cases (single token, batch, large matrices)
4. Be skipped gracefully on non-RISC-V platforms

Note: RVV kernel uses AT_DISPATCH_REDUCED_FLOATING_TYPES for x/weight (float16/bfloat16),
but bias must be float32 (matches the original API design).

Usage:
    # Run all tests
    pytest tests/test_rvv_gemm.py -v

    # Run specific test
    pytest tests/test_rvv_gemm.py::test_gemm_rvv_basic -v

    # On RISC-V hardware (e.g., Banana Pi):
    cd ~/.local_riscv_env/workspace/sglang/sgl-kernel
    pytest tests/test_rvv_gemm.py -v
"""

import platform

import pytest
import torch

# ============================================================================
# Constants
# ============================================================================

# RVV kernel uses AT_DISPATCH_REDUCED_FLOATING_TYPES for x/weight (float16/bfloat16)
# But bias must be float32 (API requirement)
TEST_DTYPE = torch.float16
BIAS_DTYPE = torch.float32


# ============================================================================
# Platform Detection
# ============================================================================


def is_riscv_platform() -> bool:
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


# ============================================================================
# Skip Conditions
# ============================================================================

# Check sgl_kernel availability once at module load
try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


def _has_weight_packed_linear() -> bool:
    """Check if weight_packed_linear op is available."""
    if not HAS_SGL_KERNEL:
        return False
    try:
        return hasattr(torch.ops.sgl_kernel, "weight_packed_linear")
    except (AttributeError, RuntimeError):
        return False


requires_sgl_kernel = pytest.mark.skipif(
    not HAS_SGL_KERNEL, reason="sgl_kernel not available"
)

requires_weight_packed_linear = pytest.mark.skipif(
    not _has_weight_packed_linear(),
    reason="weight_packed_linear not available in sgl_kernel",
)


# ============================================================================
# Reference Implementation
# ============================================================================


def torch_linear_reference(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    """
    PyTorch reference implementation for Linear layer.

    Computes: y = x @ weight.T + bias

    Args:
        x: Input tensor [M, K]
        weight: Weight tensor [N, K]
        bias: Optional bias tensor [N] (float32)

    Returns:
        Output tensor [M, N]
    """
    # Use float32 for reference computation to avoid precision issues
    x_f32 = x.float()
    weight_f32 = weight.float()

    y = torch.mm(x_f32, weight_f32.t())

    if bias is not None:
        y = y + bias.float()

    # Convert back to original dtype
    return y.to(x.dtype)


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def tolerances():
    """Fixture that provides tolerance values for float16 comparisons.

    FP16 has limited precision (~3-4 decimal digits), and for larger matrix
    multiplications, accumulated rounding errors can be significant.
    We use more relaxed tolerances that are still meaningful for LLM inference.

    Note: For very large matrices (K > 2048), errors can accumulate to ~1.0
    in worst case due to FP16 precision limits. This is expected behavior
    and does not affect inference quality significantly.
    """
    # float16 has less precision, so we need higher tolerances
    # For large GEMM operations (K > 2048), error can accumulate
    return {"atol": 1.0, "rtol": 0.1}


# ============================================================================
# Unit Tests - Basic Correctness
# ============================================================================


@requires_sgl_kernel
@requires_weight_packed_linear
class TestGemmRvvBasic:
    """Basic correctness tests for RVV GEMM kernel."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 64, 64),  # Minimal
            (1, 128, 128),  # Small
            (4, 256, 256),  # Small batch
            (8, 512, 512),  # Medium
        ],
    )
    def test_gemm_small_sizes(self, M: int, N: int, K: int, tolerances: dict):
        """Test GEMM with small matrix sizes."""
        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)  # bias must be float32

        # Reference
        y_ref = torch_linear_reference(x, weight, bias)

        # RVV implementation
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize("M", [1, 4, 16, 32])
    def test_gemm_decode_sizes(self, M: int, tolerances: dict):
        """Test GEMM with typical decode (single/few tokens) sizes."""
        # TinyLlama dimensions
        K, N = 2048, 2048

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_gemm_no_bias(self, tolerances: dict):
        """Test GEMM without bias."""
        M, N, K = 4, 256, 256

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)

        y_ref = torch_linear_reference(x, weight, None)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, None, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_gemm_single_element(self, tolerances: dict):
        """Test GEMM with 1x1 matrices (edge case)."""
        M, N, K = 1, 1, 1

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


# ============================================================================
# Unit Tests - LLM-Specific Dimensions
# ============================================================================


@requires_sgl_kernel
@requires_weight_packed_linear
class TestGemmRvvLLMSizes:
    """Tests for typical LLM layer dimensions."""

    # TinyLlama dimensions: hidden=2048, intermediate=5632, num_heads=32, head_dim=64
    TINYLLAMA_CONFIGS = [
        # (M, N, K, description)
        (1, 2048, 2048, "QKV projection decode"),
        (1, 5632, 2048, "FFN gate/up decode"),
        (1, 2048, 5632, "FFN down decode"),
    ]

    @pytest.mark.parametrize("M,N,K,desc", TINYLLAMA_CONFIGS)
    def test_tinyllama_decode(
        self, M: int, N: int, K: int, desc: str, tolerances: dict
    ):
        """Test with TinyLlama decode dimensions."""
        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_tinyllama_prefill(self, seq_len: int, tolerances: dict):
        """Test with TinyLlama prefill dimensions."""
        M = seq_len
        K, N = 2048, 2048  # QKV projection

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


# ============================================================================
# Unit Tests - Edge Cases and Numerical Stability
# ============================================================================


@requires_sgl_kernel
@requires_weight_packed_linear
class TestGemmRvvEdgeCases:
    """Edge case and numerical stability tests."""

    def test_zero_input(self, tolerances: dict):
        """Test with zero input tensor."""
        M, N, K = 4, 128, 128

        x = torch.zeros(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_zero_weight(self, tolerances: dict):
        """Test with zero weight tensor."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.zeros(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_identity_weight(self, tolerances: dict):
        """Test with identity-like weight (ones on diagonal)."""
        N = K = 128
        M = 4

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.eye(N, K, dtype=TEST_DTYPE)
        bias = torch.zeros(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_large_values(self, tolerances: dict):
        """Test with large input values (scaled for float16 range)."""
        M, N, K = 4, 128, 128

        # Use smaller scale for float16 to avoid overflow
        x = torch.randn(M, K, dtype=TEST_DTYPE) * 10
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        # Use higher tolerance for large values (accumulation error is larger)
        torch.testing.assert_close(y_rvv, y_ref, atol=0.5, rtol=0.5)

    def test_small_values(self, tolerances: dict):
        """Test with very small input values."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=TEST_DTYPE) * 1e-3
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 63, 65),  # Non-power-of-2
            (3, 127, 129),  # Odd dimensions
            (7, 255, 257),  # Prime-ish dimensions
        ],
    )
    def test_non_aligned_sizes(self, M: int, N: int, K: int, tolerances: dict):
        """Test with non-aligned (non-power-of-2) dimensions."""
        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


# ============================================================================
# Integration Tests
# ============================================================================


@requires_sgl_kernel
@requires_weight_packed_linear
class TestGemmRvvIntegration:
    """Integration tests simulating real usage patterns."""

    def test_sequential_layers(self, tolerances: dict):
        """Test sequential Linear layers like in a Transformer block."""
        hidden_size = 256
        intermediate_size = 512
        batch_size = 4

        # Simulate: hidden -> intermediate -> hidden
        x = torch.randn(batch_size, hidden_size, dtype=TEST_DTYPE)

        # Up projection
        w_up = torch.randn(intermediate_size, hidden_size, dtype=TEST_DTYPE)
        b_up = torch.randn(intermediate_size, dtype=BIAS_DTYPE)

        # Down projection
        w_down = torch.randn(hidden_size, intermediate_size, dtype=TEST_DTYPE)
        b_down = torch.randn(hidden_size, dtype=BIAS_DTYPE)

        # Reference
        h_ref = torch_linear_reference(x, w_up, b_up)
        h_ref = torch.nn.functional.gelu(h_ref)
        y_ref = torch_linear_reference(h_ref, w_down, b_down)

        # RVV implementation
        h_rvv = sgl_kernel.weight_packed_linear(x, w_up, b_up, False)
        h_rvv = torch.nn.functional.gelu(h_rvv)
        y_rvv = sgl_kernel.weight_packed_linear(h_rvv, w_down, b_down, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_qkv_projection(self, tolerances: dict):
        """Test QKV projection pattern (3 Linear layers)."""
        hidden_size = 256
        num_heads = 4
        head_dim = 64
        batch_size = 4

        x = torch.randn(batch_size, hidden_size, dtype=TEST_DTYPE)

        # Q, K, V projections (no bias)
        w_q = torch.randn(num_heads * head_dim, hidden_size, dtype=TEST_DTYPE)
        w_k = torch.randn(num_heads * head_dim, hidden_size, dtype=TEST_DTYPE)
        w_v = torch.randn(num_heads * head_dim, hidden_size, dtype=TEST_DTYPE)

        # Reference
        q_ref = torch_linear_reference(x, w_q, None)
        k_ref = torch_linear_reference(x, w_k, None)
        v_ref = torch_linear_reference(x, w_v, None)

        # RVV implementation
        q_rvv = sgl_kernel.weight_packed_linear(x, w_q, None, False)
        k_rvv = sgl_kernel.weight_packed_linear(x, w_k, None, False)
        v_rvv = sgl_kernel.weight_packed_linear(x, w_v, None, False)

        torch.testing.assert_close(q_rvv, q_ref, **tolerances)
        torch.testing.assert_close(k_rvv, k_ref, **tolerances)
        torch.testing.assert_close(v_rvv, v_ref, **tolerances)

    def test_batch_consistency(self, tolerances: dict):
        """Test that batched computation equals individual computations."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=TEST_DTYPE)
        weight = torch.randn(N, K, dtype=TEST_DTYPE)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        # Batched
        y_batch = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        # Individual rows
        y_individual = torch.stack(
            [
                sgl_kernel.weight_packed_linear(
                    x[i : i + 1], weight, bias, False
                ).squeeze(0)
                for i in range(M)
            ]
        )

        torch.testing.assert_close(y_batch, y_individual, **tolerances)


# ============================================================================
# Main Entry Point
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
