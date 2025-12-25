"""
Unit Tests for RVV GEMM kernel.

This module contains pytest-based tests for the RVV optimized GEMM
implementation used in Linear layers (QKV projection, FFN).

Usage:
    # Run all tests
    pytest tests/test_rvv_gemm.py -v

"""

import platform

import pytest
import torch

# RVV kernel uses AT_DISPATCH_REDUCED_FLOATING_TYPES for x/weight (float16/bfloat16)
# But bias must be float32 (API requirement)
BIAS_DTYPE = torch.float32


def is_riscv_platform() -> bool:
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


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


@pytest.fixture
def tolerances(dtype):
    """Fixture that provides tolerance values for comparisons."""
    if dtype == torch.bfloat16:
        # BF16 is less precise, need looser tolerances
        return {"atol": 1.5, "rtol": 0.15}
    else:
        # float16
        return {"atol": 1.0, "rtol": 0.1}


@requires_sgl_kernel
@requires_weight_packed_linear
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestGemmRvvBasic:
    """Basic correctness tests for RVV GEMM kernel."""

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 64, 64),
            (1, 128, 128),
            (4, 256, 256),
            (8, 512, 512),
        ],
    )
    def test_gemm_small_sizes(self, M: int, N: int, K: int, dtype, tolerances: dict):
        """Test GEMM with small matrix sizes."""
        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        # Reference
        y_ref = torch_linear_reference(x, weight, bias)

        # RVV implementation
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize("M", [1, 4, 16, 32])
    def test_gemm_decode_sizes(self, M: int, dtype, tolerances: dict):
        """Test GEMM with typical decode (single/few tokens) sizes."""
        K, N = 2048, 2048

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_gemm_no_bias(self, dtype, tolerances: dict):
        """Test GEMM without bias."""
        M, N, K = 4, 256, 256

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)

        y_ref = torch_linear_reference(x, weight, None)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, None, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_gemm_single_element(self, dtype, tolerances: dict):
        """Test GEMM with 1x1 matrices (edge case)."""
        M, N, K = 1, 1, 1

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


@requires_sgl_kernel
@requires_weight_packed_linear
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestGemmRvvLLMSizes:
    """Tests for typical LLM layer dimensions."""

    TINYLLAMA_CONFIGS = [
        (1, 2048, 2048, "QKV projection decode"),
        (1, 5632, 2048, "FFN gate/up decode"),
        (1, 2048, 5632, "FFN down decode"),
    ]

    @pytest.mark.parametrize("M,N,K,desc", TINYLLAMA_CONFIGS)
    def test_tinyllama_decode(
        self, M: int, N: int, K: int, desc: str, dtype, tolerances: dict
    ):
        """Test with TinyLlama decode dimensions."""
        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize("seq_len", [16, 32, 64])
    def test_tinyllama_prefill(self, seq_len: int, dtype, tolerances: dict):
        """Test with TinyLlama prefill dimensions."""
        M = seq_len
        K, N = 2048, 2048

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize("M", [128, 256])
    def test_gemm_large_prefill(self, M: int, dtype, tolerances: dict):
        """Test GEMM with larger prefill sizes (M > 64)."""
        K, N = 1024, 1024

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


@requires_sgl_kernel
@requires_weight_packed_linear
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestGemmRvvEdgeCases:
    """Edge case and numerical stability tests."""

    def test_zero_input(self, dtype, tolerances: dict):
        """Test with zero input tensor."""
        M, N, K = 4, 128, 128

        x = torch.zeros(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_zero_weight(self, dtype, tolerances: dict):
        """Test with zero weight tensor."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.zeros(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_identity_weight(self, dtype, tolerances: dict):
        """Test with identity-like weight (ones on diagonal)."""
        N = K = 128
        M = 4

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.eye(N, K, dtype=dtype)
        bias = torch.zeros(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_large_values(self, dtype, tolerances: dict):
        """Test with large input values (scaled for range)."""
        M, N, K = 4, 128, 128

        # Use smaller scale to avoid overflow
        x = torch.randn(M, K, dtype=dtype) * 10
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        # Use higher tolerance for large values
        atol = tolerances["atol"] * 2.0
        rtol = tolerances["rtol"] * 2.0
        torch.testing.assert_close(y_rvv, y_ref, atol=atol, rtol=rtol)

    def test_small_values(self, dtype, tolerances: dict):
        """Test with very small input values."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=dtype) * 1e-3
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    @pytest.mark.parametrize(
        "M,N,K",
        [
            (1, 63, 65),
            (3, 127, 129),
            (7, 255, 257),
        ],
    )
    def test_non_aligned_sizes(self, M: int, N: int, K: int, dtype, tolerances: dict):
        """Test with non-aligned (non-power-of-2) dimensions."""
        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
        bias = torch.randn(N, dtype=BIAS_DTYPE)

        y_ref = torch_linear_reference(x, weight, bias)
        y_rvv = sgl_kernel.weight_packed_linear(x, weight, bias, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)


@requires_sgl_kernel
@requires_weight_packed_linear
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
class TestGemmRvvIntegration:
    """Integration tests simulating real usage patterns."""

    def test_sequential_layers(self, dtype, tolerances: dict):
        """Test sequential Linear layers like in a Transformer block."""
        hidden_size = 256
        intermediate_size = 512
        batch_size = 4

        # Simulate: hidden -> intermediate -> hidden
        x = torch.randn(batch_size, hidden_size, dtype=dtype)

        # Up projection
        w_up = torch.randn(intermediate_size, hidden_size, dtype=dtype)
        b_up = torch.randn(intermediate_size, dtype=BIAS_DTYPE)

        # Down projection
        w_down = torch.randn(hidden_size, intermediate_size, dtype=dtype)
        b_down = torch.randn(hidden_size, dtype=BIAS_DTYPE)

        # Reference
        h_ref = torch_linear_reference(x, w_up, b_up)
        h_ref = torch.nn.functional.gelu(h_ref.float()).to(dtype)
        y_ref = torch_linear_reference(h_ref, w_down, b_down)

        # RVV implementation
        h_rvv = sgl_kernel.weight_packed_linear(x, w_up, b_up, False)
        h_rvv = torch.nn.functional.gelu(h_rvv.float()).to(dtype)
        y_rvv = sgl_kernel.weight_packed_linear(h_rvv, w_down, b_down, False)

        torch.testing.assert_close(y_rvv, y_ref, **tolerances)

    def test_qkv_projection(self, dtype, tolerances: dict):
        """Test QKV projection pattern (3 Linear layers)."""
        hidden_size = 256
        num_heads = 4
        head_dim = 64
        batch_size = 4

        x = torch.randn(batch_size, hidden_size, dtype=dtype)

        # Q, K, V projections (no bias)
        w_q = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype)
        w_k = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype)
        w_v = torch.randn(num_heads * head_dim, hidden_size, dtype=dtype)

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

    def test_batch_consistency(self, dtype, tolerances: dict):
        """Test that batched computation equals individual computations."""
        M, N, K = 4, 128, 128

        x = torch.randn(M, K, dtype=dtype)
        weight = torch.randn(N, K, dtype=dtype)
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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
