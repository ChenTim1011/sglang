"""Unit tests for RVV activation kernels (SiLU, GELU-tanh, GELU-exact).

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_activation -v
"""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import GeluAndMul, SiluAndMul, precision

torch.manual_seed(1234)


def has_sgl_kernel_activation():
    """Return True only if the RVV activation ops are registered."""
    try:
        import sgl_kernel  # noqa: F401

        _ = torch.ops.sgl_kernel.silu_and_mul_cpu
        return True
    except (ImportError, AttributeError):
        return False


@unittest.skipUnless(
    has_sgl_kernel_activation(),
    "sgl_kernel activation not available (non-RISC-V build)",
)
class TestRVVActivation(CustomTestCase):
    """Test suite for RVV activation kernels."""

    M = [128, 129, 257]
    N = [22016, 22018]
    dtype = [torch.float16, torch.bfloat16]

    def run_case_activation_silu_mul(self, m, n, dtype):
        """Run one SiLU-and-Mul case."""
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        ref = SiluAndMul(x)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def run_case_activation_gelu_tanh_mul(self, m, n, dtype):
        """Run one GELU-tanh-and-Mul case."""
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="tanh")
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def run_case_activation_gelu_mul(self, m, n, dtype):
        """Run one GELU-and-Mul case."""
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="none")
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_case_silu_and_mul(self):
        """Case: SiLU-and-Mul across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_activation_silu_mul(m, n, dt)

    def test_case_gelu_tanh_and_mul(self):
        """Case: GELU-tanh-and-Mul across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_activation_gelu_tanh_mul(m, n, dt)

    def test_case_gelu_and_mul(self):
        """Case: GELU-and-Mul across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_activation_gelu_mul(m, n, dt)


if __name__ == "__main__":
    unittest.main()
