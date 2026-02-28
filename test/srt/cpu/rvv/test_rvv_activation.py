"""Unit tests for RVV activation kernels (SiLU, GELU-tanh, GELU-exact).

Tests run against Python reference implementations from utils.py and are
skipped automatically on non-RISC-V builds where the kernels are unavailable.
"""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .utils import GeluAndMul, SiluAndMul, precision

torch.manual_seed(1234)


def _has_sgl_kernel_activation():
    """Return True only if the RVV activation ops are registered."""
    try:
        import sgl_kernel  # noqa: F401

        _ = torch.ops.sgl_kernel.silu_and_mul_cpu
        return True
    except (ImportError, AttributeError):
        return False


@unittest.skipUnless(
    _has_sgl_kernel_activation(),
    "sgl_kernel activation not available (non-RISC-V build)",
)
class TestRVVActivation(CustomTestCase):
    """RVV activation tests — SiLU-and-Mul, GELU-tanh-and-Mul, GELU-and-Mul."""

    M = [128, 129, 257]
    N = [22016, 22018]
    dtype = [torch.float16, torch.bfloat16]

    def _silu_and_mul_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.silu_and_mul_cpu(x)
        ref = SiluAndMul(x)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _gelu_tanh_and_mul_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_tanh_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="tanh")
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _gelu_and_mul_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        out = torch.ops.sgl_kernel.gelu_and_mul_cpu(x)
        ref = GeluAndMul(x, approximate="none")
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_silu_and_mul(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._silu_and_mul_test(m, n, dt)

    def test_gelu_tanh_and_mul(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._gelu_tanh_and_mul_test(m, n, dt)

    def test_gelu_and_mul(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._gelu_and_mul_test(m, n, dt)


if __name__ == "__main__":
    unittest.main()
