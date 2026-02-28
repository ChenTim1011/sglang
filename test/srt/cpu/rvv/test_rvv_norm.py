"""Unit tests for RVV norm kernels.

Tests run against Python reference implementations from utils.py and are
skipped automatically on non-RISC-V builds where the kernels are unavailable.
"""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .utils import (
    fused_rmsnorm_gated_native,
    gemma3_rmsnorm_native,
    gemma_rmsnorm_native,
    layernorm_native,
    precision,
    rmsnorm_native,
)

torch.manual_seed(1234)


def _has_sgl_kernel_norm():
    """Return True only if the RVV norm ops are registered."""
    try:
        import sgl_kernel  # noqa: F401

        _ = torch.ops.sgl_kernel.rmsnorm_cpu
        return True
    except (ImportError, AttributeError):
        return False


@unittest.skipUnless(
    _has_sgl_kernel_norm(),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNorm(CustomTestCase):
    """RVV norm tests — RMSNorm, fused-add-RMSNorm, L2Norm, Gemma variants."""

    M = [128, 129, 257]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _rmsnorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, eps)
        ref = rmsnorm_native(x, weight, eps)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _fused_add_rmsnorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def _l2norm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.l2norm_cpu(x, eps)
        # L2 norm is equivalent to rmsnorm with weight=1
        fake_weight = torch.ones(n, dtype=dtype)
        ref = rmsnorm_native(x, fake_weight, eps)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _gemma_rmsnorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
        ref = gemma_rmsnorm_native(x, weight, eps)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def _gemma_fused_add_rmsnorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = gemma_rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def _gemma3_rmsnorm_test(self, m, n, dtype):
        # Test 2D input
        x_2d = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out_2d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_2d, weight, eps)
        ref_2d = gemma3_rmsnorm_native(x_2d, weight, eps)
        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref_2d, out_2d, atol=atol, rtol=rtol)

        # Test 4D input [1, num_head, seq_len, head_dim]
        x_4d = torch.randn([1, m, 2, n], dtype=dtype)
        out_4d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_4d, weight, eps)
        ref_4d = gemma3_rmsnorm_native(x_4d, weight, eps)
        torch.testing.assert_close(ref_4d, out_4d, atol=atol, rtol=rtol)

    def test_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._rmsnorm_test(m, n, dt)

    def test_fused_add_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._fused_add_rmsnorm_test(m, n, dt)

    def test_l2norm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._l2norm_test(m, n, dt)

    def test_gemma_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._gemma_rmsnorm_test(m, n, dt)

    def test_gemma_fused_add_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._gemma_fused_add_rmsnorm_test(m, n, dt)

    def test_gemma3_rmsnorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._gemma3_rmsnorm_test(m, n, dt)


@unittest.skipUnless(
    _has_sgl_kernel_norm(),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVFusedRMSNormGated(CustomTestCase):
    """RVV fused gated RMSNorm test."""

    M = [128, 129, 257]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _norm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        gate = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
        ref = fused_rmsnorm_gated_native(x, weight, gate, eps)

        atol = rtol = precision[dtype] * 2
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_fused_rmsnorm_gated(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._norm_test(m, n, dt)


@unittest.skipUnless(
    _has_sgl_kernel_norm(),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVLayerNorm(CustomTestCase):
    """RVV LayerNorm and fused-add-LayerNorm tests."""

    M = [128, 129, 257]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def _layernorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        ref_x = x.clone()
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        torch.ops.sgl_kernel.layernorm_cpu(x, weight, eps)
        ref = layernorm_native(ref_x, weight, eps)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(x, ref, atol=atol, rtol=rtol)

    def _fused_add_layernorm_test(self, m, n, dtype):
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_layernorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = layernorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def test_layernorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._layernorm_test(m, n, dt)

    def test_fused_add_layernorm(self):
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self._fused_add_layernorm_test(m, n, dt)


if __name__ == "__main__":
    unittest.main()
