"""Unit tests for RVV norm kernels.

Tests run against local Python reference implementations and are skipped on
non-RISC-V builds where the kernels are unavailable.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_norm -v
"""

import itertools
import unittest

import torch

from sglang.test.test_utils import CustomTestCase

from .rvv_utils import (
    fused_rmsnorm_gated_native,
    gemma3_rmsnorm_native,
    gemma_rmsnorm_native,
    has_sgl_kernel_op,
    helper_non_contiguous,
    layernorm_native,
    precision,
    rmsnorm_native,
)

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormCore(CustomTestCase):
    """Test suite for RVV norm kernels."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def run_case_norm_rms(self, m, n, x=None, dtype=torch.float16):
        """Run one RMSNorm case."""
        if x is None:
            x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.rmsnorm_cpu(x, weight, eps)
        ref = rmsnorm_native(x, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def run_case_norm_rms_fused_add(self, m, n, dtype):
        """Run one fused-add-RMSNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def run_case_norm_l2(self, m, n, dtype):
        """Run one L2Norm case."""
        x = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.l2norm_cpu(x, eps)
        # L2 norm matches RMSNorm with a unit weight vector.
        fake_weight = torch.ones(n, dtype=dtype)
        ref = rmsnorm_native(x, fake_weight, eps)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def run_case_norm_gemma_rms(self, m, n, dtype):
        """Run one Gemma RMSNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
        ref = gemma_rmsnorm_native(x, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def run_case_norm_gemma_rms_fused_add(self, m, n, dtype):
        """Run one Gemma fused-add-RMSNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        torch.ops.sgl_kernel.gemma_fused_add_rmsnorm_cpu(x, residual, weight, eps)
        ref_out, ref_res = gemma_rmsnorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(x, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def run_case_norm_gemma3_rms(self, m, n, dtype):
        """Run one Gemma3 RMSNorm case for 2D and 4D tensors."""
        x_2d = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out_2d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_2d, weight, eps)
        ref_2d = gemma3_rmsnorm_native(x_2d, weight, eps)
        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(ref_2d, out_2d, atol=atol, rtol=rtol)

        # Also cover the 4D layout used by attention blocks.
        x_4d = torch.randn([1, m, 2, n], dtype=dtype)
        out_4d = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x_4d, weight, eps)
        ref_4d = gemma3_rmsnorm_native(x_4d, weight, eps)
        torch.testing.assert_close(ref_4d, out_4d, atol=atol, rtol=rtol)

    def test_case_rmsnorm(self):
        """Case: RMSNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_rms(m, n, dtype=dt)

    def test_case_fused_add_rmsnorm(self):
        """Case: fused-add-RMSNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_rms_fused_add(m, n, dt)

    def test_case_l2norm(self):
        """Case: L2Norm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_l2(m, n, dt)

    def test_case_gemma_rmsnorm(self):
        """Case: Gemma RMSNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_gemma_rms(m, n, dt)

    def test_case_gemma_fused_add_rmsnorm(self):
        """Case: Gemma fused-add-RMSNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_gemma_rms_fused_add(m, n, dt)

    def test_case_gemma3_rmsnorm(self):
        """Case: Gemma3 RMSNorm for 2D and 4D tensors."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_gemma3_rms(m, n, dt)

    def test_case_gemma3_rmsnorm_4d_multi_batch(self):
        """Case: Gemma3 RMSNorm 4D with batch_size > 1."""
        for dt in self.dtype:
            n = 4096
            x = torch.randn([2, 4, 3, n], dtype=dt)
            weight = torch.randn(n, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.gemma3_rmsnorm_cpu(x, weight, eps)
            ref = gemma3_rmsnorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_case_rmsnorm_non_contiguous(self):
        """Case: RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            with self.subTest(dtype=dt, shape=x.shape):
                self.run_case_norm_rms(x.shape[0], x.shape[1], x=x, dtype=dt)

    def test_case_fused_add_rmsnorm_non_contiguous(self):
        """Case: fused-add-RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            residual = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            ref_x, ref_res = rmsnorm_native(x.clone(), weight, eps, residual.clone())
            x_copy, res_copy = x.clone(), residual.clone()
            torch.ops.sgl_kernel.fused_add_rmsnorm_cpu(x_copy, res_copy, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(x_copy, ref_x, atol=atol, rtol=rtol)
                torch.testing.assert_close(res_copy, ref_res, atol=atol, rtol=rtol)

    def test_case_rmsnorm_small_n(self):
        """Case: RMSNorm with N smaller than one full m4 vector (< 8 for VLEN=256)."""
        for n, dt in itertools.product([1, 4, 7], self.dtype):
            with self.subTest(n=n, dtype=dt):
                self.run_case_norm_rms(4, n, dtype=dt)

    def test_case_gemma_rmsnorm_non_contiguous(self):
        """Case: Gemma RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.gemma_rmsnorm_cpu(x, weight, eps)
            ref = gemma_rmsnorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormLayer(CustomTestCase):
    """Test suite for RVV LayerNorm kernels."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def run_case_norm_layer(self, m, n, dtype):
        """Run one LayerNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, None, eps)
        ref = layernorm_native(x, weight, eps)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def run_case_norm_layer_fused_add(self, m, n, dtype):
        """Run one fused-add-LayerNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        residual = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        eps = 1e-6

        ref_x = x.clone()
        ref_residual = residual.clone()

        out = torch.ops.sgl_kernel.fused_add_layernorm_cpu(
            x, residual, weight, None, eps
        )
        ref_out, ref_res = layernorm_native(ref_x, weight, eps, ref_residual)

        atol = rtol = precision["pointwise_default"][dtype]
        torch.testing.assert_close(out, ref_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(residual, ref_res, atol=atol, rtol=rtol)

    def test_case_layernorm(self):
        """Case: LayerNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_layer(m, n, dt)

    def test_case_layernorm_with_bias(self):
        """Case: LayerNorm with non-None bias tensor."""
        for m, n, dt in itertools.product([128, 257], [4096, 4109], self.dtype):
            x = torch.randn([m, n], dtype=dt)
            weight = torch.randn(n, dtype=dt)
            bias = torch.randn(n, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, bias, eps)
            ref = layernorm_native(x, weight, eps)
            ref = ref + bias
            tol_map = precision.get("norm_layer_bias", precision["pointwise_default"])
            atol = rtol = tol_map[dt]
            if dt == torch.float16:
                atol = rtol = max(atol, 1e-2)
            with self.subTest(m=m, n=n, dtype=dt):
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_case_layernorm_non_contiguous(self):
        """Case: LayerNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.layernorm_cpu(x, weight, None, eps)
            ref = layernorm_native(x, weight, eps)
            atol = rtol = precision["pointwise_default"][dt]
            with self.subTest(dtype=dt):
                torch.testing.assert_close(out, ref, atol=atol, rtol=rtol)

    def test_case_fused_add_layernorm(self):
        """Case: fused-add-LayerNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_layer_fused_add(m, n, dt)


@unittest.skipUnless(
    has_sgl_kernel_op("rmsnorm_cpu"),
    "sgl_kernel norm not available (non-RISC-V build)",
)
class TestRVVNormFusedGated(CustomTestCase):
    """Test suite for RVV fused gated RMSNorm kernel."""

    M = [128, 129, 257, 1024, 4096]
    N = [4096, 4109]
    dtype = [torch.float16, torch.bfloat16]

    def run_case_norm_rms_gated(self, m, n, dtype):
        """Run one fused gated RMSNorm case."""
        x = torch.randn([m, n], dtype=dtype)
        weight = torch.randn(n, dtype=dtype)
        gate = torch.randn([m, n], dtype=dtype)
        eps = 1e-6

        out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
        ref = fused_rmsnorm_gated_native(x, weight, gate, eps)

        atol = rtol = precision["pointwise_default"][dtype] * 2
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_case_fused_rmsnorm_gated(self):
        """Case: fused gated RMSNorm across shape and dtype matrix."""
        for m, n, dt in itertools.product(self.M, self.N, self.dtype):
            with self.subTest(m=m, n=n, dtype=dt):
                self.run_case_norm_rms_gated(m, n, dt)

    def test_case_fused_rmsnorm_gated_non_contiguous(self):
        """Case: fused gated RMSNorm with non-contiguous input tensors."""
        for dt in self.dtype:
            x = helper_non_contiguous(torch.randn(256, 4096, dtype=dt))
            gate = torch.randn(256, 4096, dtype=dt)  # gate must be contiguous
            weight = torch.randn(4096, dtype=dt)
            eps = 1e-6
            out = torch.ops.sgl_kernel.fused_rmsnorm_gated_cpu(x, weight, gate, eps)
            ref = fused_rmsnorm_gated_native(x, weight, gate, eps)
            atol = rtol = precision["pointwise_default"][dt] * 2
            with self.subTest(dtype=dt):
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
