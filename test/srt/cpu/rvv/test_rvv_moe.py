"""Unit Tests for RVV MoE kernel.

Tests fused_experts_cpu and shared_expert_cpu (FP16/BF16) on RISC-V Vector.
N must be a multiple of BLOCK_N=64 for the RVV kernel.
"""

import itertools
import os
import sys
import unittest

import sgl_kernel  # noqa: F401
import torch
import torch.nn.functional as F

from sglang.test.test_utils import CustomTestCase

# Add workspace root to path for imports
script_dir = os.path.dirname(os.path.abspath(__file__))
workspace_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
)
if workspace_root not in sys.path:
    sys.path.insert(0, workspace_root)

try:
    from .utils import precision
except ImportError:
    from test.srt.cpu.rvv.utils import precision

try:
    from test.srt.cpu.utils import torch_naive_fused_moe
except ImportError:
    cpu_test_dir = os.path.dirname(script_dir)
    if cpu_test_dir not in sys.path:
        sys.path.insert(0, cpu_test_dir)
    from utils import torch_naive_fused_moe  # noqa: E402

try:
    from sglang.srt.layers.amx_utils import CPUQuantMethod
except ImportError:

    class CPUQuantMethod:
        UNQUANT = 0


torch.manual_seed(1234)


def fused_moe_cpu(a, w1, w2, score, topk, renormalize):
    """Run fused_experts_cpu with RVV-packed weights (is_vnni=True)."""
    topk_weights, topk_ids = torch.ops.sgl_kernel.grouped_topk_cpu(
        a, score, topk, renormalize, 1, 1, 0, None, None
    )
    packed_w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
    packed_w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)
    return torch.ops.sgl_kernel.fused_experts_cpu(
        a,
        packed_w1,
        packed_w2,
        topk_weights,
        topk_ids,
        True,
        CPUQuantMethod.UNQUANT,
        None,
        None,
        None,
        None,
        None,
        True,
    )


def shared_expert_cpu(hidden_states, w1, w2, fused_out, routed_scaling_factor):
    """Run shared_expert_cpu with RVV-packed weights."""
    packed_w1 = torch.ops.sgl_kernel.convert_weight_packed(w1)
    packed_w2 = torch.ops.sgl_kernel.convert_weight_packed(w2)
    return torch.ops.sgl_kernel.shared_expert_cpu(
        hidden_states,
        packed_w1,
        packed_w2,
        fused_out,
        routed_scaling_factor,
        False,
        False,
        False,
        None,
        None,
        None,
        True,  # is_vnni (pre-packed)
    )


class TestFusedExperts(CustomTestCase):
    """FP16/BF16 fused_experts_cpu tests (RVV packed weights)."""

    M = [1, 4, 8]
    N = [64, 128]
    K = [64, 128]
    E = [4, 8]
    topk = [2]
    renormalize = [False, True]

    def _test_fused_moe(self, m, n, k, e, topk, renormalize, dtype):
        a = torch.randn((m, k), dtype=dtype) / 10
        w1 = torch.randn((e, 2 * n, k), dtype=dtype) / 10
        w2 = torch.randn((e, k, n), dtype=dtype) / 10
        score = torch.randn((m, e), dtype=dtype)

        ref = torch_naive_fused_moe(a, w1, w2, score, topk, renormalize)
        out = fused_moe_cpu(a, w1, w2, score, topk, renormalize)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref.to(dtype), out, atol=atol, rtol=rtol)

    def test_fused_moe_bf16(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
            self.renormalize,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                e=params[3],
                topk=params[4],
                renormalize=params[5],
            ):
                self._test_fused_moe(*params, torch.bfloat16)

    def test_fused_moe_fp16(self):
        for params in itertools.product(
            self.M,
            self.N,
            self.K,
            self.E,
            self.topk,
            self.renormalize,
        ):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
                e=params[3],
                topk=params[4],
                renormalize=params[5],
            ):
                self._test_fused_moe(*params, torch.float16)


class TestSharedExpert(CustomTestCase):
    """FP16/BF16 shared_expert_cpu tests (RVV packed weights). w1: [2N,K], w2: [K,N]."""

    M = [1, 4, 8]
    N = [64]
    K = [64, 128]

    def _test_shared_expert(self, m, n, k, dtype):
        hidden = torch.randn((m, k), dtype=dtype) / 10
        w1 = torch.randn((2 * n, k), dtype=dtype) / 10
        w2 = torch.randn((k, n), dtype=dtype) / 10
        fused_out = torch.randn((m, k), dtype=dtype) / 10
        scale = 0.5

        x = hidden.float() @ w1.float().t()
        x_gate, x_up = x[:, :n], x[:, n:]
        silu_out = F.silu(x_gate) * x_up
        ref = (silu_out @ w2.float().t() + fused_out.float() * scale).to(dtype)

        out = shared_expert_cpu(hidden, w1, w2, fused_out, scale)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_shared_expert_bf16(self):
        for params in itertools.product(self.M, self.N, self.K):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
            ):
                self._test_shared_expert(*params, torch.bfloat16)

    def test_shared_expert_fp16(self):
        for params in itertools.product(self.M, self.N, self.K):
            with self.subTest(
                m=params[0],
                n=params[1],
                k=params[2],
            ):
                self._test_shared_expert(*params, torch.float16)


if __name__ == "__main__":
    unittest.main()
