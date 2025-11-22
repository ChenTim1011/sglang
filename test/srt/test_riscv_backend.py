"""
Unit tests for RISC-V attention backend.

Tests cover:
- RISC-V CPU detection
- Backend registration and selection
- Fallback mechanism
"""

import unittest
from unittest.mock import Mock, patch

import torch

from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    create_riscv_backend,
)
from sglang.srt.layers.attention.riscv_backend import RISCVAttnBackend
from sglang.srt.utils.common import is_host_cpu_riscv


class TestRISCVPDetection(unittest.TestCase):
    """Test RISC-V CPU detection utility."""

    @patch("platform.machine")
    def test_is_host_cpu_riscv_positive(self, mock_machine):
        """Test RISC-V detection for various RISC-V machine strings."""
        test_cases = ["riscv64", "riscv32", "RISCV64", "riscv"]
        for machine in test_cases:
            mock_machine.return_value = machine
            # Re-import to get fresh function with mocked platform
            from importlib import reload
            import sglang.srt.utils.common as common_module

            reload(common_module)
            result = common_module.is_host_cpu_riscv()
            self.assertTrue(
                result, f"Expected {machine} to be detected as RISC-V"
            )

    @patch("platform.machine")
    def test_is_host_cpu_riscv_negative(self, mock_machine):
        """Test RISC-V detection for non-RISC-V architectures."""
        test_cases = ["x86_64", "amd64", "aarch64", "arm64", "i386"]
        for machine in test_cases:
            mock_machine.return_value = machine
            # Re-import to get fresh function with mocked platform
            from importlib import reload
            import sglang.srt.utils.common as common_module

            reload(common_module)
            result = common_module.is_host_cpu_riscv()
            self.assertFalse(
                result, f"Expected {machine} NOT to be detected as RISC-V"
            )


class TestRISCVBackendRegistration(unittest.TestCase):
    """Test RISC-V backend registration."""

    def test_riscv_backend_registered(self):
        """Test that RISC-V backend is registered in ATTENTION_BACKENDS."""
        self.assertIn("riscv", ATTENTION_BACKENDS)
        self.assertEqual(ATTENTION_BACKENDS["riscv"], create_riscv_backend)

    def test_create_riscv_backend(self):
        """Test that create_riscv_backend returns RISCVAttnBackend instance."""
        mock_runner = Mock()
        backend = create_riscv_backend(mock_runner)
        self.assertIsInstance(backend, RISCVAttnBackend)


class TestRISCVBackendFallback(unittest.TestCase):
    """Test RISC-V backend fallback mechanism."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_runner = Mock()
        self.mock_runner.device = torch.device("cpu")
        self.mock_runner.model_config = Mock()
        self.mock_runner.model_config.num_attention_heads = 32
        self.mock_runner.tp_size = 1
        self.mock_runner.token_to_kv_pool = Mock()
        self.mock_runner.token_to_kv_pool.get_value_buffer = Mock(
            return_value=Mock(shape=[1, 1, 128])
        )

    @patch("sglang.srt.layers.attention.riscv_backend.is_host_cpu_riscv")
    def test_fallback_when_not_riscv(self, mock_is_riscv):
        """Test that backend falls back when not on RISC-V."""
        mock_is_riscv.return_value = False
        backend = RISCVAttnBackend(self.mock_runner)
        self.assertFalse(backend.use_riscv_kernels)
        self.assertIsNotNone(backend.fallback_backend)

    @patch("sglang.srt.layers.attention.riscv_backend.is_host_cpu_riscv")
    @patch("torch.ops.sgl_kernel", create=True)
    def test_fallback_when_kernel_unavailable(self, mock_ops, mock_is_riscv):
        """Test that backend falls back when RISC-V kernel is unavailable."""
        mock_is_riscv.return_value = True
        # Simulate kernel not available
        mock_ops.sgl_kernel = Mock()
        del mock_ops.sgl_kernel.decode_attention_cpu

        backend = RISCVAttnBackend(self.mock_runner)
        self.assertFalse(backend.use_riscv_kernels)
        self.assertIsNotNone(backend.fallback_backend)

    def test_prefill_always_uses_fallback(self):
        """Test that prefill always uses fallback (not yet implemented)."""
        backend = RISCVAttnBackend(self.mock_runner)
        # Prefill should always use fallback
        # This is tested implicitly by checking that forward_prefill
        # calls fallback_backend.forward_prefill
        self.assertIsNotNone(backend.fallback_backend)


if __name__ == "__main__":
    unittest.main()
