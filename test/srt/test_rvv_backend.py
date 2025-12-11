"""
Integration tests for RVV attention backend.

Tests cover:
- RISC-V CPU detection
- Backend registration and selection
- Fallback mechanism
- RVV kernel integration
"""

import unittest
from unittest.mock import MagicMock, Mock, patch

import torch

from sglang.srt.layers.attention.attention_registry import (
    ATTENTION_BACKENDS,
    create_rvv_backend,
)
from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import is_host_cpu_riscv

# Try to import sgl_kernel for extend attention tests
try:
    import sgl_kernel

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False


class TestRiscvDetection(unittest.TestCase):
    """Test RISC-V CPU detection utility."""

    @patch("platform.machine")
    def test_is_host_cpu_riscv_positive(self, mock_machine):
        """Test RISC-V detection for various RISC-V machine strings."""
        test_cases = ["riscv64", "riscv32", "RISCV64", "RISCV32"]
        for machine in test_cases:
            mock_machine.return_value = machine
            # Re-import to get fresh function with mocked platform
            from importlib import reload

            import sglang.srt.utils.common as common_module

            reload(common_module)
            result = common_module.is_host_cpu_riscv()
            self.assertTrue(result, f"Expected {machine} to be detected as RISC-V")

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
            self.assertFalse(result, f"Expected {machine} NOT to be detected as RISC-V")


class TestRVVBackendRegistration(unittest.TestCase):
    """Test RVV backend registration."""

    def test_rvv_backend_registered(self):
        """Test that RVV backend is registered in ATTENTION_BACKENDS."""
        self.assertIn("rvv", ATTENTION_BACKENDS)
        self.assertEqual(ATTENTION_BACKENDS["rvv"], create_rvv_backend)

    def test_create_rvv_backend(self):
        """Test that create_rvv_backend returns RVVAttnBackend instance."""
        mock_runner = Mock()
        # Properly configure mock to return integer values
        mock_runner.device = torch.device("cpu")
        mock_runner.model_config = Mock()
        mock_runner.model_config.num_attention_heads = 32
        mock_runner.tp_size = 1
        mock_runner.token_to_kv_pool = Mock()
        mock_runner.token_to_kv_pool.get_value_buffer = Mock(
            return_value=Mock(shape=[1, 1, 128])
        )
        backend = create_rvv_backend(mock_runner)
        self.assertIsInstance(backend, RVVAttnBackend)


class TestRVVBackendFallback(unittest.TestCase):
    """Test RVV backend fallback mechanism."""

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

    @patch("sglang.srt.utils.common.is_host_cpu_riscv")
    def test_fallback_when_not_riscv(self, mock_is_riscv):
        """Test that backend falls back when not on RISC-V."""
        mock_is_riscv.return_value = False
        backend = RVVAttnBackend(self.mock_runner)
        self.assertFalse(backend.use_rvv_kernels)
        self.assertIsNotNone(backend.fallback_backend)

    def test_fallback_when_kernel_unavailable(self):
        """Test that backend falls back when RVV kernel is unavailable.

        Note: This test is skipped on actual RISC-V hardware since the
        real RVV kernel is available and cannot be easily mocked.
        """
        import platform

        if platform.machine().lower() in ("riscv64", "riscv32", "riscv"):
            self.skipTest(
                "Cannot mock RVV kernel availability on actual RISC-V hardware"
            )

        # On non-RISC-V, we simply verify fallback backend exists
        backend = RVVAttnBackend(self.mock_runner)
        self.assertIsNotNone(backend.fallback_backend)


class TestRVVBackendIntegration(unittest.TestCase):
    def setUp(self):
        self.mock_runner = Mock()
        self.mock_runner.device = torch.device("cpu")
        self.mock_runner.model_config = Mock()
        self.mock_runner.model_config.num_attention_heads = 32
        self.mock_runner.tp_size = 1
        self.mock_runner.token_to_kv_pool = Mock()
        self.mock_runner.token_to_kv_pool.get_value_buffer = Mock(
            return_value=Mock(shape=[1, 1, 128])
        )

    @patch("sglang.srt.utils.common.is_host_cpu_riscv", return_value=True)
    @patch("importlib.util.find_spec")
    def test_prefill_cache_kernel_integration(self, mock_find_spec, mock_is_riscv):
        # Mock sgl_kernel and its ops
        mock_sgl_kernel = MagicMock()
        mock_ops = MagicMock()
        mock_ops.sgl_kernel = mock_sgl_kernel

        # Mock the kernels
        mock_sgl_kernel.decode_attention_cpu = MagicMock()
        mock_sgl_kernel.extend_attention_cpu = MagicMock()
        mock_sgl_kernel.prefill_cache_kernel = MagicMock()

        # Mock sys.modules to prevent ImportError when importing sgl_kernel
        with patch.dict("sys.modules", {"sgl_kernel": MagicMock()}):
            # Setup torch.ops
            with patch.object(torch, "ops", mock_ops):
                # Initialize backend
                backend = RVVAttnBackend(self.mock_runner)

                # Verify detection
                info = backend.get_backend_info()
                self.assertTrue(info["has_prefill_cache"])
                self.assertIsNotNone(backend.prefill_cache_fwd)

                # Setup inputs for forward_prefill
                q = torch.randn(1, 32, 128)
                k = torch.randn(1, 32, 128)
                v = torch.randn(1, 32, 128)
                layer = Mock(spec=RadixAttention)
                layer.layer_id = 0
                layer.qk_head_dim = 128
                layer.v_head_dim = 128
                layer.tp_q_head_num = 32

                forward_batch = Mock(spec=ForwardBatch)
                forward_batch.token_to_kv_pool = Mock()
                forward_batch.token_to_kv_pool.get_key_buffer.return_value = (
                    torch.randn(100, 32, 128)
                )
                forward_batch.token_to_kv_pool.get_value_buffer.return_value = (
                    torch.randn(100, 32, 128)
                )
                forward_batch.out_cache_loc = torch.tensor([1])  # Mock out_cache_loc
                forward_batch.req_pool_indices = torch.tensor([0])
                forward_batch.seq_lens = torch.tensor([10])
                forward_batch.extend_start_loc = torch.tensor([0])
                forward_batch.extend_seq_lens = torch.tensor([10])
                forward_batch.req_to_token_pool = Mock()
                forward_batch.req_to_token_pool.req_to_token = torch.zeros(
                    (1, 100), dtype=torch.int64
                )

                # Mock forward_extend to return something
                backend.forward_extend = Mock(return_value=torch.randn(1, 32, 128))

                # Run forward_prefill
                backend.forward_prefill(
                    q, k, v, layer, forward_batch, save_kv_cache=True
                )

                # Verify prefill_cache_kernel was called
                backend.prefill_cache_fwd.assert_called_once()

                # Verify forward_extend was called with save_kv_cache=False
                backend.forward_extend.assert_called_with(
                    q, k, v, layer, forward_batch, save_kv_cache=False
                )

    @patch("sglang.srt.utils.common.is_host_cpu_riscv", return_value=True)
    @patch("importlib.util.find_spec")
    def test_prefill_cache_kernel_fallback(self, mock_find_spec, mock_is_riscv):
        # Mock sgl_kernel WITHOUT prefill_cache_kernel
        mock_sgl_kernel = MagicMock()
        mock_ops = MagicMock()
        mock_ops.sgl_kernel = mock_sgl_kernel

        mock_sgl_kernel.decode_attention_cpu = MagicMock()
        mock_sgl_kernel.extend_attention_cpu = MagicMock()
        # prefill_cache_kernel is missing
        del mock_sgl_kernel.prefill_cache_kernel

        # Mock sys.modules
        with patch.dict("sys.modules", {"sgl_kernel": MagicMock()}):
            # Setup torch.ops
            with patch.object(torch, "ops", mock_ops):
                # Initialize backend
                backend = RVVAttnBackend(self.mock_runner)

                # Verify detection
                info = backend.get_backend_info()
                self.assertFalse(info["has_prefill_cache"])

                # Setup inputs
                q = torch.randn(1, 32, 128)
                k = torch.randn(1, 32, 128)
                v = torch.randn(1, 32, 128)
                layer = Mock(spec=RadixAttention)
                layer.layer_id = 0
                layer.qk_head_dim = 128
                layer.v_head_dim = 128
                layer.tp_q_head_num = 32

                forward_batch = Mock(spec=ForwardBatch)
                forward_batch.token_to_kv_pool = Mock()
                forward_batch.out_cache_loc = torch.tensor([1])  # Mock out_cache_loc

                # Mock forward_extend
                backend.forward_extend = Mock(return_value=torch.randn(1, 32, 128))

                # Run forward_prefill
                backend.forward_prefill(
                    q, k, v, layer, forward_batch, save_kv_cache=True
                )

                # Verify fallback: set_kv_buffer called
                forward_batch.token_to_kv_pool.set_kv_buffer.assert_called_once()

                # Verify forward_extend called with save_kv_cache=False
                backend.forward_extend.assert_called_with(
                    q, k, v, layer, forward_batch, save_kv_cache=False
                )


if __name__ == "__main__":
    unittest.main()
