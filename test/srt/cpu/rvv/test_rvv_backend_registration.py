"""
Test RISC-V attention backend registration and runtime behavior.

Verifies:
- Backend registration in ATTENTION_BACKENDS
- Correct backend selection on RISC-V vs non-RISC-V platforms
- Runtime fallback behavior when sgl_kernel unavailable
"""

import platform
import unittest

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS


class TestRVVBackend(unittest.TestCase):
    """Test RVV backend registration and selection."""

    def test_backend_registry_structure(self):
        """Verify ATTENTION_BACKENDS has correct structure."""
        self.assertIsInstance(ATTENTION_BACKENDS, dict)
        self.assertIn("rvv", ATTENTION_BACKENDS)

    def test_backend_selection_on_current_platform(self):
        """Test backend selection on actual hardware platform."""
        import importlib.util
        from unittest.mock import Mock

        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        mock_runner = Mock()
        mock_runner.device = "cpu"
        mock_runner.model_config = Mock()
        mock_runner.model_config.num_attention_heads = 32
        mock_runner.tp_size = 1
        mock_runner.req_to_token_pool = Mock()
        mock_runner.req_to_token_pool.size = 4
        mock_runner.token_to_kv_pool = Mock()
        mock_runner.token_to_kv_pool.full_attention_layer_id_mapping = {0: 0}
        _kv_buf = Mock()
        _kv_buf.shape = [16, 1, 128]
        _kv_buf.dtype = torch.bfloat16
        mock_runner.token_to_kv_pool.get_key_buffer = Mock(return_value=_kv_buf)
        mock_runner.token_to_kv_pool.get_value_buffer = Mock(
            return_value=Mock(shape=[16, 1, 128])
        )

        backend = RVVAttnBackend(mock_runner)

        # On RISC-V with sgl_kernel: expect use_rvv_kernels=True
        # Otherwise: expect fallback_backend to exist
        if platform.machine().lower() in ("riscv64", "riscv32"):
            if importlib.util.find_spec("sgl_kernel") is not None:
                self.assertTrue(backend.use_rvv_kernels)
            else:
                self.assertIsNotNone(backend.fallback_backend)
        else:
            self.assertIsNotNone(backend.fallback_backend)


if __name__ == "__main__":
    unittest.main()
