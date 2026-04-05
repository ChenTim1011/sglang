"""Unit tests for RVV attention backend registration and integration."""

import unittest
from math import isnan
from types import SimpleNamespace

from sglang.srt.layers.attention.attention_registry import ATTENTION_BACKENDS
from sglang.srt.utils import is_host_cpu_riscv


class _RVVBackendTestMixin:
    """Shared builders for backend tests."""

    def _make_backend_with_rvv_kernels(self, *, is_int8=False):
        """Build an RVVAttnBackend and force RVV mode on for integration tests."""
        from unittest.mock import MagicMock, patch

        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        with patch(
            "sglang.srt.layers.attention.rvv_backend.cpu_has_rvv_support",
            return_value=False,
        ):
            mock_runner = MagicMock()
            mock_runner.device = "cpu"
            mock_runner.model_config.num_attention_heads = 8
            mock_runner.tp_size = 1
            mock_runner.req_to_token_pool.size = 4
            kv_buf = MagicMock()
            kv_buf.shape = [16, 1, 64]
            kv_buf.dtype = torch.int8 if is_int8 else torch.bfloat16
            mock_runner.token_to_kv_pool.get_key_buffer.return_value = kv_buf
            mock_runner.token_to_kv_pool.get_value_buffer.return_value = MagicMock(
                shape=[16, 1, 64]
            )
            backend = RVVAttnBackend(mock_runner)

        backend.use_rvv_kernels = True
        backend.is_int8 = is_int8
        return backend

    def _make_forward_batch(self):
        from unittest.mock import MagicMock

        import torch

        batch = MagicMock()
        batch.token_to_kv_pool = MagicMock()
        batch.token_to_kv_pool.get_key_buffer.return_value = torch.randn(16, 1, 64)
        batch.token_to_kv_pool.get_value_buffer.return_value = torch.randn(16, 1, 64)
        batch.req_to_token_pool.req_to_token = torch.arange(8, dtype=torch.int32).view(
            2, 4
        )
        batch.req_pool_indices = torch.tensor([0, 1], dtype=torch.int64)
        batch.seq_lens = torch.tensor([4, 4], dtype=torch.int64)
        batch.extend_seq_lens = torch.tensor([2, 2], dtype=torch.int64)
        batch.extend_start_loc = torch.tensor([0, 2], dtype=torch.int64)
        batch.out_cache_loc = torch.tensor([3, 7], dtype=torch.int64)
        batch.encoder_out_cache_loc = torch.tensor([5, 9], dtype=torch.int64)
        return batch

    def _make_layer(self, *, is_cross_attention=False):
        from unittest.mock import MagicMock

        layer = MagicMock()
        layer.is_cross_attention = is_cross_attention
        layer.tp_q_head_num = 8
        layer.qk_head_dim = 64
        layer.v_head_dim = 64
        layer.layer_id = 0
        layer.scaling = 0.125
        layer.logit_cap = 0.0
        return layer


class TestRVVBackendInitAndRegistration(unittest.TestCase, _RVVBackendTestMixin):
    """Init and registry tests."""

    def test_registry_structure(self):
        """RVV backend must be registered so dispatchers can look it up by name."""
        self.assertIsInstance(ATTENTION_BACKENDS, dict)
        self.assertIn("rvv", ATTENTION_BACKENDS)

    def test_backend_selection(self):
        """Backend selection must match runtime platform capabilities (RVV or fallback)."""
        from unittest.mock import Mock

        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
        from sglang.srt.utils.common import cpu_has_rvv_support

        mock_runner = Mock()
        mock_runner.device = "cpu"
        mock_runner.model_config = Mock()
        mock_runner.model_config.num_attention_heads = 32
        mock_runner.tp_size = 1
        mock_runner.req_to_token_pool = Mock()
        mock_runner.req_to_token_pool.size = 4
        mock_runner.token_to_kv_pool = Mock()
        kv_buf = Mock()
        kv_buf.shape = [16, 1, 128]
        kv_buf.dtype = torch.bfloat16
        mock_runner.token_to_kv_pool.get_key_buffer = Mock(return_value=kv_buf)
        mock_runner.token_to_kv_pool.get_value_buffer = Mock(
            return_value=Mock(shape=[16, 1, 128])
        )

        backend = RVVAttnBackend(mock_runner)

        if is_host_cpu_riscv():
            if cpu_has_rvv_support():
                self.assertTrue(backend.use_rvv_kernels)
            else:
                self.assertIsNotNone(backend.fallback_backend)
        else:
            self.assertIsNotNone(backend.fallback_backend)

    def test_int8_init_falls_back_when_kernels_missing(self):
        """INT8 KV cache should fall back if RVV INT8 kernels are unavailable."""
        from unittest.mock import MagicMock, patch

        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        mock_runner = MagicMock()
        mock_runner.device = "cpu"
        mock_runner.model_config.num_attention_heads = 8
        mock_runner.tp_size = 1
        mock_runner.req_to_token_pool.size = 4
        kv_buf = MagicMock()
        kv_buf.shape = [16, 1, 64]
        kv_buf.dtype = torch.int8
        mock_runner.token_to_kv_pool.get_key_buffer.return_value = kv_buf
        mock_runner.token_to_kv_pool.get_value_buffer.return_value = MagicMock(
            shape=[16, 1, 64]
        )

        with patch(
            "sglang.srt.layers.attention.rvv_backend.cpu_has_rvv_support",
            return_value=True,
        ), patch(
            "sglang.srt.layers.attention.rvv_backend.torch.ops",
            new=SimpleNamespace(sgl_kernel=SimpleNamespace()),
        ):
            backend = RVVAttnBackend(mock_runner)

        self.assertTrue(backend.is_int8)
        self.assertFalse(backend.use_rvv_kernels)
        self.assertIsNotNone(backend.fallback_backend)

    def test_int8_requested_without_rvv_support_falls_back(self):
        """If int8 KV cache is requested, lack of RVV support should fall back."""
        from unittest.mock import MagicMock, patch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        mock_runner = MagicMock()
        mock_runner.server_args = MagicMock(kv_cache_dtype="int8")

        with patch(
            "sglang.srt.layers.attention.rvv_backend.cpu_has_rvv_support",
            return_value=False,
        ):
            backend = RVVAttnBackend(mock_runner)

        self.assertFalse(backend.use_rvv_kernels)
        self.assertIsNotNone(backend.fallback_backend)


class TestRVVForwardMetadataGuards(unittest.TestCase, _RVVBackendTestMixin):
    """Forward-metadata guard behavior."""

    def test_metadata_rejects_oversized_batch(self):
        """Batch larger than the pool must fail loudly instead of silently slicing OOB."""
        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend._attn_logits_pool = torch.empty(2, 8, 2, 65)

        forward_batch = SimpleNamespace(
            batch_size=3,
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: True),
            extend_seq_lens=None,
        )

        with self.assertRaisesRegex(RuntimeError, "exceeds pre-allocated pool size"):
            backend.init_forward_metadata(forward_batch)

    def test_metadata_tracks_max_extend_len(self):
        """max_extend_len must be cached so extend kernels know their iteration bound."""
        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend._attn_logits_pool = torch.empty(4, 8, 2, 65)

        forward_batch = SimpleNamespace(
            batch_size=2,
            forward_mode=SimpleNamespace(is_decode_or_idle=lambda: False),
            extend_seq_lens=torch.tensor([3, 7], dtype=torch.int64),
        )

        backend.init_forward_metadata(forward_batch)

        _, max_extend_len = backend.forward_metadata
        self.assertEqual(max_extend_len, 7)


class TestRVVBackendFallbackSemantics(unittest.TestCase, _RVVBackendTestMixin):
    """Fallback behavior and non-regression semantics."""

    def test_decode_no_save_uses_fallback(self):
        """save_kv_cache=False bypasses RVV; must delegate to TorchNative fallback."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()

        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)
        layer = self._make_layer()
        batch = MagicMock()

        sentinel = object()
        backend.fallback_backend.forward_decode = MagicMock(return_value=sentinel)

        result = backend.forward_decode(q, k, v, layer, batch, save_kv_cache=False)

        backend.fallback_backend.forward_decode.assert_called_once_with(
            q, k, v, layer, batch, False
        )
        self.assertIs(result, sentinel)

    def test_int8_extend_no_save_uses_fallback(self):
        """INT8 extend with save_kv_cache=False must delegate to TorchNative fallback."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels(is_int8=True)

        q = torch.randn(2, 8, 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)
        layer = self._make_layer()
        batch = MagicMock()

        sentinel = object()
        backend.fallback_backend.forward_extend = MagicMock(return_value=sentinel)

        result = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False)

        backend.fallback_backend.forward_extend.assert_called_once_with(
            q, k, v, layer, batch, False
        )
        self.assertIs(result, sentinel)

    def test_cross_attn_uses_fallback(self):
        """Cross-attention extend must never hit the RVV kernel path."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend.extend_fwd_impl = MagicMock()

        q = torch.randn(4, 8 * 64)
        k = torch.randn(4, 1, 64)
        v = torch.randn(4, 1, 64)
        layer = self._make_layer(is_cross_attention=True)
        batch = self._make_forward_batch()

        sentinel = object()
        backend.fallback_backend.forward_extend = MagicMock(return_value=sentinel)

        result = backend.forward_extend(q, k, v, layer, batch, True)

        backend.fallback_backend.forward_extend.assert_called_once_with(
            q, k, v, layer, batch, True
        )
        backend.extend_fwd_impl.assert_not_called()
        self.assertIs(result, sentinel)


class TestRVVBackendCacheWiring(unittest.TestCase, _RVVBackendTestMixin):
    """Cache location and metadata wiring tests."""

    def test_decode_cross_attn_uses_encoder_loc(self):
        """Cross-attention decode must use encoder_out_cache_loc, not out_cache_loc."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend.decode_fwd_impl = MagicMock()
        backend.forward_metadata = (torch.empty(2, 8, 2, 65), 0)

        q = torch.randn(2, 8 * 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)
        layer = self._make_layer(is_cross_attention=True)
        batch = self._make_forward_batch()

        backend.forward_decode(q, k, v, layer, batch, save_kv_cache=True)

        args = backend.decode_fwd_impl.call_args.args
        self.assertTrue(torch.equal(args[6], batch.encoder_out_cache_loc))

    def test_decode_self_attn_uses_out_loc(self):
        """Self-attention decode must use out_cache_loc, not the encoder variant."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend.decode_fwd_impl = MagicMock()
        backend.forward_metadata = (torch.empty(2, 8, 2, 65), 0)

        q = torch.randn(2, 8 * 64)
        k = torch.randn(2, 1, 64)
        v = torch.randn(2, 1, 64)
        layer = self._make_layer(is_cross_attention=False)
        batch = self._make_forward_batch()

        backend.forward_decode(q, k, v, layer, batch, save_kv_cache=True)

        args = backend.decode_fwd_impl.call_args.args
        self.assertTrue(torch.equal(args[6], batch.out_cache_loc))

    def test_extend_no_save_skips_cache(self):
        """save_kv_cache=False must skip set_kv_buffer entirely; no stale writes."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend.extend_fwd_impl = MagicMock()
        backend.forward_metadata = (None, 2)

        q = torch.randn(4, 8 * 64)
        k = torch.randn(4, 1, 64)
        v = torch.randn(4, 1, 64)
        layer = self._make_layer()
        batch = self._make_forward_batch()

        out = backend.forward_extend(q, k, v, layer, batch, save_kv_cache=False)

        batch.token_to_kv_pool.set_kv_buffer.assert_not_called()
        backend.extend_fwd_impl.assert_called_once()
        self.assertEqual(tuple(out.shape), (4, 8 * 64))

    def test_extend_save_writes_cache(self):
        """save_kv_cache=True must call set_kv_buffer exactly once per extend step."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels()
        backend.extend_fwd_impl = MagicMock()
        backend.forward_metadata = (None, 2)

        q = torch.randn(4, 8 * 64)
        k = torch.randn(4, 1, 64)
        v = torch.randn(4, 1, 64)
        layer = self._make_layer()
        batch = self._make_forward_batch()

        backend.forward_extend(q, k, v, layer, batch, save_kv_cache=True)

        batch.token_to_kv_pool.set_kv_buffer.assert_called_once_with(
            layer, batch.out_cache_loc, k, v
        )
        backend.extend_fwd_impl.assert_called_once()


class TestRVVBackendInt8ScalePlumbing(unittest.TestCase, _RVVBackendTestMixin):
    """INT8 scale resolution and buffer routing tests."""

    def test_missing_scales_use_nan_sentinel(self):
        """FP K/V with no layer scales must yield NaN sentinel, not raise."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=None,
            v_scale=None,
            k_scale_float=None,
            v_scale_float=None,
        )

        RVVAttnBackend._ensure_cached_scales(
            layer,
            torch.randn(1, 1, 8, dtype=torch.float16),
            torch.randn(1, 1, 8, dtype=torch.float16),
        )
        self.assertTrue(isnan(layer._cached_k_scale_float))
        self.assertTrue(isnan(layer._cached_v_scale_float))

    def test_missing_scales_reject_prequant_kv(self):
        """INT8 K/V without explicit scales must raise, not silently use a wrong dequant value."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=None,
            v_scale=None,
            k_scale_float=None,
            v_scale_float=None,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "pre-quantized INT8 K/V inputs require explicit dequantization scales",
        ):
            RVVAttnBackend._ensure_cached_scales(
                layer,
                torch.randint(0, 10, (1, 1, 8), dtype=torch.int8),
                torch.randint(0, 10, (1, 1, 8), dtype=torch.int8),
            )

    def test_partial_layer_scales_raise(self):
        """INT8 path must reject partially missing KV scales."""
        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=None,
            v_scale=None,
            k_scale_float=0.25,
            v_scale_float=None,
        )

        with self.assertRaisesRegex(RuntimeError, "Inconsistent INT8 KV-cache scales"):
            RVVAttnBackend._ensure_cached_scales(layer)

    def test_explicit_scales_cached(self):
        """Explicit tensor/float scales must be cached and not fallback to sentinel."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=torch.tensor(0.25),
            v_scale=None,
            k_scale_float=None,
            v_scale_float=0.5,
        )

        RVVAttnBackend._ensure_cached_scales(layer)

        self.assertAlmostEqual(layer._cached_k_scale_float, 0.25, places=6)
        self.assertAlmostEqual(layer._cached_v_scale_float, 0.5, places=6)

    def test_scales_refresh_on_change(self):
        """Updated layer scales must invalidate cached floats; stale values would corrupt dequant."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=torch.tensor(0.25),
            v_scale=torch.tensor(0.5),
            k_scale_float=None,
            v_scale_float=None,
        )

        RVVAttnBackend._ensure_cached_scales(layer)
        layer.k_scale = torch.tensor(0.75)
        layer.v_scale = torch.tensor(1.25)
        RVVAttnBackend._ensure_cached_scales(layer)

        self.assertAlmostEqual(layer._cached_k_scale_float, 0.75, places=6)
        self.assertAlmostEqual(layer._cached_v_scale_float, 1.25, places=6)

    def test_nan_cache_does_not_mask_int8(self):
        """NaN sentinel cached for FP K/V must not suppress the missing-scale error for INT8 K/V."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=None,
            v_scale=None,
            k_scale_float=None,
            v_scale_float=None,
        )

        RVVAttnBackend._ensure_cached_scales(
            layer,
            torch.randn(1, 1, 8, dtype=torch.float16),
            torch.randn(1, 1, 8, dtype=torch.float16),
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "pre-quantized INT8 K/V inputs require explicit dequantization scales",
        ):
            RVVAttnBackend._ensure_cached_scales(
                layer,
                torch.randint(0, 10, (1, 1, 8), dtype=torch.int8),
                torch.randint(0, 10, (1, 1, 8), dtype=torch.int8),
            )

    def test_default_scale_1_uses_dynamic_quantization(self):
        """k_scale=v_scale=1.0 with FP inputs is the BaseKVCacheMethod fallback (no
        calibration data in checkpoint).  The RVV backend must switch to dynamic
        per-token quantization (NaN sentinel) instead of using the 1.0 placeholder,
        which would round all sub-integer activation values to zero.
        """
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=torch.tensor(1.0),
            v_scale=torch.tensor(1.0),
            k_scale_float=1.0,
            v_scale_float=1.0,
        )

        RVVAttnBackend._ensure_cached_scales(
            layer,
            torch.randn(1, 1, 8, dtype=torch.bfloat16),
            torch.randn(1, 1, 8, dtype=torch.bfloat16),
        )
        self.assertTrue(
            isnan(layer._cached_k_scale_float),
            "k_scale=1.0 with FP inputs must switch to NaN dynamic path",
        )
        self.assertTrue(
            isnan(layer._cached_v_scale_float),
            "v_scale=1.0 with FP inputs must switch to NaN dynamic path",
        )

    def test_real_calibrated_scale_not_overridden(self):
        """A non-1.0 scale from a real calibrated checkpoint must be used as-is."""
        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        layer = SimpleNamespace(
            k_scale=torch.tensor(0.08),
            v_scale=torch.tensor(0.12),
            k_scale_float=None,
            v_scale_float=None,
        )

        RVVAttnBackend._ensure_cached_scales(
            layer,
            torch.randn(1, 1, 8, dtype=torch.bfloat16),
            torch.randn(1, 1, 8, dtype=torch.bfloat16),
        )
        self.assertAlmostEqual(layer._cached_k_scale_float, 0.08, places=5)
        self.assertAlmostEqual(layer._cached_v_scale_float, 0.12, places=5)

    def test_scale_buffers_sized_by_hidden_layers(self):
        """Scale buffers must be sized by num_hidden_layers so each layer gets its own row."""
        from unittest.mock import MagicMock, patch

        import torch

        from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend

        mock_runner = MagicMock()
        mock_runner.device = "cpu"
        mock_runner.model_config.num_attention_heads = 8
        mock_runner.model_config.num_hidden_layers = 48
        mock_runner.tp_size = 1
        mock_runner.req_to_token_pool.size = 4
        kv_buf = MagicMock()
        kv_buf.shape = [16, 2, 64]
        kv_buf.dtype = torch.int8
        mock_runner.token_to_kv_pool.get_key_buffer.return_value = kv_buf
        mock_runner.token_to_kv_pool.get_value_buffer.return_value = MagicMock(
            shape=[16, 2, 64]
        )

        ops = SimpleNamespace(
            decode_attention_int8_cpu=object(),
            extend_attention_int8_cpu=object(),
        )

        with patch(
            "sglang.srt.layers.attention.rvv_backend.cpu_has_rvv_support",
            return_value=True,
        ), patch(
            "sglang.srt.layers.attention.rvv_backend.torch.ops",
            new=SimpleNamespace(sgl_kernel=ops),
        ):
            backend = RVVAttnBackend(mock_runner)

        self.assertTrue(backend.use_rvv_kernels)
        self.assertEqual(tuple(backend._k_scale_buf.shape), (48, 16, 2))
        self.assertEqual(tuple(backend._v_scale_buf.shape), (48, 16, 2))
        self.assertEqual(backend._scale_buf_index(SimpleNamespace(layer_id=3)), 3)

    def test_decode_int8_passes_cached_scales(self):
        """INT8 decode must pass scale buffers and cached scalar scales to the kernel."""
        from unittest.mock import MagicMock

        import torch

        backend = self._make_backend_with_rvv_kernels(is_int8=True)
        backend.decode_fwd_impl = MagicMock()
        backend._k_scale_buf = torch.ones(1, 16, 1, dtype=torch.float32)
        backend._v_scale_buf = torch.full((1, 16, 1), 2.0, dtype=torch.float32)
        backend.forward_metadata = (torch.empty(2, 8, 2, 65), 0)

        q = torch.randn(2, 8 * 64)
        k = torch.randint(-8, 8, (2, 1, 64), dtype=torch.int8)
        v = torch.randint(-8, 8, (2, 1, 64), dtype=torch.int8)
        layer = self._make_layer()
        layer.k_scale = torch.tensor(0.25)
        layer.v_scale = torch.tensor(0.5)
        batch = self._make_forward_batch()

        backend.forward_decode(q, k, v, layer, batch, save_kv_cache=True)

        args = backend.decode_fwd_impl.call_args.args
        self.assertTrue(torch.equal(args[13], backend._k_scale_buf[0]))
        self.assertTrue(torch.equal(args[14], backend._v_scale_buf[0]))
        self.assertEqual(args[15], 0.25)
        self.assertEqual(args[16], 0.5)


class TestRVVLMHeadPacking(unittest.TestCase):
    """LM-head RVV packing behavior."""

    def test_non_cpu_weights_skip_packing(self):
        from unittest.mock import patch

        import torch

        from sglang.srt.layers import rvv_utils

        module = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.empty((2, 2), device="meta", dtype=torch.float16),
                requires_grad=False,
            ),
            bias=torch.nn.Parameter(
                torch.empty(2, device="meta", dtype=torch.float16),
                requires_grad=False,
            ),
        )

        with patch.object(
            rvv_utils, "cpu_has_rvv_support", return_value=True
        ), patch.object(rvv_utils, "_get_convert_weight_packed_op") as mock_get_op:
            rvv_utils._rvv_process_weight_after_loading(module, ["weight"])

        mock_get_op.assert_not_called()
        self.assertFalse(hasattr(module, "use_riscv_rvv_backend"))
        self.assertEqual(module.weight.device.type, "meta")
        self.assertEqual(module.bias.device.type, "meta")

    def test_lm_head_lazy_pack_tracks_updates(self):
        from unittest.mock import patch

        import torch

        from sglang.srt.layers import rvv_utils

        pack_calls = []

        def fake_convert(weight):
            pack_calls.append(weight.clone())
            return weight + 10

        lm_head = SimpleNamespace(
            weight=torch.nn.Parameter(
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16)
            )
        )

        with patch.object(
            rvv_utils, "cpu_has_rvv_support", return_value=True
        ), patch.object(rvv_utils, "_convert_weight_packed", side_effect=fake_convert):
            packed_1 = rvv_utils.resolve_rvv_lm_head_weight(lm_head)
            packed_2 = rvv_utils.resolve_rvv_lm_head_weight(lm_head)
            self.assertEqual(len(pack_calls), 1)
            self.assertTrue(torch.equal(packed_1, packed_2))

            with torch.no_grad():
                lm_head.weight.copy_(
                    torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16)
                )

            packed_3 = rvv_utils.resolve_rvv_lm_head_weight(lm_head)

        self.assertEqual(len(pack_calls), 2)
        self.assertTrue(
            torch.equal(
                pack_calls[0],
                torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float16),
            )
        )
        self.assertTrue(
            torch.equal(
                pack_calls[1],
                torch.tensor([[5.0, 6.0], [7.0, 8.0]], dtype=torch.float16),
            )
        )
        self.assertTrue(
            torch.equal(
                packed_3,
                torch.tensor([[15.0, 16.0], [17.0, 18.0]], dtype=torch.float16),
            )
        )
        self.assertEqual(
            lm_head._rvv_lm_head_packed_source_sig[-1], lm_head.weight._version
        )

    def test_float32_lm_head_disables_backend(self):
        from unittest.mock import patch

        import torch

        from sglang.srt.layers import rvv_utils

        lm_head = SimpleNamespace(
            weight=torch.nn.Parameter(torch.randn(2, 2, dtype=torch.float32))
        )

        with patch.object(
            rvv_utils, "cpu_has_rvv_support", return_value=True
        ), patch.object(rvv_utils, "_get_convert_weight_packed_op") as mock_get_op:
            self.assertFalse(rvv_utils.use_rvv_lm_head_backend(lm_head))
            with self.assertRaises(RuntimeError):
                rvv_utils.resolve_rvv_lm_head_weight(lm_head)

        mock_get_op.assert_not_called()

    def test_lora_lm_head_disables_backend(self):
        from unittest.mock import patch

        from sglang.srt.layers import rvv_utils

        lm_head = SimpleNamespace(
            weight=object(),
            set_lora=True,
            apply_lora=lambda *args, **kwargs: None,
        )

        with patch.object(
            rvv_utils, "cpu_has_rvv_support", return_value=True
        ), patch.object(rvv_utils, "_convert_weight_packed", return_value=object()):
            self.assertFalse(rvv_utils.use_rvv_lm_head_backend(lm_head))
            with self.assertRaises(RuntimeError):
                rvv_utils.resolve_rvv_lm_head_weight(lm_head)


if __name__ == "__main__":
    unittest.main()
