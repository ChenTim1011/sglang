from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils.common import cpu_has_rvv_support

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Number of KV splits for the decode kernel's partial-softmax reduction.
_NUM_KV_SPLITS = 2


class RVVAttnBackend(AttentionBackend):
    """Attention backend for RISC-V Vector Extension (RVV).

    Known limitations on RISC-V (features disabled via CPU_CAPABILITY_RVV guards):
      - No MoE support (fused_experts_cpu excluded from RVV build)
      - No Flash Attention (flash_attn_varlen_func unavailable)
      - No INT4 quantization (int4_scaled_mm_cpu excluded)
      - No shared memory transport (SGLANG_RISCV_NO_SHM)
      - No NUMA binding (SGLANG_RISCV_NO_NUMA)
      - No torch.compile (disabled for RISC-V)
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.device = "cpu"
        self.use_rvv_kernels = False
        self.num_head = 0
        self.v_head_dim = 0

        from sglang.srt.layers.attention.torch_native_backend import (
            TorchNativeAttnBackend,
        )

        self.fallback_backend = TorchNativeAttnBackend(model_runner)

        if cpu_has_rvv_support():
            self._try_init_rvv_kernels(model_runner)

    def _try_init_rvv_kernels(self, model_runner: ModelRunner):
        try:
            ops = torch.ops.sgl_kernel
            pool = model_runner.token_to_kv_pool
            layer_id = 0

            self.num_head = (
                model_runner.model_config.num_attention_heads // model_runner.tp_size
            )
            self.v_head_dim = pool.get_value_buffer(layer_id).shape[-1]

            # Detect INT8 KV cache mode from the buffer dtype.
            k_buffer = pool.get_key_buffer(layer_id)
            self.is_int8 = k_buffer.dtype in (torch.int8, torch.uint8)

            kernels_available = False
            if self.is_int8:
                try:
                    self.decode_fwd_impl = ops.decode_attention_int8_cpu
                    self.extend_fwd_impl = ops.extend_attention_int8_cpu
                    kernels_available = True
                except AttributeError:
                    logger.warning(
                        "[RVV] INT8 attention kernels (decode_attention_int8_cpu / "
                        "extend_attention_int8_cpu) not found in sgl_kernel. "
                        "Falling back to TorchNative. Build with RVV INT8 support to enable."
                    )
            else:
                try:
                    self.decode_fwd_impl = ops.decode_attention_cpu
                    self.extend_fwd_impl = ops.extend_attention_cpu
                    kernels_available = True
                except AttributeError:
                    logger.warning(
                        "[RVV] FP attention kernels (decode_attention_cpu / "
                        "extend_attention_cpu) not found in sgl_kernel. "
                        "Falling back to TorchNative."
                    )

            if kernels_available:
                max_bs = model_runner.req_to_token_pool.size
                self._attn_logits_pool = torch.empty(
                    (max_bs, self.num_head, _NUM_KV_SPLITS, self.v_head_dim + 1),
                    dtype=torch.float32,
                    device="cpu",
                )

                if self.is_int8:
                    max_tokens, num_kv_heads = k_buffer.shape[0], k_buffer.shape[1]
                    num_layers = model_runner.model_config.num_hidden_layers
                    self._k_scale_buf = torch.ones(
                        (num_layers, max_tokens, num_kv_heads),
                        dtype=torch.float32,
                        device="cpu",
                    )
                    self._v_scale_buf = torch.ones(
                        (num_layers, max_tokens, num_kv_heads),
                        dtype=torch.float32,
                        device="cpu",
                    )
                self.use_rvv_kernels = True
                logger.info(
                    f"[RVV] Initialized. Mode={'INT8' if self.is_int8 else 'FLOAT'}"
                )

        except (AttributeError, RuntimeError, MemoryError) as e:
            logger.warning(
                "[RVV] Init failed, falling back to TorchNative. Reason: %s",
                e,
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                "[RVV] Unexpected error during init — this is likely a bug: %s",
                e,
                exc_info=True,
            )
            raise

    def _scale_buf_index(self, layer: RadixAttention) -> int:
        return layer.layer_id

    @staticmethod
    def _ensure_cached_scales(
        layer: RadixAttention,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
    ):
        """Resolve per-layer KV scales for RVV INT8 attention."""

        def _resolve(src_attr: str, fallback_attr: str):
            src = getattr(layer, src_attr, None)
            if isinstance(src, torch.Tensor):
                return src.item(), True
            if isinstance(src, (int, float)):
                return float(src), True

            fallback = getattr(layer, fallback_attr, None)
            if fallback is not None:
                return float(fallback), True
            return None, False

        key_is_quantized = key is not None and key.dtype in (torch.int8, torch.uint8)
        value_is_quantized = value is not None and value.dtype in (
            torch.int8,
            torch.uint8,
        )

        k_scale, has_k_scale = _resolve("k_scale", "k_scale_float")
        v_scale, has_v_scale = _resolve("v_scale", "v_scale_float")

        if has_k_scale != has_v_scale:
            raise RuntimeError(
                "[RVV] Inconsistent INT8 KV-cache scales on "
                f"{type(layer).__name__}: k_scale present={has_k_scale}, "
                f"v_scale present={has_v_scale}. Both scales must be provided "
                "together for RVV INT8 attention."
            )

        if not has_k_scale:
            if key_is_quantized or value_is_quantized:
                raise RuntimeError(
                    "[RVV] Missing k_scale/v_scale on "
                    f"{type(layer).__name__}; pre-quantized INT8 K/V inputs require "
                    "explicit dequantization scales."
                )

            logger.warning_once(
                f"[RVV] Missing k_scale/v_scale on {type(layer).__name__}; using "
                "dynamic per-token quantization sentinel scales (NaN, NaN)."
            )
            k_scale = float("nan")
            v_scale = float("nan")

        layer._cached_k_scale_float, layer._cached_v_scale_float = k_scale, v_scale

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not self.use_rvv_kernels:
            return self.fallback_backend.init_forward_metadata(forward_batch)

        # Zero-allocation slice from the pre-allocated pool.
        bs = forward_batch.batch_size
        if bs > self._attn_logits_pool.shape[0]:
            raise RuntimeError(
                f"[RVV] batch_size {bs} exceeds pre-allocated pool size "
                f"{self._attn_logits_pool.shape[0]}. Re-initialize with a larger pool."
            )
        attn_logits = self._attn_logits_pool[:bs]

        max_extend_len = 0
        if not forward_batch.forward_mode.is_decode_or_idle():
            if forward_batch.extend_seq_lens is not None:
                max_extend_len = forward_batch.extend_seq_lens.max().item()

        self.forward_metadata = (attn_logits, max_extend_len)

    def init_cpu_graph_state(self, max_bs: int, max_num_tokens: int):
        del max_bs, max_num_tokens
        logger.warning_once(
            "[RVV] CPU graph / torch.compile is not supported for the RVV attention "
            "backend yet. Use eager CPU execution or switch to another backend."
        )

    def get_cpu_graph_seq_len_fill_value(self):
        logger.warning_once(
            "[RVV] get_cpu_graph_seq_len_fill_value called on unsupported RVV CPU "
            "graph path; using fill value 1 before the capture path raises."
        )
        return 1

    def init_forward_metadata_capture_cpu_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        del (
            bs,
            num_tokens,
            req_pool_indices,
            seq_lens,
            encoder_lens,
            forward_mode,
            spec_info,
        )
        raise RuntimeError(
            "[RVV] CPU graph / torch.compile is not supported for the RVV attention "
            "backend yet. Disable --enable-torch-compile or use a different "
            "attention backend."
        )

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if self.use_rvv_kernels:
            if not save_kv_cache:
                # The RVV decode kernel always writes K/V to the cache internally.
                # When save_kv_cache=False (speculative decoding verification pass),
                # fall back to TorchNative which correctly skips cache writes.
                return self.fallback_backend.forward_decode(
                    q, k, v, layer, forward_batch, save_kv_cache
                )

            loc = (
                forward_batch.encoder_out_cache_loc
                if (
                    layer.is_cross_attention
                    and forward_batch.encoder_out_cache_loc is not None
                )
                else forward_batch.out_cache_loc
            )

            current_bs = q.shape[0]
            q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
            o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            attn_logits, _ = self.forward_metadata

            args = (
                q_view,
                forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
                forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
                o_view,
                k,
                v,
                loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
            )
            if self.is_int8:
                self._ensure_cached_scales(layer, k, v)
                scale_buf_idx = self._scale_buf_index(layer)
                args += (
                    self._k_scale_buf[scale_buf_idx],
                    self._v_scale_buf[scale_buf_idx],
                )
                args += (layer._cached_k_scale_float, layer._cached_v_scale_float)

            self.decode_fwd_impl(*args)
            return o

        logger.warning_once(
            "[RVV] forward_decode: RVV kernels not active, using TorchNative fallback. "
            "Check startup logs for init failure details."
        )
        return self.fallback_backend.forward_decode(
            q, k, v, layer, forward_batch, save_kv_cache
        )

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if self.use_rvv_kernels:
            # Cross-attention requires Q and K/V from different sequences, which is
            # not supported.  Fall back to TorchNative which handles this correctly.
            if layer.is_cross_attention:
                logger.warning_once(
                    "[RVV] forward_extend: cross-attention is not supported by the RVV "
                    "extend kernel; falling back to TorchNative for cross-attention layers."
                )
                return self.fallback_backend.forward_extend(
                    q, k, v, layer, forward_batch, save_kv_cache
                )

            pool = forward_batch.token_to_kv_pool

            if self.is_int8 and not save_kv_cache:
                # The INT8 extend kernel writes quantized K/V to the cache internally
                # with no way to suppress it.  When save_kv_cache=False (speculative
                # decoding verification pass), fall back to TorchNative.
                return self.fallback_backend.forward_extend(
                    q, k, v, layer, forward_batch, save_kv_cache
                )

            if save_kv_cache and not self.is_int8:
                # FP path: Python writes K/V to the cache pool before the kernel reads it.
                # INT8 path: skipped — extend_attention_int8_cpu writes quantized K/V
                # to the cache internally via extend_set_kv_buffer_int8_quantize.
                pool.set_kv_buffer(layer, forward_batch.out_cache_loc, k, v)

            current_bs = q.shape[0]
            q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
            o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            _, max_extend_len = self.forward_metadata

            args = (
                q_view,
                k,
                v,
                o_view,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
                max_extend_len,
                layer.scaling,
                layer.logit_cap,
            )
            if self.is_int8:
                self._ensure_cached_scales(layer, k, v)
                scale_buf_idx = self._scale_buf_index(layer)
                # Order matches schema: k_scale_buf, v_scale_buf, k_scale, v_scale
                args += (
                    self._k_scale_buf[scale_buf_idx],
                    self._v_scale_buf[scale_buf_idx],
                )
                args += (layer._cached_k_scale_float, layer._cached_v_scale_float)

            self.extend_fwd_impl(*args)
            return o

        logger.warning_once(
            "[RVV] forward_extend: RVV kernels not active, using TorchNative fallback. "
            "Check startup logs for init failure details."
        )
        return self.fallback_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache
        )

    def support_triton(self):
        return False
