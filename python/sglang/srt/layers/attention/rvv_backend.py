from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_host_cpu_riscv

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

_NUM_KV_SPLITS = 8


class RVVAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.model_runner = model_runner
        self.device = "cpu"
        self.is_riscv = is_host_cpu_riscv()
        self.use_rvv_kernels = False
        self.num_head = 0
        self.v_head_dim = 0

        self._fallback_backend = None

        if self.is_riscv:
            self._try_init_rvv_kernels()

    @property
    def fallback_backend(self):
        if self._fallback_backend is None:
            self._fallback_backend = _lazy_import_torch_native()(self.model_runner)
        return self._fallback_backend

    def _try_init_rvv_kernels(self):
        try:
            ops = torch.ops.sgl_kernel

            # 1. Detect Hardware (optional — log only)
            try:
                self.vlenb = ops.get_rvv_vlenb()
                logger.info(f"[RVV] Hardware VLEN={self.vlenb * 8}bits detected.")
            except AttributeError:
                pass

            # 2. Read model shape for attn_logits pre-allocation
            pool = self.model_runner.token_to_kv_pool
            layer_id = 0
            if hasattr(pool, "full_attention_layer_id_mapping"):
                layer_id = next(iter(pool.full_attention_layer_id_mapping))

            self.num_head = (
                self.model_runner.model_config.num_attention_heads
                // self.model_runner.tp_size
            )
            self.v_head_dim = pool.get_value_buffer(layer_id).shape[-1]

            # 3. Select Kernel Implementation
            k_buffer = pool.get_key_buffer(layer_id)
            self.is_int8 = k_buffer.dtype in (torch.int8, torch.uint8)

            if self.is_int8:
                try:
                    self.decode_fwd_impl = ops.decode_attention_int8_cpu
                    self.extend_fwd_impl = ops.extend_attention_int8_cpu
                    self.use_rvv_kernels = True
                except AttributeError:
                    pass
            else:
                try:
                    self.decode_fwd_impl = ops.decode_attention_cpu
                    self.extend_fwd_impl = ops.extend_attention_cpu
                    self.use_rvv_kernels = True
                except AttributeError:
                    pass

            if self.use_rvv_kernels:
                # 4. Pre-allocate attn_logits buffer
                max_bs = self.model_runner.req_to_token_pool.size
                self._attn_logits_pool = torch.empty(
                    (max_bs, self.num_head, _NUM_KV_SPLITS, self.v_head_dim + 1),
                    dtype=torch.float32,
                    device="cpu",
                )
                logger.info(
                    f"[RVV] Initialized. Mode={'INT8' if self.is_int8 else 'FLOAT'}"
                )

        except Exception as e:
            logger.warning(f"[RVV] Init failed: {e}. Fallback to Native.")

    @staticmethod
    def _ensure_cached_scales(layer: RadixAttention):
        """Cache quantization scales on first call to avoid repeated .item() calls."""
        cached = getattr(layer, "_cached_k_scale_float", None)
        if cached is not None:
            return
        k_scale_src = getattr(layer, "k_scale", None)
        v_scale_src = getattr(layer, "v_scale", None)
        layer._cached_k_scale_float = float(
            k_scale_src.item()
            if isinstance(k_scale_src, torch.Tensor)
            else getattr(layer, "k_scale_float", 1.0)
        )
        layer._cached_v_scale_float = float(
            v_scale_src.item()
            if isinstance(v_scale_src, torch.Tensor)
            else getattr(layer, "v_scale_float", 1.0)
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not self.use_rvv_kernels:
            return self.fallback_backend.init_forward_metadata(forward_batch)

        # Reuse pre-allocated buffer (zero-alloc slice)
        bs = forward_batch.batch_size
        attn_logits = self._attn_logits_pool[:bs]

        max_extend_len = 0
        if not forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = forward_batch.extend_seq_lens.max().item()

        self.forward_metadata = (attn_logits, max_extend_len)

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
            # 1. Compute cache loc (handles cross-attention)
            loc = (
                forward_batch.encoder_out_cache_loc
                if (
                    layer.is_cross_attention
                    and forward_batch.encoder_out_cache_loc is not None
                )
                else forward_batch.out_cache_loc
            )

            # 2. Prepare Output
            current_bs = q.shape[0]
            q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
            o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            attn_logits, _ = self.forward_metadata

            # 3. Build args and dispatch (unified INT8/FP path)
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
                self._ensure_cached_scales(layer)
                args += (layer._cached_k_scale_float, layer._cached_v_scale_float)

            self.decode_fwd_impl(*args)
            return o

        # Fallback path
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
            pool = forward_batch.token_to_kv_pool

            if save_kv_cache and not self.is_int8:
                loc = (
                    forward_batch.encoder_out_cache_loc
                    if (
                        layer.is_cross_attention
                        and forward_batch.encoder_out_cache_loc is not None
                    )
                    else forward_batch.out_cache_loc
                )
                pool.set_kv_buffer(layer, loc, k, v)

            # Output Setup
            current_bs = q.shape[0]
            q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
            o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

            _, max_extend_len = self.forward_metadata

            # Build args and dispatch (unified INT8/FP path)
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
                self._ensure_cached_scales(layer)
                args += (layer._cached_k_scale_float, layer._cached_v_scale_float)

            self.extend_fwd_impl(*args)
            return o

        # Fallback path
        return self.fallback_backend.forward_extend(
            q, k, v, layer, forward_batch, save_kv_cache
        )

    def support_triton(self):
        return False


def _lazy_import_torch_native():
    """Import TorchNativeAttnBackend only when fallback is actually needed."""
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    return TorchNativeAttnBackend
