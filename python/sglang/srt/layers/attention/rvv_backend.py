from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.utils import is_host_cpu_riscv

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class RVVAttnBackend(AttentionBackend):
    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.model_runner = model_runner
        self.device = "cpu"
        self.is_riscv = is_host_cpu_riscv()
        self.use_rvv_kernels = False

        # Fallback
        self.fallback_backend = TorchNativeAttnBackend(model_runner)

        if self.is_riscv:
            self._try_init_rvv_kernels()

    def _try_init_rvv_kernels(self):
        try:
            ops = torch.ops.sgl_kernel

            # 1. Detect Hardware (optional — log only)
            if hasattr(ops, "get_rvv_vlenb"):
                self.vlenb = ops.get_rvv_vlenb()
                logger.info(f"[RVV] Hardware VLEN={self.vlenb * 8}bits detected.")

            # 2. Select Kernel Implementation
            pool = self.model_runner.token_to_kv_pool

            layer_id = 0
            if hasattr(pool, "full_attention_layer_id_mapping"):
                layer_id = next(iter(pool.full_attention_layer_id_mapping))

            k_buffer = pool.get_key_buffer(layer_id)
            self.is_int8 = k_buffer.dtype in (torch.int8, torch.uint8)

            if self.is_int8:
                if hasattr(ops, "decode_attention_int8_cpu") and hasattr(
                    ops, "extend_attention_int8_cpu"
                ):
                    self.decode_fwd_impl = ops.decode_attention_int8_cpu
                    self.extend_fwd_impl = ops.extend_attention_int8_cpu
                    self.use_rvv_kernels = True
            else:
                if hasattr(ops, "decode_attention_cpu") and hasattr(
                    ops, "extend_attention_cpu"
                ):
                    self.decode_fwd_impl = ops.decode_attention_cpu
                    self.extend_fwd_impl = ops.extend_attention_cpu
                    self.use_rvv_kernels = True

            if self.use_rvv_kernels:
                logger.info(
                    f"[RVV] Initialized. Mode={'INT8' if self.is_int8 else 'FLOAT'}"
                )

        except Exception as e:
            logger.warning(f"[RVV] Init failed: {e}. Fallback to Native.")

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not self.use_rvv_kernels:
            return self.fallback_backend.init_forward_metadata(forward_batch)

        max_extend_len = 0
        if not forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        self.forward_metadata = max_extend_len

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not self.use_rvv_kernels:
            return self.fallback_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        pool = forward_batch.token_to_kv_pool

        # 1. Compute cache loc (handles cross-attention)
        loc = (
            forward_batch.encoder_out_cache_loc
            if (
                layer.is_cross_attention
                and forward_batch.encoder_out_cache_loc is not None
            )
            else forward_batch.out_cache_loc
        )

        # 2. Save KV
        if save_kv_cache and not self.is_int8:
            pool.set_kv_buffer(layer, loc, k, v)

        # 3. Prepare Output
        current_bs = q.shape[0]
        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        attn_logits = torch.empty(
            (current_bs, layer.tp_q_head_num, 4, layer.v_head_dim + 1),
            dtype=torch.float32,
            device="cpu",
        )

        if self.is_int8:
            k_scale = float(
                layer.k_scale.item()
                if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor)
                else getattr(layer, "k_scale_float", 1.0)
            )
            v_scale = float(
                layer.v_scale.item()
                if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor)
                else getattr(layer, "v_scale_float", 1.0)
            )
            self.decode_fwd_impl(
                q_view,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
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
                k_scale,
                v_scale,
            )
        else:
            self.decode_fwd_impl(
                q_view,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
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
        return o

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not self.use_rvv_kernels:
            return self.fallback_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

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

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        max_extend_len = self.forward_metadata

        if self.is_int8:
            k_scale = float(
                layer.k_scale.item()
                if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor)
                else getattr(layer, "k_scale_float", 1.0)
            )
            v_scale = float(
                layer.v_scale.item()
                if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor)
                else getattr(layer, "v_scale_float", 1.0)
            )
            self.extend_fwd_impl(
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
                k_scale,
                v_scale,
            )
        else:
            self.extend_fwd_impl(
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
        return o

    def support_triton(self):
        return False
