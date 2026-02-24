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
            if not hasattr(torch.ops.sgl_kernel, "get_rvv_vlenb"):
                return

            # 1. Detect Hardware
            self.vlenb = torch.ops.sgl_kernel.get_rvv_vlenb()

            # 2. Config & Alignment
            self.num_head = (
                self.model_runner.model_config.num_attention_heads
                // self.model_runner.tp_size
            )
            try:
                self.v_head_dim = self.model_runner.token_to_kv_pool.get_value_buffer(
                    0
                ).shape[-1]
            except:
                self.v_head_dim = self.model_runner.model_config.head_dim

            # K1 Constraint: 256-bit alignment
            config_dtype = getattr(
                self.model_runner.model_config, "dtype", torch.float16
            )
            if not isinstance(config_dtype, torch.dtype):
                config_dtype = torch.float16

            dtype_size = torch.tensor([], dtype=config_dtype).element_size()
            if (self.v_head_dim * dtype_size) % 32 != 0:
                logger.warning(
                    f"[RVV] Head dim {self.v_head_dim} not 32-byte aligned. vlenb={self.vlenb}. Fallback."
                )
                return

            # 3. Select Kernel Implementation
            # Check for INT8 KV cache
            pool = self.model_runner.token_to_kv_pool
            k_buffer = pool.get_key_buffer(0)
            self.is_int8 = k_buffer.dtype in (torch.int8, torch.uint8)

            ops = torch.ops.sgl_kernel
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
                    f"[RVV] Initialized. VLEN={self.vlenb*8}bits. Mode={'INT8' if self.is_int8 else 'FLOAT'}"
                )

        except Exception as e:
            logger.warning(f"[RVV] Init failed: {e}. Fallback to Native.")

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        if not self.use_rvv_kernels:
            return self.fallback_backend.init_forward_metadata(forward_batch)

        bs = forward_batch.batch_size
        self.num_kv_splits = 4

        # Optimize: Contiguous allocation for K1
        # Reverted padding: using v_head_dim + 1 as per current C++ kernel expectation
        attn_logits = torch.empty(
            (bs, self.num_head, self.num_kv_splits, self.v_head_dim + 1),
            dtype=torch.float32,
            device="cpu",
        ).contiguous()

        max_extend_len = 0
        if not forward_batch.forward_mode.is_decode_or_idle():
            max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

        if self.is_int8:
            k_scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")
            v_scale = torch.tensor([1.0], dtype=torch.float32, device="cpu")
            self.forward_metadata = (attn_logits, max_extend_len, k_scale, v_scale)
        else:
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
        if not self.use_rvv_kernels:
            return self.fallback_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        pool = forward_batch.token_to_kv_pool

        # 1. Save KV
        if save_kv_cache and not self.is_int8:
            # Standard optimized path
            loc = (
                forward_batch.encoder_out_cache_loc
                if (
                    layer.is_cross_attention
                    and forward_batch.encoder_out_cache_loc is not None
                )
                else forward_batch.out_cache_loc
            )
            pool.set_kv_buffer(layer, loc, k, v)

        # 2. Prepare Output & Metadata
        current_bs = q.shape[0]
        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty((current_bs, layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)
        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        if self.is_int8:
            attn_logits, _, k_scale_tensor, v_scale_tensor = self.forward_metadata
            # Update scales
            k_scale_tensor[0] = getattr(layer, "k_scale_float", 1.0)
            if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor):
                k_scale_tensor[0] = layer.k_scale.item()

            v_scale_tensor[0] = getattr(layer, "v_scale_float", 1.0)
            if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor):
                v_scale_tensor[0] = layer.v_scale.item()

            self.decode_fwd_impl(
                q_view,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
                o_view,
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits[:current_bs],
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
                k_scale_tensor,
                v_scale_tensor,
            )
        else:
            attn_logits, _ = self.forward_metadata
            self.decode_fwd_impl(
                q_view,
                pool.get_key_buffer(layer.layer_id),
                pool.get_value_buffer(layer.layer_id),
                o_view,
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits[:current_bs],
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

        if self.is_int8:
            _, max_extend_len, k_scale_tensor, v_scale_tensor = self.forward_metadata
            # Update scales (same logic as decode)
            k_scale_tensor[0] = getattr(layer, "k_scale_float", 1.0)
            if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor):
                k_scale_tensor[0] = layer.k_scale.item()
            v_scale_tensor[0] = getattr(layer, "v_scale_float", 1.0)
            if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor):
                v_scale_tensor[0] = layer.v_scale.item()

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
                k_scale_tensor,
                v_scale_tensor,
            )
        else:
            _, max_extend_len = self.forward_metadata
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
