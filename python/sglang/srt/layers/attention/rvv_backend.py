from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.utils.common import is_host_cpu_riscv

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


class RVVAttnBackend(AttentionBackend):
    """
    RISC-V Vector (RVV) optimized attention backend.

    This backend leverages custom C++ kernels with RVV intrinsics for efficient
    attention computation on RISC-V hardware. It automatically falls back to
    TorchNativeAttnBackend for unsupported data types or configurations.
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.model_runner = model_runner
        self.is_riscv = is_host_cpu_riscv()

        # Metadata tensors must be on CPU for RVV kernels
        self.device = "cpu"

        # Backend State
        self.use_rvv_kernels = False
        self.vlenb: Optional[int] = None
        self.forward_metadata = None

        # Kernel Function Pointers
        self.decode_attention_fwd = None
        self.extend_attention_fwd = None
        self.decode_attention_fwd_int8 = None
        self.extend_attention_fwd_int8 = None

        # Fallback Backend
        self.fallback_backend = TorchNativeAttnBackend(model_runner)

        if self.is_riscv:
            self._try_load_rvv_kernels()
        else:
            logger.debug(
                "[RVV] SGLang is running on non-RISC-V hardware. RVV backend disabled."
            )

    def _try_load_rvv_kernels(self):
        """Attempt to import sgl_kernel and load RVV ops."""
        try:
            import importlib.util

            if not importlib.util.find_spec("sgl_kernel"):
                return

            # 1. Detect Vector Length
            if hasattr(torch.ops.sgl_kernel, "get_rvv_vlenb"):
                try:
                    self.vlenb = torch.ops.sgl_kernel.get_rvv_vlenb()
                    vlen_bits = torch.ops.sgl_kernel.get_rvv_vlen()
                    logger.info(
                        f"[RVV] Detected VLEN: {vlen_bits} bits ({self.vlenb} bytes)"
                    )
                except Exception:
                    self.vlenb = 16  # Default 128-bit

            # 2. Load Kernels
            ops = torch.ops.sgl_kernel
            if hasattr(ops, "decode_attention_cpu"):
                self.decode_attention_fwd = ops.decode_attention_cpu
                self.use_rvv_kernels = True

            if hasattr(ops, "extend_attention_cpu"):
                self.extend_attention_fwd = ops.extend_attention_cpu

            if hasattr(ops, "decode_attention_int8_cpu"):
                self.decode_attention_fwd_int8 = ops.decode_attention_int8_cpu

            if hasattr(ops, "extend_attention_int8_cpu"):
                self.extend_attention_fwd_int8 = ops.extend_attention_int8_cpu

            # 3. Initialize Dimensions for Metadata
            if self.use_rvv_kernels:
                self.num_head = (
                    self.model_runner.model_config.num_attention_heads
                    // self.model_runner.tp_size
                )
                # Try to get head dimension from pool, fallback to config/default
                try:
                    self.v_head_dim = (
                        self.model_runner.token_to_kv_pool.get_value_buffer(0).shape[-1]
                    )
                except Exception:
                    self.v_head_dim = self.model_runner.model_config.head_dim

                logger.info("[RVV] Kernels loaded successfully.")

        except Exception as e:
            logger.warning(f"[RVV] Failed to load kernels: {e}. Using Native fallback.")
            self.use_rvv_kernels = False

    def _check_head_dim_alignment(self, qk_head_dim: int, v_head_dim: int) -> bool:
        """Check if head dimensions align with VLEN for vectorization."""
        if not self.is_riscv or self.vlenb is None:
            return qk_head_dim % 16 == 0 and v_head_dim % 16 == 0

        # For FP16/BF16 (2 bytes), check byte alignment
        return ((qk_head_dim * 2) % self.vlenb == 0) and (
            (v_head_dim * 2) % self.vlenb == 0
        )

    def _supports_rvv_decode(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        """Check if RVV kernel supports this decode case."""
        if not self.use_rvv_kernels or self.decode_attention_fwd is None:
            return False

        if not self._check_head_dim_alignment(layer.qk_head_dim, layer.v_head_dim):
            return False

        # Critical: Check KV Cache dtype compatibility
        pool = forward_batch.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)

        if k_buffer.dtype in (torch.int8, torch.uint8):
            return self.decode_attention_fwd_int8 is not None

        # For Float path, kernel supports half, bfloat16, float32
        if k_buffer.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False

        return True

    def _supports_rvv_extend(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        """Check if RVV kernel supports this extend case."""
        if not self.use_rvv_kernels or self.extend_attention_fwd is None:
            return False

        if not self._check_head_dim_alignment(layer.qk_head_dim, layer.v_head_dim):
            return False

        pool = forward_batch.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)

        if k_buffer.dtype in (torch.int8, torch.uint8):
            return self.extend_attention_fwd_int8 is not None

        if k_buffer.dtype not in (torch.float16, torch.bfloat16, torch.float32):
            return False

        return True

    def get_backend_info(self) -> dict:
        return {
            "name": "rvv",
            "is_riscv": self.is_riscv,
            "kernels_loaded": self.use_rvv_kernels,
            "vlenb": self.vlenb,
        }

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Pre-allocate buffers needed for the forward pass."""
        if self.use_rvv_kernels:
            # Allocate logits buffer on CPU
            attn_logits = torch.empty(
                (
                    forward_batch.batch_size,
                    self.num_head,
                    8,  # num_kv_splits (hardcoded in kernel)
                    self.v_head_dim + 1,
                ),
                dtype=torch.float32,
                device="cpu",
            )

            max_extend_len = 0
            if not forward_batch.forward_mode.is_decode_or_idle():
                max_extend_len = torch.max(forward_batch.extend_seq_lens).item()

            self.forward_metadata = (attn_logits, max_extend_len)
        else:
            self.fallback_backend.init_forward_metadata(forward_batch)
            self.forward_metadata = self.fallback_backend.forward_metadata

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not self._supports_rvv_decode(layer, forward_batch):
            return self.fallback_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        pool = forward_batch.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)

        target_dtype = k_buffer.dtype
        is_int8 = target_dtype in (torch.int8, torch.uint8)

        # 1. Type Casting
        if not is_int8 and target_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            if v.dtype != target_dtype:
                v = v.to(target_dtype)

        if q.dtype == torch.float64:
            q = q.float()
            k = k.float()
            v = v.float()

        # 2. KV Cache Management
        if save_kv_cache and not is_int8:
            if (
                layer.is_cross_attention
                and forward_batch.encoder_out_cache_loc is not None
            ):
                cache_loc = forward_batch.encoder_out_cache_loc
            else:
                cache_loc = forward_batch.out_cache_loc

            pool.set_kv_buffer(layer, cache_loc, k, v)

        # 3. Output Setup
        attn_logits, _ = self.forward_metadata
        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = torch.empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        else:
            o = torch.empty_like(q).view(-1, layer.tp_q_head_num * layer.v_head_dim)

        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        # 4. Dispatch
        if is_int8:
            k_scale = getattr(layer, "k_scale_float", 1.0)
            v_scale = getattr(layer, "v_scale_float", 1.0)
            if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor):
                k_scale = layer.k_scale.item()
            if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor):
                v_scale = layer.v_scale.item()

            self.decode_attention_fwd_int8(
                q_view,
                k_buffer,
                v_buffer,
                o_view,
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
                float(k_scale),
                float(v_scale),
            )
        else:
            self.decode_attention_fwd(
                q_view,
                k_buffer,
                v_buffer,
                o_view,
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
            )

        return o.view_as(q)

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        if not self._supports_rvv_extend(layer, forward_batch):
            return self.fallback_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

        pool = forward_batch.token_to_kv_pool
        k_buffer = pool.get_key_buffer(layer.layer_id)
        v_buffer = pool.get_value_buffer(layer.layer_id)

        target_dtype = k_buffer.dtype
        is_int8 = target_dtype in (torch.int8, torch.uint8)

        # 1. Type Casting
        if not is_int8 and target_dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            if q.dtype != target_dtype:
                q = q.to(target_dtype)
            if k.dtype != target_dtype:
                k = k.to(target_dtype)
            if v.dtype != target_dtype:
                v = v.to(target_dtype)

        if q.dtype == torch.float64:
            q = q.float()
            k = k.float()
            v = v.float()

        # 2. KV Cache Management
        if save_kv_cache and not is_int8:
            if (
                layer.is_cross_attention
                and forward_batch.encoder_out_cache_loc is not None
            ):
                cache_loc = forward_batch.encoder_out_cache_loc
            else:
                cache_loc = forward_batch.out_cache_loc

            pool.set_kv_buffer(layer, cache_loc, k, v)

        # 3. Output Setup
        _, max_extend_len = self.forward_metadata
        out_dim = layer.tp_q_head_num * layer.v_head_dim
        o = torch.empty((q.shape[0], out_dim), dtype=q.dtype, device=q.device)

        q_view = q.view(-1, layer.tp_q_head_num, layer.qk_head_dim)
        o_view = o.view(-1, layer.tp_q_head_num, layer.v_head_dim)

        # 4. Dispatch
        if is_int8:
            k_scale = getattr(layer, "k_scale_float", 1.0)
            v_scale = getattr(layer, "v_scale_float", 1.0)
            if hasattr(layer, "k_scale") and isinstance(layer.k_scale, torch.Tensor):
                k_scale = layer.k_scale.item()
            if hasattr(layer, "v_scale") and isinstance(layer.v_scale, torch.Tensor):
                v_scale = layer.v_scale.item()

            self.extend_attention_fwd_int8(
                q_view,
                k,
                v,
                o_view,
                k_buffer,
                v_buffer,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                forward_batch.extend_seq_lens,
                forward_batch.extend_start_loc,
                max_extend_len,
                layer.scaling,
                layer.logit_cap,
                float(k_scale),
                float(v_scale),
            )
        else:
            self.extend_attention_fwd(
                q_view,
                k,
                v,
                o_view,
                k_buffer,
                v_buffer,
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
        """RVV backend does not support Triton."""
        return False

    def get_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens."""
        return self.fallback_backend.get_graph_seq_len_fill_value()

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens in CUDA graph."""
        return self.get_graph_seq_len_fill_value()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        pass

    def init_forward_metadata_capture_cuda_graph(
        self,
        bs: int,
        num_tokens: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        encoder_lens,
        forward_mode,
        spec_info,
    ):
        pass

    def init_forward_metadata_replay_cuda_graph(
        self,
        bs: int,
        req_pool_indices: torch.Tensor,
        seq_lens: torch.Tensor,
        seq_lens_sum: int,
        encoder_lens,
        forward_mode,
        spec_info,
        seq_lens_cpu,
    ):
        pass

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        pass
