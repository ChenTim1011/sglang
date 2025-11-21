from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class RISCVAttnBackend(AttentionBackend):
    """
    RISC-V optimized attention backend with fallback mechanism.

    This backend attempts to use RISC-V optimized kernels (with RVV intrinsics)
    when available and supported. If a case is not supported, it automatically
    falls back to torch_native backend.

    Features:
    - Selective RISC-V acceleration using fallback mechanism
    - For each case, use RISC-V kernel when supported; if not, fallback
    - Gradually expand the scope of RVV support
    """

    def __init__(self, model_runner: ModelRunner):
        super().__init__()
        self.forward_metadata = None
        self.device = model_runner.device
        self.model_runner = model_runner

        # Check if we're on RISC-V
        from sglang.srt.utils.common import is_host_cpu_riscv

        self.is_riscv = is_host_cpu_riscv()
        self.use_riscv_kernels = False
        self.decode_attention_fwd = None
        self.extend_attention_fwd = None

        # Create fallback backend (torch_native)
        self.fallback_backend = TorchNativeAttnBackend(model_runner)

        if self.is_riscv:
            print(
                "[RISC-V Backend] Detected RISC-V system, attempting to load RISC-V kernels...", flush=True)
            try:
                import sgl_kernel

                # Try to load RISC-V optimized kernels
                has_decode = hasattr(torch.ops.sgl_kernel,
                                     "decode_attention_cpu")
                has_extend = hasattr(torch.ops.sgl_kernel,
                                     "extend_attention_cpu")

                if has_decode:
                    self.decode_attention_fwd = torch.ops.sgl_kernel.decode_attention_cpu
                    self.use_riscv_kernels = True
                    print(
                        "[RISC-V Backend] ✓ RISC-V decode kernel loaded", flush=True)

                    # Initialize metadata for C++ kernels
                    self.num_head = (
                        model_runner.model_config.num_attention_heads // model_runner.tp_size
                    )
                    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
                        0).shape[-1]

                if has_extend:
                    self.extend_attention_fwd = torch.ops.sgl_kernel.extend_attention_cpu
                    print(
                        "[RISC-V Backend] ✓ RISC-V extend kernel loaded", flush=True)
                else:
                    print(
                        "[RISC-V Backend] ⚠ Extend kernel not available, will use fallback", flush=True)

            except (ImportError, AttributeError) as e:
                print(
                    f"[RISC-V Backend] ⚠ Failed to load sgl_kernel: {e}, will use fallback", flush=True)
                self.use_riscv_kernels = False
        else:
            print(
                "[RISC-V Backend] ⚠ System is not RISC-V, will use fallback backend", flush=True)

    def _supports_riscv_decode(self, layer: RadixAttention, forward_batch: ForwardBatch) -> bool:
        """
        Check if RISC-V kernel supports this decode case.

        Gradually expand support:
        - Phase 1: Basic decode (single head, standard shapes)
        - Phase 2: Multi-head, GQA
        - Phase 3: Special cases (logit_cap, etc.)
        """
        if not self.use_riscv_kernels or self.decode_attention_fwd is None:
            return False

        # Phase 1: Basic support checks
        # Check if head dimensions are supported
        if layer.qk_head_dim % 16 != 0 or layer.v_head_dim % 16 != 0:
            # RISC-V kernel may require aligned dimensions
            return False

        # Check if batch size is reasonable
        if forward_batch.batch_size > 256:
            # May need to test larger batches
            return False

        # Phase 2: Check for unsupported features
        # If logit_cap is used, may need special handling
        if layer.logit_cap > 0:
            # Currently supported, but can add more checks here
            pass

        # Phase 3: Check data types
        # RISC-V kernel supports float32, bfloat16, float16
        # Add dtype checks if needed

        return True

    def _supports_riscv_extend(self, layer: RadixAttention, forward_batch: ForwardBatch) -> bool:
        """Check if RISC-V kernel supports this extend case."""
        if not self.use_riscv_kernels or self.extend_attention_fwd is None:
            return False

        # Similar checks as decode
        if layer.qk_head_dim % 16 != 0 or layer.v_head_dim % 16 != 0:
            return False

        return True

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        if self.use_riscv_kernels:
            # Initialize metadata for C++ kernels
            bs = forward_batch.batch_size
            attn_logits = torch.zeros(
                (
                    bs,
                    self.num_head,
                    8,  # num_kv_splits (fixed for now)
                    self.v_head_dim + 1,
                ),
                dtype=torch.float32,
                device=self.device,
            )
            if forward_batch.forward_mode.is_decode_or_idle():
                max_extend_len = None
            else:
                max_extend_len = torch.max(
                    forward_batch.extend_seq_lens).item()
            self.forward_metadata = (attn_logits, max_extend_len)
        else:
            # Use fallback backend metadata
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
        """Forward decode with RISC-V kernel or fallback."""
        # Check if RISC-V kernel supports this case
        if self._supports_riscv_decode(layer, forward_batch) and self.decode_attention_fwd is not None:
            # Use RISC-V optimized kernel
            return self._riscv_decode(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # Fallback to torch_native
            return self.fallback_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

    def _riscv_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """RISC-V optimized decode using C++ kernel with RVV intrinsics."""
        attn_logits, _ = self.forward_metadata

        q = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        # Call RISC-V optimized kernel
        # This kernel internally uses RVV intrinsics when CPU_CAPABILITY_RVV is defined
        self.decode_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
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

        return o

    def forward_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """Forward extend with RISC-V kernel or fallback."""
        # Check if RISC-V kernel supports this case
        if self._supports_riscv_extend(layer, forward_batch) and self.extend_attention_fwd is not None:
            # Use RISC-V optimized kernel
            return self._riscv_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # Fallback to torch_native
            return self.fallback_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

    def _riscv_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """RISC-V optimized extend using C++ kernel with RVV intrinsics."""
        if layer.qk_head_dim != layer.v_head_dim:
            o = q.new_empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim))
        else:
            o = torch.empty_like(q)

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        # Call RISC-V optimized kernel
        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id),
            forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id),
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

    def forward_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
        **kwargs,
    ):
        """Forward prefill - currently always uses fallback."""
        # RISC-V prefill kernel not yet implemented
        # Always use fallback for now
        return self.fallback_backend.forward_prefill(
            q, k, v, layer, forward_batch, save_kv_cache, **kwargs
        )

    def support_triton(self):
        """RISC-V backend does not support Triton."""
        return False

    def get_graph_seq_len_fill_value(self):
        """Get the fill value for padded seq lens."""
        return self.fallback_backend.get_graph_seq_len_fill_value()

    def init_cuda_graph_state(self, max_bs: int, max_num_tokens: int):
        """Init CUDA graph state - not applicable for CPU backend."""
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
        """Init metadata for CUDA graph capture - not applicable for CPU backend."""
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
        """Init metadata for CUDA graph replay - not applicable for CPU backend."""
        pass

    def update_verify_buffers_to_fill_after_draft(
        self, spec_info, cuda_graph_bs
    ):
        """Update verify buffers - not applicable for CPU backend."""
        pass
