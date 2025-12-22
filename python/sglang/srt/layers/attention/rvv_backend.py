from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner


class RVVAttnBackend(AttentionBackend):
    """
    RVV optimized attention backend with fallback mechanism.

    This backend attempts to use RVV optimized kernels (with RVV intrinsics)
    when available and supported. If a case is not supported, it automatically
    falls back to torch_native backend.

    Features:
    - Selective RVV acceleration using fallback mechanism
    - For each case, use RVV kernel when supported; if not, fallback
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
        self.use_rvv_kernels = False
        self.decode_attention_fwd = None
        self.extend_attention_fwd = None

        # Create fallback backend (torch_native)
        self.fallback_backend = TorchNativeAttnBackend(model_runner)

        if self.is_riscv:
            print(
                "[RVV Backend] Detected RISC-V system, attempting to load RVV kernels...",
                flush=True,
            )
            try:
                # Check if sgl_kernel is available using importlib
                import importlib.util

                spec = importlib.util.find_spec("sgl_kernel")
                if spec is None:
                    raise ImportError("sgl_kernel not found")
                # Import sgl_kernel to access torch.ops.sgl_kernel
                # Use __import__ to avoid ruff F401 warning
                __import__("sgl_kernel")

                # Try to load RVV optimized kernels
                has_decode = hasattr(torch.ops.sgl_kernel, "decode_attention_cpu")
                has_extend = hasattr(torch.ops.sgl_kernel, "extend_attention_cpu")

                if has_decode:
                    self.decode_attention_fwd = (
                        torch.ops.sgl_kernel.decode_attention_cpu
                    )
                    self.use_rvv_kernels = True
                    print("[RVV Backend] ✓ RVV decode kernel loaded", flush=True)

                    # Initialize metadata for C++ kernels
                    self.num_head = (
                        model_runner.model_config.num_attention_heads
                        // model_runner.tp_size
                    )
                    self.v_head_dim = model_runner.token_to_kv_pool.get_value_buffer(
                        0
                    ).shape[-1]

                if has_extend:
                    self.extend_attention_fwd = (
                        torch.ops.sgl_kernel.extend_attention_cpu
                    )
                    print("[RVV Backend] ✓ RVV extend kernel loaded", flush=True)
                else:
                    print(
                        "[RVV Backend] ⚠ Extend kernel not available, will use fallback",
                        flush=True,
                    )

            except (ImportError, AttributeError) as e:
                print(
                    f"[RVV Backend] ⚠ Failed to load sgl_kernel: {e}, will use fallback",
                    flush=True,
                )
                self.use_rvv_kernels = False
        else:
            print(
                "[RVV Backend] ⚠ System is not RISC-V, will use fallback backend",
                flush=True,
            )

    def _check_head_dim_alignment(self, qk_head_dim: int, v_head_dim: int) -> bool:
        """Check if head dimensions are aligned for RVV kernel."""
        return qk_head_dim % 16 == 0 and v_head_dim % 16 == 0

    def _supports_rvv_decode(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        """
        Check if RVV kernel supports this decode case.

        Gradually expand support:
        - Phase 1: Basic decode (single head, standard shapes)
        - Phase 2: Multi-head, GQA
        - Phase 3: Special cases (logit_cap, etc.)
        """
        if not self.use_rvv_kernels or self.decode_attention_fwd is None:
            return False

        # Phase 1: Basic support checks
        if not self._check_head_dim_alignment(layer.qk_head_dim, layer.v_head_dim):
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
        # RVV kernel supports float32, bfloat16, float16
        # Add dtype checks if needed

        return True

    def _supports_rvv_extend(
        self, layer: RadixAttention, forward_batch: ForwardBatch
    ) -> bool:
        """Check if RVV kernel supports this extend case."""
        if not self.use_rvv_kernels or self.extend_attention_fwd is None:
            return False

        # Similar checks as decode
        if not self._check_head_dim_alignment(layer.qk_head_dim, layer.v_head_dim):
            print(
                f"[RVV] ⚠ Head dim not aligned: qk={layer.qk_head_dim}, v={layer.v_head_dim}, falling back",
                flush=True,
            )
            return False

        return True

    def get_backend_info(self) -> dict:
        """Get information about available RVV kernels."""
        return {
            "is_riscv": self.is_riscv,
            "use_rvv_kernels": self.use_rvv_kernels,
            "has_decode": self.decode_attention_fwd is not None,
            "has_extend": self.extend_attention_fwd is not None,
        }

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Init the metadata for a forward pass."""
        if self.use_rvv_kernels:
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
                max_extend_len = torch.max(forward_batch.extend_seq_lens).item()
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
        """Forward decode with RVV kernel or fallback."""
        # Check if RVV kernel supports this case
        if (
            self._supports_rvv_decode(layer, forward_batch)
            and self.decode_attention_fwd is not None
        ):
            # Use RVV optimized kernel
            return self._rvv_decode(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # Fallback to torch_native
            return self.fallback_backend.forward_decode(
                q, k, v, layer, forward_batch, save_kv_cache
            )

    def _rvv_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """RVV optimized decode using C++ kernel with RVV intrinsics.

        Note: The C++ kernel uses AT_DISPATCH_REDUCED_FLOATING_TYPES which only
        supports Half and BFloat16, NOT Float32. The kernel internally converts
        to float32 for computation. Tensors are converted to float16 if needed
        (e.g., when Linear outputs float32 on CPU).
        Only attn_logits must be Float32 (checked by kernel via CHECK_EQ).
        """
        # Convert to float16 if needed - RVV kernel only supports Half/BFloat16
        # Note: Transformers Linear on CPU outputs Float32 even with Float16 weights
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.half()
            k = k.half()
            v = v.half()

        attn_logits, _ = self.forward_metadata

        # Keep original dtype - kernel expects Half or BFloat16
        # (AT_DISPATCH_REDUCED_FLOATING_TYPES only supports these)
        q_reshaped = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

        if layer.qk_head_dim != layer.v_head_dim:
            o = torch.empty(
                (q_reshaped.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        else:
            o = torch.empty_like(q_reshaped)

        # Get KV buffers - convert to float16 if needed
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Convert KV buffers to float16 if needed (matching q, k, v dtype)
        if k_buffer.dtype not in (torch.float16, torch.bfloat16):
            k_buffer = k_buffer.half()
            v_buffer = v_buffer.half()

        # Call RVV optimized kernel
        # This kernel internally uses RVV intrinsics when CPU_CAPABILITY_RVV is defined
        # Kernel will convert to float32 internally for computation
        self.decode_attention_fwd(
            q_reshaped.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k_buffer,
            v_buffer,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k,
            v,
            forward_batch.out_cache_loc,
            attn_logits,  # This must be float32 (checked by kernel)
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
        """Forward extend with RVV kernel or fallback."""
        # Check if RVV kernel supports this case
        if (
            self._supports_rvv_extend(layer, forward_batch)
            and self.extend_attention_fwd is not None
        ):
            # Use RVV optimized kernel
            return self._rvv_extend(q, k, v, layer, forward_batch, save_kv_cache)
        else:
            # Fallback to torch_native
            return self.fallback_backend.forward_extend(
                q, k, v, layer, forward_batch, save_kv_cache
            )

    def _rvv_extend(
        self,
        q,
        k,
        v,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
    ):
        """RVV optimized extend using C++ kernel with RVV intrinsics.

        Note: All kernels (decode, extend, gemm) use AT_DISPATCH_REDUCED_FLOATING_TYPES
        which only supports Half and BFloat16. The RVV Zvfh extension handles FP16 natively.
        Tensors are converted to float16 if needed (e.g., when Linear outputs float32 on CPU).
        """
        # Convert to float16 if needed - RVV kernel only supports Half/BFloat16
        # Note: Transformers Linear on CPU outputs Float32 even with Float16 weights
        if q.dtype not in (torch.float16, torch.bfloat16):
            q = q.half()
            k = k.half()
            v = v.half()

        # Keep original dtype - kernel expects Half or BFloat16
        if layer.qk_head_dim != layer.v_head_dim:
            o = torch.empty(
                (q.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                dtype=q.dtype,
                device=q.device,
            )
        else:
            o = torch.empty_like(q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim))

        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        _, max_extend_len = self.forward_metadata

        # C++ kernel dtype requirements:
        # - q, k, v, k_buffer, v_buffer: Half, BFloat16, or Float32
        # - seq_lens: must be int64
        # - req_pool_indices: must be int64
        # - extend_seq_lens and extend_start_loc: must match req_to_token dtype
        req_to_token = forward_batch.req_to_token_pool.req_to_token
        index_dtype = req_to_token.dtype
        extend_seq_lens = forward_batch.extend_seq_lens.to(index_dtype)
        extend_start_loc = forward_batch.extend_start_loc.to(index_dtype)
        seq_lens = forward_batch.seq_lens.to(torch.int64)
        req_pool_indices = forward_batch.req_pool_indices.to(torch.int64)

        # Get KV buffers - convert to float16 if needed
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        # Convert KV buffers to float16 if needed (matching q, k, v dtype)
        if k_buffer.dtype not in (torch.float16, torch.bfloat16):
            k_buffer = k_buffer.half()
            v_buffer = v_buffer.half()

        self.extend_attention_fwd(
            q.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
            k,
            v,
            o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
            k_buffer,
            v_buffer,
            req_to_token,
            req_pool_indices,
            seq_lens,
            extend_seq_lens,
            extend_start_loc,
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
        """Forward prefill with optimized KV cache writing.

        Prefill phase:
        1. Compute attention output (use extend kernel)
        2. Write K/V to cache (use prefill_cache kernel if available)
        """
        # 1. Compute attention output using extend kernel
        # (extend kernel handles both prefix + new tokens efficiently)
        output = self.forward_extend(q, k, v, layer, forward_batch, save_kv_cache=False)

        # 2. Write K/V to cache using standard set_kv_buffer
        if save_kv_cache:
            forward_batch.token_to_kv_pool.set_kv_buffer(
                layer, forward_batch.out_cache_loc, k, v
            )

        return output

    def support_triton(self):
        """RVV backend does not support Triton."""
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

    def update_verify_buffers_to_fill_after_draft(self, spec_info, cuda_graph_bs):
        """Update verify buffers - not applicable for CPU backend."""
        pass
