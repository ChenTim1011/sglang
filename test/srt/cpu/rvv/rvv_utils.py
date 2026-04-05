import importlib.util

import torch
from torch.nn.functional import scaled_dot_product_attention


def has_sgl_kernel_op(op_name: str) -> bool:
    """Check if a specific operator is available in sgl_kernel."""
    if not importlib.util.find_spec("sgl_kernel"):
        return False
    try:
        getattr(torch.ops.sgl_kernel, op_name)
        return True
    except (AttributeError, RuntimeError):
        return False


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
    """RVV CPU W8A8 reference: activation is uint8, weight is int8."""
    A = A.to(torch.int32) - 128
    B = B.to(torch.int32)

    assert A.shape[-1] == B.shape[-1], "Dimension mismatch"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    M = A.numel() // A.shape[-1]
    K = A.shape[-1]
    origin_C_shape = A.shape[:-1] + (B.shape[0],)
    A = A.reshape(M, K)

    C = torch.matmul(A, B.transpose(0, 1))

    As = As.reshape(M, 1).to(torch.float32)
    C = As * C.to(torch.float32) * Bs.view(1, -1).to(torch.float32)

    if bias is not None:
        C.add_(bias.view(1, -1).to(torch.float32))

    return C.reshape(origin_C_shape).to(output_dtype)


# Numeric tolerances for shared RVV test paths: precision[key][dtype].
precision = {
    "pointwise_default": {
        torch.bfloat16: 3e-2,
        torch.float16: 1e-3,
        torch.float32: 1e-5,
    },
    # FP16 layernorm+bias path can show quantized-step residuals (~1/128) on RVV.
    "norm_layer_bias": {
        torch.bfloat16: 3e-2,
        torch.float16: 1e-2,
        torch.float32: 1e-5,
    },
    "linear_gemm": {torch.bfloat16: 1.5e-1, torch.float16: 1e-1},
    "rotary_embedding": {torch.bfloat16: 3e-2, torch.float16: 1e-2},
    "attention_decode_int8": {torch.bfloat16: 0.15, torch.float16: 0.15},
    "attention_extend_int8": {torch.bfloat16: 5e-2, torch.float16: 5e-2},
    "attention_decode_logit_cap": {torch.float16: 1e-2, torch.bfloat16: 1e-1},
    "attention_decode": {
        torch.bfloat16: 1e-1,
        torch.float16: 7e-2,
        torch.float32: 1e-5,
    },
    "attention_extend": {
        torch.bfloat16: 1e-1,
        torch.float16: 3e-3,
        torch.float32: 1e-5,
    },
}


def run_sdpa_forward_decode(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
    logit_cap=0.0,
    key: torch.Tensor = None,
    loc: torch.Tensor = None,
):
    """Reference implementation for decode attention."""
    if key is not None and loc is not None:
        k_cache[loc] = key

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        seq_len_q = 1
        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        if logit_cap > 0:
            # Manual attention: apply tanh softcap to logits before softmax.
            # q: [H_Q, seq_q, D], k: [H_KV, Tk, D], v: [H_KV, Tk, D_V]
            q = per_req_query.unsqueeze(0).float()  # [1, H_Q, seq_q, D]
            k = per_req_key.unsqueeze(0).float()  # [1, H_KV, Tk, D]
            v = per_req_value.unsqueeze(0).float()  # [1, H_KV, Tk, D_V]
            if enable_gqa:
                G = q.size(1) // k.size(1)
                k = k.repeat_interleave(G, dim=1)  # [1, H_Q, Tk, D]
                v = v.repeat_interleave(G, dim=1)  # [1, H_Q, Tk, D_V]
            scores = torch.matmul(
                q * scaling, k.transpose(-1, -2)
            )  # [1, H_Q, seq_q, Tk]
            scores = logit_cap * torch.tanh(scores / logit_cap)
            weights = torch.softmax(scores, dim=-1)
            per_req_out = (
                torch.matmul(weights, v)
                .to(per_req_query.dtype)
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
        else:
            per_req_out = (
                scaled_dot_product_attention(
                    per_req_query.unsqueeze(0),
                    per_req_key.unsqueeze(0),
                    per_req_value.unsqueeze(0),
                    enable_gqa=enable_gqa,
                    scale=scaling,
                    is_causal=causal,
                )
                .squeeze(0)
                .movedim(query.dim() - 2, 0)
            )
        output[start_q:end_q, :, :] = per_req_out
        start_q, start_kv = end_q, end_kv

    return output


def run_sdpa_forward_extend(
    query: torch.Tensor,
    output: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    extend_prefix_lens: torch.Tensor,
    extend_seq_lens: torch.Tensor,
    scaling=None,
    enable_gqa=False,
    causal=False,
):
    """Reference implementation for extend attention."""
    assert seq_lens.shape[0] == extend_prefix_lens.shape[0]
    assert seq_lens.shape[0] == extend_seq_lens.shape[0]

    # [num_tokens, num_heads, head_size] -> [num_heads, num_tokens, head_size]
    query = query.movedim(0, query.dim() - 2)

    start_q, start_kv = 0, 0
    for seq_idx in range(seq_lens.shape[0]):
        extend_seq_len_q = extend_seq_lens[seq_idx]
        prefill_seq_len_q = extend_prefix_lens[seq_idx]

        seq_len_kv = seq_lens[seq_idx]
        end_q = start_q + extend_seq_len_q
        end_kv = start_kv + seq_len_kv

        per_req_query = query[:, start_q:end_q, :]
        per_req_query_redudant = torch.empty(
            (per_req_query.shape[0], seq_len_kv, per_req_query.shape[2]),
            dtype=per_req_query.dtype,
            device=per_req_query.device,
        )

        per_req_query_redudant[:, prefill_seq_len_q:, :] = per_req_query

        # get key and value from cache. per_req_tokens contains the kv cache
        # index for each token in the sequence.
        req_pool_idx = req_pool_indices[seq_idx]
        per_req_tokens = req_to_token[req_pool_idx, :seq_len_kv]
        per_req_key = k_cache[per_req_tokens].movedim(0, query.dim() - 2)
        per_req_value = v_cache[per_req_tokens].movedim(0, query.dim() - 2)

        per_req_out_redudant = (
            scaled_dot_product_attention(
                per_req_query_redudant.unsqueeze(0),
                per_req_key.unsqueeze(0),
                per_req_value.unsqueeze(0),
                enable_gqa=enable_gqa,
                scale=scaling,
                is_causal=causal,
            )
            .squeeze(0)
            .movedim(query.dim() - 2, 0)
        )
        output[start_q:end_q] = per_req_out_redudant[prefill_seq_len_q:]
        start_q, start_kv = end_q, end_kv
    return output


def rmsnorm_native(x, weight, eps, residual=None):
    """Reference RMSNorm: out = x * rsqrt(mean(x^2) + eps) * weight."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x if residual is None else (x, residual)


def gemma_rmsnorm_native(x, weight, eps, residual=None):
    """Reference Gemma RMSNorm: scale = (1 + weight)."""
    orig_dtype = x.dtype
    if residual is not None:
        x = x + residual
        residual = x
    x = x.float()
    variance = x.pow(2).mean(dim=-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = x * (1.0 + weight.float())
    x = x.to(orig_dtype)
    return x if residual is None else (x, residual)


def gemma3_rmsnorm_native(x, weight, eps):
    """Reference Gemma3 RMSNorm: supports 2D and 4D inputs."""
    output = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    output = output * (1.0 + weight.float())
    return output.type_as(x)


def fused_rmsnorm_gated_native(x, weight, gate, eps):
    """Reference fused gated RMSNorm: rms_norm(x) * weight * silu(gate)."""
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    x = weight * x.to(input_dtype)
    x = x * torch.nn.functional.silu(gate.to(torch.float32))
    return x.to(input_dtype)


def layernorm_native(x, weight, eps, residual=None):
    """Reference LayerNorm: mean+var two pass."""
    orig_dtype = x.dtype
    x = x.to(torch.float32)
    if residual is not None:
        x = x + residual.to(torch.float32)
        residual = x.to(orig_dtype)
    variance, mean = torch.var_mean(x, dim=-1, keepdim=True, correction=0)
    x = (x - mean) * torch.rsqrt(variance + eps)
    x = x.to(orig_dtype) * weight
    return x if residual is None else (x, residual)


def helper_non_contiguous(t: torch.Tensor) -> torch.Tensor:
    """Return a non-contiguous view of t by striding the batch dimension 2x."""
    buf = torch.empty(t.shape[0] * 2, *t.shape[1:], dtype=t.dtype)
    buf[::2] = t
    return buf[::2]
