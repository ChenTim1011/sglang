import torch
from torch.nn.functional import scaled_dot_product_attention

# Precision tolerances for RVV tests
precision = {
    torch.bfloat16: 3e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
}

# Specific tolerances for GEMM which naturally accumulates more floating point errors
gemm_precision = {
    torch.bfloat16: 1.5e-1,
    torch.float16: 1e-1,
}

# Tolerances for fused sigmoid-mul which involves non-linear accumulation
sigmoid_mul_precision = {
    torch.bfloat16: 0.5,
    torch.float16: 2e-1,
}

# Quantization error tolerances for INT8 decode/extend attention
int8_decode_precision = {
    torch.bfloat16: 0.15,
    torch.float16: 0.15,
}

int8_extend_precision = {
    torch.bfloat16: 0.5,
    torch.float16: 0.5,
}


def native_w8a8_per_token_matmul(A, B, As, Bs, bias, output_dtype=torch.bfloat16):
    """
    Reference INT8 GEMM implementation aligned with RVV kernel logic.

    This implementation matches the calculation order in RVV C++ code:
    - INT8 matrix multiplication
    - Dequantization with combined scales: C * (As * Bs)
    - Optional bias addition

    Args:
        A: Input tensor [M, K] int8
        B: Weight tensor [N, K] int8
        As: Per-token activation scales [M, 1] or [M] float32
        Bs: Per-column weight scales [N, 1] or [N] float32
        bias: Optional bias [N] float32
        output_dtype: Output data type

    Returns:
        Output tensor [M, K] in output_dtype
    """
    A = A.to(torch.float32)
    B = B.to(torch.float32)

    # Handle scale shapes: ensure [M, 1] and [1, N] for broadcasting
    # per_token_quant_int8 returns [M, 1], need to handle both [M] and [M, 1]
    if As.dim() == 1:
        As = As.unsqueeze(-1)  # [M] -> [M, 1]
    # Bs can be [N] or [N, 1], need to convert to [1, N] for broadcasting
    if Bs.dim() == 2:
        Bs = Bs.squeeze(-1)  # [N, 1] -> [N]
    Bs = Bs.unsqueeze(0)  # [N] -> [1, N]

    assert A.shape[-1] == B.shape[-1], f"Dimension mismatch: A={A.shape}, B={B.shape}"
    assert B.ndim == 2 and B.is_contiguous(), "B must be a 2D contiguous tensor"

    # Reshape input
    M = A.numel() // A.shape[-1]
    B = B.t()  # Transpose weight matrix [N, K] -> [K, N]
    N, K = B.shape[1], B.shape[0]
    origin_C_shape = A.shape[:-1] + (N,)
    A = A.reshape(M, -1)

    # INT8 GEMM: [M, K] @ [K, N] = [M, N]
    C = torch.matmul(A, B)  # [M, N] int32 accumulator

    # Dequantize: C * (As * Bs)
    # As: [M, 1], Bs: [1, N] -> broadcast to [M, N]
    combined_scales = As * Bs  # [M, 1] * [1, N] = [M, N]
    C = C * combined_scales

    if bias is not None:
        C.add_(bias.view(1, -1))

    return C.reshape(origin_C_shape).to(output_dtype)


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


def SiluAndMul(x):
    """Reference implementation for SiluAndMul activation."""
    x = torch.chunk(x, 2, dim=-1)
    return torch.nn.functional.silu(x[0]) * x[1]


def torch_naive_fused_moe(a, w1, w2, score, topk, renormalize):
    """Reference implementation for fused MoE."""
    B, D = a.shape
    a = a.view(B, -1, D).repeat(1, topk, 1).reshape(-1, D)
    out = torch.zeros(B * topk, w2.shape[1], dtype=a.dtype, device=a.device)
    score = torch.softmax(score, dim=-1, dtype=torch.float32)
    topk_weight, topk_ids = torch.topk(score, topk)

    if renormalize:
        topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

    topk_weight = topk_weight.view(-1)
    topk_ids = topk_ids.view(-1)
    for i in range(w1.shape[0]):
        mask = topk_ids == i
        if mask.sum():
            out[mask] = SiluAndMul(a[mask] @ w1[i].transpose(0, 1)) @ w2[i].transpose(
                0, 1
            )
    return (
        out.view(B, -1, w2.shape[1]) * topk_weight.view(B, -1, 1).to(out.dtype)
    ).sum(dim=1)
