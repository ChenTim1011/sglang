"""
RVV-specific test utilities.

This module provides RVV-aligned helper functions that match
the C++ implementation exactly (epsilon=1e-7, RVV quantization logic).
"""

import torch

# Precision tolerances for RVV tests
precision = {
    torch.bfloat16: 1e-2,
    torch.float16: 1e-3,
    torch.float32: 1e-5,
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
