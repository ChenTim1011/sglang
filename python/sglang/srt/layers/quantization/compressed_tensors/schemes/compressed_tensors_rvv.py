# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch

from sglang.srt.layers.quantization.utils import replace_parameter, unpack_cols
from sglang.srt.layers.rvv_utils import _rvv_process_int4_weight_after_loading


def compressed_uint4b8_to_rvv_signed_int4_bytes(
    packed_weight: torch.Tensor,
    input_size_per_partition: int,
    input_permutation: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Convert compressed-tensors uint4b8 column-pack to RVV signed INT4 bytes."""
    if input_size_per_partition % 8 != 0:
        raise ValueError(
            "compressed-tensors INT4 weights must have K divisible by 8 before "
            "converting to the RVV INT4 layout."
        )

    output_size_per_partition = packed_weight.shape[0]
    unpacked = unpack_cols(
        packed_weight,
        num_bits=4,
        size_k=output_size_per_partition,
        size_n=input_size_per_partition,
    ).to(torch.uint8)
    if input_permutation is not None:
        unpacked = unpacked.index_select(1, input_permutation.to(unpacked.device))

    signed_nibbles = unpacked ^ 0x8
    return (signed_nibbles[:, 0::2] | (signed_nibbles[:, 1::2] << 4)).contiguous()


def process_weights_after_loading_rvv_wna16(scheme, layer: torch.nn.Module) -> None:
    """Load WNA16 checkpoint weights and prepare RVV W4A8 dynamic compute."""
    if scheme.num_bits != 4 or not scheme.symmetric:
        raise NotImplementedError(
            "RVV compressed-tensors WNA16 currently supports only symmetric W4A16."
        )

    c = scheme.kernel_config
    group_size = c.group_size if c.group_size != -1 else c.partition_weight_shape[0]
    input_permutation = None
    if scheme.has_g_idx:
        input_permutation = torch.argsort(getattr(layer, scheme.w_gidx_name)).to(
            torch.int64
        )
        sorted_g_idx = getattr(layer, scheme.w_gidx_name)[input_permutation]
        expected_g_idx = torch.arange(
            c.partition_weight_shape[0],
            device=getattr(layer, scheme.w_q_name).device,
            dtype=sorted_g_idx.dtype,
        ).div_(group_size, rounding_mode="floor")
        if not torch.equal(sorted_g_idx, expected_g_idx):
            raise NotImplementedError(
                "RVV compressed-tensors W4A16 requires activation-order g_idx "
                "groups to sort into contiguous group_size blocks."
            )

    layer.rvv_g_idx_sort_indices = input_permutation
    rvv_weight = compressed_uint4b8_to_rvv_signed_int4_bytes(
        getattr(layer, scheme.w_q_name).data.contiguous(),
        c.partition_weight_shape[0],
        input_permutation=input_permutation,
    )
    replace_parameter(
        layer,
        scheme.w_q_name,
        torch.nn.Parameter(rvv_weight, requires_grad=False),
    )
    _rvv_process_int4_weight_after_loading(
        layer, scheme.w_q_name, scheme.w_s_name, group_size
    )
    if not getattr(layer, "use_riscv_rvv_int4_w4a8_dynamic_linear_backend", False):
        raise RuntimeError(
            "[RVV] compressed-tensors W4A16 selected on a RISC-V host, but RVV "
            "W4A8 dynamic INT4 packing was not enabled. Rebuild sgl-kernel "
            "with convert_weight_w4a8_dynamic_packed support."
        )
    scheme.rvv_group_size = group_size


def apply_weights_rvv_wna16(
    scheme, layer: torch.nn.Module, x: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    """Apply a WNA16 scheme through the RVV W4A8 dynamic runtime policy."""
    x_shape = x.shape
    if len(x_shape) == 3:
        x = x.view(-1, x.shape[-1])
    input_permutation = getattr(layer, "rvv_g_idx_sort_indices", None)
    if input_permutation is not None:
        input_permutation = input_permutation.to(x.device)

    w4a8_input_permutation = input_permutation
    if input_permutation is not None and x.size(0) != 1:
        x = x.index_select(-1, input_permutation)
        w4a8_input_permutation = None
    output = torch.ops.sgl_kernel.weight_w4a8_dynamic_linear(
        x,
        getattr(layer, "_rvv_int4_w4a8_dynamic_w_q"),
        getattr(layer, "_rvv_int4_w4a8_dynamic_w_s"),
        bias,
        w4a8_input_permutation,
        scheme.rvv_group_size,
        True,
    )

    if len(x_shape) == 3:
        output = output.view(x_shape[0], x_shape[1], -1)
    return output
