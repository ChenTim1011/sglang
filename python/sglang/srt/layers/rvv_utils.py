import logging
from typing import Optional

import torch

from sglang.srt.utils import cpu_has_rvv_support

logger = logging.getLogger(__name__)

try:
    _convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
except AttributeError:
    _convert_weight_packed = None


def rvv_linear_forward(
    x: torch.Tensor,
    layer: torch.nn.Module,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Shared RVV linear dispatch: handles both BF16 packed and INT8 W8A8 paths.

    Callers must verify use_riscv_rvv_backend(layer) is True before calling.
    """
    x_shapes = x.shape
    if len(x_shapes) == 3:
        x = x.view(-1, x.shape[-1])

    if hasattr(layer, "weight_scale"):
        output = torch.ops.sgl_kernel.int8_scaled_mm_with_quant(
            x,
            layer.weight,
            layer.weight_scale,
            bias,
            x.dtype,
            True,  # is_packed
        )
    else:
        output = torch.ops.sgl_kernel.weight_packed_linear(
            x,
            layer.weight,
            bias,
            True,  # is_packed
        )

    if len(x_shapes) == 3:
        output = output.view(x_shapes[0], x_shapes[1], -1)
    return output


def _rvv_process_weight_after_loading(module, weight_names) -> None:
    """Process weights for RVV backend.

    1. Convert weights to RVV packed format via convert_weight_packed.
    2. Mark the module with use_riscv_rvv_backend = True.

    For INT8 quantization, use --quantization w8a8_int8 which goes through
    the standard W8A8Int8LinearMethod path.
    """
    if not cpu_has_rvv_support():
        return

    if hasattr(module, "bias") and module.bias is not None:
        if module.bias.dtype != torch.float32:
            module.bias = torch.nn.Parameter(module.bias.float(), requires_grad=False)

    if _convert_weight_packed is not None:
        for weight_name in weight_names:
            weight_tensor = getattr(module, weight_name)
            weight_tensor.data = _convert_weight_packed(weight_tensor.data)

    module.use_riscv_rvv_backend = True


class RvvPackWeightMethod:
    """Mirrors AMX's PackWeightMethod for the RVV backend."""

    def __init__(self, weight_names):
        self.weight_names = weight_names

    def process_weights_after_loading(self, module) -> None:
        _rvv_process_weight_after_loading(module, self.weight_names)
