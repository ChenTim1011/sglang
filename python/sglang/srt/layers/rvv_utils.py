import logging

import torch

from sglang.srt.utils import cpu_has_rvv_support

logger = logging.getLogger(__name__)


def _rvv_process_weight_after_loading(module, weight_names) -> None:
    """
    Process weights for RVV backend:
    1. Check RVV availability.
    2. Convert weights to packed format if kernel is available.
    3. Set backend flag.
    """
    if not cpu_has_rvv_support():
        return

    if hasattr(module, "bias") and module.bias is not None:
        if module.bias.dtype != torch.float32:
            module.bias = torch.nn.Parameter(module.bias.float(), requires_grad=False)

    if hasattr(torch.ops.sgl_kernel, "convert_weight_packed"):
        for weight_name in weight_names:
            weight_tensor = getattr(module, weight_name)
            packed_data = torch.ops.sgl_kernel.convert_weight_packed(weight_tensor.data)
            weight_tensor.data = packed_data

    module.use_riscv_rvv_backend = True
