import logging
from typing import Optional

import torch

from sglang.srt.utils import cpu_has_rvv_support

logger = logging.getLogger(__name__)

# hasattr() on torch.ops namespaces always returns False — use try/except.
try:
    _convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
except (AttributeError, RuntimeError):
    # RuntimeError can occur when the sgl_kernel extension is present but was
    # compiled without RVV support (e.g., a CPU-only wheel for another arch).
    _convert_weight_packed = None
    import platform as _platform

    if _platform.machine() == "riscv64":
        logger.warning(
            "[RVV] sgl_kernel.convert_weight_packed not found. "
            "Weight packing will be disabled; performance will be degraded. "
            "Ensure sgl-kernel was built with RVV support."
        )


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

    try:
        if getattr(layer, "weight_scale", None) is not None:
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
    except Exception as exc:
        raise RuntimeError(
            f"[RVV] rvv_linear_forward failed for {type(layer).__name__}: "
            f"x.shape={x.shape} x.dtype={x.dtype} "
            f"weight.shape={layer.weight.shape} weight.dtype={layer.weight.dtype}"
        ) from exc

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
    else:
        logger.warning(
            "[RVV] convert_weight_packed unavailable; skipping weight packing for %s. "
            "Module will NOT use RVV backend.",
            type(module).__name__,
        )


def probe_rvv_op(op_name: str) -> bool:
    """Return True if the named sgl_kernel RVV op is available.

    Use this for per-op probes after cpu_has_rvv_support() has already
    returned True.  Necessary because individual ops may be absent even
    when the base kernel is present (e.g., built without a specific kernel).

    hasattr() on torch.ops namespaces always returns False — use try/except.
    """
    try:
        getattr(torch.ops.sgl_kernel, op_name)
        return True
    except (AttributeError, RuntimeError):
        # RuntimeError can occur when the op is missing from the registry on some
        # torch builds (e.g., built without that specific kernel).
        return False
