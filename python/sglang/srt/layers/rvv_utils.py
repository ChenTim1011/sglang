import logging

import torch

from sglang.srt.utils import cpu_has_rvv_support, is_host_cpu_riscv

logger = logging.getLogger(__name__)

# hasattr() on torch.ops namespaces always returns False — use try/except.
try:
    _convert_weight_packed = torch.ops.sgl_kernel.convert_weight_packed
except (AttributeError, RuntimeError):
    # RuntimeError can occur when the sgl_kernel extension is present but was
    # compiled without RVV support (e.g., a CPU-only wheel for another arch).
    _convert_weight_packed = None

    if is_host_cpu_riscv():
        logger.warning(
            "[RVV] sgl_kernel.convert_weight_packed not found. "
            "Weight packing will be disabled; performance will be degraded. "
            "Ensure sgl-kernel was built with RVV support."
        )


class PackRVVWeightMethod:
    """
    Creates a new nn.Parameter for the packed weight so that the original
    embedding weight (tied via tie_weights()) is not modified in-place.
    Called by LMHead.__init__ when running on a RVV-capable host.
    """

    def __init__(self, weight_names):
        self.weight_names = weight_names

    def process_weights_after_loading(self, module) -> None:
        if not cpu_has_rvv_support() or _convert_weight_packed is None:
            return
        for name in self.weight_names:
            w = getattr(module, name)
            packed = torch.nn.Parameter(
                _convert_weight_packed(w.data), requires_grad=False
            )
            packed.__dict__ = (
                w.__dict__
            )  # preserve weight_attrs (input_dim, output_dim…)
            setattr(module, name, packed)
        module.use_riscv_rvv_backend = True


def _rvv_process_weight_after_loading(module, weight_names) -> None:
    """Process weights for RVV backend.

    1. Convert weights to RVV packed format via convert_weight_packed.
    2. Mark the module with use_riscv_rvv_backend = True.

    For INT8 quantization, use --quantization w8a8_int8 which goes through
    the standard W8A8Int8LinearMethod path.
    """
    if not cpu_has_rvv_support():
        return

    if getattr(module, "bias", None) is not None:
        if module.bias.dtype != torch.float32:
            module.bias = torch.nn.Parameter(module.bias.float(), requires_grad=False)

    if _convert_weight_packed is None:
        raise RuntimeError(
            "[RVV] sgl_kernel.convert_weight_packed unavailable; "
            "cannot pack weights for RVV backend. "
            "Rebuild sgl-kernel with RVV support."
        )
    for weight_name in weight_names:
        weight_tensor = getattr(module, weight_name)
        weight_tensor.data = _convert_weight_packed(weight_tensor.data)
    module.use_riscv_rvv_backend = True
