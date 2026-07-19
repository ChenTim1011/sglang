"""CPU RVV TorchInductor policy for supported text-model linear regions.

This file intentionally keeps SGLang serving policy outside model definitions.
PyTorch Inductor owns kernel selection/codegen; this policy only decides which
SGLang callsites should enter Inductor and when to fall back.
"""

import logging
import time
import weakref
from dataclasses import dataclass
from typing import NamedTuple, Optional

import psutil
import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

try:
    from torch._inductor import inductor_prims as _inductor_prims

    _RVV_PACK_BF16_WEIGHT_OP = getattr(_inductor_prims, "rvv_pack_bf16_weight", None)
    _RVV_PACKED_BF16_LINEAR_OP = getattr(
        _inductor_prims, "rvv_packed_bf16_linear", None
    )
    _RVV_PACKED_WEIGHT_BLOCK_N = getattr(
        _inductor_prims, "RVV_BF16_PACKED_WEIGHT_BLOCK_N", 32
    )
except ImportError:
    _RVV_PACK_BF16_WEIGHT_OP = None
    _RVV_PACKED_BF16_LINEAR_OP = None
    _RVV_PACKED_WEIGHT_BLOCK_N = 32

_RVV_REGIONAL_LINEAR_TARGETS = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")
_RVV_REGIONAL_TARGETS = (*_RVV_REGIONAL_LINEAR_TARGETS, "lm_head")
_RVV_REGIONAL_COMPILED_BY_SHAPE = {}
_RVV_PACKED_WEIGHT_BUFFER = "_sglang_rvv_packed_weight"
_RVV_PACKED_MEMORY_RESERVE_BYTES = 1024 * 1024 * 1024


class _RegionalShapeKey(NamedTuple):
    dtype: torch.dtype
    device_type: str
    projection: str
    rows: int
    out_features: int
    in_features: int
    mode: str


@dataclass(frozen=True)
class _PackedWeightSource:
    tensor_ref: weakref.ReferenceType
    data_ptr: int
    shape: tuple[int, ...]
    stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device

    @classmethod
    def capture(cls, weight: torch.Tensor):
        return cls(
            tensor_ref=weakref.ref(weight),
            data_ptr=weight.untyped_storage().data_ptr(),
            shape=tuple(weight.shape),
            stride=tuple(weight.stride()),
            dtype=weight.dtype,
            device=weight.device,
        )

    def matches(self, weight: torch.Tensor) -> bool:
        try:
            return (
                self.tensor_ref() is weight
                and self.data_ptr == weight.untyped_storage().data_ptr()
                and self.shape == tuple(weight.shape)
                and self.stride == tuple(weight.stride())
                and self.dtype == weight.dtype
                and self.device == weight.device
            )
        except RuntimeError:
            return False


@dataclass(frozen=True)
class _RegionalModelAdapter:
    name: str
    model_type_names: tuple[str, ...]
    projection_type_names: tuple[tuple[str, tuple[str, ...]], ...] = ()
    requires_tp1: bool = False
    requires_unquantized: bool = False

    def projection_types(self, name: str) -> tuple[str, ...]:
        return dict(self.projection_type_names).get(name, ())


_TORCH_NATIVE_ADAPTER = _RegionalModelAdapter(
    name="torch_native",
    model_type_names=(
        "TorchNativeLlamaForCausalLM",
        "TorchNativePhi3ForCausalLM",
    ),
)
_QWEN_PROJECTION_TYPES = (
    ("qkv_proj", ("QKVParallelLinear",)),
    ("o_proj", ("RowParallelLinear",)),
    ("gate_up_proj", ("MergedColumnParallelLinear",)),
    ("down_proj", ("RowParallelLinear",)),
)
_QWEN2_ADAPTER = _RegionalModelAdapter(
    name="qwen2",
    model_type_names=("Qwen2ForCausalLM",),
    projection_type_names=_QWEN_PROJECTION_TYPES,
    requires_tp1=True,
    requires_unquantized=True,
)
_QWEN3_ADAPTER = _RegionalModelAdapter(
    name="qwen3",
    model_type_names=("Qwen3ForCausalLM",),
    projection_type_names=_QWEN_PROJECTION_TYPES,
    requires_tp1=True,
    requires_unquantized=True,
)
_REGIONAL_MODEL_ADAPTERS = (
    _TORCH_NATIVE_ADAPTER,
    _QWEN2_ADAPTER,
    _QWEN3_ADAPTER,
)


def _resolve_model_adapter(model: nn.Module) -> Optional[_RegionalModelAdapter]:
    model_type = type(model).__name__
    return next(
        (
            adapter
            for adapter in _REGIONAL_MODEL_ADAPTERS
            if model_type in adapter.model_type_names
        ),
        None,
    )


def _iter_projection_modules(model: nn.Module):
    layer_model = getattr(model, "model", None)
    layers = getattr(layer_model, "layers", None)
    if layers is None:
        return
    for layer in layers:
        self_attn = getattr(layer, "self_attn", None)
        mlp = getattr(layer, "mlp", None)
        for owner, name in (
            (self_attn, "qkv_proj"),
            (self_attn, "o_proj"),
            (mlp, "gate_up_proj"),
            (mlp, "down_proj"),
        ):
            yield name, getattr(owner, name, None) if owner is not None else None


def _validate_adapter_projection(
    adapter: _RegionalModelAdapter,
    name: str,
    module: nn.Module,
) -> None:
    if adapter is _TORCH_NATIVE_ADAPTER:
        if not isinstance(module, nn.Linear):
            raise RuntimeError(
                f"RVV Inductor {adapter.name} adapter expected nn.Linear for {name}"
            )
        return

    expected_types = adapter.projection_types(name)
    if type(module).__name__ not in expected_types:
        raise RuntimeError(
            f"RVV Inductor {adapter.name} adapter does not support "
            f"{name} type {type(module).__name__}"
        )
    if adapter.requires_tp1 and getattr(module, "tp_size", None) != 1:
        raise RuntimeError(
            f"RVV Inductor {adapter.name} adapter currently requires TP=1 for {name}"
        )
    quant_method = getattr(module, "quant_method", None)
    if (
        adapter.requires_unquantized
        and type(quant_method).__name__ != "UnquantizedLinearMethod"
    ):
        raise RuntimeError(
            f"RVV Inductor {adapter.name} adapter currently requires unquantized {name}"
        )


def _linear_shape(input: torch.Tensor, weight: torch.Tensor):
    if input.ndim != 2 or weight.ndim != 2:
        return None
    m, k = input.shape
    n, weight_k = weight.shape
    if k != weight_k:
        return None
    return int(m), int(n), int(k)


def _regional_bucket_rows(rows: int, *, allow_small_batch: bool = False) -> int:
    if rows == 1:
        return 1
    buckets = (2, 4, 8, 16, 32, 64, 128) if allow_small_batch else (16, 32, 64, 128)
    for bucket in buckets:
        if rows <= bucket:
            return bucket
    return rows


def _should_use_rvv_inductor_regional_shape(
    name: str,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    *,
    allow_small_batch: bool = False,
) -> bool:
    shape = _linear_shape(input, weight)
    if shape is None:
        return False
    m, n, k = shape
    supported_rows = m == 1 or (allow_small_batch and 2 <= m <= 15)
    if name != "lm_head":
        supported_rows = supported_rows or 16 <= m <= 128
    return (
        name in _RVV_REGIONAL_TARGETS
        and bias is None
        and input.device.type == "cpu"
        and weight.device.type == "cpu"
        and input.dtype == torch.bfloat16
        and weight.dtype == torch.bfloat16
        and supported_rows
        and n >= 1024
        and k >= 1024
        and m * n * k >= 1_000_000
    )


def _rvv_qkv_region(input: torch.Tensor, weight: torch.Tensor):
    return F.linear(input, weight, None)


def _rvv_o_proj_region(input: torch.Tensor, weight: torch.Tensor):
    return F.linear(input, weight, None)


def _rvv_gate_up_region(input: torch.Tensor, weight: torch.Tensor):
    return F.linear(input, weight, None)


def _rvv_down_region(input: torch.Tensor, weight: torch.Tensor):
    return F.linear(input, weight, None)


def _rvv_lm_head_region(input: torch.Tensor, weight: torch.Tensor):
    return F.linear(input, weight, None)


def _rvv_qkv_packed_region(
    input: torch.Tensor, packed_weight: torch.Tensor, out_features: int
):
    return _RVV_PACKED_BF16_LINEAR_OP(input, packed_weight, out_features)


def _rvv_o_proj_packed_region(
    input: torch.Tensor, packed_weight: torch.Tensor, out_features: int
):
    return _RVV_PACKED_BF16_LINEAR_OP(input, packed_weight, out_features)


def _rvv_gate_up_packed_region(
    input: torch.Tensor, packed_weight: torch.Tensor, out_features: int
):
    return _RVV_PACKED_BF16_LINEAR_OP(input, packed_weight, out_features)


def _rvv_down_packed_region(
    input: torch.Tensor, packed_weight: torch.Tensor, out_features: int
):
    return _RVV_PACKED_BF16_LINEAR_OP(input, packed_weight, out_features)


def _rvv_lm_head_packed_region(
    input: torch.Tensor, packed_weight: torch.Tensor, out_features: int
):
    return _RVV_PACKED_BF16_LINEAR_OP(input, packed_weight, out_features)


_RVV_REGIONAL_FUNCTIONS = {
    "qkv_proj": _rvv_qkv_region,
    "o_proj": _rvv_o_proj_region,
    "gate_up_proj": _rvv_gate_up_region,
    "down_proj": _rvv_down_region,
    "lm_head": _rvv_lm_head_region,
}
_RVV_EXPLICIT_PACKED_FUNCTIONS = {
    "qkv_proj": _rvv_qkv_packed_region,
    "o_proj": _rvv_o_proj_packed_region,
    "gate_up_proj": _rvv_gate_up_packed_region,
    "down_proj": _rvv_down_packed_region,
    "lm_head": _rvv_lm_head_packed_region,
}


def _rvv_cpu_capability() -> str:
    try:
        return str(torch.backends.cpu.get_cpu_capability()).upper()
    except Exception:
        return ""


def _is_cpu_device(value) -> bool:
    if isinstance(value, torch.device):
        return value.type == "cpu"
    return str(value) == "cpu"


def _is_bfloat16_dtype(value) -> bool:
    return value == torch.bfloat16 or str(value) in {"bfloat16", "torch.bfloat16"}


def _explicit_packed_ops_available() -> bool:
    return callable(_RVV_PACK_BF16_WEIGHT_OP) and callable(_RVV_PACKED_BF16_LINEAR_OP)


def _explicit_packing_enabled(server_args) -> bool:
    return _explicit_packing_requested(server_args) and _explicit_packed_ops_available()


def _simple_rvv_inductor_requested(server_args) -> bool:
    return bool(
        server_args is not None
        and getattr(server_args, "enable_cpu_rvv_inductor", False)
    )


def _regional_policy_requested(server_args) -> bool:
    return _simple_rvv_inductor_requested(server_args) or bool(
        server_args is not None
        and getattr(server_args, "cpu_compile_mode", "off") == "regional"
    )


def _explicit_packing_requested(server_args) -> bool:
    return _simple_rvv_inductor_requested(server_args) or bool(
        server_args is not None
        and getattr(server_args, "cpu_rvv_packed_weight_mode", "off") == "explicit"
    )


def _packed_weight_memory_budget_mib(server_args):
    if server_args is None:
        return None
    budget = getattr(server_args, "cpu_rvv_memory_budget_mib", None)
    if budget is not None:
        return budget
    return getattr(server_args, "cpu_rvv_packed_weight_max_mib", None)


def _resolve_packed_weight_budget(server_args):
    configured_mib = _packed_weight_memory_budget_mib(server_args)
    if configured_mib is not None:
        if configured_mib < 0:
            raise ValueError("cpu_rvv_memory_budget_mib must be non-negative")
        return (
            int(configured_mib * 1024 * 1024),
            "user",
            None,
            None,
            None,
        )

    memory = psutil.virtual_memory()
    reserve_bytes = max(
        _RVV_PACKED_MEMORY_RESERVE_BYTES,
        int(memory.total * 0.10),
    )
    return (
        max(0, int(memory.available) - reserve_bytes),
        "available_memory",
        int(memory.available),
        int(memory.total),
        reserve_bytes,
    )


def _expected_packed_weight_nbytes(module: nn.Module) -> int:
    weight = getattr(module, "weight", None)
    if not (
        isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.device.type == "cpu"
        and weight.dtype == torch.bfloat16
    ):
        return 0
    out_features, in_features = weight.shape
    padded_out_features = (
        (int(out_features) + _RVV_PACKED_WEIGHT_BLOCK_N - 1)
        // _RVV_PACKED_WEIGHT_BLOCK_N
        * _RVV_PACKED_WEIGHT_BLOCK_N
    )
    return padded_out_features * int(in_features) * weight.element_size()


def _register_explicit_packed_weight(module: nn.Module) -> int:
    weight = getattr(module, "weight", None)
    if not (
        _explicit_packed_ops_available()
        and isinstance(weight, torch.Tensor)
        and weight.ndim == 2
        and weight.device.type == "cpu"
        and weight.dtype == torch.bfloat16
    ):
        return 0
    if _RVV_PACKED_WEIGHT_BUFFER in module._buffers:
        return 0

    with torch.inference_mode(False), torch.no_grad():
        packed_weight = _RVV_PACK_BF16_WEIGHT_OP(weight.detach())
    module.register_buffer(
        _RVV_PACKED_WEIGHT_BUFFER,
        packed_weight,
        persistent=False,
    )
    module._sglang_rvv_packed_source = _PackedWeightSource.capture(weight)
    return packed_weight.numel() * packed_weight.element_size()


def _valid_explicit_packed_weight(module: nn.Module):
    weight = getattr(module, "weight", None)
    packed_weight = module._buffers.get(_RVV_PACKED_WEIGHT_BUFFER)
    if not isinstance(weight, torch.Tensor) or packed_weight is None:
        return None
    source = getattr(module, "_sglang_rvv_packed_source", None)
    if not isinstance(source, _PackedWeightSource) or not source.matches(weight):
        return None
    return packed_weight


def _compile_rvv_regional_linear(shape_key):
    compiled = _RVV_REGIONAL_COMPILED_BY_SHAPE.get(shape_key)
    if compiled is None:
        logger.info(
            "Creating RVV Inductor regional callable: projection=%s, "
            "M=%d, N=%d, K=%d, mode=%s, device=%s.",
            shape_key.projection,
            shape_key.rows,
            shape_key.out_features,
            shape_key.in_features,
            shape_key.mode,
            shape_key.device_type,
        )
        functions = (
            _RVV_EXPLICIT_PACKED_FUNCTIONS
            if shape_key.mode == "explicit_packed"
            else _RVV_REGIONAL_FUNCTIONS
        )
        target = functions[shape_key.projection]
        compiled = torch.compile(
            target,
            dynamic=False,
            fullgraph=True,
        )
        _RVV_REGIONAL_COMPILED_BY_SHAPE[shape_key] = compiled
    return compiled


@dataclass(frozen=True)
class _RvvInductorLinearPolicy:
    projection: str
    allow_small_batch: bool = False

    def apply(
        self,
        module: nn.Module,
        input: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        weight = module.weight
        if (
            torch.compiler.is_compiling()
            or not _should_use_rvv_inductor_regional_shape(
                self.projection,
                input,
                weight,
                None,
                allow_small_batch=self.allow_small_batch,
            )
            or (
                bias is not None
                and (
                    not isinstance(bias, torch.Tensor)
                    or bias.device != input.device
                    or bias.dtype != input.dtype
                )
            )
        ):
            return F.linear(input, weight, bias)

        m, n, k = _linear_shape(input, weight)
        bucket_m = _regional_bucket_rows(m, allow_small_batch=self.allow_small_batch)
        compile_input = input
        if bucket_m != m:
            compile_input = F.pad(input, (0, 0, 0, bucket_m - m))
        packed_weight = _valid_explicit_packed_weight(module)
        mode = "explicit_packed" if packed_weight is not None else "row_major"
        shape_key = _RegionalShapeKey(
            input.dtype,
            input.device.type,
            self.projection,
            bucket_m,
            n,
            k,
            mode,
        )
        compiled = _compile_rvv_regional_linear(shape_key)
        if packed_weight is None:
            output = compiled(compile_input, weight)
        else:
            output = compiled(compile_input, packed_weight, n)
        output = output if bucket_m == m else output[:m]
        return output if bias is None else output + bias


def _install_rvv_inductor_linear_policy(
    module: nn.Module,
    projection: str,
    *,
    allow_small_batch: bool = False,
) -> bool:
    if module is None or getattr(module, "_sglang_cpu_linear_policy", None) is not None:
        return False
    if not isinstance(getattr(module, "weight", None), torch.Tensor):
        return False
    module._sglang_cpu_linear_policy = _RvvInductorLinearPolicy(
        projection=projection,
        allow_small_batch=allow_small_batch,
    )
    return True


def invalidate_rvv_inductor_packed_weights(model: nn.Module, *_) -> None:
    """Invalidate packed side tensors after a public weight lifecycle event."""
    for module in model.modules():
        if _RVV_PACKED_WEIGHT_BUFFER in module._buffers:
            module._sglang_rvv_packed_source = None


def _should_install_rvv_inductor_regional_policy(
    model: nn.Module,
    server_args,
    force: bool,
) -> bool:
    if _resolve_model_adapter(model) is None:
        return False
    if force:
        return True
    if server_args is None:
        return False
    return (
        _is_cpu_device(getattr(server_args, "device", None))
        and _is_bfloat16_dtype(getattr(server_args, "dtype", None))
        and getattr(server_args, "attention_backend", None) == "torch_native"
        and _regional_policy_requested(server_args)
        and not bool(getattr(server_args, "enable_lora", False))
        and _rvv_cpu_capability() == "RVV"
    )


def _simple_rvv_inductor_unavailable_reason(model: nn.Module, server_args):
    if not _simple_rvv_inductor_requested(server_args):
        return None
    if _resolve_model_adapter(model) is None:
        return "the loaded model does not have a supported RVV adapter"
    if not _is_cpu_device(getattr(server_args, "device", None)):
        return "--device must be cpu"
    if not _is_bfloat16_dtype(getattr(server_args, "dtype", None)):
        return "--dtype must be bfloat16"
    if getattr(server_args, "attention_backend", None) != "torch_native":
        return "--attention-backend must be torch_native"
    if bool(getattr(server_args, "enable_lora", False)):
        return "LoRA is not supported by the RVV regional policy"
    capability = _rvv_cpu_capability()
    if capability != "RVV":
        return f"the detected CPU capability is {capability}, not RVV"
    return None


def _validate_tied_lm_head(model: nn.Module) -> None:
    config = getattr(model, "config", None)
    if not bool(getattr(config, "tie_word_embeddings", False)):
        return
    layer_model = getattr(model, "model", None)
    embed_tokens = getattr(layer_model, "embed_tokens", None)
    if getattr(model, "lm_head", None) is not embed_tokens:
        raise RuntimeError(
            "RVV Inductor regional policy requires tied lm_head and "
            "embed_tokens to remain the same module"
        )


def install_rvv_inductor_regional_policy(
    model: nn.Module,
    server_args=None,
    *,
    force: bool = False,
) -> int:
    if getattr(model, "_sglang_rvv_inductor_regional_policy_installed", False):
        return 0
    unavailable_reason = (
        None if force else _simple_rvv_inductor_unavailable_reason(model, server_args)
    )
    if unavailable_reason is not None:
        raise RuntimeError(
            "--enable-cpu-rvv-inductor cannot be used because " + unavailable_reason
        )
    if not _should_install_rvv_inductor_regional_policy(model, server_args, force):
        return 0

    adapter = _resolve_model_adapter(model)
    if adapter is None:
        return 0
    _validate_tied_lm_head(model)
    allow_small_batch = _simple_rvv_inductor_requested(server_args) or bool(
        server_args is not None
        and getattr(server_args, "cpu_rvv_regional_small_batch", False)
    )

    layer_model = getattr(model, "model", None)
    layers = getattr(layer_model, "layers", None)
    if layers is None:
        return 0
    projection_modules = list(_iter_projection_modules(model))
    for name, module in projection_modules:
        _validate_adapter_projection(adapter, name, module)

    pack_started = time.perf_counter()
    packed_bytes = 0
    packed_tensors = 0
    explicit_packing_enabled = _explicit_packing_enabled(server_args)
    explicit_packing_requested = _explicit_packing_requested(server_args)
    if explicit_packing_requested:
        (
            max_bytes,
            budget_source,
            available_bytes,
            total_memory_bytes,
            reserve_bytes,
        ) = _resolve_packed_weight_budget(server_args)
    else:
        max_bytes = None
        budget_source = "disabled"
        available_bytes = None
        total_memory_bytes = None
        reserve_bytes = None
    eligible_bytes = 0
    eligible_tensors = 0
    skipped_bytes = 0
    skipped_tensors = 0
    packing_plan = []
    if explicit_packing_enabled:
        modules_to_pack = [module for _, module in projection_modules]
        modules_to_pack.append(model.lm_head)
        for module in modules_to_pack:
            expected_nbytes = _expected_packed_weight_nbytes(module)
            if expected_nbytes:
                eligible_tensors += 1
                eligible_bytes += expected_nbytes
                packing_plan.append((module, expected_nbytes))

        for module, expected_nbytes in packing_plan:
            if max_bytes is not None and packed_bytes + expected_nbytes > max_bytes:
                skipped_tensors += 1
                skipped_bytes += expected_nbytes
                continue
            packed_nbytes = _register_explicit_packed_weight(module)
            if packed_nbytes:
                packed_tensors += 1
                packed_bytes += packed_nbytes
    model._sglang_rvv_explicit_pack_stats = {
        "requested": explicit_packing_requested,
        "available": _explicit_packed_ops_available(),
        "enabled": explicit_packing_enabled,
        "seconds": time.perf_counter() - pack_started,
        "tensors": packed_tensors,
        "bytes": packed_bytes,
        "eligible_tensors": eligible_tensors,
        "eligible_bytes": eligible_bytes,
        "skipped_tensors": skipped_tensors,
        "skipped_bytes": skipped_bytes,
        "max_bytes": max_bytes,
        "budget_source": budget_source,
        "available_bytes": available_bytes,
        "total_memory_bytes": total_memory_bytes,
        "reserve_bytes": reserve_bytes,
    }

    lm_head_installed = _install_rvv_inductor_linear_policy(
        getattr(model, "lm_head", None),
        "lm_head",
        allow_small_batch=allow_small_batch,
    )

    installed = 0
    for name, module in projection_modules:
        installed += int(
            _install_rvv_inductor_linear_policy(
                module,
                name,
                allow_small_batch=allow_small_batch,
            )
        )

    if installed or lm_head_installed:
        model._sglang_rvv_inductor_regional_policy_installed = True
        model.register_load_state_dict_post_hook(invalidate_rvv_inductor_packed_weights)
        logger.info(
            "Installed RVV Inductor regional policy with the %s adapter on %d "
            "projection modules (lm_head=%s).",
            adapter.name,
            installed,
            lm_head_installed,
        )
        if allow_small_batch:
            logger.info(
                "Enabled RVV Inductor regional routing for M=2-15 "
                "(buckets 2, 4, 8, 16)."
            )
        if explicit_packing_requested and not explicit_packing_enabled:
            logger.warning(
                "Explicit RVV packed weights were requested, but the installed "
                "PyTorch does not provide the required primitives; using row-major "
                "weights."
            )
        elif packed_tensors or skipped_tensors:
            logger.info(
                "Prepared %d explicit RVV packed weights (%.2f MiB) in %.3f s; "
                "skipped %d because of the memory budget.",
                packed_tensors,
                packed_bytes / (1024 * 1024),
                model._sglang_rvv_explicit_pack_stats["seconds"],
                skipped_tensors,
            )
            if skipped_tensors and budget_source == "available_memory":
                logger.warning(
                    "RVV packed-weight coverage was limited to protect host "
                    "memory: packed %.2f MiB of %.2f MiB. Set "
                    "--cpu-rvv-memory-budget-mib only after checking peak RSS.",
                    packed_bytes / (1024 * 1024),
                    eligible_bytes / (1024 * 1024),
                )
    return installed
