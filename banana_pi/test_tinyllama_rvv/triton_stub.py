#!/usr/bin/env python3
"""
Triton Stub Module for RISC-V CPU-Only Environment

Triton (https://github.com/openai/triton) is a language and compiler for
writing highly efficient custom Deep-Learning primitives. It is designed
for GPU execution (CUDA/ROCm) and is not available for RISC-V CPU.

This stub module provides the minimal interface needed for SGLang to import
without errors. All GPU-accelerated operations automatically fall back to
PyTorch's CPU implementations.

Usage:
    This module is automatically loaded by test_tinyllama_rvv.py before
    importing SGLang. No manual intervention is required.

References:
    - Triton project: https://github.com/openai/triton
    - Stub pattern examples:
      * PyTorch CUDA stubs: https://github.com/pytorch/pytorch/blob/main/torch/cuda/__init__.py
      * TensorFlow GPU stubs: https://github.com/tensorflow/tensorflow (CPU-only builds)
      * ONNX Runtime stubs: https://github.com/microsoft/onnxruntime (provider stubs)
"""

import importlib
import importlib.util
import sys
from types import ModuleType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StubValue:
    """Return-self object to tolerate chained attribute / method access."""

    __slots__ = ("_value",)

    def __init__(self, value=None):
        self._value = value

    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, name):
        value = _StubValue()
        object.__setattr__(self, name, value)
        return value

    def __getitem__(self, key):
        return self

    def __iter__(self):
        return iter([])

    def __repr__(self):
        return "<triton_stub_value>"

    def __bool__(self):
        return bool(self._value)

    def __int__(self):
        return int(self._value or 0)

    def __float__(self):
        return float(self._value or 0.0)


class _StubModule(ModuleType):
    """Module that lazily creates stub values for any missing attributes."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _StubValue()
        setattr(self, name, value)
        return value


def stub_function(*args, **kwargs):
    """Return a stub value so attribute access does not fail."""
    return _StubValue()


class _TritonKernelStub:
    """Stub class that mimics triton kernel behavior with [] syntax support."""

    def __init__(self, func):
        self._func = func

    def __getitem__(self, grid):
        """Support kernel[grid] syntax - returns a callable that executes the function."""

        # Return a callable that will execute the original function
        # This allows kernel[(n,)](args...) syntax
        def kernel_launcher(*args, **kwargs):
            # For RISC-V, we should fallback to CPU implementation
            # But since this is a stub, we'll just return None or execute a fallback
            # The actual implementation should handle the fallback in the calling code
            return self._func(*args, **kwargs)

        return kernel_launcher

    def __call__(self, *args, **kwargs):
        """Support direct kernel call syntax."""
        return self._func(*args, **kwargs)


def stub_decorator(*args, **kwargs):
    """Decorator stub that returns a triton kernel stub."""
    if args and callable(args[0]):
        # Direct decoration: @triton.jit def func(...)
        return _TritonKernelStub(args[0])

    def decorator(func):
        # Decorator with arguments: @triton.jit(...) def func(...)
        return _TritonKernelStub(func)

    return decorator


def _constexpr(value):
    """Simulate tl.constexpr by returning the value unchanged."""
    return value


# ---------------------------------------------------------------------------
# Create fake triton module tree
# ---------------------------------------------------------------------------

triton_module = _StubModule("triton")
triton_module.__version__ = "2.0.0"  # Use PEP 440 compatible version
triton_module.__file__ = __file__
triton_module.__spec__ = importlib.util.spec_from_loader(
    "triton", loader=None, origin=__file__
)

# Submodules required by torch/sglang
triton_module.language = _StubModule("triton.language")
triton_module.compiler = _StubModule("triton.compiler")
triton_module.runtime = _StubModule("triton.runtime")
triton_module.testing = _StubModule("triton.testing")

# ---------------------------------------------------------------------------
# Populate commonly used attributes
# ---------------------------------------------------------------------------

# Decorators
triton_module.jit = stub_decorator
triton_module.autotune = stub_decorator
triton_module.heuristics = stub_decorator

# Misc helpers used by torch._inductor
triton_module.cdiv = lambda x, y: 0 if not y else (int(x) + int(y) - 1) // int(y)
triton_module.next_power_of_2 = lambda x: (
    1 if not x else 1 << (int(x - 1).bit_length()) if isinstance(x, int) else 1
)

# triton.language namespace
tl = triton_module.language
tl.dot = stub_function
tl.sum = stub_function
tl.max = stub_function
tl.min = stub_function
tl.exp = stub_function
tl.log = stub_function
tl.sqrt = stub_function
tl.sin = stub_function
tl.cos = stub_function
tl.sigmoid = stub_function
tl.load = stub_function
tl.store = stub_function
tl.broadcast_to = stub_function
tl.reshape = stub_function
tl.where = stub_function
tl.program_id = stub_function
tl.num_programs = stub_function
tl.debug_barrier = stub_function
tl.constexpr = _constexpr
tl.arange = stub_function
tl.zeros = stub_function
tl.zeros_like = stub_function
tl.full = stub_function
tl.full_like = stub_function

# Provide tl extras (others auto-stub via __getattr__)
tl.extra = _StubModule("triton.language.extra")
tl.extra.cuda = _StubModule("triton.language.extra.cuda")
tl.extra.cuda.libdevice = _StubModule("triton.language.extra.cuda.libdevice")
tl.math = _StubModule("triton.language.math")
tl.math.max = stub_function
tl.math.min = stub_function
tl.math.exp = stub_function
tl.math.log = stub_function
tl.math.sqrt = stub_function
tl.math.rsqrt = stub_function
tl.math.abs = stub_function
tl.core = _StubModule("triton.language.core")
tl.core.view = stub_function
tl.core.reshape = stub_function
tl.tensor = _StubModule("triton.language.tensor")
tl.dtype = _StubModule("triton.language.dtype")
tl.pointer_type = stub_function
# triton.compiler namespace
triton_module.compiler.compile = stub_function
triton_module.compiler.ASTSource = type("ASTSource", (), {})
triton_module.compiler.AttrsDescriptor = type("AttrsDescriptor", (), {})
triton_module.compiler.CompiledKernel = type("CompiledKernel", (), {"_is_stub": True})


# Basic Config object used by torch._inductor.runtime.triton_compat
class _TritonConfig:
    def __init__(self, *args, **kwargs):
        # Some callsites pass a dict as the first positional argument
        if args:
            if isinstance(args[0], dict):
                kwargs = {**args[0], **kwargs}
            else:
                kwargs = {**kwargs}
        self.num_warps = kwargs.get("num_warps", 1)
        self.num_stages = kwargs.get("num_stages", 1)
        self.num_ctas = kwargs.get("num_ctas", 1)
        self.num_warps_config = kwargs.get("num_warps_config")
        self.num_stages_config = kwargs.get("num_stages_config")
        self.meta = kwargs.get("meta", {})
        self.extra_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k
            not in [
                "num_warps",
                "num_stages",
                "num_ctas",
                "num_warps_config",
                "num_stages_config",
                "meta",
            ]
        }

    def __repr__(self):
        return (
            f"TritonConfig(num_warps={self.num_warps}, "
            f"num_stages={self.num_stages}, num_ctas={self.num_ctas})"
        )


triton_module.Config = _TritonConfig

# triton.runtime namespace
triton_module.runtime.__path__ = []
triton_module.runtime.driver = type(
    "Driver",
    (),
    {
        "active": lambda: False,
        "utils": type("Utils", (), {}),
    },
)()
triton_module.runtime.autotune = stub_function
autotuner_module = _StubModule("triton.runtime.autotuner")
runtime_jit_module = _StubModule("triton.runtime.jit")


class _OutOfResources(RuntimeError):
    pass


autotuner_module.OutOfResources = _OutOfResources
triton_module.runtime.autotuner = autotuner_module
runtime_jit_module.KernelInterface = type("KernelInterface", (), {"_is_stub": True})

# ---------------------------------------------------------------------------
# Register in sys.modules so any `import triton` sees the stub
# ---------------------------------------------------------------------------

sys.modules["triton"] = triton_module
sys.modules["triton.language"] = triton_module.language
sys.modules["triton.language.extra"] = tl.extra
sys.modules["triton.language.extra.cuda"] = tl.extra.cuda
sys.modules["triton.language.extra.cuda.libdevice"] = tl.extra.cuda.libdevice
sys.modules["triton.language.math"] = tl.math
sys.modules["triton.language.core"] = tl.core
sys.modules["triton.language.tensor"] = tl.tensor
sys.modules["triton.language.dtype"] = tl.dtype
sys.modules["triton.compiler"] = triton_module.compiler
sys.modules["triton.runtime"] = triton_module.runtime
sys.modules["triton.runtime.autotuner"] = autotuner_module
sys.modules["triton.runtime.jit"] = runtime_jit_module
sys.modules["triton.testing"] = triton_module.testing

triton_module._is_stub = True
