#!/usr/bin/env python3
"""
vLLM Stub Module for RISC-V CPU-Only Environment

vLLM (https://github.com/vllm-project/vllm) is a high-throughput and
memory-efficient inference and serving engine for LLMs. It is designed
for GPU execution and is not available for RISC-V CPU.

This stub module provides the minimal interface needed for SGLang to import
vLLM modules without errors. All GPU-accelerated operations automatically
fall back to PyTorch's CPU implementations or SGLang's own implementations.

Usage:
    This module should be loaded before importing SGLang modules that depend
    on vLLM. It can be loaded automatically by test_tinyllama_riscv.py or
    manually imported.

References:
    - vLLM project: https://github.com/vllm-project/vllm
    - Similar to triton_stub.py for Triton compatibility
"""

import sys
from types import ModuleType


class _StubModule(ModuleType):
    """Stub module that allows attribute access without errors."""

    def __init__(self, name):
        super().__init__(name)
        self.__path__ = []

    def __getattr__(self, name):
        # Return a stub value for any attribute access
        stub_value = _StubValue()
        setattr(self, name, stub_value)
        return stub_value


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

    def __setattr__(self, name, value):
        object.__setattr__(self, name, value)


# Create vllm module structure
vllm_module = _StubModule("vllm")
vllm_module.__version__ = "0.0.0-stub"

# Create vllm._custom_ops module (used in layernorm.py for HIP)
vllm_module._custom_ops = _StubModule("vllm._custom_ops")

# Create vllm.model_executor structure
vllm_module.model_executor = _StubModule("vllm.model_executor")
vllm_module.model_executor.layers = _StubModule("vllm.model_executor.layers")
vllm_module.model_executor.layers.layernorm = _StubModule(
    "vllm.model_executor.layers.layernorm"
)

# Import SGLang's own RMSNorm and GemmaRMSNorm implementations
# These are the actual implementations that will be used on RISC-V
try:
    from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm

    # Export SGLang's implementations as vLLM's classes
    # This allows code that imports from vllm to use SGLang's implementations
    vllm_module.model_executor.layers.layernorm.RMSNorm = RMSNorm
    vllm_module.model_executor.layers.layernorm.GemmaRMSNorm = GemmaRMSNorm
except ImportError:
    # If SGLang's layernorm is not available, create stub classes
    class _StubRMSNorm:
        """Stub RMSNorm class."""

        def __init__(self, *args, **kwargs):
            pass

    class _StubGemmaRMSNorm:
        """Stub GemmaRMSNorm class."""

        def __init__(self, *args, **kwargs):
            pass

    vllm_module.model_executor.layers.layernorm.RMSNorm = _StubRMSNorm
    vllm_module.model_executor.layers.layernorm.GemmaRMSNorm = _StubGemmaRMSNorm

# Create vllm.distributed structure (used in parallel_state.py)
vllm_module.distributed = _StubModule("vllm.distributed")
vllm_module.distributed.parallel_state = _StubModule("vllm.distributed.parallel_state")

# Register in sys.modules so any `import vllm` sees the stub
sys.modules["vllm"] = vllm_module
sys.modules["vllm._custom_ops"] = vllm_module._custom_ops
sys.modules["vllm.model_executor"] = vllm_module.model_executor
sys.modules["vllm.model_executor.layers"] = vllm_module.model_executor.layers
sys.modules["vllm.model_executor.layers.layernorm"] = (
    vllm_module.model_executor.layers.layernorm
)
sys.modules["vllm.distributed"] = vllm_module.distributed
sys.modules["vllm.distributed.parallel_state"] = vllm_module.distributed.parallel_state

vllm_module._is_stub = True
