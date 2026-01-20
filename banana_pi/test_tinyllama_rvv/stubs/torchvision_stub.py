#!/usr/bin/env python3
"""
Torchvision Stub Module for RISC-V CPU-Only Environment

Torchvision wheels are often unavailable for RISC-V.
This stub provides minimal interfaces (Transforms, InterpolationMode) required
by SGLang's multimodal utilities, allowing text-only models to run without
building the full vision stack.
"""

import sys
from types import ModuleType


class _StubModule(ModuleType):
    __version__ = "0.0.0-stub"
    _is_stub = True

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        value = _StubModule(name)
        setattr(self, name, value)
        return value


class _StubValue:
    def __call__(self, *args, **kwargs):
        return self

    def __getattr__(self, _):
        return _StubValue()


# Create module structure
torchvision = _StubModule("torchvision")
torchvision.transforms = _StubModule("torchvision.transforms")
torchvision.transforms.functional = _StubModule("torchvision.transforms.functional")


# Define InterpolationMode class
class InterpolationMode:
    BICUBIC = "bicubic"
    NEAREST = "nearest"
    BILINEAR = "bilinear"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"


torchvision.transforms.functional.InterpolationMode = InterpolationMode

# Register in sys.modules
sys.modules["torchvision"] = torchvision
sys.modules["torchvision.transforms"] = torchvision.transforms
sys.modules["torchvision.transforms.functional"] = torchvision.transforms
sys.modules["torchvision.transforms.functional"].InterpolationMode = InterpolationMode
