#!/usr/bin/env python3
"""
Verify RVV support. Aligns with cpu_has_rvv_support() in sglang.srt.utils.common.
Exit code: 0 on success, non-zero on failure.
"""

import os
import sys

import torch


def main():
    if "riscv" not in __import__("platform").machine().lower():
        print("✗ Not RISC-V CPU")
        return 1

    if os.getenv("SGLANG_DISABLE_RVV_KERNELS", "").lower() in ("1", "true", "yes"):
        print("✗ SGLANG_DISABLE_RVV_KERNELS is set")
        return 1

    try:
        torch.ops.sgl_kernel.weight_packed_linear  # Same probe as cpu_has_rvv_support
    except (AttributeError, Exception) as e:
        print(f"✗ RVV kernels not available: {e}")
        return 1

    print("✓ RVV support verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
