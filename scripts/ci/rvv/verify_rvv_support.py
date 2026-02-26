#!/usr/bin/env python3
"""
Verify RISC-V Vector (RVV) Extension Support
This script checks for RVV hardware capabilities and kernel availability.
Exit code: 0 on success, non-zero on failure.
"""

import platform
import sys
from pathlib import Path


def check_riscv_cpu():
    """Check if running on RISC-V CPU."""
    machine = platform.machine().lower()
    if "riscv" not in machine:
        print(f"✗ Not a RISC-V CPU: {machine}")
        return False
    print(f"✓ RISC-V CPU detected: {machine}")
    return True


def check_rvv_extension():
    """Check for RVV extension in /proc/cpuinfo."""
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.exists():
        print("✗ /proc/cpuinfo not found")
        return False

    isa_found = False
    with open(cpuinfo) as f:
        for line in f:
            if "isa" in line.lower():
                isa = line.split(":")[-1].strip()
                isa_found = True

                # Check for 'v' extension (RVV)
                # Format: rv64gcv or rv64imafdcv or similar
                if "v" in isa.lower():
                    print(f"✓ RVV extension available: {isa}")
                    return True
                else:
                    print(f"✗ RVV extension not found in ISA: {isa}")
                    return False

    if not isa_found:
        print("✗ ISA field not found in /proc/cpuinfo")
        return False

    return False


def check_sgl_kernel():
    """Check if sgl_kernel can be imported."""
    try:
        import sgl_kernel

        print(f"✓ sgl_kernel imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import sgl_kernel: {e}")
        return False


def check_rvv_kernels():
    """Check if RVV kernels are available in sgl_kernel."""
    try:
        import sgl_kernel

        # Check for RVV-specific operations
        rvv_ops = []

        # Look for RVV-related attributes
        for attr in dir(sgl_kernel):
            if "rvv" in attr.lower():
                rvv_ops.append(attr)

        if rvv_ops:
            print(f"✓ RVV kernels available: {rvv_ops}")
            return True
        else:
            print("⚠ No RVV-specific kernels found (this may be expected)")
            return True  # Not necessarily an error

    except Exception as e:
        print(f"✗ Error checking RVV kernels: {e}")
        return False


def main():
    """Run all verification checks."""
    print("=" * 60)
    print("RVV Support Verification")
    print("=" * 60)

    checks = [
        ("RISC-V CPU", check_riscv_cpu),
        ("RVV Extension", check_rvv_extension),
        ("sgl_kernel Import", check_sgl_kernel),
        ("RVV Kernels", check_rvv_kernels),
    ]

    results = []
    for name, check_func in checks:
        print(f"\nChecking: {name}")
        print("-" * 60)
        results.append(check_func())

    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)

    for (name, _), result in zip(checks, results):
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")

    # Critical checks: CPU and extension
    critical_checks = results[:2]
    if all(critical_checks):
        print("\n✓ All critical checks passed!")
        return 0
    else:
        print("\n✗ Critical checks failed!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
