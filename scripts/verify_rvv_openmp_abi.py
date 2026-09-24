#!/usr/bin/env python3
"""Fail unless the running RVV stack uses libgomp and only libgomp."""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
from pathlib import Path


def shared_object_for(module_file: str) -> Path:
    package_dir = Path(module_file).resolve().parent
    candidates = sorted(package_dir.glob("common_ops*.so"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"expected one sgl_kernel common_ops shared object, found {candidates}"
        )
    return candidates[0]


def command_output(*args: str) -> str:
    return subprocess.check_output(args, text=True, stderr=subprocess.STDOUT)


def op_registered(namespace, name: str) -> bool:
    try:
        getattr(namespace, name)
    except (AttributeError, RuntimeError):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-preload",
        action="store_true",
        help="do not reject a non-empty LD_PRELOAD (for diagnosis only)",
    )
    args = parser.parse_args()

    if platform.machine() != "riscv64":
        raise RuntimeError(f"expected riscv64, got {platform.machine()}")
    if os.environ.get("LD_PRELOAD") and not args.allow_preload:
        raise RuntimeError(
            f"LD_PRELOAD must be empty, got {os.environ['LD_PRELOAD']!r}"
        )

    import sgl_kernel
    import torch

    extension = shared_object_for(sgl_kernel.__file__)
    dynamic = command_output("readelf", "-d", str(extension))
    symbols = command_output("readelf", "-Ws", str(extension))
    maps = Path("/proc/self/maps").read_text()

    checks = {
        "extension_needs_libgomp": "libgomp.so.1" in dynamic,
        "extension_needs_no_libomp": "libomp.so" not in dynamic,
        "extension_has_no_kmpc": "__kmpc_" not in symbols,
        "extension_uses_gomp": "GOMP_" in symbols,
        "process_loaded_libgomp": "libgomp.so" in maps,
        "process_loaded_no_libomp": "libomp.so" not in maps,
        "decode_attention_registered": op_registered(
            torch.ops.sgl_kernel, "decode_attention_cpu"
        ),
        "extend_attention_registered": op_registered(
            torch.ops.sgl_kernel, "extend_attention_cpu"
        ),
    }
    for name, passed in checks.items():
        print(f"{name}={passed}")
    print(f"torch_version={torch.__version__}")
    print(f"torch_revision={torch.version.git_version}")
    print(f"extension={extension}")

    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print(f"ABI gate failed: {', '.join(failed)}", file=sys.stderr)
        return 1
    print("RVV OpenMP ABI gate: PASS (libgomp-only)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
