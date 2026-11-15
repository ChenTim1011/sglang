"""Check a native RISC-V runner without requiring PyTorch or RVV hardware."""

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx", default="g++")
    parser.add_argument("--objdump", default="objdump")
    parser.add_argument("--cross-only", action="store_true")
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).resolve().parent
    record = {
        "host": platform.machine(),
        "cross_only": args.cross_only,
        "native_scalar_executed": False,
        "rvv_executed": False,
        "scope": "Runner/toolchain smoke; no SGLang or TorchInductor integration",
        "commands": [],
    }
    result_path = out / "result.json"

    def run(command, name):
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        (out / f"{name}.log").write_text(result.stdout + result.stderr)
        record["commands"].append({"command": command, "exit_code": result.returncode})
        if result.returncode:
            raise RuntimeError(f"{name} failed; see {out / f'{name}.log'}")
        return result.stdout

    try:
        if not args.cross_only and platform.machine() != "riscv64":
            raise RuntimeError("Native CI requires a riscv64 host")
        cpuinfo = Path("/proc/cpuinfo")
        if cpuinfo.exists():
            (out / "cpuinfo.txt").write_text(cpuinfo.read_text())
        run([args.cxx, "--version"], "compiler")
        common = [args.cxx, "-std=c++17", "-O2", "-mabi=lp64d", "-ffp-contract=off"]
        scalar = out / "scalar_smoke"
        run(
            common
            + [
                "-march=rv64gc",
                "-fno-tree-vectorize",
                "-fno-tree-slp-vectorize",
                str(source / "scalar_smoke.cpp"),
                "-o",
                str(scalar),
            ],
            "scalar_compile",
        )
        if not args.cross_only:
            run([str(scalar)], "scalar_execution")
            record["native_scalar_executed"] = True
            record["scalar_sizes_passed"] = 7
        obj = out / "rvv_compile_only.o"
        run(
            common
            + [
                "-march=rv64gcv_zvl128b",
                "-c",
                str(source / "rvv_compile_only.cpp"),
                "-o",
                str(obj),
            ],
            "rvv_compile",
        )
        assembly = run(
            [args.objdump, "-d", "--disassemble=rvv_compile_only", str(obj)],
            "rvv_disassembly",
        )
        checks = {
            "setup": bool(re.search(r"\bvset(?:i)?vli\b", assembly)),
            "load": "vle32.v" in assembly,
            "mul": bool(re.search(r"\bvfmul\.(?:vv|vf)\b", assembly)),
            "add": bool(re.search(r"\bvfadd\.(?:vv|vf)\b", assembly)),
            "store": "vse32.v" in assembly,
        }
        record["rvv_instruction_checks"] = checks
        if not all(checks.values()):
            raise RuntimeError(f"Missing RVV instructions: {checks}")
        record["source_sha256"] = {
            name: hashlib.sha256((source / name).read_bytes()).hexdigest()
            for name in ("smoke.py", "scalar_smoke.cpp", "rvv_compile_only.cpp")
        }
        record["status"] = "passed"
    except Exception as error:
        record["status"] = "failed"
        record["error"] = str(error)
        raise
    finally:
        result_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2), flush=True)
        summary = os.environ.get("GITHUB_STEP_SUMMARY")
        if summary:
            with open(summary, "a") as stream:
                stream.write("## RISC-V runner smoke\n\n```json\n")
                stream.write(json.dumps(record, indent=2) + "\n```\n")


if __name__ == "__main__":
    main()
