#!/usr/bin/env python3
"""Scan RVV graph/profile logs for obvious graph health regressions."""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

BAD_PATTERNS = (
    "convert_weight_packed",
    "convert_weight_w4a8_dynamic_packed",
    "bytecode_tracing",
    "torch._dynamo.exc",
    "BackendCompilerFailed",
    "Graph break",
    "graph break",
)

GOOD_PATTERNS = (
    "weight_packed_linear",
    "weight_w4a8_dynamic_linear",
    "decode_attention_cpu",
    "Torch-Compiled Region",
    "compiled_autograd",
)


def iter_files(paths: list[Path]) -> list[Path]:
    out: list[Path] = []
    for path in paths:
        if path.is_dir():
            out.extend(
                p
                for p in sorted(path.rglob("*"))
                if p.is_file()
                and p.suffix in {".log", ".txt", ".json", ".gz", ".tsv", ".md"}
            )
        elif path.is_file():
            out.append(path)
    return out


def read_text(path: Path, max_bytes: int) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        return f.read(max_bytes)


def count_trace_events(text: str, patterns: tuple[str, ...]) -> dict[str, int]:
    try:
        data = json.loads(text)
    except Exception:
        return {}
    events = data.get("traceEvents", [])
    counts = {p: 0 for p in patterns}
    for event in events:
        name = str(event.get("name", ""))
        for pattern in patterns:
            if pattern in name:
                counts[pattern] += 1
    return {k: v for k, v in counts.items() if v}


def scan_file(path: Path, max_bytes: int) -> tuple[dict[str, int], dict[str, int]]:
    text = read_text(path, max_bytes)
    bad = {p: text.count(p) for p in BAD_PATTERNS if text.count(p)}
    good = {p: text.count(p) for p in GOOD_PATTERNS if text.count(p)}
    bad.update(count_trace_events(text, BAD_PATTERNS))
    good.update(count_trace_events(text, GOOD_PATTERNS))
    return bad, good


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="+", type=Path, help="Logs/traces or directories"
    )
    parser.add_argument("--max-bytes", type=int, default=64 * 1024 * 1024)
    parser.add_argument("--warn-only", action="store_true")
    args = parser.parse_args()

    files = iter_files(args.paths)
    if not files:
        print("No files to scan.")
        return 0

    any_bad = False
    print("file\tbad_patterns\tgood_patterns")
    for path in files:
        bad, good = scan_file(path, args.max_bytes)
        if bad:
            any_bad = True
        bad_s = ",".join(f"{k}:{v}" for k, v in sorted(bad.items())) or "-"
        good_s = ",".join(f"{k}:{v}" for k, v in sorted(good.items())) or "-"
        print(f"{path}\t{bad_s}\t{good_s}")

    if any_bad and not args.warn_only:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
