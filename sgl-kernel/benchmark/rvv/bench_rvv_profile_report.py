#!/usr/bin/env python3
"""Summarize RVV profile traces/logs into a small markdown/text report."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path

INTERESTING = (
    "weight_packed_linear",
    "weight_w4a8_dynamic_linear",
    "decode_attention_cpu",
    "rms_norm",
    "silu",
    "swiglu",
    "aten::",
    "Torch-Compiled Region",
)


def iter_files(paths: list[Path]) -> list[Path]:
    files: list[Path] = []
    for path in paths:
        if path.is_dir():
            files.extend(p for p in sorted(path.rglob("*")) if p.is_file())
        elif path.is_file():
            files.append(path)
    return files


def read_text(path: Path, max_bytes: int) -> str:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8", errors="ignore") as f:
            return f.read(max_bytes)
    with path.open("rt", encoding="utf-8", errors="ignore") as f:
        return f.read(max_bytes)


def scan_trace(text: str) -> tuple[Counter[str], dict[str, float]]:
    counts: Counter[str] = Counter()
    durations: dict[str, float] = defaultdict(float)
    try:
        data = json.loads(text)
    except Exception:
        return counts, durations
    for event in data.get("traceEvents", []):
        name = str(event.get("name", ""))
        if not any(p in name for p in INTERESTING):
            continue
        counts[name] += 1
        dur = event.get("dur")
        if isinstance(dur, (int, float)):
            durations[name] += float(dur) / 1000.0
    return counts, durations


def scan_text(text: str) -> Counter[str]:
    counts: Counter[str] = Counter()
    for pattern in INTERESTING:
        n = text.count(pattern)
        if n:
            counts[pattern] = n
    return counts


def render_report(
    files: list[Path],
    text_counts: Counter[str],
    trace_counts: Counter[str],
    trace_durations_ms: dict[str, float],
    top_k: int,
) -> str:
    lines: list[str] = []
    lines.append("# RVV Profile Report")
    lines.append("")
    lines.append(f"- files_scanned: {len(files)}")
    lines.append("")
    lines.append("## Text Pattern Counts")
    lines.append("")
    lines.append("| pattern | count |")
    lines.append("|---|---:|")
    for name, count in text_counts.most_common(top_k):
        lines.append(f"| `{name}` | {count} |")
    if not text_counts:
        lines.append("| - | 0 |")
    lines.append("")
    lines.append("## Trace Events")
    lines.append("")
    lines.append("| event | count | total_ms |")
    lines.append("|---|---:|---:|")
    ranked = sorted(
        trace_counts,
        key=lambda n: (trace_durations_ms.get(n, 0.0), trace_counts[n]),
        reverse=True,
    )[:top_k]
    for name in ranked:
        lines.append(
            f"| `{name}` | {trace_counts[name]} | {trace_durations_ms.get(name, 0.0):.3f} |"
        )
    if not ranked:
        lines.append("| - | 0 | 0.000 |")
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--top-k", type=int, default=30)
    parser.add_argument("--max-bytes", type=int, default=128 * 1024 * 1024)
    parser.add_argument("--markdown-out", type=Path)
    parser.add_argument("--warn-only", action="store_true")
    parser.add_argument(
        "--health-check",
        action="store_true",
        help="Also run check_rvv_graph_health.py on the same paths.",
    )
    args = parser.parse_args()

    files = iter_files(args.paths)
    text_counts: Counter[str] = Counter()
    trace_counts: Counter[str] = Counter()
    trace_durations_ms: dict[str, float] = defaultdict(float)

    for path in files:
        if path.suffix not in {".log", ".txt", ".json", ".gz", ".tsv", ".md"}:
            continue
        text = read_text(path, args.max_bytes)
        text_counts.update(scan_text(text))
        counts, durations = scan_trace(text)
        trace_counts.update(counts)
        for name, dur in durations.items():
            trace_durations_ms[name] += dur

    report = render_report(
        files, text_counts, trace_counts, trace_durations_ms, args.top_k
    )
    print(report)
    if args.markdown_out:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(report + "\n", encoding="utf-8")

    if args.health_check:
        import subprocess
        import sys

        checker = Path(__file__).with_name("check_rvv_graph_health.py")
        cmd = [sys.executable, str(checker), *map(str, args.paths)]
        if args.warn_only:
            cmd.append("--warn-only")
        return subprocess.call(cmd)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
