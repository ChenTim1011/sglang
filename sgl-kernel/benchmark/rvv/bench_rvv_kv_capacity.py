"""Benchmark BF16-KV vs INT8-KV request capacity under a fixed KV memory budget."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass

import requests

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from _rvv_bench_utils import (  # noqa: E402
    _CTX_128,
    _CTX_512,
    BASE_URL,
    BF16_MODEL,
    _ensure_server_port_ready,
    _kill_lingering_server,
    _kv_bytes_per_token,
    _launch,
    _max_tokens_for_memory_budget,
)

REQUEST_TIMEOUT = 1800


@dataclass
class CapacityRow:
    name: str
    kv_cache_dtype: str
    kv_memory_mb: int
    max_total_tokens: int
    batch_size: int
    active_tokens: int
    success: bool
    latency_s: float
    error: str = ""


def _build_prompt(prompt_tokens: int) -> tuple[str, str]:
    if prompt_tokens <= 128:
        return _CTX_128, "~128"
    if prompt_tokens <= 512:
        return _CTX_512, "~512"

    repeats = max(1, (prompt_tokens + 511) // 512)
    return ((_CTX_512 + " ") * repeats).strip(), f"~{repeats * 512}"


def _run_generate_batch(
    prompt: str, batch_size: int, output_len: int
) -> tuple[bool, float, str]:
    payload = {
        "text": [prompt] * batch_size,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": False,
    }

    requests.post(f"{BASE_URL}/flush_cache", timeout=REQUEST_TIMEOUT).raise_for_status()

    start = time.perf_counter()
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
        latency_s = time.perf_counter() - start
        if response.status_code != 200:
            try:
                body = response.json()
            except Exception:
                body = response.text.strip()
            return False, latency_s, str(body)

        # Force JSON decode so we fail loudly on malformed responses.
        json.loads(response.text)
        return True, latency_s, ""
    except Exception as e:
        latency_s = time.perf_counter() - start
        return False, latency_s, str(e)


def _print_rows(rows: list[CapacityRow], prompt_label: str, output_len: int) -> None:
    print("=" * 96)
    print(
        f"RVV KV Capacity Sweep: fixed budget, prompt={prompt_label}, output_len={output_len}"
    )
    print("=" * 96)
    print(
        f"{'Mode':<12} {'KV':<8} {'BudgetMB':>9} {'MaxTok':>9} {'Batch':>7} "
        f"{'ActTok':>9} {'OK':>4} {'Latency(s)':>11}"
    )
    for row in rows:
        print(
            f"{row.name:<12} {row.kv_cache_dtype:<8} {row.kv_memory_mb:>9} "
            f"{row.max_total_tokens:>9} {row.batch_size:>7} {row.active_tokens:>9} "
            f"{('yes' if row.success else 'no'):>4} {row.latency_s:>11.3f}"
        )
        if row.error:
            print(f"  error: {row.error}")

    print("-" * 96)
    by_mode: dict[str, list[CapacityRow]] = {}
    for row in rows:
        by_mode.setdefault(row.name, []).append(row)

    for name, mode_rows in by_mode.items():
        success_rows = [r for r in mode_rows if r.success]
        if not success_rows:
            print(f"{name}: no successful batch.")
            continue
        best = success_rows[-1]
        print(
            f"{name}: max successful batch={best.batch_size}, "
            f"active_tokens={best.active_tokens}, latency={best.latency_s:.3f}s"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare BF16-KV vs INT8-KV under a fixed KV memory budget."
    )
    parser.add_argument(
        "--model",
        default=BF16_MODEL,
        help="Model name to launch.",
    )
    parser.add_argument(
        "--kv-memory-mb",
        type=int,
        default=256,
        help="Fixed KV-cache budget in MB.",
    )
    parser.add_argument(
        "--context-len",
        type=int,
        default=512,
        help="Approximate prompt length bucket.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=32,
        help="Decode length per request.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 16, 32, 48, 64],
        help="Batch sizes to sweep in ascending order.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.5,
        help="Forwarded to server launch.",
    )
    args = parser.parse_args()

    prompt, prompt_label = _build_prompt(args.context_len)
    active_tokens_per_request = args.context_len + args.output_len
    modes = [
        ("BF16 KV", "", 2),
        ("INT8 KV", "int8", 1),
    ]

    rows: list[CapacityRow] = []

    for name, kv_cache_dtype, kv_dtype_bytes in modes:
        max_total_tokens = _max_tokens_for_memory_budget(
            args.kv_memory_mb, kv_dtype_bytes
        )

        print(
            f"[run] {name}: kv_memory_mb={args.kv_memory_mb}, "
            f"max_total_tokens={max_total_tokens}",
            flush=True,
        )

        _kill_lingering_server()
        _ensure_server_port_ready()
        process = _launch(
            model=args.model,
            kv_cache_dtype=kv_cache_dtype,
            max_total_tokens=max_total_tokens,
            extra_args=[
                "--attention-backend",
                "rvv",
                "--trust-remote-code",
                "--mem-fraction-static",
                str(args.mem_fraction_static),
            ],
        )

        try:
            time.sleep(2)
            for batch_size in args.batch_sizes:
                active_tokens = batch_size * active_tokens_per_request
                success, latency_s, error = _run_generate_batch(
                    prompt=prompt,
                    batch_size=batch_size,
                    output_len=args.output_len,
                )
                rows.append(
                    CapacityRow(
                        name=name,
                        kv_cache_dtype=kv_cache_dtype or "bf16",
                        kv_memory_mb=args.kv_memory_mb,
                        max_total_tokens=max_total_tokens,
                        batch_size=batch_size,
                        active_tokens=active_tokens,
                        success=success,
                        latency_s=latency_s,
                        error=error,
                    )
                )
                if not success:
                    break
        finally:
            try:
                process.kill()
            except Exception:
                pass
            _kill_lingering_server()

    _print_rows(rows, prompt_label=prompt_label, output_len=args.output_len)


if __name__ == "__main__":
    main()
