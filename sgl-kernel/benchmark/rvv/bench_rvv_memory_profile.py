"""Profile BF16 vs INT8-KV memory behavior on the RVV backend."""

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
    _launch,
    _MemoryTraceMonitor,
)

REQUEST_TIMEOUT = 1800


@dataclass
class MemoryProfileResult:
    name: str
    kv_cache_dtype: str
    prompt_tokens_label: str
    output_tokens: int
    baseline_mb: float
    prefill_rss_mb: float
    prefill_delta_mb: float
    peak_mb: float
    decode_delta_mb: float
    decode_steps: int
    stable_tail_std_mb: float


def _build_prompt(prompt_tokens: int) -> tuple[str, str]:
    if prompt_tokens <= 128:
        return _CTX_128, "~128"
    if prompt_tokens <= 512:
        return _CTX_512, "~512"

    repeats = max(1, (prompt_tokens + 511) // 512)
    return ((_CTX_512 + " ") * repeats).strip(), f"~{repeats * 512}"


def _stream_generate(prompt: str, output_tokens: int, pid: int) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }

    requests.post(f"{BASE_URL}/flush_cache", timeout=REQUEST_TIMEOUT).raise_for_status()

    monitor = _MemoryTraceMonitor(pid)
    monitor.pre_request_sample()

    response = requests.post(
        f"{BASE_URL}/generate",
        json=payload,
        stream=True,
        timeout=REQUEST_TIMEOUT,
    )
    response.raise_for_status()

    decode_step = 0
    for chunk in response.iter_lines(decode_unicode=False):
        chunk = chunk.decode("utf-8")
        if not chunk or not chunk.startswith("data:"):
            continue
        if chunk == "data: [DONE]":
            break
        json.loads(chunk[5:].strip())
        monitor.sample(decode_step)
        decode_step += 1

    analysis = monitor.analyze()
    analysis["decode_steps"] = decode_step
    return analysis


def _run_one_mode(
    name: str,
    model: str,
    kv_cache_dtype: str,
    prompt: str,
    prompt_tokens_label: str,
    output_tokens: int,
    mem_fraction_static: float,
    max_total_tokens: int | None,
) -> MemoryProfileResult:
    _kill_lingering_server()
    _ensure_server_port_ready()

    extra_args = [
        "--attention-backend",
        "rvv",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(mem_fraction_static),
    ]
    process = _launch(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        max_total_tokens=max_total_tokens,
        extra_args=extra_args,
    )

    try:
        time.sleep(2)
        analysis = _stream_generate(
            prompt=prompt,
            output_tokens=output_tokens,
            pid=process.pid,
        )
        prefill_rss_mb = analysis["prefill_rss"]
        peak_mb = analysis["peak_mb"]
        prefill_delta_mb = analysis["prefill_delta_mb"]
        decode_delta_mb = (
            peak_mb - prefill_rss_mb if peak_mb >= 0 and prefill_rss_mb >= 0 else -1.0
        )
        return MemoryProfileResult(
            name=name,
            kv_cache_dtype=kv_cache_dtype or "bf16",
            prompt_tokens_label=prompt_tokens_label,
            output_tokens=output_tokens,
            baseline_mb=analysis["baseline_mb"],
            prefill_rss_mb=prefill_rss_mb,
            prefill_delta_mb=prefill_delta_mb,
            peak_mb=peak_mb,
            decode_delta_mb=decode_delta_mb,
            decode_steps=analysis["decode_steps"],
            stable_tail_std_mb=analysis["stability_std"],
        )
    finally:
        try:
            process.kill()
        except Exception:
            pass
        _kill_lingering_server()


def _print_summary(results: list[MemoryProfileResult]) -> None:
    print("=" * 88)
    print("RVV Memory Profile: BF16 KV vs INT8 KV")
    print("=" * 88)
    print(
        f"{'Mode':<14} {'KV':<8} {'Prompt':<8} {'Out':>5} "
        f"{'Baseline':>10} {'Prefill':>10} {'PrefillΔ':>10} "
        f"{'Peak':>10} {'DecodeΔ':>10} {'Steps':>7}"
    )
    for row in results:
        print(
            f"{row.name:<14} {row.kv_cache_dtype:<8} {row.prompt_tokens_label:<8} "
            f"{row.output_tokens:>5} {row.baseline_mb:>10.1f} {row.prefill_rss_mb:>10.1f} "
            f"{row.prefill_delta_mb:>10.1f} {row.peak_mb:>10.1f} "
            f"{row.decode_delta_mb:>10.1f} {row.decode_steps:>7}"
        )

    if len(results) != 2:
        return

    bf16 = next((r for r in results if r.kv_cache_dtype != "int8"), None)
    int8 = next((r for r in results if r.kv_cache_dtype == "int8"), None)
    if bf16 is None or int8 is None:
        return

    print("-" * 88)
    print(
        "INT8-KV vs BF16-KV: "
        f"prefillΔ {int8.prefill_delta_mb - bf16.prefill_delta_mb:+.1f} MB, "
        f"peak {int8.peak_mb - bf16.peak_mb:+.1f} MB, "
        f"decodeΔ {int8.decode_delta_mb - bf16.decode_delta_mb:+.1f} MB"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile RSS trace for BF16 KV and INT8 KV on RVV."
    )
    parser.add_argument(
        "--model",
        default=BF16_MODEL,
        help="Model name to launch.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Approximate prompt length bucket. Uses built-in synthetic prompts.",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=128,
        help="Number of decode tokens to stream.",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=0.5,
        help="Forwarded to server launch.",
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=None,
        help="Optional max-total-tokens override for the server.",
    )
    args = parser.parse_args()

    prompt, prompt_tokens_label = _build_prompt(args.prompt_tokens)
    modes = [
        ("BF16 KV", ""),
        ("INT8 KV", "int8"),
    ]

    results = []
    for name, kv_cache_dtype in modes:
        print(f"[run] {name} (kv_cache_dtype={kv_cache_dtype or 'bf16'})", flush=True)
        result = _run_one_mode(
            name=name,
            model=args.model,
            kv_cache_dtype=kv_cache_dtype,
            prompt=prompt,
            prompt_tokens_label=prompt_tokens_label,
            output_tokens=args.tokens,
            mem_fraction_static=args.mem_fraction_static,
            max_total_tokens=args.max_total_tokens,
        )
        results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
