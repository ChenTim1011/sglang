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
    _MemoryMonitor,
    _MemoryTraceMonitor,
)

DEFAULT_REQUEST_TIMEOUT = 14400
DEFAULT_WATCHDOG_TIMEOUT = 14400


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
    ttft_s: float
    decode_time_s: float
    decode_tok_s: float
    stable_tail_std_mb: float
    target_prompt_tokens: int
    actual_prompt_tokens: int


def _build_prompt(prompt_tokens: int, tokenizer=None) -> tuple[str, str, int]:
    if tokenizer is not None:
        seed = (
            "RISC-V vector processors execute long-context language model "
            "inference with quantized weights and key value cache tensors. "
            "This synthetic document is repeated only to allocate real KV cache "
            "inside the running SGLang server. "
        )
        seed_ids = tokenizer.encode(seed, add_special_tokens=False)
        if not seed_ids:
            raise RuntimeError("Tokenizer produced an empty seed prompt.")
        repeats = max(1, (prompt_tokens + len(seed_ids) - 1) // len(seed_ids))
        token_ids = (seed_ids * repeats)[:prompt_tokens]
        prompt = tokenizer.decode(token_ids, skip_special_tokens=True)
        actual = len(tokenizer.encode(prompt, add_special_tokens=False))
        return prompt, str(actual), actual

    if prompt_tokens <= 128:
        return _CTX_128, "~128", 128
    if prompt_tokens <= 512:
        return _CTX_512, "~512", 512

    repeats = max(1, (prompt_tokens + 511) // 512)
    return ((_CTX_512 + " ") * repeats).strip(), f"~{repeats * 512}", repeats * 512


def _stream_generate(
    prompt: str, output_tokens: int, pid: int, request_timeout: int
) -> dict:
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
            "ignore_eos": True,
        },
        "stream": True,
    }

    requests.post(f"{BASE_URL}/flush_cache", timeout=request_timeout).raise_for_status()

    trace_monitor = _MemoryTraceMonitor(pid)
    baseline_mb = trace_monitor.pre_request_sample()
    peak_monitor = _MemoryMonitor(pid, interval=0.1)
    peak_monitor.start(max(baseline_mb, 0.0))

    request_start = time.perf_counter()
    first_token_time = -1.0
    last_token_time = -1.0
    decode_step = 0
    try:
        response = requests.post(
            f"{BASE_URL}/generate",
            json=payload,
            stream=True,
            timeout=request_timeout,
        )
        response.raise_for_status()

        for chunk in response.iter_lines(decode_unicode=False):
            chunk = chunk.decode("utf-8")
            if not chunk or not chunk.startswith("data:"):
                continue
            if chunk == "data: [DONE]":
                break
            json.loads(chunk[5:].strip())
            now = time.perf_counter()
            if first_token_time < 0:
                first_token_time = now
            last_token_time = now
            trace_monitor.sample(decode_step)
            decode_step += 1
    finally:
        peak_mb = peak_monitor.stop()
    analysis = trace_monitor.analyze()
    analysis["peak_mb"] = max(analysis["peak_mb"], peak_mb)
    analysis["decode_steps"] = decode_step
    analysis["ttft_s"] = (
        first_token_time - request_start if first_token_time >= 0 else -1.0
    )
    analysis["decode_time_s"] = (
        last_token_time - first_token_time
        if last_token_time >= 0 and first_token_time >= 0 and decode_step > 1
        else -1.0
    )
    analysis["decode_tok_s"] = (
        (decode_step - 1) / analysis["decode_time_s"]
        if analysis["decode_time_s"] > 0 and decode_step > 1
        else -1.0
    )
    return analysis


def _run_one_mode(
    name: str,
    model: str,
    kv_cache_dtype: str,
    prompt: str,
    prompt_tokens_label: str,
    target_prompt_tokens: int,
    actual_prompt_tokens: int,
    output_tokens: int,
    mem_fraction_static: float,
    max_total_tokens: int | None,
    quantization: str,
    request_timeout: int,
    watchdog_timeout: int,
) -> MemoryProfileResult:
    _kill_lingering_server()
    _ensure_server_port_ready()

    extra_args = [
        "--attention-backend",
        "rvv",
        "--trust-remote-code",
        "--mem-fraction-static",
        str(mem_fraction_static),
        "--watchdog-timeout",
        str(watchdog_timeout),
    ]
    process = _launch(
        model=model,
        kv_cache_dtype=kv_cache_dtype,
        quant_arg=quantization,
        max_total_tokens=max_total_tokens,
        extra_args=extra_args,
    )

    try:
        time.sleep(2)
        analysis = _stream_generate(
            prompt=prompt,
            output_tokens=output_tokens,
            pid=process.pid,
            request_timeout=request_timeout,
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
            ttft_s=analysis["ttft_s"],
            decode_time_s=analysis["decode_time_s"],
            decode_tok_s=analysis["decode_tok_s"],
            stable_tail_std_mb=analysis["stability_std"],
            target_prompt_tokens=target_prompt_tokens,
            actual_prompt_tokens=actual_prompt_tokens,
        )
    finally:
        try:
            process.kill()
        except Exception:
            pass
        _kill_lingering_server()


def _print_summary(results: list[MemoryProfileResult]) -> None:
    print("=" * 88)
    print("RVV Memory Profile: measured RSS, BF16 KV vs INT8 KV")
    print("=" * 88)
    print(
        f"{'Mode':<14} {'KV':<8} {'Target':>8} {'Actual':>8} {'Out':>5} "
        f"{'Baseline':>10} {'Prefill':>10} {'PrefillΔ':>10} "
        f"{'Peak':>10} {'DecodeΔ':>10} {'Steps':>7} {'TTFT(s)':>9} {'Tok/s':>8}"
    )
    for row in results:
        print(
            f"{row.name:<14} {row.kv_cache_dtype:<8} {row.target_prompt_tokens:>8} "
            f"{row.actual_prompt_tokens:>8} {row.output_tokens:>5} "
            f"{row.baseline_mb:>10.1f} {row.prefill_rss_mb:>10.1f} "
            f"{row.prefill_delta_mb:>10.1f} {row.peak_mb:>10.1f} "
            f"{row.decode_delta_mb:>10.1f} {row.decode_steps:>7} "
            f"{row.ttft_s:>9.2f} {row.decode_tok_s:>8.3f}"
        )

    print("-" * 88)
    by_target: dict[int, dict[str, MemoryProfileResult]] = {}
    for row in results:
        by_target.setdefault(row.target_prompt_tokens, {})[row.kv_cache_dtype] = row

    print(
        f"{'Target':>8} {'Actual':>8} {'BF16Δ':>10} {'INT8Δ':>10} "
        f"{'SavedΔ':>10} {'Saved%':>8} {'BF16tok/s':>10} {'INT8tok/s':>10}"
    )
    for target in sorted(by_target):
        rows = by_target[target]
        bf16 = rows.get("bf16")
        int8 = rows.get("int8")
        if bf16 is None or int8 is None:
            continue
        saved = bf16.prefill_delta_mb - int8.prefill_delta_mb
        saved_pct = (
            100.0 * saved / bf16.prefill_delta_mb if bf16.prefill_delta_mb > 0 else 0.0
        )
        print(
            f"{target:>8} {int8.actual_prompt_tokens:>8} {bf16.prefill_delta_mb:>10.1f} "
            f"{int8.prefill_delta_mb:>10.1f} {saved:>10.1f} {saved_pct:>7.1f}% "
            f"{bf16.decode_tok_s:>10.3f} {int8.decode_tok_s:>10.3f}"
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
        "--quantization",
        default="",
        help="Optional quantization argument forwarded to sglang serve.",
    )
    parser.add_argument(
        "--prompt-tokens",
        type=int,
        default=512,
        help="Approximate prompt length bucket. Uses built-in synthetic prompts.",
    )
    parser.add_argument(
        "--prompt-token-list",
        type=int,
        nargs="+",
        default=None,
        help="Run a sweep over these target prompt token counts.",
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
    parser.add_argument(
        "--request-timeout",
        type=int,
        default=DEFAULT_REQUEST_TIMEOUT,
        help="HTTP timeout in seconds for long prefill streaming requests.",
    )
    parser.add_argument(
        "--watchdog-timeout",
        type=int,
        default=DEFAULT_WATCHDOG_TIMEOUT,
        help="Server watchdog timeout in seconds for long CPU prefill runs.",
    )
    parser.add_argument(
        "--no-tokenizer",
        action="store_true",
        help="Use the older approximate repeated-text prompt builder.",
    )
    args = parser.parse_args()

    tokenizer = None
    if not args.no_tokenizer:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompt_token_targets = args.prompt_token_list or [args.prompt_tokens]
    modes = [
        ("BF16 KV", ""),
        ("INT8 KV", "int8"),
    ]

    results = []
    for target_prompt_tokens in prompt_token_targets:
        prompt, prompt_tokens_label, actual_prompt_tokens = _build_prompt(
            target_prompt_tokens, tokenizer=tokenizer
        )
        for name, kv_cache_dtype in modes:
            print(
                f"[run] target_prompt_tokens={target_prompt_tokens} "
                f"actual_prompt_tokens={actual_prompt_tokens} {name} "
                f"(kv_cache_dtype={kv_cache_dtype or 'bf16'})",
                flush=True,
            )
            result = _run_one_mode(
                name=name,
                model=args.model,
                kv_cache_dtype=kv_cache_dtype,
                prompt=prompt,
                prompt_tokens_label=prompt_tokens_label,
                target_prompt_tokens=target_prompt_tokens,
                actual_prompt_tokens=actual_prompt_tokens,
                output_tokens=args.tokens,
                mem_fraction_static=args.mem_fraction_static,
                max_total_tokens=args.max_total_tokens,
                quantization=args.quantization,
                request_timeout=args.request_timeout,
                watchdog_timeout=args.watchdog_timeout,
            )
            results.append(result)

    _print_summary(results)


if __name__ == "__main__":
    main()
