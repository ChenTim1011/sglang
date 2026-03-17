"""Benchmark quantization modes on decode throughput and TTFT."""

import argparse
import statistics
import sys
import time
from typing import List, Optional, Tuple

import requests
from _rvv_bench_utils import (
    _CTX_128,
    _SHORT_PROMPT,
    BASE_URL,
    BF16_MODEL,
    W8A8_MODEL,
    BenchConfig,
    _ensure_server_port_ready,
    _kill_lingering_server,
    _launch,
)

try:
    from sglang.srt.utils import kill_process_tree
except ImportError:
    print("sglang not found")
    sys.exit(1)

POST_KILL_SLEEP = 180

# Keep max_tokens fixed so batch-size scaling reflects decode behavior.
DECODE_CONFIGS = [
    BenchConfig(_CTX_128, 64, 1, "BS=1   long-ctx/128 (64 tok)"),
    BenchConfig(_CTX_128, 64, 4, "BS=4   long-ctx/128 (64 tok)"),
    BenchConfig(_CTX_128, 64, 8, "BS=8   long-ctx/128 (64 tok)"),
    BenchConfig(_CTX_128, 64, 16, "BS=16  long-ctx/128 (64 tok)"),
]

QUICK_CONFIGS = [
    BenchConfig(_SHORT_PROMPT, 16, 1, "BS=1  short-ctx/5  [quick]"),
    BenchConfig(_SHORT_PROMPT, 8, 4, "BS=4  short-ctx/5  [quick]"),
]

# Minimal W8A8 config to verify N-major GEMM loop impact.
VERIFY_NMAJOR_CONFIGS = [
    BenchConfig(_SHORT_PROMPT, 16, 1, "BS=1   short-ctx  (control)"),
    BenchConfig(_SHORT_PROMPT, 16, 4, "BS=4   short-ctx  (N-major)"),
    BenchConfig(_SHORT_PROMPT, 8, 8, "BS=8   short-ctx  (N-major)"),
]


def _measure(
    cfg: BenchConfig,
    warmup: int = 2,
    repeat: int = 2,
    request_timeout: int = 1800,
) -> Tuple[Optional[float], Optional[float]]:

    prompts = [cfg.prompt] * cfg.batch_size

    for _ in range(warmup):
        try:
            requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"max_new_tokens": cfg.max_tokens},
                },
                timeout=request_timeout,
            )
        except Exception:
            pass

    samples: List[float] = []
    for i in range(repeat):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"max_new_tokens": cfg.max_tokens},
                },
                timeout=request_timeout,
            )
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            tps = cfg.batch_size * cfg.max_tokens / elapsed
            samples.append(tps)
        except Exception as e:
            print(f"    [run {i+1} ERROR] {e}")

    if not samples:
        return None, None
    median = statistics.median(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return median, std


def _measure_ttft(
    prompt: str,
    warmup: int = 1,
    repeat: int = 2,
    request_timeout: int = 300,
) -> Tuple[Optional[float], Optional[float]]:
    for _ in range(warmup):
        try:
            requests.post(
                f"{BASE_URL}/generate",
                json={"text": prompt, "sampling_params": {"max_new_tokens": 1}},
                timeout=request_timeout,
            )
        except Exception:
            pass

    samples: List[float] = []
    for i in range(repeat):
        try:
            t0 = time.perf_counter()
            resp = requests.post(
                f"{BASE_URL}/generate",
                json={"text": prompt, "sampling_params": {"max_new_tokens": 1}},
                timeout=request_timeout,
            )
            elapsed = time.perf_counter() - t0
            resp.raise_for_status()
            samples.append(elapsed)
        except Exception as e:
            print(f"    [TTFT run {i+1} ERROR] {e}")

    if not samples:
        return None, None
    median = statistics.median(samples)
    std = statistics.stdev(samples) if len(samples) > 1 else 0.0
    return median, std


def run_mode(
    name: str,
    model: str,
    quant_arg: str,
    configs: List[BenchConfig],
    kv_cache_dtype: str = "",
    decode_repeat: int = 2,
    prefill_repeat: int = 2,
    request_timeout: int = 1800,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(
        f"  Mode: {name}" + (f" [kv={kv_cache_dtype}]" if kv_cache_dtype else ""),
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    _, port = _ensure_server_port_ready(BASE_URL)

    proc = _launch(model, kv_cache_dtype, quant_arg)
    ttft_short = ttft_long = None
    ttft_short_std = ttft_long_std = None
    tps_results: dict = {}

    try:
        print("  TTFT short-ctx/5   ...", end=" ", flush=True)
        ttft_short, ttft_short_std = _measure_ttft(_SHORT_PROMPT, repeat=prefill_repeat)
        print(
            f"{ttft_short*1000:.0f} ms ±{ttft_short_std*1000:.0f}"
            if ttft_short is not None and ttft_short_std is not None
            else "FAILED"
        )

        print("  TTFT long-ctx/128  ...", end=" ", flush=True)
        ttft_long, ttft_long_std = _measure_ttft(_CTX_128, repeat=prefill_repeat)
        print(
            f"{ttft_long*1000:.0f} ms ±{ttft_long_std*1000:.0f}"
            if ttft_long is not None and ttft_long_std is not None
            else "FAILED"
        )

        for cfg in configs:
            print(f"  {cfg.desc} ...", end=" ", flush=True)
            median, std = _measure(
                cfg, repeat=decode_repeat, request_timeout=request_timeout
            )
            if median is not None:
                print(
                    f"{median:.2f} tok/s ±{std:.2f}",
                    flush=True,
                )
            else:
                print("FAILED", flush=True)
            tps_results[cfg.desc] = (median, std)

    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)

    return {
        "ttft_short_ms": ttft_short * 1000 if ttft_short else None,
        "ttft_short_std_ms": (
            ttft_short_std * 1000 if ttft_short_std is not None else None
        ),
        "ttft_long_ms": ttft_long * 1000 if ttft_long else None,
        "ttft_long_std_ms": ttft_long_std * 1000 if ttft_long_std is not None else None,
        "tps_results": tps_results,  # dict: desc → (median, std)
        "kv_cache_dtype": kv_cache_dtype,
    }


def print_summary(all_results: dict, configs: List[BenchConfig]):
    mode_names = list(all_results.keys())
    col_w = 22

    # ── Throughput ──────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  THROUGHPUT  (tok/s decode, higher is better)  median ± stdev, 3 runs")
    print(f"{'='*80}")
    header = f"{'Config':<36}" + "".join(f"{m:<{col_w}}" for m in mode_names)
    print(header)
    print("-" * (36 + col_w * len(mode_names)))

    bf16_tps = {
        desc: v[0]
        for desc, v in all_results.get("BF16 packed", {}).get("tps_results", {}).items()
    }

    for cfg in configs:
        row = f"{cfg.desc:<36}"
        baseline = bf16_tps.get(cfg.desc)
        for name in mode_names:
            entry = all_results.get(name, {}).get("tps_results", {}).get(cfg.desc)
            if entry is None or entry[0] is None:
                row += f"{'N/A':<{col_w}}"
            else:
                median, std = entry
                if name == "BF16 packed" or baseline is None:
                    cell = f"{median:.2f}±{std:.2f}"
                else:
                    cell = f"{median:.2f}±{std:.2f}({median/baseline:.2f}x)"
                row += f"{cell:<{col_w}}"
        print(row)
    print("\nSpeedup relative to BF16 packed baseline.")

    # ── TTFT ────────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("  TTFT  (time-to-first-token = prefill latency, lower is better)")
    print(f"{'='*80}")
    print(f"  {'Mode':<22} {'TTFT short-ctx/5':>18}  {'TTFT long-ctx/128':>18}")
    print(f"  {'-'*60}")
    for name in mode_names:
        r = all_results.get(name, {})
        ts = r.get("ttft_short_ms")
        tl = r.get("ttft_long_ms")
        print(
            f"  {name:<22} " f"{f'{ts:.0f} ms':>18}  " f"{f'{tl:.0f} ms':>18}"
            if ts and tl
            else f"  {name:<22} {'N/A':>18}  {'N/A':>18}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Decode throughput + TTFT benchmark for RVV quantization modes."
    )
    parser.add_argument("--quick", action="store_true", help="Short run for CI")
    parser.add_argument(
        "--verify-nmajor",
        action="store_true",
        help="Verify N-major GEMM loop: W8A8 only, BS=1/4/8 short-ctx (fastest run)",
    )
    parser.add_argument(
        "--decode-repeat",
        type=int,
        default=2,
        help="Decode throughput repeat count per config (default: 2)",
    )
    parser.add_argument(
        "--prefill-repeat",
        type=int,
        default=2,
        help="Prefill(TTFT) repeat count per context (default: 2)",
    )
    args = parser.parse_args()

    if args.decode_repeat < 1 or args.prefill_repeat < 1:
        raise ValueError("--decode-repeat and --prefill-repeat must be >= 1")

    if args.verify_nmajor:
        MODES = [("W8A8", W8A8_MODEL, "w8a8_int8", "")]
        configs = VERIFY_NMAJOR_CONFIGS
    else:
        MODES = [
            ("BF16 packed", BF16_MODEL, "", ""),
            ("INT8 KV cache", BF16_MODEL, "", "int8"),
            ("W8A8", W8A8_MODEL, "w8a8_int8", ""),
            ("W8A8+INT8KV", W8A8_MODEL, "w8a8_int8", "int8"),
        ]
        configs = QUICK_CONFIGS if args.quick else DECODE_CONFIGS

    _kill_lingering_server()

    print(f"\n{'='*60}", flush=True)
    print("  BENCHMARK CONFIGURATION", flush=True)
    print(f"{'='*60}", flush=True)
    for name, model, quant_arg, kv_dtype in MODES:
        print(
            f"  {name:<16} model={model}  "
            f"quant={quant_arg or 'none'}  kv={kv_dtype or 'bf16'}  "
            f"metrics=throughput+TTFT",
            flush=True,
        )
    print(
        "  Configs: "
        f"{len(configs)}  |  Decode: 2 warmup + {args.decode_repeat} repeat"
        " → median ± stdev"
        f"  |  Prefill(TTFT): 1 warmup + {args.prefill_repeat} repeat",
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    all_results = {}
    for name, model, quant_arg, kv_dtype in MODES:
        all_results[name] = run_mode(
            name,
            model,
            quant_arg,
            configs,
            kv_dtype,
            decode_repeat=args.decode_repeat,
            prefill_repeat=args.prefill_repeat,
        )

    print_summary(all_results, configs)
    print("\n[SUCCESS] RVV quant throughput benchmark completed.", flush=True)


if __name__ == "__main__":
    main()
