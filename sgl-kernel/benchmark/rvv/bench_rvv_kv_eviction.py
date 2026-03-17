"""Track KV eviction behavior by measuring RSS over sequential requests."""

import argparse
import sys
import time
from typing import List

import requests as req_lib
from _rvv_bench_utils import (
    _CTX_128,
    BASE_URL,
    BF16_MODEL,
    _kill_lingering_server,
    _launch,
    _process_tree_rss_mb,
)

try:
    from sglang.srt.utils import kill_process_tree
except ImportError:
    print("sglang not found")
    sys.exit(1)

POST_KILL_SLEEP = 180


def run_kv_eviction_test(
    prompt_base: str,
    n_requests: int,
    pid: int,
    request_timeout: int = 600,
) -> dict:
    """Send sequential requests and track RSS after each one."""
    samples: List[tuple] = []  # (req_idx, rss_mb)
    initial_rss = _process_tree_rss_mb(pid)
    max_rss_seen = initial_rss

    for i in range(n_requests):
        prompt = f"{prompt_base} [req-{i}]"
        try:
            req_lib.post(
                f"{BASE_URL}/generate",
                json={"text": prompt, "sampling_params": {"max_new_tokens": 8}},
                timeout=request_timeout,
            )
        except Exception as e:
            print(f"\n    [req {i} ERROR] {e}", flush=True)
            break

        rss = _process_tree_rss_mb(pid)
        samples.append((i, rss))
        max_rss_seen = max(max_rss_seen, rss)

        bar_range = max(max_rss_seen - initial_rss, 1.0)
        bar_len = int((rss - initial_rss) / bar_range * 20)
        print(
            f"\r    req {i+1:>4}/{n_requests}:  {rss:>7.0f} MB"
            f"  [{'#' * bar_len:<20}]  Δ{rss - initial_rss:+.0f} MB",
            end="",
            flush=True,
        )

    print(flush=True)

    if len(samples) < 4:
        return {
            "samples": samples,
            "initial_rss": initial_rss,
            "plateau_mb": -1,
            "growth_mb": -1,
            "stabilize_at": None,
            "n_requests": len(samples),
        }

    tail_start = max(1, int(len(samples) * 0.7))
    tail_rss = [r for _, r in samples[tail_start:]]
    plateau_mb = sum(tail_rss) / len(tail_rss)
    growth_mb = plateau_mb - initial_rss

    stabilize_at = None
    for req_idx, rss in samples:
        if plateau_mb > 0 and rss >= 0.95 * plateau_mb:
            stabilize_at = req_idx
            break

    return {
        "samples": samples,
        "initial_rss": initial_rss,
        "plateau_mb": plateau_mb,
        "growth_mb": growth_mb,
        "stabilize_at": stabilize_at,
        "n_requests": len(samples),
    }


def print_eviction_summary(results: dict, name: str):
    plateau = results.get("plateau_mb", -1)
    growth = results.get("growth_mb", -1)
    stabilize = results.get("stabilize_at")
    initial = results.get("initial_rss", -1)

    print(f"\n{'='*70}")
    print(f"  KV EVICTION TEST — {name}")
    print(f"{'='*70}")
    print(f"  Initial RSS  : {initial:.0f} MB")
    if plateau >= 0:
        stab_str = f"req #{stabilize}" if stabilize is not None else "not reached"
        print(f"  Plateau RSS  : {plateau:.0f} MB  (Δ+{growth:.0f} MB)")
        print(f"  Stabilized at: {stab_str}")
        if stabilize is None:
            print("  WARNING: KV pool never reached 95% of plateau.")
            print("  Increase --requests or reduce pool size.")

    samples = results.get("samples", [])
    if samples:
        stride = max(1, len(samples) // 10)
        print(f"\n  Per-request RSS (every {stride} reqs):")
        for req_idx, rss in samples[::stride]:
            bar_range = max(plateau - initial, 1.0) if plateau > 0 else 1.0
            bar_len = int((rss - initial) / bar_range * 24)
            marker = "▶" if req_idx == stabilize else " "
            print(
                f"    {marker} req {req_idx:>4}: {rss:>7.0f} MB"
                f"  [{'#' * bar_len:<24}]  Δ{rss - initial:+.0f} MB"
            )


def main():
    parser = argparse.ArgumentParser(
        description="KV eviction pool saturation test: track RSS until pool fills."
    )
    parser.add_argument(
        "--requests",
        type=int,
        default=60,
        help="Number of sequential requests (default 60; ~6 min on K1)",
    )
    parser.add_argument(
        "--kv-cache-dtype",
        default="",
        help="KV cache dtype: '' for BF16 (default) or 'int8'",
    )
    parser.add_argument(
        "--model",
        default=BF16_MODEL,
        help=f"Model to use (default: {BF16_MODEL})",
    )
    args = parser.parse_args()

    _kill_lingering_server()

    dtype_label = "INT8" if args.kv_cache_dtype == "int8" else "BF16"
    n = args.requests

    # Cap pool so it saturates during the test: 70% of total workload
    tokens_per_req = 140  # 128 ctx + suffix + 8 output
    max_total_tokens = max(512, int(n * tokens_per_req * 0.7))

    print(f"\nKV Eviction Test: {dtype_label} KV  |  {n} requests", flush=True)
    print(f"  Pool capped at {max_total_tokens} tokens (70% of workload)", flush=True)

    proc = _launch(
        args.model,
        kv_cache_dtype=args.kv_cache_dtype,
        max_total_tokens=max_total_tokens,
    )
    try:
        print(f"  Startup RSS: {_process_tree_rss_mb(proc.pid):.0f} MB", flush=True)
        results = run_kv_eviction_test(_CTX_128, n, proc.pid)
    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)

    print_eviction_summary(results, dtype_label)
    print("\n[SUCCESS] RVV KV eviction test completed.", flush=True)


if __name__ == "__main__":
    main()
