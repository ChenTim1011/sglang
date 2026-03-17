"""Probe KV capacity limits with single long-context requests."""

import argparse
import sys
import time
from typing import List

import requests
from _rvv_bench_utils import (
    _CTX_512,
    _KV_HEAD_DIM,
    _KV_N_KV_HEADS,
    _KV_N_LAYERS,
    BASE_URL,
    BF16_MODEL,
    _ensure_server_port_ready,
    _kill_lingering_server,
    _launch,
    _max_tokens_for_memory_budget,
    _process_tree_rss_mb,
)

try:
    from sglang.srt.utils import kill_process_tree
except ImportError:
    print("sglang not found")
    sys.exit(1)

POST_KILL_SLEEP = 180


# KV capacity math


def _kv_bytes_per_token_bf16() -> int:
    return _KV_N_LAYERS * 2 * _KV_N_KV_HEADS * _KV_HEAD_DIM * 2  # BF16


def _kv_bytes_per_token_int8() -> int:
    return _KV_N_LAYERS * 2 * _KV_N_KV_HEADS * _KV_HEAD_DIM * 1  # INT8


# Prompt builder
# Build a large prompt corpus to slice from.
_LONG_PROMPT_BASE = (_CTX_512 + " ") * 30  # ~15,360 tokens worth of chars

# Calibrated from observed K1 tokenization ratio for Qwen2.5-1.5B.
_CHARS_PER_TOKEN = 4.88  # Qwen2.5-1.5B calibrated constant (not 4.0)


def _build_prompt(context_len_tokens: int) -> str:
    """Build a prompt approximating context_len_tokens actual tokens.

    Uses Qwen2.5-1.5B calibrated ratio: ~4.88 chars/token (measured on K1).
    Empirically: asking for N_estimated tokens yields ~N_estimated actual tokens.
    """
    n_chars = int(context_len_tokens * _CHARS_PER_TOKEN)
    return _LONG_PROMPT_BASE[:n_chars]


# Single-request probe


def _probe_request(
    context_len: int,
    output_len: int,
    request_timeout: float,
) -> dict:
    """Send a single long-context request and return outcome.

    Returns dict with:
      success: bool
      latency_s: float
      error: str or None  (e.g. "HTTP 500 ...", "client timeout")
    """
    prompt = _build_prompt(context_len)
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": prompt,
                "sampling_params": {"max_new_tokens": output_len},
            },
            timeout=request_timeout,
        )
        elapsed = time.perf_counter() - t0
        if resp.status_code == 200:
            return {"success": True, "latency_s": elapsed, "error": None}
        else:
            return {
                "success": False,
                "latency_s": elapsed,
                "error": f"HTTP {resp.status_code}: {resp.text[:200]}",
            }
    except requests.exceptions.Timeout:
        elapsed = time.perf_counter() - t0
        return {"success": False, "latency_s": elapsed, "error": "client timeout"}
    except Exception as e:
        elapsed = time.perf_counter() - t0
        return {"success": False, "latency_s": elapsed, "error": str(e)[:200]}


# Sweep test


def run_kv_capacity_limit_test(
    kv_memory_mb: int,
    context_lengths: List[int],
    output_len: int,
    kv_cache_dtype: str,
    request_timeout: float,
) -> List[dict]:
    """For each context length, send a single request and check if it succeeds.

    The test reveals the exact context length at which the KV pool runs out.
    Returns list of result dicts per context length.
    """
    kv_dtype_bytes = 1 if kv_cache_dtype == "int8" else 2
    max_total_tokens = _max_tokens_for_memory_budget(kv_memory_mb, kv_dtype_bytes)
    dtype_label = "INT8" if kv_cache_dtype == "int8" else "BF16"

    print(f"\n{'='*68}", flush=True)
    print(
        f"  KV Capacity Limit Sweep  |  KV: {dtype_label}  |  budget: {kv_memory_mb} MB",
        flush=True,
    )
    print(
        f"  Bytes/token: {_kv_bytes_per_token_bf16() if kv_dtype_bytes==2 else _kv_bytes_per_token_int8():,} B"
        f"  |  Max tokens in pool: {max_total_tokens:,}",
        flush=True,
    )
    print(
        f"  Output len: {output_len} tok  |  Timeout: {request_timeout:.0f}s",
        flush=True,
    )
    print(f"{'='*68}", flush=True)

    _ensure_server_port_ready(BASE_URL)

    proc = _launch(
        BF16_MODEL,
        kv_cache_dtype=kv_cache_dtype,
        max_total_tokens=max_total_tokens,
        extra_args=[
            "--disable-radix-cache"
        ],  # IMPORTANT: prevents prefix-cache KV reuse
        # Without this, request N reuses cached tokens from request N-1, masking OOM
    )
    startup_rss = _process_tree_rss_mb(proc.pid)
    print(f"  Startup RSS: {startup_rss:.0f} MB", flush=True)
    print(
        f"  {'Context':>9}  {'Total tok':>9}  {'Fits?':>8}  {'Result':>9}  {'Latency':>9}  {'Note'}",
        flush=True,
    )
    print(f"  {'-'*65}", flush=True)

    results = []
    try:
        for ctx_len in context_lengths:
            total_tok = ctx_len + output_len
            fits_in_pool = total_tok <= max_total_tokens
            fits_label = "YES" if fits_in_pool else "NO (>pool)"

            result = _probe_request(ctx_len, output_len, request_timeout)
            rss = _process_tree_rss_mb(proc.pid)

            if result["success"]:
                status = "SUCCESS"
                note = f"lat={result['latency_s']:.1f}s"
            else:
                status = "FAIL"
                note = f"err={result['error'][:40] if result['error'] else 'unknown'}"

            # Flag expected vs actual
            mismatch = ""
            if fits_in_pool and not result["success"]:
                mismatch = " ← unexpected fail"
            elif not fits_in_pool and result["success"]:
                mismatch = " ← unexpected success?"

            print(
                f"  {ctx_len:>9}  {total_tok:>9}  {fits_label:>8}  {status:>9}  "
                f"{result['latency_s']:>8.1f}s  {note}{mismatch}",
                flush=True,
            )

            results.append(
                {
                    "context_len": ctx_len,
                    "total_tokens": total_tok,
                    "fits_theoretical": fits_in_pool,
                    "success": result["success"],
                    "latency_s": result["latency_s"],
                    "error": result["error"],
                    "rss_mb": rss,
                    "kv_cache_dtype": dtype_label,
                }
            )
    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)

    return results


# ----- comparison summary -----


def print_capacity_comparison(
    bf16_results: List[dict],
    int8_results: List[dict],
    kv_memory_mb: int,
    output_len: int,
):
    """Side-by-side comparison: which contexts BF16/INT8 can serve."""
    print(f"\n{'='*80}")
    print("  KV CAPACITY LIMIT COMPARISON  (BF16 KV vs INT8 KV)")
    print(f"  Budget: {kv_memory_mb} MB  |  Output: {output_len} tok")
    print(f"{'='*80}")

    # KV math summary
    bf16_max = _max_tokens_for_memory_budget(kv_memory_mb, 2)
    int8_max = _max_tokens_for_memory_budget(kv_memory_mb, 1)
    print(
        f"  BF16 KV pool max tokens : {bf16_max:,}  ({_kv_bytes_per_token_bf16()} B/token)"
    )
    print(
        f"  INT8 KV pool max tokens : {int8_max:,}  ({_kv_bytes_per_token_int8()} B/token)"
    )
    print(f"  INT8 capacity ratio     : {int8_max/bf16_max:.2f}x\n")

    print(
        f"  {'Context':>9}  {'BF16 result':>12}  {'BF16 lat':>9}  "
        f"{'INT8 result':>12}  {'INT8 lat':>9}  {'Advantage'}"
    )
    print(f"  {'-'*72}")

    bf16_map = {r["context_len"]: r for r in bf16_results}
    int8_map = {r["context_len"]: r for r in int8_results}
    all_ctx = sorted(set(list(bf16_map) + list(int8_map)))

    for ctx in all_ctx:
        b = bf16_map.get(ctx)
        i = int8_map.get(ctx)

        b_res = "SUCCESS" if b and b["success"] else ("FAIL" if b else "N/A")
        b_lat = f"{b['latency_s']:.1f}s" if b and b["success"] else "—"
        i_res = "SUCCESS" if i and i["success"] else ("FAIL" if i else "N/A")
        i_lat = f"{i['latency_s']:.1f}s" if i and i["success"] else "—"

        advantage = ""
        if b and i:
            if not b["success"] and i["success"]:
                advantage = "✓ INT8 only"
            elif b["success"] and i["success"]:
                advantage = "  both ok"
            elif not b["success"] and not i["success"]:
                advantage = "  both fail"

        print(
            f"  {ctx:>9}  {b_res:>12}  {b_lat:>9}  "
            f"{i_res:>12}  {i_lat:>9}  {advantage}"
        )

    # Find crossover
    bf16_limit = next(
        (r["context_len"] for r in bf16_results if not r["success"]), None
    )
    int8_limit = next(
        (r["context_len"] for r in int8_results if not r["success"]), None
    )

    print(f"\n{'='*80}")
    print("  CONCLUSION")
    print(f"{'='*80}")
    print(
        f"  BF16 KV fails at context ≥ : {bf16_limit or 'not reached (increase context lengths)'}"
    )
    print(
        f"  INT8 KV fails at context ≥ : {int8_limit or 'not reached (increase context lengths)'}"
    )
    if bf16_limit and int8_limit:
        ratio = int8_limit / bf16_limit
        print(
            f"\n  INT8 KV supports {ratio:.2f}x longer contexts within the same {kv_memory_mb} MB budget."
            f"\n  This directly translates to serving {ratio:.2f}x more tokens per request."
        )
    elif bf16_limit and not int8_limit:
        print(
            f"\n  INT8 KV can serve all tested context lengths; BF16 fails at {bf16_limit} tokens."
            f"\n  → Use --context-lengths with larger values to find INT8's limit."
        )
    elif not bf16_limit:
        print(
            "\n  BF16 succeeded for all tested context lengths."
            "\n  → Use smaller --kv-memory-mb or larger --context-lengths to trigger the limit."
        )

    print(
        "\n  WHY THIS TEST IS MORE RELIABLE THAN CONCURRENT LOAD ON CPU K1:"
        "\n    On CPU, concurrent requests are processed sequentially (not in parallel)."
        "\n    This means concurrent load test latency scales with N × single_request_lat,"
        "\n    making both BF16 and INT8 fail equally (CPU-bound, not KV-bound)."
        "\n    The single long-context test directly triggers KV pool exhaustion,"
        "\n    which is the actual constraint INT8 KV is designed to address."
    )


# ----- main -----


def main():
    parser = argparse.ArgumentParser(
        description=(
            "KV cache capacity limit test: finds max context length BF16 vs INT8 KV can serve.\n"
            "\n"
            "On K1 CPU, this is the correct way to demonstrate INT8 KV advantage.\n"
            "Concurrent load tests fail to show it because CPU compute (not KV memory)\n"
            "is the bottleneck for concurrent requests.\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Test only 2 context lengths (below and above BF16 limit)",
    )
    parser.add_argument(
        "--kv-memory-mb",
        type=int,
        default=64,
        help="KV pool budget in MB (default 64; smaller = faster test, easier to hit limit)",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=8,
        help="Output tokens per request (default 8; keep small to minimize test time)",
    )
    parser.add_argument(
        "--context-lengths",
        type=int,
        nargs="+",
        help=(
            "List of context lengths to sweep (tokens). "
            "Default: auto-compute based on kv-memory-mb to span BF16/INT8 boundary."
        ),
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds (default 600; long to avoid false failures)",
    )
    args = parser.parse_args()

    # Auto-compute context lengths to straddle the BF16 KV limit
    if args.context_lengths is None:
        bf16_max_ctx = (
            _max_tokens_for_memory_budget(args.kv_memory_mb, 2) - args.output_len
        )
        int8_max_ctx = (
            _max_tokens_for_memory_budget(args.kv_memory_mb, 1) - args.output_len
        )

        if args.quick:
            # Two points: just below BF16 limit and just above it (but within INT8)
            args.context_lengths = [
                int(bf16_max_ctx * 0.85),  # well within BF16 → both succeed
                int(bf16_max_ctx * 1.10),  # just above BF16 → BF16 fails, INT8 succeeds
            ]
        else:
            # Sweep from 80% of BF16 limit to 120% of BF16 limit (in INT8 range)
            step = max(100, bf16_max_ctx // 8)
            start = max(200, int(bf16_max_ctx * 0.7))
            end = min(int8_max_ctx - args.output_len, int(bf16_max_ctx * 1.4))
            args.context_lengths = list(range(start, end + step, step))

    # Print plan
    bf16_max = _max_tokens_for_memory_budget(args.kv_memory_mb, 2)
    int8_max = _max_tokens_for_memory_budget(args.kv_memory_mb, 1)
    print(f"\nKV Capacity Limit Test  (budget={args.kv_memory_mb} MB)")
    print(f"  BF16 pool: {bf16_max:,} tokens max  (28KB/tok)")
    print(f"  INT8 pool: {int8_max:,} tokens max  (14KB/tok)")
    print(f"  Context lengths to test: {args.context_lengths}")
    print(f"  Output len: {args.output_len}  Timeout: {args.request_timeout:.0f}s")

    _kill_lingering_server()

    common = dict(
        kv_memory_mb=args.kv_memory_mb,
        context_lengths=args.context_lengths,
        output_len=args.output_len,
        request_timeout=args.request_timeout,
    )

    # BF16 sweep
    bf16_results = run_kv_capacity_limit_test(kv_cache_dtype="", **common)

    # INT8 sweep
    int8_results = run_kv_capacity_limit_test(kv_cache_dtype="int8", **common)

    print_capacity_comparison(
        bf16_results, int8_results, args.kv_memory_mb, args.output_len
    )

    print("\n[SUCCESS] KV capacity limit test completed.", flush=True)


if __name__ == "__main__":
    main()
