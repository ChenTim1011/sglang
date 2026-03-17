"""Profile RVV kernels and report per-component time breakdown."""

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

import requests

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server
except ImportError:
    pass  # Optional: for --launch-server only

SGL_KERNEL_PREFIX = "sgl-kernel::"
BF16_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
W8A8_MODEL = "RedHatAI/Qwen2.5-1.5B-quantized.w8a8"

# Component -> RVV source file mapping for follow-up optimization.
COMPONENT_TO_FILE = {
    "weight_packed_linear": "gemm.cpp",
    "decode_attention_cpu": "decode.cpp",
    "extend_attention_cpu": "expand.cpp",
    "rotary_embedding_cpu": "rope.cpp",
    "rmsnorm_cpu": "norm.cpp",
    "fused_add_rmsnorm_cpu": "norm.cpp",
    "silu_and_mul_cpu": "activation.cpp",
    "int8_scaled_mm_cpu": "gemm_int8.cpp",
}
SERVER_START_TIMEOUT = 1800
POST_KILL_SLEEP = 15


def _load_trace(path: Path) -> dict:
    """Load Chrome trace JSON (plain or gzipped)."""
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as f:
            return json.load(f)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _get_all_events(trace: dict) -> list[dict]:
    """Get all trace events (for computing total span)."""
    return [
        e
        for e in trace.get("traceEvents", trace.get("events", []))
        if isinstance(e, dict)
    ]


def _trace_span_us(events: list[dict]) -> float:
    """Compute wall-clock span of trace: max(ts+dur) - min(ts) in microseconds."""
    if not events:
        return 0.0
    min_ts = min(e.get("ts", 0) for e in events)
    max_end = max(e.get("ts", 0) + (e.get("dur", 0) or 0) for e in events)
    return max(0.0, max_end - min_ts)


def _extract_sgl_kernel_events(trace: dict) -> list[dict]:
    """Extract events with name starting with sgl-kernel::."""
    events = trace.get("traceEvents", trace.get("events", []))
    return [
        e
        for e in events
        if isinstance(e, dict) and e.get("name", "").startswith(SGL_KERNEL_PREFIX)
    ]


def _aggregate_component_times(events: list[dict]) -> dict[str, float]:
    """Aggregate duration (us) by kernel name. Returns {name: total_us}."""
    by_name: dict[str, float] = {}
    for e in events:
        name = e.get("name", "")
        dur = e.get("dur", 0) or 0
        if name and dur > 0:
            by_name[name] = by_name.get(name, 0) + dur
    return by_name


def parse_trace(path: Path) -> tuple[dict[str, float], float, float]:
    """
    Parse a Chrome trace file.
    Returns: (by_name, sgl_kernel_total_us, trace_span_us).
    trace_span_us = wall-clock span of the trace (for computing sgl-kernel % of total).
    """
    trace = _load_trace(path)
    all_events = _get_all_events(trace)
    sgl_events = _extract_sgl_kernel_events(trace)
    by_name = _aggregate_component_times(sgl_events)
    sgl_total_us = sum(by_name.values())
    span_us = _trace_span_us(all_events)
    return by_name, sgl_total_us, span_us


def find_traces_in_dir(output_dir: Path) -> list[Path]:
    """Find .trace.json or .trace.json.gz in output_dir."""
    found = []
    for p in output_dir.rglob("*"):
        if p.suffix == ".json" and ".trace" in p.name:
            found.append(p)
        elif p.suffix == ".gz" and p.name.endswith(".trace.json.gz"):
            found.append(p)
    return sorted(found, key=lambda x: x.stat().st_mtime, reverse=True)


def _stage_from_trace_name(name: str) -> str | None:
    """Infer PREFILL / DECODE from trace filename.
    SGLang produces: rvv-*-EXTEND.trace.json.gz (prefill), rvv-*-DECODE.trace.json.gz (decode).
    Returns None if filename has no stage suffix (e.g. rvv-*-TP-0.trace.json.gz from manual stop).
    """
    n = name.upper()
    if "EXTEND" in n:
        return "PREFILL"
    if "DECODE" in n:
        return "DECODE"
    n_lower = name.lower()
    if "prefill" in n_lower:
        return "PREFILL"
    if "decode" in n_lower:
        return "DECODE"
    return None


def _stage_from_trace_content(by_name: dict[str, float]) -> str:
    """Infer PREFILL / DECODE from sgl-kernel event dominance.
    Used when filename has no stage suffix (manual stop_profile without stage).
    """
    extend_us = sum(us for k, us in by_name.items() if "extend_attention" in k.lower())
    decode_us = sum(us for k, us in by_name.items() if "decode_attention" in k.lower())
    if decode_us > extend_us and decode_us > 0:
        return "DECODE"
    if extend_us > decode_us and extend_us > 0:
        return "PREFILL"
    return "MIXED"


def infer_stage(path: Path, by_name: dict[str, float] | None = None) -> str:
    """Infer PREFILL / DECODE / MIXED from trace filename, or from content when filename has no stage."""
    stage = _stage_from_trace_name(path.name)
    if stage is not None:
        return stage
    if by_name is None:
        by_name, _, _ = parse_trace(path)
    return _stage_from_trace_content(by_name)


def run_profile(
    url: str,
    output_dir: Path,
    num_steps: int = 1,
    profile_by_stage: bool = False,
    profile_prefix: str = "rvv",
    prompt: str = "The quick brown fox",
    max_new_tokens: int = 4,
    trace_wait_timeout: int = 3600,
) -> Path:
    """
    Start profiler, send requests to trigger forwards, then stop and flush traces.
    Returns the output_dir where traces are written.

    NOTE: We poll the output directory until the trace file appears.
    Do NOT use /dev/shm — large traces can exhaust K1's RAM (OOM).  Use /tmp (default).
    """
    output_dir = output_dir / str(int(time.time()))
    output_dir.mkdir(parents=True, exist_ok=True)

    json_data = {
        "output_dir": str(output_dir),
        "activities": ["CPU"],
        "profile_by_stage": profile_by_stage,
        "profile_prefix": profile_prefix,
        "with_stack": False,
        "record_shapes": False,
        # SGLang bug workaround: when profile_by_stage=True and num_steps is absent,
        # profiler_prefill_ct stays None → TypeError on first batch.
        # Pass a large sentinel so the stage counters are initialized but auto-stop
        # never fires (we control profiling duration via manual /stop_profile).
        "num_steps": 9999 if profile_by_stage else None,
    }
    r = requests.post(f"{url}/start_profile", json=json_data, timeout=60)
    r.raise_for_status()
    print("  Profiler started. Sending requests...")

    generate_ok = 0
    for i in range(num_steps):
        try:
            requests.post(
                f"{url}/generate",
                json={
                    "text": prompt,
                    "sampling_params": {"max_new_tokens": max_new_tokens},
                },
                timeout=300,
            )
            print(f"  Request {i + 1}/{num_steps} done.")
            generate_ok += 1
        except requests.exceptions.ConnectionError as e:
            print(
                f"  [ERROR] Request {i} failed (Connection refused — server likely crashed): {e}"
            )
            print("  Aborting: no trace will be produced.")
            raise SystemExit(1) from e
        except Exception as e:
            print(f"  [WARN] Request {i} failed: {e}")

    if generate_ok == 0:
        print("  [ERROR] All generate requests failed. Aborting.")
        raise SystemExit(1)

    # Fire-and-forget: the server writes the trace asynchronously; on slow flash
    # storage it may take many minutes.  A short timeout here is intentional.
    print("  Triggering stop_profile (polling for trace file)...")
    try:
        requests.post(f"{url}/stop_profile", timeout=30)
        print("  stop_profile completed quickly.")
        return output_dir
    except requests.exceptions.ConnectionError as e:
        print(
            f"  [ERROR] stop_profile failed (Connection refused — server likely crashed): {e}"
        )
        print("  Aborting: no trace will be produced.")
        raise SystemExit(1) from e
    except Exception:
        print(
            "  stop_profile still writing trace (expected on slow storage) — polling..."
        )

    deadline = time.time() + trace_wait_timeout
    poll_interval = 15
    min_traces = 2 if profile_by_stage else 1
    # When profile_by_stage, DECODE trace may never appear; accept 1 trace after grace period
    grace_seconds = 60 if profile_by_stage else 0
    first_trace_time = None

    while time.time() < deadline:
        traces = find_traces_in_dir(output_dir)
        if len(traces) >= min_traces:
            for t in traces[:min_traces]:
                print(f"  Trace ready: {t.name}")
            return output_dir
        if traces:
            if first_trace_time is None:
                first_trace_time = time.time()
            elapsed = time.time() - first_trace_time
            if grace_seconds > 0 and elapsed >= grace_seconds:
                print(
                    f"  Found {len(traces)} trace(s); accepting after {grace_seconds}s (DECODE may not be produced)"
                )
                for t in traces[:1]:
                    print(f"  Trace ready: {t.name}")
                return output_dir
            print(
                f"  Found {len(traces)} trace(s), waiting for {min_traces}... ({int(grace_seconds - elapsed)}s grace)"
            )
        else:
            first_trace_time = None
        remaining = int(deadline - time.time())
        print(f"  Waiting for trace file(s)... ({remaining}s remaining)", flush=True)
        time.sleep(poll_interval)

    raise TimeoutError(f"No trace file in {output_dir} after {trace_wait_timeout}s")


def print_breakdown(
    by_name: dict[str, float],
    sgl_total_us: float,
    trace_span_us: float = 0.0,
    title: str = "Component breakdown",
):
    """Print a formatted table of component times and percentages."""
    if sgl_total_us <= 0:
        print(f"{title}: no sgl-kernel events found")
        return

    # Sort by duration descending
    items = sorted(by_name.items(), key=lambda x: -x[1])
    sgl_total_ms = sgl_total_us / 1000.0

    print(f"\n{title}")
    print("=" * 70)
    if trace_span_us > 0:
        span_ms = trace_span_us / 1000.0
        pct_of_total = 100.0 * sgl_total_us / trace_span_us
        print(f"  Total trace span (wall): {span_ms:.1f} ms")
        print(
            f"  sgl-kernel total:        {sgl_total_ms:.1f} ms  →  {pct_of_total:.1f}% of total (RVV kernel 實際占比)"
        )
        print("-" * 70)
    print(f"{'Component':<45} {'Time (ms)':>10} {'%':>8}")
    print("-" * 70)
    for name, us in items:
        short = name.replace(SGL_KERNEL_PREFIX, "")
        ms = us / 1000.0
        pct = 100.0 * us / sgl_total_us
        rvv_file = COMPONENT_TO_FILE.get(short, "")
        file_str = f"  → {rvv_file}" if rvv_file else ""
        print(f"{short:<45} {ms:>10.2f} {pct:>7.1f}%{file_str}")
    print("-" * 70)
    print(f"{'TOTAL (sgl-kernel)':<45} {sgl_total_ms:>10.2f} {100.0:>7.1f}%")
    print()


def run_with_launch(
    num_steps: int,
    profile_by_stage: bool,
    output_dir: Path,
    max_new_tokens: int = 4,
    w8a8: bool = False,
) -> Path:
    """Launch server, run profile, kill server. Returns output_dir with traces."""
    try:
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server
    except ImportError as e:
        print(f"[ERROR] requires sglang: {e}")
        sys.exit(1)

    model = W8A8_MODEL if w8a8 else BF16_MODEL
    url = DEFAULT_URL_FOR_TEST
    other_args = [
        "--dtype",
        "bfloat16",
        "--device",
        "cpu",
        "--watchdog-timeout",
        "900",
        "--attention-backend",
        "rvv",
    ]
    if w8a8:
        other_args += ["--quantization", "w8a8_int8"]

    proc = popen_launch_server(
        model,
        url,
        timeout=SERVER_START_TIMEOUT,
        other_args=other_args,
        env={**os.environ},
    )
    try:
        # 300s for profile_by_stage (trace written per stage), 120s otherwise
        trace_wait = 300 if profile_by_stage else 120
        output_dir = run_profile(
            url=url,
            output_dir=output_dir,
            num_steps=num_steps,
            profile_by_stage=profile_by_stage,
            profile_prefix="rvv",
            max_new_tokens=max_new_tokens,
            trace_wait_timeout=trace_wait,
        )
        return output_dir
    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)


def main():
    parser = argparse.ArgumentParser(
        description="Profile RVV components (launch server, profile, kill)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/tmp/rvv_profile",
        help="Profile output directory (default: /tmp to avoid OOM from /dev/shm ramdisk)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=1,
        help="Number of generate requests to profile (keep small to limit trace size)",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Max new tokens per request (default: 4; use 16–32 with --profile-by-stage for more decode steps)",
    )
    parser.add_argument(
        "--profile-by-stage",
        action="store_true",
        help="Profile prefill and decode separately (recommended for decode bottleneck analysis)",
    )
    parser.add_argument(
        "--test", action="store_true", help="Smoke test: 2 steps, no profile_by_stage"
    )
    parser.add_argument(
        "--w8a8",
        action="store_true",
        help=f"Profile W8A8 quantized model ({W8A8_MODEL}) with --quantization w8a8_int8",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    num_steps = 2 if args.test else args.num_steps
    profile_by_stage = False if args.test else args.profile_by_stage
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else 4
    model_label = W8A8_MODEL if args.w8a8 else BF16_MODEL

    print(
        f"Launching server + profile: model={model_label}, w8a8={args.w8a8}, "
        f"{num_steps} steps, max_new_tokens={max_new_tokens}, "
        f"profile_by_stage={profile_by_stage}, output={output_dir}"
    )
    output_dir = run_with_launch(
        num_steps,
        profile_by_stage,
        output_dir,
        max_new_tokens=max_new_tokens,
        w8a8=args.w8a8,
    )

    traces = find_traces_in_dir(output_dir)
    if not traces:
        print("No trace files generated. Check server logs.")
        sys.exit(1)
    print(f"Traces saved to {output_dir}")
    for t in traces[:5]:
        by_name, sgl_total_us, trace_span_us = parse_trace(t)
        stage = infer_stage(t, by_name=by_name)
        print(f"\n--- {t.name}  [{stage}] ---")
        title = f"Component breakdown ({stage})"
        print_breakdown(by_name, sgl_total_us, trace_span_us, title=title)


if __name__ == "__main__":
    main()
