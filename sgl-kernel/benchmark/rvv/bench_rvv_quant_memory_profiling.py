# Decode throughput + TTFT + memory profiling for RVV CPU backend.
#
# Experiment purpose:
#   Compare four quantization configurations (BF16, W8A8, INT8-KV, W8A8+INT8KV) on the
#   RVV backend to evaluate memory savings, throughput trade-offs, and decode stability
#   for long-context inference on RISC-V CPU (e.g. SpacemiT K1).
#
# What this script does:
#   For each mode (BF16 packed, W8A8, INT8 KV cache, W8A8+INT8KV):
#     a. Launch SGLang server with mode-specific env/args (--device cpu, --dtype bfloat16)
#     b. Measure TTFT (time-to-first-token) for short-ctx/5 and long-ctx/128 prompts
#     c. If --memory-profile: run one streaming request (long-ctx/128, N tok out),
#        sample RSS at each streaming chunk; report peak, prefill RSS, decode stability
#     d. If --kv-eviction: send N sequential requests with unique prompts; track RSS until
#        KV pool fills and plateaus. Unique suffix [req-{i}] prevents prefix cache reuse.
#     e. Run decode throughput configs (short/long context, BS=1/BS=4); poll peak RSS
#     f. Kill server, wait for port free, repeat for next mode
#   Print summary tables: throughput (tok/s), TTFT (ms), memory (RSS), memory/KV traces.
#
# Metrics:
#   - Throughput: output tokens per second (decode phase)
#   - TTFT: time from request start to first token (prefill-dominated)
#   - RSS: process-tree resident memory (main + all child processes; matches htop)
#   - Memory trace (--memory-profile): peak RSS, prefill RSS, decode-phase stability
#   - KV eviction (--kv-eviction): plateau RSS, stabilize_at (request where pool fills)
#
# KV eviction test logic:
#   Request 0..K:   RSS rises as KV pool fills (new entries, no eviction yet)
#   Request K+1..N: RSS plateaus — pool full, eviction matches allocation
#   INT8 KV plateau ≈ 50% below BF16 plateau (half the pool capacity used)
#
# QUANTIZATION MODES (two independent axes):
#
#   Axis 1 — Weight + activation (select via --bf16-model / --w8a8-model):
#     • BF16:  Qwen/Qwen2.5-1.5B-Instruct — bfloat16 weights, bfloat16 activations (baseline)
#     • W8A8:  RedHatAI/Qwen2.5-1.5B-quantized.w8a8 — pre-calibrated INT8
#              weights + per-token dynamic activation quantization (--quantization w8a8_int8)
#
#   Axis 2 — KV cache (applies to any model):
#     • BF16 KV:  bfloat16 key/value cache (default)
#     • INT8 KV:  INT8 KV cache (--kv-cache-dtype int8), ~2x smaller pool
#
#   Four combinations:
#     | Mode          | Model  | Weight+Act | KV cache | Args                         |
#     |---------------|--------|------------|----------|------------------------------|
#     | BF16 packed   | Qwen   | BF16       | BF16     | (none)                       |
#     | INT8 KV cache | Qwen   | BF16       | INT8     | --kv-cache-dtype int8        |
#     | W8A8          | NM-q   | INT8       | BF16     | --quantization w8a8_int8     |
#     | W8A8+INT8KV   | NM-q   | INT8       | INT8     | w8a8_int8 + kv-cache int8    |
#
#   Trade-offs:
#     • W8A8: faster INT8 GEMM, ~2x lower weight memory; requires pre-quantized model
#     • INT8 KV: lower KV pool memory (~2x); works with any BF16 model dynamically
#     • W8A8+INT8KV: max memory savings, max speedup; cumulative quality impact
#
# Usage:
#   python bench_rvv_quant_memory_profiling.py                        # BF16 modes only (2 modes)
#   python bench_rvv_quant_memory_profiling.py --w8a8-model           # W8A8 modes only (2 modes)
#   python bench_rvv_quant_memory_profiling.py --bf16-model --w8a8-model  # all 4 modes
#   python bench_rvv_quant_memory_profiling.py --quick                # shorter run (2 configs)
#   python bench_rvv_quant_memory_profiling.py --memory-profile       # full run + memory trace per mode
#   python bench_rvv_quant_memory_profiling.py --memory-profile --memory-profile-tokens 256
#   python bench_rvv_quant_memory_profiling.py --quick --kv-eviction              # KV eviction only
#   python bench_rvv_quant_memory_profiling.py --memory-profile --kv-eviction     # full + both memory tests
#   python bench_rvv_quant_memory_profiling.py --quick --kv-eviction --kv-eviction-requests 20

import argparse
import json
import os
import platform
import socket
import sys
import threading
import time
from dataclasses import dataclass
from typing import List, Optional
from urllib.parse import urlparse

import psutil
import requests

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server
except ImportError:
    print("sglang not found")
    sys.exit(1)

BF16_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
W8A8_MODEL = "RedHatAI/Qwen2.5-1.5B-quantized.w8a8"
BASE_URL = DEFAULT_URL_FOR_TEST

# ----- prompts -----

_SHORT_PROMPT = "The quick fox: "  # ~5 tokens
_CTX_128 = ("the quick brown fox jumps over the lazy dog " * 19).strip()  # ~128 tokens

# ----- benchmark configs -----


@dataclass
class BenchConfig:
    prompt: str
    max_tokens: int
    batch_size: int
    desc: str


DECODE_CONFIGS = [
    # Short context — nearly pure decode batching
    BenchConfig(_SHORT_PROMPT, 64, 1, "BS=1  short-ctx/5   (64 tok out)"),
    BenchConfig(_SHORT_PROMPT, 16, 4, "BS=4  short-ctx/5   (16 tok out)"),
    # Long context — INT8 KV advantage visible; NOTE: prefill ~1 min on K1
    BenchConfig(_CTX_128, 32, 1, "BS=1  long-ctx/128  (32 tok out)"),
    BenchConfig(_CTX_128, 16, 4, "BS=4  long-ctx/128  (16 tok out)"),
]

QUICK_CONFIGS = [
    BenchConfig(_SHORT_PROMPT, 16, 1, "BS=1 short-ctx/5   [quick]"),
    BenchConfig(_CTX_128, 8, 1, "BS=1 mid-ctx/128   [quick]"),
]

# ----- memory helpers -----


def _process_tree_rss_mb(pid: int) -> float:
    """Return total RSS (MB) of process and all descendants.
    SGLang uses multiprocessing; model + KV cache live in child processes.
    Single-process RSS (~613 MB) underreports; htop shows ~13 GB (process tree).
    """
    try:
        proc = psutil.Process(pid)
        total = proc.memory_info().rss
        for child in proc.children(recursive=True):
            try:
                total += child.memory_info().rss
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return total / (1024 * 1024)
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return -1.0


# ----- memory monitor -----


class _MemoryTraceMonitor:
    """Record (decode_step, rss_mb) samples during a streaming request.
    Used to analyze: peak memory, prefill vs decode memory, decode stability.

    Lifecycle:
      1. Call pre_request_sample() BEFORE sending the HTTP request — records true baseline.
      2. Call sample(step) for each SSE chunk received (decode_step 0, 1, 2, ...).
         step=0 is after prefill + 1st token; the delta from baseline shows prefill cost.
      3. Call analyze() to get the full breakdown.
    """

    def __init__(self, pid: int):
        self.pid = pid
        self._baseline_mb: float = -1.0  # RSS before request is sent
        self._decode_samples: List[tuple[int, float]] = []  # (step, rss_mb), step >= 0

    def pre_request_sample(self) -> float:
        """Sample RSS BEFORE sending request — captures true prefill baseline.
        Call this once right before the POST, not during streaming.
        """
        rss = _process_tree_rss_mb(self.pid)
        self._baseline_mb = max(rss, 0.0)
        return rss

    def sample(self, decode_step: int) -> float:
        """Record RSS at current decode step (step >= 0). Returns rss_mb."""
        rss = _process_tree_rss_mb(self.pid)
        if rss >= 0:
            self._decode_samples.append((decode_step, rss))
        return rss

    def analyze(self) -> dict:
        """Return peak_mb, baseline_mb, prefill_rss, prefill_delta_mb,
        decode_stable, stability_std, samples (decode only).
        """
        if not self._decode_samples:
            return {
                "peak_mb": -1,
                "baseline_mb": self._baseline_mb,
                "prefill_rss": -1,
                "prefill_delta_mb": -1,
                "decode_stable": False,
                "stability_std": -1,
                "samples": [],
            }

        # Peak across ALL samples (including baseline if we have it)
        all_rss = [self._baseline_mb] + [r for _, r in self._decode_samples]
        peak_mb = max(r for r in all_rss if r >= 0)

        # prefill_rss = RSS at step 0 (first token arrived = prefill done + 1 decode)
        prefill_rss = self._decode_samples[0][1]
        baseline_mb = self._baseline_mb
        prefill_delta_mb = (prefill_rss - baseline_mb) if baseline_mb >= 0 else -1

        # Decode stability: last 50% of decode samples (after memory "settles")
        if len(self._decode_samples) >= 4:
            half = len(self._decode_samples) // 2
            tail_rss = [r for _, r in self._decode_samples[half:]]
            mean_r = sum(tail_rss) / len(tail_rss)
            std = (sum((x - mean_r) ** 2 for x in tail_rss) / len(tail_rss)) ** 0.5
            # Stable if std < 5% of mean or < 50 MB absolute
            decode_stable = std < 0.05 * mean_r or std < 50
        else:
            std = -1
            decode_stable = False

        return {
            "peak_mb": peak_mb,
            "baseline_mb": baseline_mb,
            "prefill_rss": prefill_rss,
            "prefill_delta_mb": prefill_delta_mb,
            "decode_stable": decode_stable,
            "stability_std": std,
            "samples": self._decode_samples,  # (step, rss_mb), step >= 0
        }


class _MemoryMonitor:
    """Poll server process RSS in a background thread to capture peak during decode."""

    def __init__(
        self, pid: int, interval: float = 0.2
    ):  # 0.5→0.2 for better prefill coverage
        self.pid = pid
        self.interval = interval
        self.baseline_mb: float = -1.0
        self.peak_mb: float = -1.0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self, baseline_mb: float):
        self.baseline_mb = baseline_mb
        self.peak_mb = baseline_mb
        self._thread.start()

    def stop(self) -> float:
        """Stop monitoring and return peak RSS in MB."""
        self._stop.set()
        self._thread.join()
        return self.peak_mb

    def _run(self):
        try:
            while not self._stop.is_set():
                rss = _process_tree_rss_mb(self.pid)
                if rss >= 0 and rss > self.peak_mb:
                    self.peak_mb = rss
                self._stop.wait(self.interval)
        except Exception:
            pass


# ----- bandwidth estimate -----

# E2E model (Qwen2.5-1.5B) architecture constants (for bandwidth estimation)
_E2E_N_LAYERS = 24
_E2E_N_KV_HEADS = 8  # GQA: 8 KV heads
_E2E_HEAD_DIM = 48
_E2E_WEIGHT_BYTES_BF16 = 3_000_000_000  # ~3 GB BF16 weights (1.5B params)


def _estimate_memory_bandwidth_gbs(
    input_seq_len: int,
    n_decode_steps: int,
    decode_time_s: float,
    kv_dtype_bytes: int = 2,  # 2 = BF16, 1 = INT8
) -> float:
    """Estimate effective memory bandwidth (GB/s) during decode phase.

    Each decode step reads:
      - Model weights (dominant for small models)
      - KV cache: n_layers × 2(K+V) × n_kv_heads × head_dim × avg_seq × dtype_bytes

    NOTE: On small E2E models, KV << weights. INT8 KV saves pool capacity.
    """
    if decode_time_s <= 0 or n_decode_steps <= 0:
        return -1.0
    avg_seq = input_seq_len + n_decode_steps / 2  # average KV length during decode
    kv_bytes_per_step = (
        _E2E_N_LAYERS * 2 * _E2E_N_KV_HEADS * _E2E_HEAD_DIM * avg_seq * kv_dtype_bytes
    )
    total_bytes_per_step = _E2E_WEIGHT_BYTES_BF16 + kv_bytes_per_step
    latency_per_step_s = decode_time_s / n_decode_steps
    return total_bytes_per_step / latency_per_step_s / (1024**3)


# ----- measurement -----


def _measure_ttft(prompt: str) -> Optional[float]:
    """TTFT approximation: time to generate exactly 1 token after prefill (seconds)."""
    try:
        t0 = time.perf_counter()
        requests.post(
            f"{BASE_URL}/generate",
            json={"text": prompt, "sampling_params": {"max_new_tokens": 1}},
            timeout=300,
        )
        return time.perf_counter() - t0
    except Exception as e:
        print(f"    [TTFT ERROR] {e}")
        return None


def _run_memory_profiling(
    prompt: str,
    max_new_tokens: int,
    pid: int,
    request_timeout: int = 600,
) -> Optional[dict]:
    """Run a streaming request and sample RSS at each decode step.

    Timeline:
      pre_request_sample()  ← baseline before prefill
      POST /generate        ← prefill starts
      chunk 0 arrives       ← prefill done; step=0 RSS shows prefill cost
      chunk 1, 2, ...       ← one new KV pair added per step; RSS grows slowly
      [DONE]                ← decode finished

    Returns analysis dict with: peak_mb, baseline_mb, prefill_rss,
    prefill_delta_mb, decode_stable, stability_std, samples,
    ttft_s, decode_time_s, n_decode_steps.
    """
    monitor = _MemoryTraceMonitor(pid)

    # ── Sample BEFORE sending request ──────────────────────────────────────
    monitor.pre_request_sample()

    url = f"{BASE_URL}/generate"
    payload = {
        "text": prompt,
        "sampling_params": {"max_new_tokens": max_new_tokens},
        "stream": True,
    }

    t_request_start = time.perf_counter()
    t_first_token: Optional[float] = None
    t_end: float = t_request_start

    try:
        with requests.post(
            url, json=payload, stream=True, timeout=request_timeout
        ) as resp:
            resp.raise_for_status()
            decode_step = 0
            for line in resp.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in chunk:
                    return None
                if t_first_token is None:
                    t_first_token = time.perf_counter()  # ← prefill latency ends here
                # Sample RSS after each decode step (step=0 = after prefill + 1st token)
                monitor.sample(decode_step)
                decode_step += 1
            t_end = time.perf_counter()

        result = monitor.analyze()
        result["ttft_s"] = (t_first_token - t_request_start) if t_first_token else -1
        result["decode_time_s"] = (t_end - t_first_token) if t_first_token else -1
        result["n_decode_steps"] = decode_step
        return result
    except Exception as e:
        print(f"    [Memory profiling ERROR] {e}")
        return None


def _run_kv_eviction_test(
    prompt_base: str,
    n_requests: int,
    pid: int,
    request_timeout: int = 600,
) -> dict:
    """Send n_requests sequential requests and track RSS after each one.

    Each request uses a UNIQUE prompt (unique suffix) so SGLang's prefix cache
    cannot reuse existing KV entries — each request genuinely fills new slots.

    Expected RSS progression:
      Requests 0..K   : RSS rises as KV pool fills (new entries, no eviction yet)
      Requests K+1..N : RSS plateaus — pool is full, eviction now matches allocation
      INT8 KV plateau ≈ 50% below BF16 plateau (half the pool capacity used)

    Uses max_new_tokens=8 (fast decode) so the test finishes in reasonable time.
    On K1 (E2E model): ~6s per request × 60 requests ≈ 6 min per mode.
    """
    samples: List[tuple[int, float]] = []  # (req_idx, rss_mb)
    initial_rss = _process_tree_rss_mb(pid)
    max_rss_seen = initial_rss

    for i in range(n_requests):
        # Unique prompt → forces new KV allocation every request
        prompt = f"{prompt_base} [req-{i}]"
        try:
            requests.post(
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

        # Inline progress bar (overwrites same line until done)
        bar_range = max(max_rss_seen - initial_rss, 1.0)
        bar_len = int((rss - initial_rss) / bar_range * 20)
        print(
            f"\r    req {i+1:>4}/{n_requests}:  {rss:>7.0f} MB"
            f"  [{('#' * bar_len):<20}]  Δ{rss - initial_rss:+.0f} MB",
            end="",
            flush=True,
        )

    print(flush=True)  # newline after progress loop

    if len(samples) < 4:
        return {
            "samples": samples,
            "initial_rss": initial_rss,
            "plateau_mb": -1,
            "growth_mb": -1,
            "stabilize_at": None,
            "n_requests": len(samples),
        }

    # ── Plateau = mean RSS of last 30% of requests ──────────────────────────
    # Assumption: by last 30%, the KV pool is full and eviction is steady-state.
    tail_start = max(1, int(len(samples) * 0.7))
    tail_rss = [r for _, r in samples[tail_start:]]
    plateau_mb = sum(tail_rss) / len(tail_rss)
    growth_mb = plateau_mb - initial_rss

    # ── Stabilization point: first request where RSS ≥ 95% of plateau ───────
    # This marks the request where KV pool first reached (near-)capacity.
    # A 5% tolerance handles measurement noise from psutil RSS sampling.
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


def _measure(
    cfg: BenchConfig, warmup: int = 1, request_timeout: int = 1800
) -> Optional[float]:
    """Returns tok/s (output tokens only), or None on error."""
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
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None

    return cfg.batch_size * cfg.max_tokens / elapsed


SERVER_START_TIMEOUT = 1800
POST_KILL_SLEEP = 180


def _port_in_use(host: str, port: int) -> bool:
    """Return True if something is listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect((host, port))
            return True
        except (socket.error, OSError):
            return False


def _wait_for_port_free(host: str, port: int, timeout: float = 300) -> bool:
    """Wait until port is free (connection refused). Returns True if free, False on timeout."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if not _port_in_use(host, port):
            return True
        time.sleep(5)
    return False


def _kill_lingering_server():
    """Kill any leftover sglang serve processes from a previous interrupted run."""
    parsed = urlparse(BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 21000

    killed_any = False
    for proc in psutil.process_iter(["pid", "cmdline"]):
        try:
            cmdline = proc.info.get("cmdline") or []
            if any("sglang" in c for c in cmdline) and (
                any("serve" in c for c in cmdline)
                or any("launch_server" in c for c in cmdline)
            ):
                print(
                    f"[cleanup] Killing lingering server pid={proc.pid}: "
                    f"{' '.join(cmdline[:4])}",
                    flush=True,
                )
                kill_process_tree(proc.pid)
                killed_any = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

    if killed_any:
        time.sleep(5)  # Let OS reclaim socket FDs
        if not _wait_for_port_free(host, port, timeout=60):
            print(f"[cleanup] Warning: port {port} still in use after 60s", flush=True)
        else:
            print(f"[cleanup] Port {port} is free.", flush=True)


def _launch(
    model: str,
    kv_cache_dtype: str = "",
    quant_arg: str = "",
    max_total_tokens: Optional[int] = None,
):
    other_args = [
        "--dtype",
        "bfloat16",
        "--device",
        "cpu",
        "--watchdog-timeout",
        "2400",
    ]
    if kv_cache_dtype:
        other_args += ["--kv-cache-dtype", kv_cache_dtype]
    if quant_arg:
        other_args += ["--quantization", quant_arg]
    if max_total_tokens is not None:
        other_args += ["--max-total-tokens", str(max_total_tokens)]
    return popen_launch_server(
        model, BASE_URL, timeout=SERVER_START_TIMEOUT, other_args=other_args
    )


# ----- main -----

# (name, model, quant_arg, kv_cache_dtype)
# Select modes via --bf16-model / --w8a8-model CLI flags (see main()).
_BF16_MODES = [
    ("BF16 packed", BF16_MODEL, "", ""),  # BF16 weights, BF16 KV
    ("INT8 KV cache", BF16_MODEL, "", "int8"),  # BF16 weights, INT8 KV
]
_W8A8_MODES = [
    ("W8A8", W8A8_MODEL, "w8a8_int8", ""),  # INT8 weights+act, BF16 KV
    ("W8A8+INT8KV", W8A8_MODEL, "w8a8_int8", "int8"),  # INT8 weights+act, INT8 KV
]


def run_mode(
    name: str,
    model: str,
    quant_arg: str,
    configs: List[BenchConfig],
    kv_cache_dtype: str = "",
    request_timeout: int = 1800,
    memory_profile: bool = False,
    memory_profile_tokens: int = 128,
    kv_eviction: bool = False,
    kv_eviction_requests: int = 60,
) -> dict:
    print(f"\n{'='*60}", flush=True)
    print(
        f"  Mode: {name}" + (f" [kv={kv_cache_dtype}]" if kv_cache_dtype else ""),
        flush=True,
    )
    print(f"{'='*60}", flush=True)

    # Ensure port is free before launch (prevents bind failure after mode switch)
    parsed = urlparse(BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 21000
    if _port_in_use(host, port):
        print(f"  Waiting for port {port} to be free...", flush=True)
        if not _wait_for_port_free(host, port):
            raise RuntimeError(
                f"Port {port} still in use after 300s; cannot launch server"
            )
        print(f"  Port {port} is free.", flush=True)

    # KV eviction test: limit pool so it saturates during the test run.
    # Each request uses ~140 tokens (128 ctx + ~4 suffix + 8 output).
    # Target capacity = 70% of total test workload → pool fills at ~req 42 of 60.
    max_total_tokens = None
    if kv_eviction:
        tokens_per_req = 140  # conservative estimate: 128 ctx + suffix + 8 output
        max_total_tokens = max(512, int(kv_eviction_requests * tokens_per_req * 0.7))
        print(
            f"  KV pool capped at {max_total_tokens} tokens"
            f" (70% of {kv_eviction_requests} × {tokens_per_req} tok/req)"
            f" to force pool saturation.",
            flush=True,
        )

    proc = _launch(model, kv_cache_dtype, quant_arg, max_total_tokens)
    try:
        # RSS right after server ready: process tree (model + KV pool + children)
        baseline_rss = _process_tree_rss_mb(proc.pid)
        print(
            f"  RSS at startup:  {baseline_rss:.0f} MB  (process tree: model + KV pool)",
            flush=True,
        )

        ttft_short = None
        ttft_long = None
        print(f"  TTFT short-ctx/5   ...", end=" ", flush=True)
        ttft_short = _measure_ttft(_SHORT_PROMPT)
        print(f"{ttft_short*1000:.0f} ms" if ttft_short else "FAILED")

        print(f"  TTFT long-ctx/128  ...", end=" ", flush=True)
        ttft_long = _measure_ttft(_CTX_128)
        print(f"{ttft_long*1000:.0f} ms" if ttft_long else "FAILED")

        memory_trace = None
        if memory_profile:
            print(
                f"  Memory profiling (long-ctx/128, {memory_profile_tokens} tok out) ...",
                flush=True,
            )
            memory_trace = _run_memory_profiling(
                _CTX_128,
                memory_profile_tokens,
                proc.pid,
                request_timeout=request_timeout,
            )
            if memory_trace:
                baseline = memory_trace["baseline_mb"]
                prefill = memory_trace["prefill_rss"]
                delta = memory_trace["prefill_delta_mb"]
                peak = memory_trace["peak_mb"]
                stable = "✓" if memory_trace["decode_stable"] else "✗"
                std = memory_trace["stability_std"]
                ttft = memory_trace.get("ttft_s", -1)
                dec_t = memory_trace.get("decode_time_s", -1)
                n_steps = memory_trace.get("n_decode_steps", 0)

                # KV dtype bytes: 1 for int8, 2 for bf16
                kv_bytes = 1 if kv_cache_dtype == "int8" else 2
                bw = _estimate_memory_bandwidth_gbs(128, n_steps, dec_t, kv_bytes)

                print(f"    Baseline (pre-request): {baseline:.0f} MB", flush=True)
                print(
                    f"    After prefill (step 0): {prefill:.0f} MB  "
                    f"[prefill alloc Δ={delta:+.0f} MB]",
                    flush=True,
                )
                peak_src = "decode" if peak > prefill else "prefill"
                print(
                    f"    Peak RSS:               {peak:.0f} MB  (from {peak_src} phase)",
                    flush=True,
                )
                std_str = f"  σ={std:.0f} MB" if std >= 0 else ""
                print(
                    f"    Decode stability:       {stable}{std_str}  "
                    f"({n_steps} steps, {dec_t:.0f}s)",
                    flush=True,
                )
                if bw > 0:
                    print(f"    Est. memory bandwidth:  {bw:.2f} GB/s", flush=True)
                if ttft > 0:
                    print(f"    TTFT (from trace):      {ttft*1000:.0f} ms", flush=True)

                # Abbreviated per-step trace (print every stride steps)
                samples = memory_trace.get("samples", [])
                if samples:
                    stride = max(1, len(samples) // 10)  # ~10 points
                    print(f"    Per-step RSS (every {stride} steps):", flush=True)
                    for step, rss in samples[::stride]:
                        bar = "#" * int((rss - baseline) / max(peak - baseline, 1) * 20)
                        print(
                            f"      step {step:>4}: {rss:>7.0f} MB  [{bar:<20}]",
                            flush=True,
                        )
            else:
                print("  FAILED", flush=True)

        # ── KV eviction test ─────────────────────────────────────────────────
        kv_eviction_trace = None
        if kv_eviction:
            print(
                f"  KV eviction test ({kv_eviction_requests} sequential requests,"
                f" 8 tok each, unique prompts) ...",
                flush=True,
            )
            kv_eviction_trace = _run_kv_eviction_test(
                _CTX_128,
                kv_eviction_requests,
                proc.pid,
                request_timeout=request_timeout,
            )
            if kv_eviction_trace and kv_eviction_trace["plateau_mb"] >= 0:
                plateau = kv_eviction_trace["plateau_mb"]
                growth = kv_eviction_trace["growth_mb"]
                stabilize = kv_eviction_trace["stabilize_at"]
                stab_str = (
                    f"req #{stabilize}" if stabilize is not None else "not reached"
                )
                print(
                    f"  KV pool plateau: {plateau:.0f} MB  "
                    f"(Δ+{growth:.0f} MB from initial)  "
                    f"stabilized at {stab_str}",
                    flush=True,
                )
                if stabilize is None:
                    print(
                        f"  WARNING: KV pool never reached 95% of plateau. "
                        f"Increase --kv-eviction-requests or check pool capacity.",
                        flush=True,
                    )
            else:
                print("  FAILED", flush=True)

        # Start memory monitor — tracks RSS while decode benchmarks run
        monitor = _MemoryMonitor(proc.pid)
        monitor.start(baseline_rss)

        tps_results = {}
        for cfg in configs:
            print(f"  {cfg.desc} ...", end=" ", flush=True)
            tps = _measure(cfg, request_timeout=request_timeout)
            print(f"{tps:.2f} tok/s" if tps is not None else "FAILED", flush=True)
            tps_results[cfg.desc] = tps

        peak_rss = monitor.stop()
        delta = peak_rss - baseline_rss
        print(
            f"  Peak RSS during decode: {peak_rss:.0f} MB  (Δ{delta:+.0f} MB vs startup)",
            flush=True,
        )

    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)

    return {
        "rss_mb": baseline_rss,
        "peak_rss_mb": peak_rss,
        "ttft_short_ms": ttft_short * 1000 if ttft_short else None,
        "ttft_long_ms": ttft_long * 1000 if ttft_long else None,
        "tps": tps_results,
        "memory_trace": memory_trace if memory_profile else None,
        "kv_eviction_trace": kv_eviction_trace if kv_eviction else None,
        "kv_cache_dtype": kv_cache_dtype,
    }


def print_summary(all_results: dict, configs: List[BenchConfig]):
    col_w = 16
    mode_names = list(all_results.keys())

    # --- Throughput ---
    print(f"\n{'='*70}")
    print("  THROUGHPUT  (tok/s decode, higher is better)")
    print(f"{'='*70}")
    header = f"{'Config':<38}" + "".join(f"{m:<{col_w}}" for m in mode_names)
    print(header)
    print("-" * (38 + col_w * len(mode_names)))

    bf16_tps = all_results.get("BF16 packed", {}).get("tps", {})
    for cfg in configs:
        row = f"{cfg.desc:<38}"
        baseline = bf16_tps.get(cfg.desc)
        for name in mode_names:
            tps = all_results.get(name, {}).get("tps", {}).get(cfg.desc)
            if tps is None:
                row += f"{'N/A':<{col_w}}"
            elif name == "BF16 packed" or baseline is None:
                row += f"{tps:.2f} tok/s{'':<{col_w - 11}}"
            else:
                row += f"{tps:.2f}({tps/baseline:.2f}x){'':<{col_w - 14}}"
        print(row)
    print("\nSpeedup is relative to BF16 packed baseline.")

    # --- TTFT ---
    print(f"\n{'='*70}")
    print("  TTFT  (time-to-first-token, lower is better)")
    print(f"{'='*70}")
    print(f"  {'Mode':<22} {'TTFT short-ctx/5':>18}  {'TTFT long-ctx/128':>18}")
    print(f"  {'-'*60}")
    for name in mode_names:
        r = all_results.get(name, {})
        ts = r.get("ttft_short_ms")
        tl = r.get("ttft_long_ms")
        ts_str = f"{ts:.0f} ms" if ts is not None else "N/A"
        tl_str = f"{tl:.0f} ms" if tl is not None else "N/A"
        print(f"  {name:<22} {ts_str:>18}  {tl_str:>18}")

    # --- Memory ---
    print(f"\n{'='*70}")
    print("  MEMORY  (RSS = model + KV pool + overhead)")
    print(f"{'='*70}")
    print(
        f"  {'Mode':<22} {'Startup RSS':>12}  {'Peak RSS':>10}  {'Delta':>8}  {'vs BF16':>10}"
    )
    print(f"  {'-'*70}")
    bf16_rss = all_results.get("BF16 packed", {}).get("rss_mb", 0)
    for name in mode_names:
        r = all_results.get(name, {})
        rss = r.get("rss_mb", -1)
        peak = r.get("peak_rss_mb", -1)
        rss_str = f"{rss:.0f} MB" if rss > 0 else "N/A"
        peak_str = f"{peak:.0f} MB" if peak > 0 else "N/A"
        delta_str = f"{peak - rss:+.0f} MB" if rss > 0 and peak > 0 else ""
        vs_str = (
            f"-{bf16_rss - rss:.0f} MB"
            if rss > 0 and bf16_rss > 0 and name != "BF16 packed"
            else ""
        )
        print(
            f"  {name:<22} {rss_str:>12}  {peak_str:>10}  {delta_str:>8}  {vs_str:>10}"
        )

    # --- Memory trace (when --memory-profile) ---
    if any(all_results.get(n, {}).get("memory_trace") for n in mode_names):
        print(f"\n{'='*80}")
        print("  MEMORY TRACE  (baseline → prefill → decode peak → stability)")
        print(f"{'='*80}")
        print("  Baseline   = RSS before request is sent")
        print("  Prefill Δ  = RSS growth during prefill (first token - baseline)")
        print("  Peak       = max RSS during entire request (OOM risk indicator)")
        print("  Stable     = decode-phase RSS settles (last 50% of steps, σ < 5%)")
        print(
            "  BW         = estimated effective memory bandwidth (weights + KV reads)"
        )
        print(
            f"\n  {'Mode':<22} {'Baseline':>10}  {'Prefill Δ':>10}  "
            f"{'Peak':>10}  {'Stable':>8}  {'BW':>10}"
        )
        print(f"  {'-'*76}")
        for name in mode_names:
            t = all_results.get(name, {}).get("memory_trace")
            if not t:
                continue
            baseline = t.get("baseline_mb", -1)
            delta = t.get("prefill_delta_mb", -1)
            peak = t.get("peak_mb", -1)
            stable = "✓" if t.get("decode_stable") else "✗"

            # bandwidth
            kv_dtype = all_results.get(name, {}).get("kv_cache_dtype", "")
            kv_bytes = 1 if kv_dtype == "int8" else 2
            bw = _estimate_memory_bandwidth_gbs(
                128,
                t.get("n_decode_steps", 0),
                t.get("decode_time_s", -1),
                kv_bytes,
            )
            base_str = f"{baseline:.0f} MB" if baseline >= 0 else "N/A"
            delta_str = f"{delta:+.0f} MB" if delta >= 0 else "N/A"
            peak_str = f"{peak:.0f} MB" if peak >= 0 else "N/A"
            bw_str = f"{bw:.2f} GB/s" if bw > 0 else "N/A"
            print(
                f"  {name:<22} {base_str:>10}  {delta_str:>10}  "
                f"{peak_str:>10}  {stable:>8}  {bw_str:>10}"
            )

        # Footnote on KV eviction behavior
        print()
        print("  NOTE: Within a single streaming request, KV grows monotonically")
        print("  (one KV pair added per decode step). True KV eviction only happens")
        print("  between requests when the pool is full.")
        print("  Use --kv-eviction to run the multi-request pool saturation test.")

    # --- KV eviction test (when --kv-eviction) ---
    if any(all_results.get(n, {}).get("kv_eviction_trace") for n in mode_names):
        print(f"\n{'='*80}")
        print("  KV EVICTION TEST  (sequential requests → pool fills → plateau)")
        print(f"{'='*80}")
        print("  Initial RSS  = RSS before first request (server idle)")
        print("  Plateau RSS  = RSS after KV pool saturates (eviction steady-state)")
        print("  KV pool Δ   = Plateau - Initial ≈ KV pool capacity in use")
        print("  Stable at    = first request where RSS reached 95% of plateau")
        print("  INT8 KV plateau should be ~50% below BF16 plateau (half pool size)")
        print(
            f"\n  {'Mode':<22} {'Initial':>10}  {'Plateau':>10}  "
            f"{'KV pool Δ':>10}  {'Stable at':>10}  {'Ratio vs BF16':>14}"
        )
        print(f"  {'-'*80}")
        bf16_plateau = (
            all_results.get("BF16 packed", {}).get("kv_eviction_trace") or {}
        ).get("plateau_mb", 0)
        bf16_initial = (
            all_results.get("BF16 packed", {}).get("kv_eviction_trace") or {}
        ).get("initial_rss", 0)
        for name in mode_names:
            t = all_results.get(name, {}).get("kv_eviction_trace")
            if not t:
                continue
            initial = t.get("initial_rss", -1)
            plateau = t.get("plateau_mb", -1)
            growth = t.get("growth_mb", -1)
            stabilize = t.get("stabilize_at")

            init_str = f"{initial:.0f} MB" if initial >= 0 else "N/A"
            plat_str = f"{plateau:.0f} MB" if plateau >= 0 else "N/A"
            growth_str = f"+{growth:.0f} MB" if growth >= 0 else "N/A"
            stab_str = f"req #{stabilize}" if stabilize is not None else "N/A"

            # KV pool growth ratio vs BF16 baseline (expect ~0.5x for INT8 KV)
            if name != "BF16 packed" and growth >= 0 and bf16_plateau > bf16_initial:
                bf16_growth = bf16_plateau - bf16_initial
                ratio = growth / bf16_growth if bf16_growth > 0 else -1
                ratio_str = f"{ratio:.2f}x" if ratio >= 0 else "N/A"
            else:
                ratio_str = "(baseline)"

            print(
                f"  {name:<22} {init_str:>10}  {plat_str:>10}  "
                f"{growth_str:>10}  {stab_str:>10}  {ratio_str:>14}"
            )

        # Show abbreviated per-mode RSS progression (every 10 requests)
        print()
        for name in mode_names:
            t = all_results.get(name, {}).get("kv_eviction_trace")
            if not t or not t.get("samples"):
                continue
            samples = t["samples"]
            initial = t.get("initial_rss", samples[0][1])
            plateau = t.get("plateau_mb", samples[-1][1])
            stride = max(1, len(samples) // 10)
            print(f"  {name} — per-request RSS (every {stride} reqs):")
            for req_idx, rss in samples[::stride]:
                bar_range = max(plateau - initial, 1.0)
                bar_len = int((rss - initial) / bar_range * 24)
                marker = "▶" if req_idx == t.get("stabilize_at") else " "
                print(
                    f"    {marker} req {req_idx:>4}: {rss:>7.0f} MB"
                    f"  [{('#' * bar_len):<24}]  Δ{rss - initial:+.0f} MB"
                )
            print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Memory + throughput profiling for RVV quantization modes.\n"
            "Select model(s) to benchmark via --bf16-model and/or --w8a8-model.\n"
            "Default (no flag): runs BF16 modes only."
        )
    )
    parser.add_argument("--quick", action="store_true", help="Short run for CI")
    parser.add_argument(
        "--bf16-model",
        action="store_true",
        help=f"Run BF16 modes using {BF16_MODEL} (BF16 packed, INT8 KV cache). Default if no model flag given.",
    )
    parser.add_argument(
        "--w8a8-model",
        action="store_true",
        help=(
            f"Run W8A8 modes using {W8A8_MODEL} + --quantization w8a8_int8 (W8A8, W8A8+INT8KV). "
            "Note: base model (non-Instruct) — sufficient for memory/throughput benchmarking."
        ),
    )
    parser.add_argument(
        "--memory-profile",
        action="store_true",
        help="Sample RSS at each decode step; report peak, prefill, decode stability",
    )
    parser.add_argument(
        "--memory-profile-tokens",
        type=int,
        default=128,
        help=(
            "Number of decode tokens for memory profiling request (default 128). "
            "More tokens → better stability analysis. ~0.7s/tok on K1 (E2E model)."
        ),
    )
    parser.add_argument(
        "--kv-eviction",
        action="store_true",
        help=(
            "Run KV pool saturation test: send N sequential requests with unique prompts, "
            "track RSS until pool fills and plateaus. Compares BF16 vs INT8 KV pool size."
        ),
    )
    parser.add_argument(
        "--kv-eviction-requests",
        type=int,
        default=60,
        help=(
            "Number of sequential requests for KV eviction test (default 60). "
            "Each uses 8 output tokens. On K1: ~6s/req → 60 reqs ≈ 6 min per mode."
        ),
    )
    args = parser.parse_args()

    # Build modes based on which model flags were requested.
    # Default (neither flag) → BF16 only.
    use_bf16 = args.bf16_model or not args.w8a8_model
    use_w8a8 = args.w8a8_model
    MODES = []
    if use_bf16:
        MODES += _BF16_MODES
    if use_w8a8:
        MODES += _W8A8_MODES

    # Kill any leftover server from a previous interrupted run (port conflict guard)
    _kill_lingering_server()

    configs = QUICK_CONFIGS if args.quick else DECODE_CONFIGS
    request_timeout = 1800

    print(f"Platform: {platform.machine()}", flush=True)
    if use_bf16:
        print(f"BF16 model: {BF16_MODEL}", flush=True)
    if use_w8a8:
        print(f"W8A8 model: {W8A8_MODEL}", flush=True)
    print(
        f"Configs:  {len(configs)} shapes, modes: {[m[0] for m in MODES]}", flush=True
    )
    if args.memory_profile:
        print(
            f"Memory profiling: enabled (long-ctx/128, {args.memory_profile_tokens} tok out)",
            flush=True,
        )
    if args.kv_eviction:
        print(
            f"KV eviction test: enabled ({args.kv_eviction_requests} sequential requests per mode)",
            flush=True,
        )

    all_results = {}
    for name, model, quant_arg, kv_dtype in MODES:
        all_results[name] = run_mode(
            name,
            model,
            quant_arg,
            configs,
            kv_dtype,
            request_timeout=request_timeout,
            memory_profile=args.memory_profile,
            memory_profile_tokens=args.memory_profile_tokens,
            kv_eviction=args.kv_eviction,
            kv_eviction_requests=args.kv_eviction_requests,
        )

    print_summary(all_results, configs)


if __name__ == "__main__":
    main()
