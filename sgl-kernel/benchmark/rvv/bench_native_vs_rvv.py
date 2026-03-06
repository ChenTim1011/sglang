# Native PyTorch backend vs RVV optimized backend
#
# Experiment purpose:
#   Quantify the speedup from RVV-optimized kernels vs native PyTorch on RISC-V CPU (e.g. SpacemiT K1).
#   Measure decode throughput, TTFT, and prefill throughput.
#
# Two model suites:
#   (1) Qwen2.5 Instruct (BF16): Native PyTorch, RVV BF16, RVV INT8-KV (--kv-cache-dtype int8).
#   (2) Qwen2.5 w8a8 (pre-quantized W8): W8A8 (BF16 KV), W8A8+INT8KV (INT8 KV).
# W8A8 here means a model that is already weight-8 (e.g. RedHatAI/Qwen2.5-1.5B-quantized.w8a8),
# not dynamic weight quantization.
#
# Metrics:
#   - Decode throughput (tok/s), TTFT (ms), Prefill throughput (tok/s).
#
# Usage:
#   python bench_native_vs_rvv.py           # full run (both suites)
#   python bench_native_vs_rvv.py --quick  # shorter run for smoke test
#   python bench_native_vs_rvv.py --instruct-only  # only Qwen2.5 Instruct suite
#   python bench_native_vs_rvv.py --w8a8-only    # only Qwen2.5 w8a8 suite

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server
except ImportError:
    print("sglang not found")
    sys.exit(1)

MODEL_INSTRUCT = "Qwen/Qwen2.5-1.5B-Instruct"
MODEL_W8A8 = "RedHatAI/Qwen2.5-1.5B-quantized.w8a8"
BASE_URL = DEFAULT_URL_FOR_TEST

# ── prompts ──────────────────────────────────────────────────────────────────

_SHORT = "The quick fox:"  # ~5 tokens
_CTX_32 = ("the quick brown fox jumps over the lazy dog " * 4).strip()  # ~32 tokens

# ── configs ───────────────────────────────────────────────────────────────────


@dataclass
class DecodeConfig:
    prompt: str
    max_tokens: int
    batch_size: int
    desc: str


@dataclass
class PrefillConfig:
    prompt: str
    desc: str


# max_tokens = output length per sequence; unified to 32 for all so throughput comparison is fair (same decode length).
DECODE_CONFIGS = [
    DecodeConfig(_SHORT, 16, 1, "decode BS=1  short-ctx/5  (16 tok)"),
    DecodeConfig(_SHORT, 16, 4, "decode BS=4  short-ctx/5  (16 tok)"),
    DecodeConfig(_SHORT, 16, 8, "decode BS=8  short-ctx/5  (16 tok)"),
    DecodeConfig(_CTX_32, 16, 1, "decode BS=1  ctx/32  (16 tok)"),
    DecodeConfig(_CTX_32, 16, 4, "decode BS=4  ctx/32  (16 tok)"),
    DecodeConfig(_CTX_32, 16, 8, "decode BS=8  ctx/32  (16 tok)"),
]

PREFILL_CONFIGS = [
    PrefillConfig(_CTX_32, "prefill ctx/32"),
]

# Quick mode: shorter runs for smoke test
QUICK_DECODE_CONFIGS = [
    DecodeConfig(_SHORT, 16, 1, "decode BS=1  short-ctx/5  (16 tok)  [quick]"),
]
QUICK_PREFILL_CONFIGS = [
    PrefillConfig(_CTX_32, "prefill ctx/32 [quick]"),
]

# ── modes ─────────────────────────────────────────────────────────────────────

# (name, model, extra_env, attention_backend, other_args)
# other_args: extra server CLI args, e.g. ["--kv-cache-dtype", "int8"]
INSTRUCT_MODES = [
    (
        "Native PyTorch",
        MODEL_INSTRUCT,
        {"SGLANG_DISABLE_RVV_KERNELS": "1"},
        "torch_native",
        [],
    ),
    ("RVV BF16", MODEL_INSTRUCT, {}, "rvv", []),
    ("RVV INT8-KV", MODEL_INSTRUCT, {}, "rvv", ["--kv-cache-dtype", "int8"]),
]
W8A8_MODES = [
    ("W8A8", MODEL_W8A8, {}, "rvv", ["--quantization", "w8a8_int8"]),
    (
        "W8A8+INT8KV",
        MODEL_W8A8,
        {},
        "rvv",
        ["--quantization", "w8a8_int8", "--kv-cache-dtype", "int8"],
    ),
]

# ── helpers ───────────────────────────────────────────────────────────────────


def _measure_decode(cfg: DecodeConfig, warmup: int = 1) -> Optional[float]:
    prompts = [cfg.prompt] * cfg.batch_size
    for _ in range(warmup):
        try:
            requests.post(
                f"{BASE_URL}/generate",
                json={
                    "text": prompts,
                    "sampling_params": {"max_new_tokens": cfg.max_tokens},
                },
                timeout=300,
            )
        except Exception:
            pass
    try:
        t0 = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": prompts,
                "sampling_params": {"max_new_tokens": cfg.max_tokens},
            },
            timeout=600,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return cfg.batch_size * cfg.max_tokens / elapsed
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None


def _measure_prefill(
    cfg: PrefillConfig, warmup: int = 1
) -> Tuple[Optional[float], Optional[float]]:
    """Returns (prefill_tps, ttft_seconds).
    TTFT = elapsed time for max_new_tokens=1 (prefill-dominated).
    """
    prompt_tokens = len(cfg.prompt.split()) * 3 // 4

    for _ in range(warmup):
        try:
            requests.post(
                f"{BASE_URL}/generate",
                json={"text": cfg.prompt, "sampling_params": {"max_new_tokens": 1}},
                timeout=600,
            )
        except Exception:
            pass
    try:
        t0 = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}/generate",
            json={"text": cfg.prompt, "sampling_params": {"max_new_tokens": 1}},
            timeout=600,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return prompt_tokens / elapsed, elapsed
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None, None


SERVER_START_TIMEOUT = 1800
POST_KILL_SLEEP = 600
# PyTorch/CPU server can be slow to release the port after kill; wait long enough.
PORT_FREE_TIMEOUT = 900


def _port_in_use(host: str, port: int) -> bool:
    """Return True if something is listening on host:port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.settimeout(1)
            s.connect((host, port))
            return True
        except (socket.error, OSError):
            return False


def _kill_process_on_port(port: int) -> bool:
    """Try to kill the process holding the given TCP port (Linux). Returns True if something was killed or attempted."""
    killed = False
    # Prefer fuser -k (sends SIGKILL to processes using port/tcp)
    try:
        r = subprocess.run(
            ["fuser", "-k", f"{port}/tcp"],
            capture_output=True,
            timeout=10,
        )
        if r.returncode == 0:
            killed = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    if not killed:
        # Fallback: parse ss -tlnp for PID(s) listening on port, then kill -9
        try:
            r = subprocess.run(
                ["ss", "-tlnp"],
                capture_output=True,
                timeout=5,
                text=True,
            )
            if r.returncode == 0 and r.stdout:
                for line in r.stdout.splitlines():
                    if f":{port}" in line or f":{port} " in line:
                        for m in re.finditer(r"pid=(\d+)", line):
                            pid = m.group(1)
                            subprocess.run(
                                ["kill", "-9", pid], capture_output=True, timeout=2
                            )
                            killed = True
                            break
                        break
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    return killed


def _wait_for_port_free(
    host: str, port: int, timeout: float = PORT_FREE_TIMEOUT
) -> bool:
    """Wait until port is free (connection refused). Returns True if free, False on timeout."""
    start = time.perf_counter()
    while time.perf_counter() - start < timeout:
        if not _port_in_use(host, port):
            return True
        time.sleep(5)
    return False


def _launch(
    model: str,
    extra_env: dict,
    attention_backend: str,
    server_extra_args: Optional[list] = None,
    retry: int = 1,
):
    """Launch server. Retries once after 60s sleep on timeout."""
    env = {**os.environ, **extra_env}
    other_args = [
        "--dtype",
        "bfloat16",
        "--device",
        "cpu",
        "--watchdog-timeout",
        "1800",
        "--attention-backend",
        attention_backend,
    ]
    if server_extra_args:
        other_args.extend(server_extra_args)
    for attempt in range(retry + 1):
        try:
            return popen_launch_server(
                model,
                BASE_URL,
                timeout=SERVER_START_TIMEOUT,
                other_args=other_args,
                env=env,
            )
        except TimeoutError:
            if attempt < retry:
                print(
                    "  [WARN] Server startup timeout, waiting 60s before retry...",
                    flush=True,
                )
                time.sleep(60)
            else:
                raise


# ── run one mode ──────────────────────────────────────────────────────────────


def run_mode(
    name: str,
    model: str,
    extra_env: dict,
    attention_backend: str,
    server_extra_args: list,
    decode_cfgs,
    prefill_cfgs,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Mode: {name}")
    print(f"  Model: {model}")
    mode_str = ", ".join(f"{k}={v}" for k, v in extra_env.items()) or "(default)"
    print(f"  Env:  {mode_str}")
    print(f"  Attention backend: {attention_backend}")
    if server_extra_args:
        print(f"  Server args: {' '.join(server_extra_args)}")
    print(f"{'='*60}")

    # Ensure port is free before launch (prevents bind failure after mode switch)
    parsed = urlparse(BASE_URL)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 21000
    if _port_in_use(host, port):
        print(f"  Port {port} is in use; attempting to free it...", flush=True)
        _kill_process_on_port(port)
        time.sleep(5)
        print(
            f"  Waiting for port {port} to be free (timeout={PORT_FREE_TIMEOUT}s)...",
            flush=True,
        )
        if not _wait_for_port_free(host, port):
            raise RuntimeError(
                f"Port {port} still in use after {PORT_FREE_TIMEOUT}s; cannot launch server. "
                f"On the host, run: fuser -k {port}/tcp  or  kill -9 $(lsof -ti :{port})"
            )
        print(f"  Port {port} is free.", flush=True)

    proc = _launch(model, extra_env, attention_backend, server_extra_args)
    results = {"decode": {}, "prefill": {}, "ttft": {}}
    try:
        print("  [decode]")
        for cfg in decode_cfgs:
            print(f"    {cfg.desc} ...", end=" ", flush=True)
            tps = _measure_decode(cfg)
            print(f"{tps:.2f} tok/s" if tps else "FAILED")
            results["decode"][cfg.desc] = tps

        print("  [prefill + TTFT]")
        for cfg in prefill_cfgs:
            print(f"    {cfg.desc} ...", end=" ", flush=True)
            tps, ttft = _measure_prefill(cfg)
            if tps is not None:
                print(f"{tps:.2f} tok/s  TTFT={ttft*1000:.0f} ms")
            else:
                print("FAILED")
            results["prefill"][cfg.desc] = tps
            results["ttft"][cfg.desc] = ttft * 1000 if ttft is not None else None  # ms
    finally:
        kill_process_tree(proc.pid)
        time.sleep(POST_KILL_SLEEP)

    return results


# ── summary ───────────────────────────────────────────────────────────────────


def print_summary(
    all_results: dict,
    decode_cfgs,
    prefill_cfgs,
    mode_names,
    baseline_name: Optional[str] = "Native PyTorch",
):
    """Print decode/prefill/TTFT table. Speedup is vs baseline_name if present."""
    col_w = 18

    # --- Decode throughput ---
    print(f"\n{'='*70}")
    if baseline_name and baseline_name in mode_names:
        print(f"  DECODE THROUGHPUT  (tok/s, speedup vs {baseline_name})")
    else:
        print("  DECODE THROUGHPUT  (tok/s)")
    print(f"{'='*70}")
    header = f"{'Config':<42}" + "".join(f"{n:<{col_w}}" for n in mode_names)
    print(header)
    print("-" * (42 + col_w * len(mode_names)))

    baseline_decode = (
        all_results.get(baseline_name, {}).get("decode", {}) if baseline_name else {}
    )
    for cfg in decode_cfgs:
        row = f"  {cfg.desc:<40}"
        base_tps = baseline_decode.get(cfg.desc)
        for name in mode_names:
            tps = all_results.get(name, {}).get("decode", {}).get(cfg.desc)
            if tps is None:
                row += f"{'N/A':<{col_w}}"
            elif name == baseline_name or base_tps is None:
                row += f"{tps:.2f} tok/s{'':<{col_w - 12}}"
            else:
                row += f"{tps:.2f}({tps/base_tps:.1f}x){'':<{col_w - 14}}"
        print(row)

    # --- Prefill throughput + TTFT ---
    print(f"\n{'='*70}")
    if baseline_name and baseline_name in mode_names:
        print(f"  PREFILL THROUGHPUT + TTFT  (speedup vs {baseline_name})")
    else:
        print("  PREFILL THROUGHPUT + TTFT")
    print(f"{'='*70}")
    print(f"{'Config':<42}" + "".join(f"{n:<{col_w}}" for n in mode_names))
    print("-" * (42 + col_w * len(mode_names)))

    baseline_prefill = (
        all_results.get(baseline_name, {}).get("prefill", {}) if baseline_name else {}
    )
    baseline_ttft = (
        all_results.get(baseline_name, {}).get("ttft", {}) if baseline_name else {}
    )
    for cfg in prefill_cfgs:
        row = f"  {cfg.desc:<40}"
        base_tps = baseline_prefill.get(cfg.desc)
        for name in mode_names:
            tps = all_results.get(name, {}).get("prefill", {}).get(cfg.desc)
            if tps is None:
                row += f"{'N/A':<{col_w}}"
            elif name == baseline_name or base_tps is None:
                row += f"{tps:.2f} tok/s{'':<{col_w - 12}}"
            else:
                row += f"{tps:.2f}({tps/base_tps:.1f}x){'':<{col_w - 14}}"
        print(row)

        ttft_row = f"  {'  TTFT':<40}"
        base_t = baseline_ttft.get(cfg.desc)
        for name in mode_names:
            t = all_results.get(name, {}).get("ttft", {}).get(cfg.desc)
            if t is None:
                ttft_row += f"{'N/A':<{col_w}}"
            elif name == baseline_name or base_t is None:
                ttft_row += f"{t:.0f} ms{'':<{col_w - 7}}"
            else:
                ttft_row += (
                    f"{t:.0f} ms ({base_t/t:.1f}x faster){'':<{max(0, col_w - 20)}}"
                )
        print(ttft_row)

    if baseline_name and baseline_name in mode_names:
        print()
        print(f"Speedup is relative to {baseline_name}.")


# ── main ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--quick", action="store_true", help="Shorter run for smoke test"
    )
    parser.add_argument(
        "--instruct-only",
        action="store_true",
        help="Run only Qwen2.5 Instruct suite (Native, RVV BF16, RVV INT8-KV)",
    )
    parser.add_argument(
        "--w8a8-only",
        action="store_true",
        help="Run only Qwen2.5 w8a8 suite (W8A8, W8A8+INT8KV)",
    )
    args = parser.parse_args()

    if args.quick:
        decode_cfgs, prefill_cfgs = QUICK_DECODE_CONFIGS, QUICK_PREFILL_CONFIGS
    else:
        decode_cfgs, prefill_cfgs = DECODE_CONFIGS, PREFILL_CONFIGS

    run_instruct = args.instruct_only or not args.w8a8_only
    run_w8a8 = args.w8a8_only or not args.instruct_only

    print(f"Decode configs: {len(decode_cfgs)}, Prefill configs: {len(prefill_cfgs)}")

    if run_instruct:
        print(f"\n>>> Suite: Qwen2.5 Instruct ({MODEL_INSTRUCT})")
        print(f"Modes: {[n for n, *_ in INSTRUCT_MODES]}")
        all_instruct = {}
        for name, model, extra_env, attn_backend, server_args in INSTRUCT_MODES:
            all_instruct[name] = run_mode(
                name,
                model,
                extra_env,
                attn_backend,
                server_args,
                decode_cfgs,
                prefill_cfgs,
            )
        print_summary(
            all_instruct,
            decode_cfgs,
            prefill_cfgs,
            mode_names=[n for n, *_ in INSTRUCT_MODES],
            baseline_name="Native PyTorch",
        )

    if run_w8a8:
        print(f"\n>>> Suite: Qwen2.5 w8a8 ({MODEL_W8A8})")
        print(f"Modes: {[n for n, *_ in W8A8_MODES]}")
        all_w8a8 = {}
        for name, model, extra_env, attn_backend, server_args in W8A8_MODES:
            all_w8a8[name] = run_mode(
                name,
                model,
                extra_env,
                attn_backend,
                server_args,
                decode_cfgs,
                prefill_cfgs,
            )
        print_summary(
            all_w8a8,
            decode_cfgs,
            prefill_cfgs,
            mode_names=[n for n, *_ in W8A8_MODES],
            baseline_name="W8A8",
        )


if __name__ == "__main__":
    main()
