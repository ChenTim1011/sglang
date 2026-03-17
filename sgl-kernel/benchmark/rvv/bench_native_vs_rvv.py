"""Compare native PyTorch and RVV backends on decode/prefill/TTFT."""

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional, Tuple
from urllib.parse import urlparse

import requests
from _rvv_bench_utils import (
    BASE_URL,
)
from _rvv_bench_utils import BF16_MODEL as MODEL_INSTRUCT
from _rvv_bench_utils import (
    _port_in_use,
    _wait_for_port_free,
)

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import popen_launch_server
except ImportError:
    print("sglang not found")
    sys.exit(1)

# Prompts

_SHORT = "The quick fox:"  # ~5 tokens
_CTX_32 = ("the quick brown fox jumps over the lazy dog " * 4).strip()  # ~32 tokens

# Deterministic greedy sampling improves run-to-run comparability.
BENCH_SAMPLING_PARAMS = {
    "temperature": 0.0,
    "top_k": 1,
    "top_p": 1.0,
    "repetition_penalty": 1.0,
    "ignore_eos": True,
}

# Configs


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


# Keep output length fixed to make throughput comparisons fair.
DECODE_CONFIGS = [
    DecodeConfig(_CTX_32, 16, 1, "decode BS=1  ctx/32  (16 tok)"),
    DecodeConfig(_CTX_32, 16, 4, "decode BS=4  ctx/32  (16 tok)"),
    DecodeConfig(_CTX_32, 16, 8, "decode BS=8  ctx/32  (16 tok)"),
]

PREFILL_CONFIGS = [
    PrefillConfig(_CTX_32, "prefill ctx/32"),
]

# Short smoke-test configs.
QUICK_DECODE_CONFIGS = [
    DecodeConfig(_SHORT, 16, 1, "decode BS=1  short-ctx/5  (16 tok)  [quick]"),
]
QUICK_PREFILL_CONFIGS = [
    PrefillConfig(_CTX_32, "prefill ctx/32 [quick]"),
]

# Modes: (name, model, extra_env, attention_backend, other_args)
INSTRUCT_MODES = [
    (
        "Native PyTorch",
        MODEL_INSTRUCT,
        {"SGLANG_DISABLE_RVV_KERNELS": "1"},
        "torch_native",
        [],
    ),
    ("RVV BF16", MODEL_INSTRUCT, {}, "rvv", []),
    ("RVV W8A8", MODEL_INSTRUCT, {}, "rvv", ["--quantization", "w8a8_int8"]),
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
                    "sampling_params": {
                        **BENCH_SAMPLING_PARAMS,
                        "max_new_tokens": cfg.max_tokens,
                    },
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
                "sampling_params": {
                    **BENCH_SAMPLING_PARAMS,
                    "max_new_tokens": cfg.max_tokens,
                },
            },
            timeout=1200,
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
                json={
                    "text": cfg.prompt,
                    "sampling_params": {
                        **BENCH_SAMPLING_PARAMS,
                        "max_new_tokens": 1,
                    },
                },
                timeout=1200,
            )
        except Exception:
            pass
    try:
        t0 = time.perf_counter()
        r = requests.post(
            f"{BASE_URL}/generate",
            json={
                "text": cfg.prompt,
                "sampling_params": {
                    **BENCH_SAMPLING_PARAMS,
                    "max_new_tokens": 1,
                },
            },
            timeout=1200,
        )
        elapsed = time.perf_counter() - t0
        r.raise_for_status()
        return prompt_tokens / elapsed, elapsed
    except Exception as e:
        print(f"    [ERROR] {e}")
        return None, None


SERVER_START_TIMEOUT = 3600
POST_KILL_SLEEP = 600
PORT_FREE_TIMEOUT = 900


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
        if not _wait_for_port_free(host, port, timeout=PORT_FREE_TIMEOUT):
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
        help="Run only Qwen2.5 Instruct suite (Native, RVV BF16, RVV W8A8)",
    )
    args = parser.parse_args()

    if args.quick:
        decode_cfgs, prefill_cfgs = QUICK_DECODE_CONFIGS, QUICK_PREFILL_CONFIGS
    else:
        decode_cfgs, prefill_cfgs = DECODE_CONFIGS, PREFILL_CONFIGS

    run_instruct = True

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


if __name__ == "__main__":
    main()
