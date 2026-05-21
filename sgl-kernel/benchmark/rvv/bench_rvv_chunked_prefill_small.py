#!/usr/bin/env python3
"""Small mixed long/short request benchmark for RVV chunked prefill."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse

import requests


def wait_ready(base_url: str, timeout_s: int) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        for endpoint in ("/get_model_info", "/model_info"):
            try:
                if requests.get(base_url + endpoint, timeout=5).ok:
                    return
            except requests.RequestException:
                pass
        time.sleep(5)
    raise TimeoutError(f"server did not become ready within {timeout_s}s")


def launch_server(
    args: argparse.Namespace, mode: str, log_path: Path
) -> subprocess.Popen:
    parsed = urlparse(args.base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 30000
    cmd = [
        sys.executable,
        "-m",
        "sglang.launch_server",
        "--model-path",
        args.model,
        "--host",
        host,
        "--port",
        str(port),
        "--dtype",
        "bfloat16",
        "--device",
        "cpu",
        "--attention-backend",
        "rvv",
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--max-running-requests",
        str(args.max_running_requests),
        "--max-total-tokens",
        str(args.max_total_tokens),
        "--watchdog-timeout",
        "2400",
    ]
    if args.quantization:
        cmd += ["--quantization", args.quantization]
    if mode == "chunked":
        cmd += ["--chunked-prefill-size", str(args.chunked_prefill_size)]
    else:
        cmd += ["--chunked-prefill-size", "-1"]
    print("launch:", " ".join(cmd), flush=True)
    log_f = log_path.open("w", encoding="utf-8")
    return subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, text=True)


def stop_server(proc: subprocess.Popen | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=30)


def flush_cache(base_url: str) -> None:
    try:
        requests.post(base_url + "/flush_cache", timeout=30)
    except requests.RequestException:
        pass


def make_prompt(tokens: int, label: str) -> str:
    words = (f"{label} prefill scheduling benchmark " * tokens).split()
    return " ".join(words[:tokens])


def generate(
    base_url: str, name: str, prompt_tokens: int, output_len: int
) -> dict[str, object]:
    prompt = make_prompt(prompt_tokens, name)
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "top_k": 1,
            "top_p": 1.0,
            "ignore_eos": True,
            "max_new_tokens": output_len,
        },
    }
    start = time.perf_counter()
    r = requests.post(base_url + "/generate", json=payload, timeout=1800)
    latency = time.perf_counter() - start
    r.raise_for_status()
    return {
        "request": name,
        "prompt_tokens_approx": prompt_tokens,
        "output_len": output_len,
        "latency_s": latency,
    }


def run_mode(
    args: argparse.Namespace, mode: str, out_dir: Path
) -> list[dict[str, object]]:
    proc = launch_server(args, mode, out_dir / f"{mode}.server.log")
    rows: list[dict[str, object]] = []
    try:
        wait_ready(args.base_url, args.server_ready_timeout)
        flush_cache(args.base_url)
        jobs: list[tuple[str, int, int]] = [
            ("long", args.long_prompt_tokens, args.long_output_len)
        ]
        for i in range(args.num_short):
            jobs.append((f"short-{i}", args.short_prompt_tokens, args.short_output_len))
        start = time.perf_counter()
        with ThreadPoolExecutor(max_workers=args.max_workers) as pool:
            futures = [
                pool.submit(generate, args.base_url, name, prompt_tokens, output_len)
                for name, prompt_tokens, output_len in jobs
            ]
            for fut in as_completed(futures):
                row = fut.result()
                row["mode"] = mode
                row["elapsed_from_batch_start_s"] = time.perf_counter() - start
                rows.append(row)
                print(f"{mode}\t{row['request']}\t{row['latency_s']:.3f}s", flush=True)
    finally:
        stop_server(proc)
    return sorted(rows, key=lambda r: str(r["request"]))


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    jsonl = out_dir / "chunked_prefill_results.jsonl"
    tsv = out_dir / "summary.tsv"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with tsv.open("w", encoding="utf-8") as f:
        f.write(
            "mode\trequest\tprompt_tokens_approx\toutput_len\tlatency_s\telapsed_from_batch_start_s\n"
        )
        for row in rows:
            f.write(
                f"{row['mode']}\t{row['request']}\t{row['prompt_tokens_approx']}\t"
                f"{row['output_len']}\t{row['latency_s']:.6f}\t"
                f"{row['elapsed_from_batch_start_s']:.6f}\n"
            )
    print(f"summary: {tsv}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model", default=os.getenv("BF16_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
    )
    parser.add_argument("--quantization", default="")
    parser.add_argument(
        "--base-url", default=os.getenv("BASE_URL", "http://127.0.0.1:30000")
    )
    parser.add_argument(
        "--out-dir", type=Path, default=Path("/tmp/rvv_chunked_prefill_small")
    )
    parser.add_argument(
        "--mode", choices=["both", "chunked", "unchunked"], default="both"
    )
    parser.add_argument("--chunked-prefill-size", type=int, default=256)
    parser.add_argument("--long-prompt-tokens", type=int, default=768)
    parser.add_argument("--long-output-len", type=int, default=8)
    parser.add_argument("--short-prompt-tokens", type=int, default=64)
    parser.add_argument("--short-output-len", type=int, default=16)
    parser.add_argument("--num-short", type=int, default=3)
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=float(os.getenv("SERVER_MEM_FRACTION_STATIC", "0.28")),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=int(os.getenv("SERVER_MAX_RUNNING_REQUESTS", "4")),
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=int(os.getenv("SERVER_MAX_TOTAL_TOKENS", "3072")),
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=int(os.getenv("SERVER_READY_TIMEOUT", "1200")),
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["unchunked", "chunked"] if args.mode == "both" else [args.mode]
    rows: list[dict[str, object]] = []
    for mode in modes:
        rows.extend(run_mode(args, mode, args.out_dir))
    write_outputs(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
