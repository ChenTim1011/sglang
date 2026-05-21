#!/usr/bin/env python3
"""Prefix/radix cache benchmark for a single BF3 RVV SGLang server."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def build_prompt(prefix_tokens: int, question_tokens: int, idx: int) -> str:
    prefix = (
        "shared system instruction for radix prefix cache " * prefix_tokens
    ).split()
    question = (
        "unique question asks for a short deterministic answer " * question_tokens
    ).split()
    return " ".join(
        prefix[:prefix_tokens] + [f"case-{idx}"] + question[:question_tokens]
    )


def wait_ready(base_url: str, timeout_s: int) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        for endpoint in ("/get_model_info", "/model_info"):
            try:
                r = requests.get(base_url + endpoint, timeout=5)
                if r.ok:
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
    if mode == "radix_off":
        cmd += ["--disable-radix-cache"]
    if args.server_arg:
        cmd += args.server_arg
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


def timed_generate(base_url: str, prompt: str, output_len: int) -> tuple[float, int]:
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
    data = r.json()
    text = data[0].get("text", "") if isinstance(data, list) else data.get("text", "")
    return latency, len(str(text).split())


def run_mode(
    args: argparse.Namespace, mode: str, out_dir: Path
) -> list[dict[str, object]]:
    server_log = out_dir / f"{mode}.server.log"
    proc = launch_server(args, mode, server_log)
    rows: list[dict[str, object]] = []
    try:
        wait_ready(args.base_url, args.server_ready_timeout)
        flush_cache(args.base_url)
        for i in range(args.num_requests):
            if mode == "flush_each":
                flush_cache(args.base_url)
            prompt = build_prompt(args.prefix_tokens, args.question_tokens, i)
            latency_s, output_words = timed_generate(
                args.base_url, prompt, args.output_len
            )
            rows.append(
                {
                    "mode": mode,
                    "request": i,
                    "latency_s": latency_s,
                    "prefix_tokens_approx": args.prefix_tokens,
                    "question_tokens_approx": args.question_tokens,
                    "output_words": output_words,
                    "server_log": str(server_log),
                }
            )
            print(f"{mode}\t{i}\t{latency_s:.3f}s", flush=True)
    finally:
        stop_server(proc)
    return rows


def write_outputs(rows: list[dict[str, object]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    jsonl = out_dir / "prefix_cache_results.jsonl"
    tsv = out_dir / "summary.tsv"
    with jsonl.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")
    with tsv.open("w", encoding="utf-8") as f:
        f.write(
            "mode\trequest\tlatency_s\tprefix_tokens_approx\tquestion_tokens_approx\toutput_words\tserver_log\n"
        )
        for row in rows:
            f.write(
                f"{row['mode']}\t{row['request']}\t{row['latency_s']:.6f}\t"
                f"{row['prefix_tokens_approx']}\t{row['question_tokens_approx']}\t"
                f"{row['output_words']}\t{row['server_log']}\n"
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
    parser.add_argument("--out-dir", type=Path, default=Path("/tmp/rvv_prefix_cache"))
    parser.add_argument("--num-requests", type=int, default=6)
    parser.add_argument("--prefix-tokens", type=int, default=384)
    parser.add_argument("--question-tokens", type=int, default=32)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument(
        "--mode",
        choices=["both", "radix_on", "radix_off", "flush_each"],
        default="both",
    )
    parser.add_argument(
        "--mem-fraction-static",
        type=float,
        default=float(os.getenv("SERVER_MEM_FRACTION_STATIC", "0.28")),
    )
    parser.add_argument(
        "--max-running-requests",
        type=int,
        default=int(os.getenv("SERVER_MAX_RUNNING_REQUESTS", "2")),
    )
    parser.add_argument(
        "--max-total-tokens",
        type=int,
        default=int(os.getenv("SERVER_MAX_TOTAL_TOKENS", "2048")),
    )
    parser.add_argument(
        "--server-ready-timeout",
        type=int,
        default=int(os.getenv("SERVER_READY_TIMEOUT", "1200")),
    )
    parser.add_argument("--server-arg", action="append", default=[])
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    modes = ["radix_on", "radix_off"] if args.mode == "both" else [args.mode]
    rows: list[dict[str, object]] = []
    for mode in modes:
        rows.extend(run_mode(args, mode, args.out_dir))
    write_outputs(rows, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
