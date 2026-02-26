# Benchmark RVV vs torch_native end-to-end throughput (Llama-3.2-1B-Instruct)
#
# Servers are launched sequentially (torch_native first, then rvv) to avoid OOM
# on memory-constrained hardware (Banana Pi BPI-F3, ~8 GB RAM).

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass

import requests
from utils import IS_CI, print_benchmark_result

os.environ.setdefault("SGLANG_USE_CPU_ENGINE", "1")

try:
    from sglang.srt.utils import kill_process_tree
    from sglang.test.test_utils import DEFAULT_URL_FOR_TEST, popen_launch_server

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("sglang not found")
    sys.exit(1)


MODEL = "meta-llama/Llama-3.2-1B-Instruct"
BASE_URL = DEFAULT_URL_FOR_TEST


@dataclass
class BenchmarkConfig:
    prompt: str
    max_tokens: int
    batch_size: int
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float  # TPOT in ms (time per output token)
    torch_ms: float  # TPOT in ms for torch_native
    speedup: float  # torch_ms / rvv_ms
    throughput_rvv: float  # tokens/s


STANDARD_CONFIGS = [
    BenchmarkConfig(
        "Write a short story about a robot learning to paint: ",
        128,
        1,
        "Single request (128 tok)",
    ),
    BenchmarkConfig(
        "Explain quantum computing in simple terms: ",
        64,
        2,
        "Batch x2 (64 tok)",
    ),
    BenchmarkConfig(
        "The quick brown fox jumps over",
        50,
        1,
        "Latency (50 tok)",
    ),
]

CI_CONFIGS = [
    BenchmarkConfig(
        "Write a short story: ",
        32,
        1,
        "CI Quick (32 tok)",
    ),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


def _measure(base_url, config):
    """Send a single request and return (tpot_ms, throughput_tok_s)."""
    prompts = [config.prompt] * config.batch_size
    start = time.perf_counter()
    response = requests.post(
        base_url + "/generate",
        json={
            "text": prompts,
            "sampling_params": {"temperature": 0, "max_new_tokens": config.max_tokens},
        },
    ).json()
    elapsed = time.perf_counter() - start

    if isinstance(response, list):
        total_tokens = sum(
            (
                len(r.get("meta_info", {}).get("completion_tokens", []))
                if isinstance(r.get("meta_info", {}).get("completion_tokens"), list)
                else r.get("meta_info", {}).get("completion_tokens", 0)
            )
            for r in response
        )
    else:
        ct = response.get("meta_info", {}).get("completion_tokens", 0)
        total_tokens = len(ct) if isinstance(ct, list) else ct

    tpot_ms = (elapsed / total_tokens * 1000) if total_tokens > 0 else 0
    throughput = total_tokens / elapsed if elapsed > 0 else 0
    return tpot_ms, throughput


def _run_all_configs(backend, configs, dtype):
    """Launch one server, run all configs, kill server. Returns list of (tpot_ms, throughput)."""
    process = popen_launch_server(
        MODEL,
        BASE_URL,
        timeout=1800,
        other_args=[
            "--attention-backend",
            backend,
            "--trust-remote-code",
            "--mem-fraction-static",
            "0.6",
            "--max-total-tokens",
            "512",
            "--device",
            "cpu",
            "--dtype",
            dtype,
        ],
    )
    results = []
    try:
        for config in configs:
            results.append(_measure(BASE_URL, config))
    finally:
        kill_process_tree(process.pid)
    return results


def run_benchmark(configs, dtype="bfloat16"):
    """Run torch_native then rvv sequentially; return list of BenchmarkResult."""
    print(f"\n[1/2] torch_native backend (dtype={dtype})...")
    native_results = _run_all_configs("torch_native", configs, dtype)

    print(f"\n[2/2] rvv backend (dtype={dtype})...")
    rvv_results = _run_all_configs("rvv", configs, dtype)

    results = []
    for i, config in enumerate(configs):
        torch_ms, _ = native_results[i]
        rvv_ms, throughput = rvv_results[i]
        speedup = torch_ms / rvv_ms if rvv_ms > 0 else float("inf")
        results.append(
            BenchmarkResult(
                config=config,
                rvv_ms=rvv_ms,
                torch_ms=torch_ms,
                speedup=speedup,
                throughput_rvv=throughput,
            )
        )
    return results


def print_result(result):
    c = result.config
    params = f"BS={c.batch_size}, max_tok={c.max_tokens}"
    print_benchmark_result(
        c.description,
        params,
        result.rvv_ms,
        result.torch_ms,
        result.speedup,
        throughput_rvv=result.throughput_rvv,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV vs Torch-native end-to-end throughput"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    parser.add_argument(
        "--dtype",
        default=os.environ.get("SGLANG_DTYPE", "bfloat16"),
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    configs = CI_CONFIGS if args.quick else ALL_CONFIGS

    print("=" * 60)
    print("RVV End-to-End Throughput Benchmark")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"Model   : {MODEL}")
    print(f"dtype   : {args.dtype}")
    print(f"CI Mode : {IS_CI}")
    print(f"Metric  : TPOT (ms/tok) — lower is better; speedup = native/rvv")
    print("-" * 60)

    results = []
    try:
        results = run_benchmark(configs, dtype=args.dtype)
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

    print()
    for result in results:
        print_result(result)

    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print("=" * 60)
        print(f"Average Speedup: {avg_speedup:.2f}x")
        print(f"Total Benchmarks: {len(results)}")
        print("=" * 60)


if __name__ == "__main__":
    main()
