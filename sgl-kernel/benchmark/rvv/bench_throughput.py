# Benchmark Pure PyTorch vs Full RVV end-to-end throughput (Qwen2.5-1.5B)
#
# 2-way comparison:
#   1. Pure PyTorch (torch_native + SGLANG_DISABLE_RVV_KERNELS=1)
#   2. Full RVV (rvv attention + all RVV kernels)
#
# Servers are launched sequentially to avoid OOM on memory-constrained

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


MODEL = "Qwen/Qwen2.5-1.5B"
BASE_URL = DEFAULT_URL_FOR_TEST


@dataclass
class BenchmarkConfig:
    prompt: str
    max_tokens: int
    batch_size: int
    description: str
    input_len: int = 0  # if >0, pad prompt to this many tokens (prefill-heavy)


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    pytorch_ms: float  # TPOT in ms — pure PyTorch
    rvv_ms: float  # TPOT in ms — full RVV
    speedup: float  # pytorch_ms / rvv_ms
    throughput_rvv: float  # tokens/s for full RVV


# Decode-heavy: matches original bench_throughput configs
DECODE_CONFIGS = [
    BenchmarkConfig("The quick fox: ", 128, 1, "Single request (128 tok)"),
    BenchmarkConfig("The quick fox: ", 64, 2, "Batch x2 (64 tok)"),
    BenchmarkConfig("The quick fox: ", 50, 1, "Latency (50 tok)"),
    BenchmarkConfig("The quick fox: ", 32, 4, "Batch x4 (32 tok)"),
    BenchmarkConfig("The quick fox: ", 32, 8, "Batch x8 (32 tok)"),
]

# Prefill-heavy: long prompt, short output
_LONG_PROMPT = (
    "The following is a detailed technical document about RISC-V vector extensions, "
)
PREFILL_CONFIGS = [
    BenchmarkConfig(_LONG_PROMPT, 4, 1, "Prefill BS=1 (128-tok prompt)", input_len=128),
    BenchmarkConfig(_LONG_PROMPT, 4, 1, "Prefill BS=1 (256-tok prompt)", input_len=256),
]

CI_CONFIGS = [
    BenchmarkConfig("The quick fox: ", 8, 1, "Decode BS=1 (8 tok out)"),
    BenchmarkConfig(_LONG_PROMPT, 4, 1, "Prefill BS=1 (256-tok prompt)", input_len=256),
]

ALL_CONFIGS = CI_CONFIGS if IS_CI else (DECODE_CONFIGS + PREFILL_CONFIGS)


def _pad_prompt_to_len(prompt: str, target_tokens: int) -> str:
    """Return a prompt padded to approximately target_tokens tokens."""
    filler = (
        "the quick brown fox jumps over the lazy dog "
        "in a warm summer afternoon near the river "
    )
    filler_words = filler.split()
    base_words = prompt.split()
    target_words = int(target_tokens / 1.3)
    extra_needed = max(0, target_words - len(base_words))
    repetitions = (extra_needed + len(filler_words) - 1) // len(filler_words)
    return prompt.rstrip() + " " + (filler * repetitions).rstrip()


def _measure(config):
    """Send one request, return (tpot_ms, throughput_tok_s)."""
    prompt = config.prompt
    if config.input_len > 0:
        prompt = _pad_prompt_to_len(prompt, config.input_len)

    prompts = [prompt] * config.batch_size
    start = time.perf_counter()
    response = requests.post(
        BASE_URL + "/generate",
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


def _run_configs(backend, configs, dtype, env_override=None):
    """Launch one server, run all configs, kill server."""
    old_env = {}
    if env_override:
        for key, val in env_override.items():
            old_env[key] = os.environ.get(key)
            os.environ[key] = val

    try:
        process = popen_launch_server(
            MODEL,
            BASE_URL,
            timeout=2400,
            other_args=[
                "--attention-backend",
                backend,
                "--trust-remote-code",
                "--mem-fraction-static",
                "0.6",
                "--max-total-tokens",
                "1024",
                "--device",
                "cpu",
                "--dtype",
                dtype,
                "--skip-server-warmup",
                "--watchdog-timeout",
                "1800",
            ],
        )
        results = []
        try:
            for config in configs:
                results.append(_measure(config))
        finally:
            kill_process_tree(process.pid)
    finally:
        if env_override:
            for key, val in old_env.items():
                if val is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = val

    return results


def run_benchmark(configs, dtype="bfloat16"):
    """Run 2-way benchmark: Pure PyTorch vs Full RVV."""

    print(
        f"\n[1/2] Pure PyTorch baseline (SGLANG_DISABLE_RVV_KERNELS=1, dtype={dtype})..."
    )
    pytorch_results = _run_configs(
        "torch_native",
        configs,
        dtype,
        env_override={"SGLANG_DISABLE_RVV_KERNELS": "1"},
    )

    print(f"\n[2/2] Full RVV backend (dtype={dtype})...")
    rvv_results = _run_configs("rvv", configs, dtype)

    results = []
    for i, config in enumerate(configs):
        pytorch_ms, _ = pytorch_results[i]
        rvv_ms, throughput = rvv_results[i]
        speedup = pytorch_ms / rvv_ms if rvv_ms > 0 else float("inf")
        results.append(
            BenchmarkResult(
                config=config,
                pytorch_ms=pytorch_ms,
                rvv_ms=rvv_ms,
                speedup=speedup,
                throughput_rvv=throughput,
            )
        )
    return results


def print_result(result):
    c = result.config
    regime = f"[PREFILL in_len={c.input_len}]" if c.input_len > 0 else "[DECODE]"
    params = f"BS={c.batch_size}, max_tok={c.max_tokens}"
    w = 42

    print(f"\n  {c.description:<{w}} {regime} {params}")
    print(f"    Pure PyTorch:  {result.pytorch_ms:>10.3f} ms/tok")
    print(
        f"    Full RVV:      {result.rvv_ms:>10.3f} ms/tok "
        f"({result.throughput_rvv:.1f} tok/s)"
    )
    print(f"    Speedup:       {result.speedup:>10.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark Pure PyTorch vs Full RVV end-to-end throughput"
    )
    parser.add_argument("--quick", action="store_true", help="Run CI configs only")
    parser.add_argument(
        "--dtype",
        default=os.environ.get("SGLANG_DTYPE", "bfloat16"),
        help="Model dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    configs = CI_CONFIGS if args.quick else ALL_CONFIGS

    print("=" * 70)
    print("RVV End-to-End Throughput Benchmark (Pure PyTorch vs Full RVV)")
    print("=" * 70)
    print(f"Platform : {platform.machine()}")
    print(f"Model    : {MODEL}")
    print(f"dtype    : {args.dtype}")
    print(f"CI Mode  : {IS_CI}")
    print(f"Metric   : TPOT (ms/tok) — lower is better")
    print()
    print("Backends:")
    print("  1. Pure PyTorch — torch_native + SGLANG_DISABLE_RVV_KERNELS=1")
    print("  2. Full RVV     — rvv attention + all RVV kernels")
    print("-" * 70)

    results = []
    try:
        results = run_benchmark(configs, dtype=args.dtype)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    print()
    for result in results:
        print_result(result)

    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print()
        print("=" * 70)
        print(f"  Avg Total Speedup: {avg_speedup:.2f}x  (Full RVV vs Pure PyTorch)")
        print(f"  Total Benchmarks:  {len(results)}")
        print("=" * 70)


if __name__ == "__main__":
    main()
