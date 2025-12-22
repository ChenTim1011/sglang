"""
Benchmark script comparing RVV and torch_native attention backends (Decode, FP16/BF16).

This script benchmarks through SGLang's attention backend system,
comparing `attention-backend=rvv` vs `attention-backend=torch_native`.

Usage:
    python3 bench_rvv_decode.py [--quick]
"""

import argparse
import os
import platform
import sys
import time
from dataclasses import dataclass
from unittest.mock import Mock

import torch

# ============================================================================
# Configuration & Environment Setup
# ============================================================================

IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))


def setup_triton_stub():
    """Setup triton_stub for RISC-V environments."""
    try:
        import triton

        return
    except ImportError:
        pass

    possible_stub_paths = [
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "banana_pi",
            "test_tinyllama_rvv",
            "triton_stub.py",
        ),
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "..",
            "..",
            "banana_pi",
            "test_tinyllama_rvv",
            "triton_stub.py",
        ),
    ]

    triton_stub_path = None
    for path in possible_stub_paths:
        if os.path.exists(path):
            triton_stub_path = path
            break

    if triton_stub_path:
        stub_namespace = {
            "__file__": triton_stub_path,
            "__name__": "triton_stub",
            "__package__": "",
        }
        stub_namespace.update(sys.modules)
        with open(triton_stub_path, "r") as f:
            exec(compile(f.read(), triton_stub_path, "exec"), stub_namespace)


setup_triton_stub()

try:
    from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    HAS_SGLANG = True
except ImportError:
    HAS_SGLANG = False
    print("SGLang not found")
    sys.exit(1)


# ============================================================================
# Benchmark Data Classes
# ============================================================================


@dataclass
class BenchmarkConfig:
    num_requests: int
    num_heads: int
    head_dim: int
    seq_len: int
    description: str
    kv_dtype: torch.dtype = torch.float16


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float
    throughput_rvv: float


# ============================================================================
# Benchmark Configurations
# ============================================================================

STANDARD_CONFIGS = [
    BenchmarkConfig(1, 8, 64, 128, "Small Batch (BS=1)"),
    BenchmarkConfig(4, 8, 64, 128, "Medium Batch (BS=4)"),
    BenchmarkConfig(32, 8, 64, 128, "Large Batch (BS=32)"),
    BenchmarkConfig(1, 32, 64, 128, "TinyLlama Decode (BS=1)"),
    BenchmarkConfig(8, 32, 64, 128, "TinyLlama Decode (BS=8)"),
    BenchmarkConfig(1, 32, 64, 2048, "TinyLlama Decode (BS=1, Seq=2048)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 4, 32, 32, "CI Quick Test"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS


# ============================================================================
# Mocking Utilities
# ============================================================================


def create_mock_runner(num_heads, head_dim, v_head_dim, dtype=torch.float16):
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.tp_size = 1
    mock_runner.kv_cache_dtype = "auto"

    mock_runner.token_to_kv_pool = Mock()
    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(10000, num_heads, head_dim, dtype=dtype)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(10000, num_heads, v_head_dim, dtype=dtype)
    )

    return mock_runner


def create_mock_layer(num_heads, head_dim, v_head_dim):
    mock_layer = Mock()
    mock_layer.tp_q_head_num = num_heads
    mock_layer.qk_head_dim = head_dim
    mock_layer.v_head_dim = v_head_dim
    mock_layer.layer_id = 0
    mock_layer.scaling = 1.0 / (head_dim**0.5)
    mock_layer.logit_cap = 50.0
    return mock_layer


def create_mock_forward_batch(
    num_requests, num_heads, head_dim, v_head_dim, max_seq_len, dtype=torch.float16
):
    mock_batch = Mock()
    mock_batch.batch_size = num_requests
    mock_batch.out_cache_loc = torch.zeros(num_requests, dtype=torch.int64)
    mock_batch.seq_lens = torch.ones(num_requests, dtype=torch.int64) * max_seq_len
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    mock_batch.req_to_token_pool = Mock()
    mock_batch.req_to_token_pool.req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.int64
    )

    mock_batch.token_to_kv_pool = Mock()
    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(
            max_seq_len * num_requests + 100, num_heads, head_dim, dtype=dtype
        )
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(
            max_seq_len * num_requests + 100, num_heads, v_head_dim, dtype=dtype
        )
    )
    return mock_batch


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_single_backend(backend_name, config, num_iterations=100, warmup=10):
    v_head_dim = config.head_dim

    mock_runner = create_mock_runner(
        config.num_heads, config.head_dim, v_head_dim, config.kv_dtype
    )
    mock_layer = create_mock_layer(config.num_heads, config.head_dim, v_head_dim)

    if backend_name == "rvv":
        backend = RVVAttnBackend(mock_runner)
    elif backend_name == "torch_native":
        backend = TorchNativeAttnBackend(mock_runner)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    dtype = torch.float16
    q = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    k = torch.randn(config.num_requests, config.num_heads, config.head_dim, dtype=dtype)
    v = torch.randn(config.num_requests, config.num_heads, v_head_dim, dtype=dtype)

    forward_batch = create_mock_forward_batch(
        config.num_requests,
        config.num_heads,
        config.head_dim,
        v_head_dim,
        config.seq_len,
        config.kv_dtype,
    )

    backend.init_forward_metadata(forward_batch)

    for _ in range(warmup):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 10 if quick else 100
    warmup = 2 if quick else 10

    rvv_time = run_single_backend("rvv", config, iterations, warmup)
    torch_time = run_single_backend("torch_native", config, iterations, warmup)

    speedup = torch_time / rvv_time
    throughput = config.num_requests / rvv_time

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=speedup,
        throughput_rvv=throughput,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    print(
        f"  {c.description:<35} | BS={c.num_requests}, H={c.num_heads}, D={c.head_dim}"
    )
    print(f"    RVV:   {result.rvv_ms:8.3f} ms ({result.throughput_rvv:.1f} req/s)")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Decode Attention (FP16/BF16)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Decode Attention Benchmark (FP16)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print(f"Quick Mode: {args.quick}")
    print("-" * 60)

    results = []
    for config in configs:
        try:
            result = run_benchmark(config, quick=args.quick)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")

    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print(f"Average Speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
