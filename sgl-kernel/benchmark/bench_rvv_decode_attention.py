#!/usr/bin/env python3
"""
Benchmark script comparing RVV and torch_native attention backends.

This script benchmarks through SGLang's attention backend system,
comparing `attention-backend=rvv` vs `attention-backend=torch_native`.

Usage:
    python3 bench_rvv_decode_attention.py [--quick]
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

# CI environment detection
# This checks if the script is running in a Continuous Integration environment
# (like GitHub Actions). If true, we typically run a smaller set of tests.
IS_CI = (
    os.getenv("CI", "false").lower() == "true"
    or os.getenv("GITHUB_ACTIONS", "false").lower() == "true"
)

# Add parent directory to path to import sglang
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "python"))


def setup_triton_stub():
    """
    Setup triton_stub for RISC-V environments where full Triton is not available.
    This allows SGLang to import without failing on missing triton dependency.
    """
    try:
        import triton

        print(
            f"[INFO] Using real triton module (version: {getattr(triton, '__version__', 'unknown')})"
        )
        return
    except ImportError:
        pass

    print("[INFO] Real triton not available, attempting to use triton_stub...")

    # Check multiple possible locations for triton_stub.py
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
        # Execute triton_stub.py directly - it registers itself in sys.modules
        stub_namespace = {
            "__file__": triton_stub_path,
            "__name__": "triton_stub",
            "__package__": "",
        }
        stub_namespace.update(sys.modules)
        with open(triton_stub_path, "r") as f:
            exec(compile(f.read(), triton_stub_path, "exec"), stub_namespace)
        print(f"[INFO] Using triton_stub from {triton_stub_path}")
    else:
        print(f"[WARNING] triton not available and triton_stub not found")


# Setup environment
setup_triton_stub()

try:
    from sglang.srt.layers.attention.rvv_backend import RVVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend

    HAS_SGLANG = True
except ImportError as e:
    HAS_SGLANG = False
    print(f"ERROR: SGLang not available: {e}")
    sys.exit(1)


def is_riscv_platform():
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


# ============================================================================
# Benchmark Data Classes
# ============================================================================


@dataclass
class BenchmarkConfig:
    """Configuration for a single benchmark."""

    num_requests: int
    num_heads: int
    head_dim: int
    seq_len: int
    description: str


@dataclass
class BenchmarkResult:
    """Result of a single benchmark."""

    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float
    throughput_rvv: float


# ============================================================================
# Benchmark Configurations
# ============================================================================

# Standard configurations
STANDARD_CONFIGS = [
    BenchmarkConfig(1, 8, 64, 128, "Small Batch (BS=1)"),
    BenchmarkConfig(4, 8, 64, 128, "Medium Batch (BS=4)"),
    BenchmarkConfig(32, 8, 64, 128, "Large Batch (BS=32)"),
]

# TinyLlama configurations (2048 hidden, 32 heads -> 64 head_dim)
TINYLLAMA_CONFIGS = [
    BenchmarkConfig(1, 32, 64, 128, "TinyLlama Decode (BS=1)"),
    BenchmarkConfig(8, 32, 64, 128, "TinyLlama Decode (BS=8)"),
]

# CI configurations (faster)
CI_CONFIGS = [
    BenchmarkConfig(1, 4, 32, 32, "CI Quick Test"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS + TINYLLAMA_CONFIGS


# ============================================================================
# Mocking Utilities
# ============================================================================


def create_mock_runner(num_heads, head_dim, v_head_dim):
    """Create a mock ModelRunner for testing."""
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.tp_size = 1

    # Mock token_to_kv_pool
    mock_runner.token_to_kv_pool = Mock()
    # Return buffers large enough for testing
    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(10000, num_heads, head_dim, dtype=torch.float16)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(10000, num_heads, v_head_dim, dtype=torch.float16)
    )

    return mock_runner


def create_mock_layer(num_heads, head_dim, v_head_dim):
    """Create a mock RadixAttention layer."""
    mock_layer = Mock()
    mock_layer.tp_q_head_num = num_heads
    mock_layer.qk_head_dim = head_dim
    mock_layer.v_head_dim = v_head_dim
    mock_layer.layer_id = 0
    mock_layer.scaling = 1.0 / (head_dim**0.5)
    mock_layer.logit_cap = 50.0
    return mock_layer


def create_mock_forward_batch(
    num_requests, num_heads, head_dim, v_head_dim, max_seq_len
):
    """Create a mock ForwardBatch."""
    mock_batch = Mock()
    mock_batch.batch_size = num_requests
    mock_batch.out_cache_loc = torch.zeros(num_requests, dtype=torch.int64)
    mock_batch.seq_lens = torch.ones(num_requests, dtype=torch.int64) * max_seq_len
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    # Mock req_to_token_pool
    mock_batch.req_to_token_pool = Mock()
    # Map requests to tokens [0..seq_len]
    mock_batch.req_to_token_pool.req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.int64
    )

    # Mock token_to_kv_pool
    mock_batch.token_to_kv_pool = Mock()
    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(
            max_seq_len * num_requests + 100, num_heads, head_dim, dtype=torch.float16
        )
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(
            max_seq_len * num_requests + 100, num_heads, v_head_dim, dtype=torch.float16
        )
    )
    return mock_batch


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_single_backend(backend_name, config, num_iterations=100, warmup=10):
    """Run benchmark for a single backend."""
    v_head_dim = config.head_dim

    mock_runner = create_mock_runner(config.num_heads, config.head_dim, v_head_dim)
    mock_layer = create_mock_layer(config.num_heads, config.head_dim, v_head_dim)

    if backend_name == "rvv":
        backend = RVVAttnBackend(mock_runner)
    elif backend_name == "torch_native":
        backend = TorchNativeAttnBackend(mock_runner)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # Data
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
    )

    backend.init_forward_metadata(forward_batch)

    # Warmup
    for _ in range(warmup):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(num_iterations):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    """Run benchmark for a configuration."""
    iterations = 10 if quick else 100
    warmup = 2 if quick else 10

    # Run RVV
    rvv_time = run_single_backend("rvv", config, iterations, warmup)

    # Run Torch Native
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
    """Print result in a formatted way."""
    c = result.config
    print(
        f"  {c.description:<30} | BS={c.num_requests}, H={c.num_heads}, D={c.head_dim}"
    )
    print(f"    RVV:   {result.rvv_ms:8.3f} ms ({result.throughput_rvv:.1f} req/s)")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVV Decode Attention")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    # Determine configs
    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Decode Attention Benchmark")
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

    # Summary
    if results:
        avg_speedup = sum(r.speedup for r in results) / len(results)
        print(f"Average Speedup: {avg_speedup:.2f}x")


if __name__ == "__main__":
    main()
