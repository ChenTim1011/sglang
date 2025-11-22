#!/usr/bin/env python3
"""
Benchmark script comparing RISC-V and torch_native attention backends.

This script benchmarks through SGLang's attention backend system,
comparing `attention-backend=riscv` vs `attention-backend=torch_native`.

Usage:
    python3 bench_decode_attention_backend.py [--num-iterations N] [--warmup N]
"""

import argparse
import time
import torch
import sys
import os
import platform
from unittest.mock import Mock

# Add parent directory to path
sys.path.insert(0, os.path.join(
    os.path.dirname(__file__), "..", "..", "python"))

# IMPORTANT: Load triton_stub BEFORE importing SGLang (for RISC-V environments)
# This must happen before any SGLang imports that may use triton
# Priority: 1. Try real triton first, 2. Fall back to triton_stub if not available
try:
    import triton
    HAS_TRITON = True
    triton_version = getattr(triton, "__version__", "unknown")
    triton_file = getattr(triton, "__file__", "unknown")
    print(f"[INFO] Using real triton module (version: {triton_version})")
    print(f"[INFO] Triton path: {triton_file}")
except ImportError:
    HAS_TRITON = False
    print("[INFO] Real triton not available, attempting to use triton_stub...")

    # Try to use triton_stub for RISC-V environments
    # Check multiple possible locations
    possible_stub_paths = [
        os.path.join(
            os.path.dirname(
                __file__), "..", "..", "banana_pi", "test_tinyllama_riscv", "triton_stub.py"
        ),
        os.path.join(
            os.path.dirname(
                __file__), "..", "..", "..", "banana_pi", "test_tinyllama_riscv", "triton_stub.py"
        ),
    ]

    triton_stub_path = None
    for path in possible_stub_paths:
        if os.path.exists(path):
            triton_stub_path = path
            break

    if triton_stub_path:
        # Execute triton_stub.py directly - it registers itself in sys.modules
        # Create a namespace with __file__ set correctly
        stub_namespace = {"__file__": triton_stub_path,
                          "__name__": "triton_stub", "__package__": ""}
        stub_namespace.update(sys.modules)
        with open(triton_stub_path, "r") as f:
            triton_stub_code = f.read()
        # Execute in the namespace - triton_stub will register itself in sys.modules
        exec(compile(triton_stub_code, triton_stub_path, "exec"), stub_namespace)
        print(f"[INFO] Using triton_stub from {triton_stub_path}")
    else:
        print(f"[WARNING] triton not available and triton_stub not found")
        print(f"  Checked paths: {possible_stub_paths}")

# Now import SGLang (triton_stub should be loaded if needed)
try:
    from sglang.srt.layers.attention.riscv_backend import RISCVAttnBackend
    from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
    HAS_SGLANG = True
except ImportError as e:
    HAS_SGLANG = False
    print(f"ERROR: SGLang not available: {e}")
    print("Make sure SGLang Python package is installed and PYTHONPATH is set correctly")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def is_riscv_platform():
    """Check if running on RISC-V platform."""
    machine = platform.machine().lower()
    return machine in ("riscv64", "riscv32", "riscv")


def create_mock_runner(num_heads, head_dim, v_head_dim):
    """Create a mock ModelRunner for testing."""
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")

    # Mock model_config
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.tp_size = 1

    # Mock token_to_kv_pool
    mock_runner.token_to_kv_pool = Mock()
    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(128, num_heads, head_dim, dtype=torch.float16)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(
            128, num_heads, v_head_dim, dtype=torch.float16)
    )

    return mock_runner


def create_mock_layer(num_heads, head_dim, v_head_dim):
    """Create a mock RadixAttention layer."""
    mock_layer = Mock()
    mock_layer.tp_q_head_num = num_heads
    mock_layer.qk_head_dim = head_dim
    mock_layer.v_head_dim = v_head_dim
    mock_layer.layer_id = 0
    mock_layer.scaling = 1.0 / (head_dim ** 0.5)
    mock_layer.logit_cap = 50.0
    return mock_layer


def create_mock_forward_batch(num_requests, num_heads, head_dim, v_head_dim, max_seq_len):
    """Create a mock ForwardBatch."""
    mock_batch = Mock()
    mock_batch.batch_size = num_requests
    mock_batch.out_cache_loc = torch.zeros(num_requests, dtype=torch.int64)
    mock_batch.seq_lens = torch.ones(num_requests, dtype=torch.int64)
    mock_batch.req_pool_indices = torch.arange(num_requests, dtype=torch.int64)

    # Mock req_to_token_pool
    mock_batch.req_to_token_pool = Mock()
    mock_batch.req_to_token_pool.req_to_token = torch.zeros(
        num_requests, max_seq_len, dtype=torch.int64
    )

    # Mock token_to_kv_pool
    mock_batch.token_to_kv_pool = Mock()
    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=torch.randn(
            max_seq_len, num_heads, head_dim, dtype=torch.float16)
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=torch.randn(
            max_seq_len, num_heads, v_head_dim, dtype=torch.float16)
    )

    return mock_batch


def benchmark_backend(
    backend_name,
    num_requests,
    num_heads,
    head_dim,
    max_seq_len,
    num_iterations=100,
    warmup=10,
):
    """Benchmark a specific attention backend."""
    v_head_dim = head_dim  # Assume same for simplicity

    # Create mock objects
    mock_runner = create_mock_runner(num_heads, head_dim, v_head_dim)
    mock_layer = create_mock_layer(num_heads, head_dim, v_head_dim)

    # Create backend
    if backend_name == "riscv":
        backend = RISCVAttnBackend(mock_runner)
    elif backend_name == "torch_native":
        backend = TorchNativeAttnBackend(mock_runner)
    else:
        raise ValueError(f"Unknown backend: {backend_name}")

    # Create test data
    dtype = torch.float16
    q = torch.randn(num_requests, num_heads, head_dim, dtype=dtype)
    k = torch.randn(num_requests, num_heads, head_dim, dtype=dtype)
    v = torch.randn(num_requests, num_heads, v_head_dim, dtype=dtype)

    # Create forward batch
    forward_batch = create_mock_forward_batch(
        num_requests, num_heads, head_dim, v_head_dim, max_seq_len
    )

    # Initialize forward metadata
    backend.init_forward_metadata(forward_batch)

    # Warmup
    for _ in range(warmup):
        try:
            backend.forward_decode(q, k, v, mock_layer,
                                   forward_batch, save_kv_cache=True)
        except Exception as e:
            print(f"WARNING: Warmup failed: {e}")
            return None

    # Benchmark
    start_time = time.perf_counter()

    for _ in range(num_iterations):
        try:
            backend.forward_decode(q, k, v, mock_layer,
                                   forward_batch, save_kv_cache=True)
        except Exception as e:
            print(f"ERROR: Benchmark failed: {e}")
            return None

    end_time = time.perf_counter()

    total_time = end_time - start_time
    avg_time = total_time / num_iterations
    throughput = num_requests / avg_time

    return {
        "total_time": total_time,
        "avg_time": avg_time,
        "throughput": throughput,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RISC-V vs torch_native attention backends"
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=1,
        help="Number of requests (default: 1)",
    )
    parser.add_argument(
        "--num-heads",
        type=int,
        default=4,
        help="Number of attention heads (default: 4)",
    )
    parser.add_argument(
        "--head-dim",
        type=int,
        default=32,
        help="Head dimension (default: 32)",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=32,
        help="Maximum sequence length (default: 32)",
    )
    parser.add_argument(
        "--num-iterations",
        type=int,
        default=100,
        help="Number of benchmark iterations (default: 100)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Number of warmup iterations (default: 10)",
    )

    args = parser.parse_args()

    if not is_riscv_platform():
        print("WARNING: Not running on RISC-V platform")
        print(f"  Machine: {platform.machine()}")

    print("=" * 60)
    print("Attention Backend Benchmark (RISC-V vs torch_native)")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"Parameters:")
    print(f"  num_requests: {args.num_requests}")
    print(f"  num_heads: {args.num_heads}")
    print(f"  head_dim: {args.head_dim}")
    print(f"  max_seq_len: {args.max_seq_len}")
    print(f"  num_iterations: {args.num_iterations}")
    print(f"  warmup: {args.warmup}")
    print()

    # Benchmark RISC-V backend
    print("Benchmarking attention-backend=riscv...")
    riscv_results = benchmark_backend(
        "riscv",
        args.num_requests,
        args.num_heads,
        args.head_dim,
        args.max_seq_len,
        args.num_iterations,
        args.warmup,
    )

    if riscv_results is None:
        print("ERROR: Failed to benchmark RISC-V backend")
        sys.exit(1)

    print(f"  Average time: {riscv_results['avg_time']*1000:.3f} ms")
    print(f"  Throughput: {riscv_results['throughput']:.2f} requests/sec")
    print()

    # Benchmark torch_native backend
    print("Benchmarking attention-backend=torch_native...")
    torch_native_results = benchmark_backend(
        "torch_native",
        args.num_requests,
        args.num_heads,
        args.head_dim,
        args.max_seq_len,
        args.num_iterations,
        args.warmup,
    )

    if torch_native_results is None:
        print("ERROR: Failed to benchmark torch_native backend")
        sys.exit(1)

    print(f"  Average time: {torch_native_results['avg_time']*1000:.3f} ms")
    print(
        f"  Throughput: {torch_native_results['throughput']:.2f} requests/sec")
    print()

    # Calculate speedup
    speedup = torch_native_results["avg_time"] / riscv_results["avg_time"]
    print("=" * 60)
    print("Comparison Results")
    print("=" * 60)
    print(
        f"attention-backend=riscv:      {riscv_results['avg_time']*1000:.3f} ms")
    print(
        f"attention-backend=torch_native: {torch_native_results['avg_time']*1000:.3f} ms")
    print(f"Speedup:                      {speedup:.2f}x")
    print("=" * 60)


if __name__ == "__main__":
    main()
