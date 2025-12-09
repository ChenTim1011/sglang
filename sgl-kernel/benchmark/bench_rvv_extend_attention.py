#!/usr/bin/env python3
"""
Benchmark script for RVV extend attention kernel.

This script benchmarks the RVV optimized extend attention implementation
against PyTorch's native backend through SGLang's attention backend system.

Usage:
    python3 bench_rvv_extend_attention.py [--quick]
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
    num_reqs: int
    seq_len: int
    extend_len: int
    num_heads: int
    head_dim: int
    mode: str  # "extend" or "prefill"
    description: str


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# ============================================================================
# Benchmark Configurations
# ============================================================================

STANDARD_CONFIGS = [
    BenchmarkConfig(1, 128, 32, 8, 64, "extend", "Extend (Seq=128, Ext=32)"),
    BenchmarkConfig(1, 128, 128, 8, 64, "prefill", "Prefill (Seq=128)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 32, 8, 4, 32, "extend", "CI Extend"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = STANDARD_CONFIGS

# ============================================================================
# Mocking Utilities
# ============================================================================


class MockRunner:
    def __init__(self, num_heads, head_dim):
        self.device = "cpu"
        self.model_config = argparse.Namespace(num_attention_heads=num_heads)
        self.tp_size = 1
        self.token_to_kv_pool = MockTokenToKVPool(num_heads, head_dim)


class MockTokenToKVPool:
    def __init__(self, num_heads, head_dim, max_tokens=10000, dtype=torch.float16):
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.k_buffer = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype)
        self.v_buffer = torch.randn(max_tokens, num_heads, head_dim, dtype=dtype)

    def get_key_buffer(self, layer_id):
        return self.k_buffer

    def get_value_buffer(self, layer_id):
        return self.v_buffer

    def set_kv_buffer(self, layer, loc, k, v):
        # Handle both layer object and layer_id
        if hasattr(loc, "__len__"):
            self.k_buffer[loc] = k
            self.v_buffer[loc] = v


class MockLayer:
    def __init__(self, num_heads, head_dim):
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_heads
        self.tp_v_head_num = num_heads
        self.head_dim = head_dim
        self.qk_head_dim = head_dim
        self.v_head_dim = head_dim
        self.layer_id = 0
        self.scaling = 1.0 / (head_dim**0.5)
        self.logit_cap = 50.0
        self.is_cross_attention = False
        self.attn_type = None


class MockForwardMode:
    def is_decode_or_idle(self):
        return False

    def is_extend(self):
        return True


class MockForwardBatch:
    def __init__(self, num_reqs, seq_len, extend_len, num_heads, head_dim):
        self.batch_size = num_reqs
        self.req_pool_indices = torch.arange(num_reqs)
        self.seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int64)
        self.extend_seq_lens = torch.full((num_reqs,), extend_len, dtype=torch.int64)
        self.extend_prefix_lens = self.seq_lens - self.extend_seq_lens
        self.extend_start_loc = torch.arange(num_reqs) * extend_len
        max_tokens = num_reqs * seq_len * 2
        self.req_to_token_pool = MockReqToTokenPool(num_reqs, seq_len, max_tokens)
        self.token_to_kv_pool = MockTokenToKVPool(num_heads, head_dim, max_tokens)
        self.forward_mode = MockForwardMode()
        self.out_cache_loc = torch.arange(num_reqs * extend_len)


class MockReqToTokenPool:
    def __init__(self, num_reqs, seq_len, max_tokens):
        self.req_to_token = torch.zeros(num_reqs, seq_len, dtype=torch.int64)
        for i in range(num_reqs):
            self.req_to_token[i] = torch.arange(seq_len) % max_tokens


# ============================================================================
# Benchmark Functions
# ============================================================================


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    runner = MockRunner(config.num_heads, config.head_dim)
    layer = MockLayer(config.num_heads, config.head_dim)

    if backend_name == "rvv":
        backend = RVVAttnBackend(runner)
    else:
        backend = TorchNativeAttnBackend(runner)

    actual_extend_len = (
        config.seq_len if config.mode == "prefill" else config.extend_len
    )
    actual_seq_len = config.seq_len

    batch = MockForwardBatch(
        config.num_reqs,
        actual_seq_len,
        actual_extend_len,
        config.num_heads,
        config.head_dim,
    )
    backend.init_forward_metadata(batch)

    total_extend_len = config.num_reqs * actual_extend_len
    dtype = torch.float16
    q = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)
    k = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)
    v = torch.randn(total_extend_len, config.num_heads, config.head_dim, dtype=dtype)

    for _ in range(warmup):
        backend.forward_extend(q, k, v, layer, batch)

    start = time.time()
    for _ in range(num_iterations):
        backend.forward_extend(q, k, v, layer, batch)
    end = time.time()

    return (end - start) / num_iterations


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 5 if quick else 20
    warmup = 2 if quick else 5

    rvv_time = run_single_backend("rvv", config, iterations, warmup)
    torch_time = run_single_backend("torch_native", config, iterations, warmup)

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=torch_time / rvv_time,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    print(f"  {c.description:<30} | Mode={c.mode}, Seq={c.seq_len}, Ext={c.extend_len}")
    print(f"    RVV:   {result.rvv_ms:8.3f} ms")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark RVV Extend Attention")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Extend Attention Benchmark")
    print("=" * 60)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode: {IS_CI}")
    print("-" * 60)

    results = []
    for config in configs:
        try:
            result = run_benchmark(config, quick=args.quick)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {config.description}: {e}")


if __name__ == "__main__":
    main()
