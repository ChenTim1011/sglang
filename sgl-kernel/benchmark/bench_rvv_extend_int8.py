"""
Benchmark script for RVV extend attention kernel (INT8).

This script benchmarks the RVV optimized extend attention implementation
against PyTorch's native backend (FP16, as baseline) through SGLang's attention backend system.

Usage:
    python3 bench_rvv_extend_int8.py [--quick]
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
    kv_dtype: torch.dtype = torch.int8


@dataclass
class BenchmarkResult:
    config: BenchmarkConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# ============================================================================
# Benchmark Configurations
# ============================================================================

INT8_CONFIGS = [
    # Small / Standard
    BenchmarkConfig(
        1,
        128,
        32,
        8,
        64,
        "extend",
        "INT8 Extend (Seq=128, Ext=32)",
    ),
    BenchmarkConfig(1, 128, 128, 8, 64, "prefill", "INT8 Prefill (Seq=128)"),
    # TinyLlama
    BenchmarkConfig(1, 2048, 2048, 8, 64, "prefill", "INT8 Prefill (Seq=2048)"),
    BenchmarkConfig(1, 4096, 4096, 8, 64, "prefill", "INT8 Prefill (Seq=4096)"),
    BenchmarkConfig(1, 2048, 128, 8, 64, "extend", "INT8 Extend (Seq=2048, Ext=128)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 32, 8, 4, 32, "extend", "CI INT8 Extend"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = INT8_CONFIGS

# ============================================================================
# Mocking Utilities
# ============================================================================


class MockRunner:
    def __init__(self, num_heads, head_dim, kv_dtype=torch.float16):
        self.device = "cpu"
        self.model_config = argparse.Namespace(num_attention_heads=num_heads)
        self.tp_size = 1
        self.token_to_kv_pool = MockTokenToKVPool(num_heads, head_dim, dtype=kv_dtype)
        self.kv_cache_dtype = "int8" if kv_dtype == torch.int8 else "auto"


class MockTokenToKVPool:
    def __init__(self, num_heads, head_dim, max_tokens=10000, dtype=torch.float16):
        self.max_tokens = max_tokens
        self.num_heads = num_heads
        self.head_dim = head_dim

        if dtype == torch.int8:
            # Generate float, then quantize to int8 range
            self.k_buffer = (
                torch.randn(max_tokens, num_heads, head_dim, dtype=torch.float32) * 50
            ).to(torch.int8)
            self.v_buffer = (
                torch.randn(max_tokens, num_heads, head_dim, dtype=torch.float32) * 50
            ).to(torch.int8)
        else:
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
    def __init__(
        self, num_reqs, seq_len, extend_len, num_heads, head_dim, kv_dtype=torch.float16
    ):
        self.batch_size = num_reqs
        self.req_pool_indices = torch.arange(num_reqs)
        self.seq_lens = torch.full((num_reqs,), seq_len, dtype=torch.int64)
        self.extend_seq_lens = torch.full((num_reqs,), extend_len, dtype=torch.int64)
        self.extend_prefix_lens = self.seq_lens - self.extend_seq_lens
        self.extend_start_loc = torch.arange(num_reqs) * extend_len
        max_tokens = num_reqs * seq_len * 2
        self.req_to_token_pool = MockReqToTokenPool(num_reqs, seq_len, max_tokens)
        self.token_to_kv_pool = MockTokenToKVPool(
            num_heads, head_dim, max_tokens, dtype=kv_dtype
        )
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


class RVVAttnBackendInt8(RVVAttnBackend):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.k_scale = 0.01
        self.v_scale = 0.01
        self.has_int8_kernel = hasattr(
            torch.ops.sgl_kernel, "extend_attention_int8_cpu"
        )

    def forward_extend(self, q, k, v, layer, forward_batch):
        if not self.has_int8_kernel:
            # Assume it exists for now since we are testing it.
            pass
        # Dispatch to INT8 kernel manually
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

        o_extend = torch.empty_like(v)

        # Note: k and v here are the EXTEND part (new tokens), so they are still float/half
        # The PREFIX part comes from k_buffer/v_buffer which are INT8

        torch.ops.sgl_kernel.extend_attention_int8_cpu(
            q,
            k,
            v,
            o_extend,
            k_buffer,
            v_buffer,
            forward_batch.req_to_token_pool.req_to_token,
            forward_batch.req_pool_indices,
            forward_batch.seq_lens,
            forward_batch.extend_seq_lens,
            forward_batch.extend_start_loc,
            torch.max(forward_batch.extend_seq_lens).item(),
            layer.scaling,
            layer.logit_cap,
            self.k_scale,
            self.v_scale,
        )
        return o_extend


def run_single_backend(backend_name, config, num_iterations=20, warmup=5):
    runner = MockRunner(config.num_heads, config.head_dim, config.kv_dtype)
    layer = MockLayer(config.num_heads, config.head_dim)

    if backend_name == "rvv":
        if config.kv_dtype == torch.int8:
            backend = RVVAttnBackendInt8(runner)
        else:
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
        config.kv_dtype,
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
    iterations = 10 if quick else 100
    warmup = 2 if quick else 10

    rvv_time = run_single_backend("rvv", config, iterations, warmup)

    # Use FP16 for torch comparison even if config is INT8 (since Torch has no INT8 kernel)
    if config.kv_dtype == torch.int8:
        config_fp16 = BenchmarkConfig(
            config.num_reqs,
            config.seq_len,
            config.extend_len,
            config.num_heads,
            config.head_dim,
            config.mode,
            config.description,
            torch.float16,
        )
        torch_time = run_single_backend("torch_native", config_fp16, iterations, warmup)
    else:
        torch_time = run_single_backend("torch_native", config, iterations, warmup)

    return BenchmarkResult(
        config=config,
        rvv_ms=rvv_time * 1000,
        torch_ms=torch_time * 1000,
        speedup=torch_time / rvv_time,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    dtype_str = "INT8" if c.kv_dtype == torch.int8 else "FP16"
    print(
        f"  {c.description:<35} | Mode={c.mode}, Seq={c.seq_len}, Ext={c.extend_len}, {dtype_str}"
    )
    print(f"    RVV:   {result.rvv_ms:8.3f} ms")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Extend Attention (INT8)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Extend Attention Benchmark (INT8)")
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
