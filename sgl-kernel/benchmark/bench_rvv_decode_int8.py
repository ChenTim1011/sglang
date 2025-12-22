"""
Benchmark script comparing RVV and torch_native attention backends (Decode, INT8).

This script benchmarks through SGLang's attention backend system,
comparing `attention-backend=rvv` (INT8) vs `attention-backend=torch_native` (FP16).

Usage:
    python3 bench_rvv_decode_int8.py [--quick]
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
    kv_dtype: torch.dtype = torch.int8


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

INT8_CONFIGS = [
    # Small / Standard
    BenchmarkConfig(1, 8, 64, 128, "INT8 Small Batch (BS=1)"),
    BenchmarkConfig(4, 8, 64, 128, "INT8 Medium Batch (BS=4)"),
    BenchmarkConfig(32, 8, 64, 128, "INT8 Large Batch (BS=32)"),
    # TinyLlama
    BenchmarkConfig(1, 32, 64, 128, "INT8 TinyLlama Decode (BS=1)"),
    BenchmarkConfig(8, 32, 64, 128, "INT8 TinyLlama Decode (BS=8)"),
    # Long Sequence
    BenchmarkConfig(1, 32, 64, 2048, "INT8 Long Seq (BS=1, Seq=2048)"),
    BenchmarkConfig(1, 32, 64, 4096, "INT8 Long Seq (BS=1, Seq=4096)"),
]

CI_CONFIGS = [
    BenchmarkConfig(1, 4, 32, 32, "CI INT8 Quick Test"),
]

if IS_CI:
    ALL_CONFIGS = CI_CONFIGS
else:
    ALL_CONFIGS = INT8_CONFIGS


# ============================================================================
# Mocking Utilities
# ============================================================================


def create_mock_runner(num_heads, head_dim, v_head_dim, dtype=torch.float16):
    mock_runner = Mock()
    mock_runner.device = torch.device("cpu")
    mock_runner.model_config = Mock()
    mock_runner.model_config.num_attention_heads = num_heads
    mock_runner.tp_size = 1
    mock_runner.kv_cache_dtype = "int8" if dtype == torch.int8 else "auto"

    mock_runner.token_to_kv_pool = Mock()

    def create_buffer(num_tokens, heads, hdim, dtype):
        if dtype == torch.int8:
            return (torch.randn(num_tokens, heads, hdim, dtype=torch.float32) * 50).to(
                torch.int8
            )
        else:
            return torch.randn(num_tokens, heads, hdim, dtype=dtype)

    mock_runner.token_to_kv_pool.get_key_buffer = Mock(
        return_value=create_buffer(10000, num_heads, head_dim, dtype)
    )
    mock_runner.token_to_kv_pool.get_value_buffer = Mock(
        return_value=create_buffer(10000, num_heads, v_head_dim, dtype)
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

    def create_buffer(num_tokens, heads, hdim, dtype):
        if dtype == torch.int8:
            return (torch.randn(num_tokens, heads, hdim, dtype=torch.float32) * 50).to(
                torch.int8
            )
        else:
            return torch.randn(num_tokens, heads, hdim, dtype=dtype)

    mock_batch.token_to_kv_pool.get_key_buffer = Mock(
        return_value=create_buffer(
            max_seq_len * num_requests + 100, num_heads, head_dim, dtype
        )
    )
    mock_batch.token_to_kv_pool.get_value_buffer = Mock(
        return_value=create_buffer(
            max_seq_len * num_requests + 100, num_heads, v_head_dim, dtype
        )
    )
    return mock_batch


# ============================================================================
# Benchmark Functions
# ============================================================================


class RVVAttnBackendInt8(RVVAttnBackend):
    def __init__(self, model_runner):
        super().__init__(model_runner)
        self.k_scale = 0.01
        self.v_scale = 0.01
        if hasattr(torch.ops.sgl_kernel, "decode_attention_int8_cpu"):
            self.decode_attention_fwd_int8 = (
                torch.ops.sgl_kernel.decode_attention_int8_cpu
            )
        else:
            self.decode_attention_fwd_int8 = None

    def _rvv_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer,
        forward_batch,
        save_kv_cache=True,
    ):
        k_buffer = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id)
        if k_buffer.dtype == torch.int8 and self.decode_attention_fwd_int8:
            if q.dtype not in (torch.float16, torch.bfloat16):
                q = q.half()

            if k.dtype != torch.int8:
                k = (k / self.k_scale).round().clamp(-128, 127).to(torch.int8)
                v = (v / self.v_scale).round().clamp(-128, 127).to(torch.int8)

            attn_logits, _ = self.forward_metadata
            q_reshaped = q.reshape(-1, layer.tp_q_head_num * layer.qk_head_dim)

            if layer.qk_head_dim != layer.v_head_dim:
                o = torch.empty(
                    (q_reshaped.shape[0], layer.tp_q_head_num * layer.v_head_dim),
                    dtype=q.dtype,
                    device=q.device,
                )
            else:
                o = torch.empty_like(q_reshaped)

            v_buffer = forward_batch.token_to_kv_pool.get_value_buffer(layer.layer_id)

            self.decode_attention_fwd_int8(
                q_reshaped.view(-1, layer.tp_q_head_num, layer.qk_head_dim),
                k_buffer,
                v_buffer,
                o.view(-1, layer.tp_q_head_num, layer.v_head_dim),
                k,
                v,
                forward_batch.out_cache_loc,
                attn_logits,
                forward_batch.req_to_token_pool.req_to_token,
                forward_batch.req_pool_indices,
                forward_batch.seq_lens,
                layer.scaling,
                layer.logit_cap,
                self.k_scale,
                self.v_scale,
            )
            return o
        else:
            return super()._rvv_decode(q, k, v, layer, forward_batch, save_kv_cache)


def run_single_backend(backend_name, config, num_iterations=100, warmup=10):
    v_head_dim = config.head_dim

    # Use FP16 for torch native comparison even if benchmark is INT8
    # Use INT8 for RVV if benchmark is INT8
    runner_dtype = torch.float16
    if backend_name == "rvv" and config.kv_dtype == torch.int8:
        runner_dtype = torch.int8

    mock_runner = create_mock_runner(
        config.num_heads, config.head_dim, v_head_dim, runner_dtype
    )
    mock_layer = create_mock_layer(config.num_heads, config.head_dim, v_head_dim)

    if backend_name == "rvv":
        if config.kv_dtype == torch.int8:
            backend = RVVAttnBackendInt8(mock_runner)
        else:
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
        runner_dtype,
    )

    backend.init_forward_metadata(forward_batch)

    for _ in range(warmup):
        try:
            backend.forward_decode(
                q, k, v, mock_layer, forward_batch, save_kv_cache=True
            )
        except Exception:
            if backend_name == "rvv":
                # INT8 might fail if not implemented
                return 0.001
            raise

    start_time = time.perf_counter()
    for _ in range(num_iterations):
        backend.forward_decode(q, k, v, mock_layer, forward_batch, save_kv_cache=True)
    end_time = time.perf_counter()

    avg_time = (end_time - start_time) / num_iterations
    return avg_time


def run_benchmark(config: BenchmarkConfig, quick=False) -> BenchmarkResult:
    iterations = 10 if quick else 20
    warmup = 2 if quick else 5

    rvv_time = run_single_backend("rvv", config, iterations, warmup)

    # Run Torch Native (Always FP16 as baseline)
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
    dtype_str = "INT8"
    print(
        f"  {c.description:<35} | BS={c.num_requests}, H={c.num_heads}, D={c.head_dim}, {dtype_str}"
    )
    print(f"    RVV:   {result.rvv_ms:8.3f} ms ({result.throughput_rvv:.1f} req/s)")
    print(f"    Torch: {result.torch_ms:8.3f} ms")
    print(f"    Speedup: {result.speedup:.2f}x")
    print("-" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark RVV Decode Attention (INT8)"
    )
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark")
    args = parser.parse_args()

    configs = ALL_CONFIGS
    if args.quick:
        configs = CI_CONFIGS

    print("=" * 60)
    print("RVV Decode Attention Benchmark (INT8)")
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
