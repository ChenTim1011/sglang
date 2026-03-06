# Benchmark RVV vs torch_native rotary embedding kernel
#
# Covers 2D, 4D (neox / non-neox) and 3D (DeepSeek V2 style).
# Dtypes tested: BF16 (primary for RoPE).

import platform
import sys
from dataclasses import dataclass

import torch
from utils import (
    IS_CI,
    benchmark_function,
    print_benchmark_result,
    print_benchmark_summary,
)

from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler

try:
    import sgl_kernel  # noqa: F401

    HAS_SGL_KERNEL = True
except ImportError:
    HAS_SGL_KERNEL = False
    print("sgl_kernel not found")
    sys.exit(1)


@dataclass
class RopeConfig:
    head_size: int
    rotary_dim: int
    max_pos: int
    base: int
    is_neox: bool
    dtype: torch.dtype
    batch_size: int
    seq_len: int
    num_q_heads: int
    num_kv_heads: int
    dims: int  # 2, 3, or 4
    description: str


@dataclass
class BenchmarkResult:
    config: RopeConfig
    rvv_ms: float
    torch_ms: float
    speedup: float


# fmt: off
STANDARD_CONFIGS = [
    # 2D non-neox
    RopeConfig(128, 128, 2048, 10000, False, torch.bfloat16, 2, 512, 32, 8, 2, "2D non-neox B=2 S=512"),
    RopeConfig(128, 128, 2048, 10000, False, torch.bfloat16, 1, 32, 16, 4, 2, "2D non-neox B=1 S=32"),
    # 2D neox
    RopeConfig(256, 128, 4096, 10000, True, torch.bfloat16, 2, 512, 32, 8, 2, "2D neox B=2 S=512"),
    RopeConfig(64, 64, 32, 8000, True, torch.bfloat16, 32, 32, 1, 1, 2, "2D neox B=32 S=32"),
    # 4D non-neox
    RopeConfig(128, 128, 2048, 10000, False, torch.bfloat16, 2, 512, 32, 8, 4, "4D non-neox B=2 S=512"),
    RopeConfig(512, 128, 311, 10000, False, torch.bfloat16, 3, 39, 4, 2, 4, "4D non-neox B=3 S=39"),
    # 4D neox
    RopeConfig(256, 128, 4096, 10000, True, torch.bfloat16, 2, 512, 32, 8, 4, "4D neox B=2 S=512"),
    RopeConfig(512, 128, 311, 10000, True, torch.bfloat16, 3, 39, 4, 2, 4, "4D neox B=3 S=39"),
    # 3D DeepSeek V2
    RopeConfig(64, 64, 256, 16, False, torch.bfloat16, 1, 1024, 16, 1, 3, "3D DeepSeek-V2 S=1024"),
    RopeConfig(64, 64, 256, 16, False, torch.bfloat16, 1, 128, 16, 1, 3, "3D DeepSeek-V2 S=128"),
]

CI_CONFIGS = [
    RopeConfig(128, 128, 2048, 10000, False, torch.bfloat16, 1, 32, 16, 4, 2, "2D non-neox"),
    RopeConfig(64, 64, 32, 8000, True, torch.bfloat16, 4, 32, 1, 1, 2, "2D neox"),
    RopeConfig(128, 128, 2048, 10000, False, torch.bfloat16, 1, 32, 16, 4, 4, "4D non-neox"),
    RopeConfig(64, 64, 256, 16, False, torch.bfloat16, 1, 128, 16, 1, 3, "3D DeepSeek-V2"),
]
# fmt: on

ALL_CONFIGS = CI_CONFIGS if IS_CI else STANDARD_CONFIGS

WARMUP = 2 if IS_CI else 5
REPEAT = 5 if IS_CI else 20


def _create_rope_and_inputs(cfg: RopeConfig):
    """Create RotaryEmbedding, positions, query, key for a given config."""
    set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

    if cfg.dims == 3:
        # DeepSeek V2 style
        qk_rope_head_dim = cfg.rotary_dim
        qk_nope_head_dim = 128
        k_dim = 576

        freqs = torch.rand(cfg.max_pos, qk_rope_head_dim // 2)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(cfg.dtype)
        positions = torch.randint(0, cfg.max_pos, (cfg.seq_len,))

        rope = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            cfg.rotary_dim,
            cfg.max_pos,
            cfg.base,
            cfg.is_neox,
            1.0,
            cfg.dtype,
            device="cpu",
        )
        rope.register_buffer("cos_sin_cache", cos_sin_cache)

        q = torch.randn(
            cfg.seq_len,
            cfg.num_q_heads,
            qk_nope_head_dim + qk_rope_head_dim,
            dtype=cfg.dtype,
        )
        k = torch.randn(cfg.seq_len, 1, k_dim, dtype=cfg.dtype)
        _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
        k_pe = k[:, :, k_dim - qk_rope_head_dim :]
        return rope, positions, q_pe, k_pe, cos_sin_cache
    else:
        rope = RotaryEmbedding(
            cfg.head_size,
            cfg.rotary_dim,
            cfg.max_pos,
            cfg.base,
            cfg.is_neox,
            cfg.dtype,
        ).to("cpu")

        pos_ids = torch.arange(cfg.seq_len, device="cpu").repeat(cfg.batch_size)
        query = torch.randn(
            cfg.batch_size * cfg.seq_len,
            cfg.num_q_heads * cfg.head_size,
            dtype=cfg.dtype,
        )
        key = torch.randn(
            cfg.batch_size * cfg.seq_len,
            cfg.num_kv_heads * cfg.head_size,
            dtype=cfg.dtype,
        )
        if cfg.dims == 4:
            query = query.view(
                cfg.batch_size, cfg.seq_len, cfg.num_q_heads, cfg.head_size
            )
            key = key.view(cfg.batch_size, cfg.seq_len, cfg.num_kv_heads, cfg.head_size)
        cos_sin_cache = rope.cos_sin_cache.to(cfg.dtype)
        return rope, pos_ids, query, key, cos_sin_cache


def run_benchmark(cfg: RopeConfig) -> BenchmarkResult:
    rope, positions, query, key, cos_sin_cache = _create_rope_and_inputs(cfg)

    # RVV path
    def rvv_fn():
        q, k = query.clone(), key.clone()
        return torch.ops.sgl_kernel.rotary_embedding_cpu(
            positions, q, k, rope.head_size, cos_sin_cache, cfg.is_neox
        )

    # Reference path
    def ref_fn():
        q, k = query.clone(), key.clone()
        return rope.forward_native(positions, q, k)

    rvv_ms, _ = benchmark_function(rvv_fn, warmup=WARMUP, repeat=REPEAT)
    torch_ms, _ = benchmark_function(ref_fn, warmup=WARMUP, repeat=REPEAT)
    return BenchmarkResult(
        config=cfg,
        rvv_ms=rvv_ms,
        torch_ms=torch_ms,
        speedup=torch_ms / rvv_ms,
    )


def print_result(result: BenchmarkResult):
    c = result.config
    neox_str = "neox" if c.is_neox else "non-neox"
    shape_str = f"B={c.batch_size} S={c.seq_len} {str(c.dtype).split('.')[-1]}"
    label = f"rope {c.dims}D {neox_str:>8}  {c.description}"
    print_benchmark_result(
        label, shape_str, result.rvv_ms, result.torch_ms, result.speedup
    )


def main():
    print("=" * 70)
    print("RVV Rotary Embedding Kernel Benchmark")
    print("=" * 70)
    print(f"Platform: {platform.machine()}")
    print(f"CI Mode:  {IS_CI}")
    print("-" * 70)

    results = []
    for cfg in ALL_CONFIGS:
        try:
            result = run_benchmark(cfg)
            results.append(result)
            print_result(result)
        except Exception as e:
            print(f"FAILED {cfg.description}: {e}")

    print_benchmark_summary(results)


if __name__ == "__main__":
    main()
