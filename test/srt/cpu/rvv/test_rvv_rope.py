"""Unit tests for RVV rotary embedding kernel.

Tests run against RotaryEmbedding.forward_native / DeepseekScalingRotaryEmbedding
and are skipped automatically on non-RISC-V builds.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_rope -v
"""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

from .rvv_utils import has_sgl_kernel_op, precision

torch.manual_seed(1234)


@unittest.skipUnless(
    has_sgl_kernel_op("rotary_embedding_cpu"),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRopeCore(CustomTestCase):
    """Test suite for RVV RoPE kernel compatibility checks."""

    # (head_size, rotary_dim, max_pos, base, is_neox, dtype, device, batch, seq, q_heads, kv_heads)
    test_config = [
        (64, 64, 32, 8000, True, torch.bfloat16, "cpu", 32, 32, 1, 1),
        (64, 64, 32, 8000, True, torch.float16, "cpu", 32, 32, 1, 1),  # FP16 neox
        (256, 128, 4096, 10000, True, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (512, 128, 311, 10000, True, torch.bfloat16, "cpu", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.float16, "cpu", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.bfloat16, "cpu", 3, 39, 4, 2),
        # Non-neox tail case: rotary_dim=96 → embed_dim=48, vl_max=32 → tail of 16
        # Exercises the j += vl loop when embed_dim % vl_max != 0 in compute_body.
        (128, 96, 2048, 10000, False, torch.bfloat16, "cpu", 2, 64, 8, 2),
        (128, 96, 2048, 10000, False, torch.float16, "cpu", 2, 64, 8, 2),
    ]

    def run_case_rope_core(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        dims,
        is_neox_style,
        device,
        dtype,
    ):
        """Run one RoPE case for either 2D or 4D tensor layout."""
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))
        torch.manual_seed(100)
        rope_ref = RotaryEmbedding(
            head_size,
            rotary_dim,
            max_position_embeddings,
            base,
            is_neox_style,
            dtype,
        ).to(device)
        pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
        query = torch.randn(
            batch_size * seq_len,
            num_q_heads * head_size,
            dtype=dtype,
            device=device,
        )
        key = torch.randn(
            batch_size * seq_len,
            num_kv_heads * head_size,
            dtype=dtype,
            device=device,
        )
        if dims == 3:
            # 3D layout: [num_tokens, num_heads, head_size] — requires num_kv_heads == 1
            query = query.view(batch_size * seq_len, num_q_heads, head_size)
            key = key.view(batch_size * seq_len, num_kv_heads, head_size)
        elif dims == 4:
            query = query.view(batch_size, seq_len, num_q_heads, head_size)
            key = key.view(batch_size, seq_len, num_kv_heads, head_size)
        query_ref, key_ref = query.clone(), key.clone()
        query_cpu, key_cpu = query.clone(), key.clone()

        query_ref_out, key_ref_out = rope_ref.forward_native(
            pos_ids, query_ref, key_ref
        )
        query_cpu_out, key_cpu_out = torch.ops.sgl_kernel.rotary_embedding_cpu(
            pos_ids,
            query_cpu,
            key_cpu,
            rope_ref.head_size,
            rope_ref.cos_sin_cache.to(query.dtype),
            rope_ref.is_neox_style,
        )
        atol = rtol = precision["rope"][dtype]
        torch.testing.assert_close(query_ref_out, query_cpu_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(key_ref_out, key_cpu_out, atol=atol, rtol=rtol)

    def test_case_rope_2d(self):
        """Case: 2D RoPE across representative shape and dtype matrix."""
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self.run_case_rope_core(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=2,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )

    def test_case_rope_4d(self):
        """Case: 4D RoPE across representative shape and dtype matrix."""
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self.run_case_rope_core(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=4,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )


@unittest.skipUnless(
    has_sgl_kernel_op("rotary_embedding_cpu"),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRopeDeepseekV2(CustomTestCase):
    """Test suite for RVV DeepSeek V2 RoPE variant."""

    def test_case_deepseek_v2_rope(self):
        """Case: DeepSeek V2 RoPE path with shared KV-head layout."""
        num_head = 16
        seq_len = 1024
        q_head_dim = 192
        qk_nope_head_dim = 128
        qk_rope_head_dim = 64
        max_pos = 256
        k_dim = 576
        rotary_dim = 64
        is_neox_style = False
        set_global_server_args_for_scheduler(ServerArgs(model_path="dummy"))

        freqs = torch.rand(max_pos, qk_rope_head_dim // 2)
        cos = freqs.cos() * 0.7
        sin = freqs.sin() * 0.7
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(torch.bfloat16)
        positions = torch.randint(0, max_pos, (seq_len,))

        rope = DeepseekScalingRotaryEmbedding(
            qk_rope_head_dim,
            rotary_dim,
            max_pos,
            16,
            is_neox_style,
            1.0,
            torch.bfloat16,
            device="cpu",
        )
        rope.register_buffer("cos_sin_cache", cos_sin_cache)

        for dtype in [torch.bfloat16, torch.float16]:
            # cos_sin_cache and rope buffer must match the query dtype.
            cs_cache = cos_sin_cache.to(dtype)
            rope.register_buffer("cos_sin_cache", cs_cache)

            with torch.no_grad():
                q = torch.randn(seq_len, num_head, q_head_dim, dtype=dtype)
                q_clone = q.clone()
                k = torch.randn(seq_len, 1, k_dim, dtype=dtype)
                k_clone = k.clone()
                _, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
                _, q_pe_clone = q_clone.split(
                    [qk_nope_head_dim, qk_rope_head_dim], dim=-1
                )
                k_pe = k[:, :, k_dim - qk_rope_head_dim :]
                k_pe_clone = k_clone[:, :, k_dim - qk_rope_head_dim :]

                q_pe, k_pe = rope.forward_native(
                    query=q_pe,
                    key=k_pe,
                    positions=positions,
                )

                q_pe_clone, k_pe_clone = torch.ops.sgl_kernel.rotary_embedding_cpu(
                    positions,
                    q_pe_clone,
                    k_pe_clone,
                    rope.head_size,
                    cs_cache,
                    False,
                )

                atol = rtol = precision["rope"][q_pe.dtype]
                torch.testing.assert_close(q_pe, q_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone, atol=atol, rtol=rtol)


@unittest.skipUnless(
    has_sgl_kernel_op("rotary_embedding_cpu"),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRope3D(TestRVVRopeCore):
    """Test suite for the 3D RoPE kernel path (num_kv_heads == 1, non-neox only)."""

    # Config: (head_size, rotary_dim, max_pos, base, is_neox, dtype, device, batch, seq, q_heads, kv_heads)
    # 3D requires num_kv_heads == 1 (enforced by TORCH_CHECK in rope.cpp)
    test_config_3d = [
        (128, 64, 2048, 10000, False, torch.bfloat16, "cpu", 2, 32, 8, 1),
        (64, 64, 512, 8000, False, torch.float16, "cpu", 4, 16, 4, 1),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 64, 16, 1),
    ]

    def test_case_rope_3d(self):
        """3D RoPE kernel correctness (non-neox, num_kv_heads=1)."""
        for cfg in self.test_config_3d:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(head_size=hs, rotary_dim=rd, batch=bs, seq=sl, dtype=dt):
                self.run_case_rope_core(
                    head_size=hs,
                    rotary_dim=rd,
                    max_position_embeddings=mp,
                    base=base,
                    batch_size=bs,
                    seq_len=sl,
                    num_q_heads=qh,
                    num_kv_heads=kvh,
                    dims=3,
                    is_neox_style=neox,
                    device=dev,
                    dtype=dt,
                )

    def test_case_rope_3d_neox_raises(self):
        """3D RoPE must raise TORCH_CHECK when is_neox=True."""
        with self.assertRaises(RuntimeError):
            self.run_case_rope_core(
                head_size=128,
                rotary_dim=64,
                max_position_embeddings=2048,
                base=10000,
                batch_size=2,
                seq_len=32,
                num_q_heads=8,
                num_kv_heads=1,
                dims=3,
                is_neox_style=True,
                device="cpu",
                dtype=torch.bfloat16,
            )


if __name__ == "__main__":
    unittest.main()
