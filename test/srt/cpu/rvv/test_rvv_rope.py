"""Unit tests for RVV rotary embedding kernel.

Tests run against RotaryEmbedding.forward_native / DeepseekScalingRotaryEmbedding
and are skipped automatically on non-RISC-V builds.
"""

import unittest

import torch

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
)
from sglang.srt.server_args import ServerArgs, set_global_server_args_for_scheduler
from sglang.test.test_utils import CustomTestCase

from .utils import precision

torch.manual_seed(1234)


def _has_sgl_kernel_rope():
    """Return True only if the RVV rotary_embedding_cpu op is registered."""
    try:
        import sgl_kernel  # noqa: F401

        _ = torch.ops.sgl_kernel.rotary_embedding_cpu
        return True
    except (ImportError, AttributeError):
        return False


@unittest.skipUnless(
    _has_sgl_kernel_rope(),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVRope(CustomTestCase):
    """RVV RoPE tests — 2D/4D, neox/non-neox configs."""

    # (head_size, rotary_dim, max_pos, base, is_neox, dtype, device, batch, seq, q_heads, kv_heads)
    test_config = [
        (64, 64, 32, 8000, True, torch.bfloat16, "cpu", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (512, 128, 311, 10000, True, torch.bfloat16, "cpu", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cpu", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.bfloat16, "cpu", 3, 39, 4, 2),
    ]

    def _single_test(
        self,
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        is_neox_style,
        dtype,
        device,
        batch_size,
        seq_len,
        num_q_heads,
        num_kv_heads,
        dims,
    ):
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
        if dims == 4:
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
        atol = rtol = 5e-2
        torch.testing.assert_close(query_ref_out, query_cpu_out, atol=atol, rtol=rtol)
        torch.testing.assert_close(key_ref_out, key_cpu_out, atol=atol, rtol=rtol)

    def test_rope_2d(self):
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self._single_test(
                    hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh, dims=2
                )

    def test_rope_4d(self):
        for cfg in self.test_config:
            hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh = cfg
            with self.subTest(
                head_size=hs, rotary_dim=rd, is_neox=neox, batch=bs, seq=sl
            ):
                self._single_test(
                    hs, rd, mp, base, neox, dt, dev, bs, sl, qh, kvh, dims=4
                )


@unittest.skipUnless(
    _has_sgl_kernel_rope(),
    "sgl_kernel rotary_embedding_cpu not available (non-RISC-V build)",
)
class TestRVVDeepseekV2Rope(CustomTestCase):
    """RVV RoPE 3D test — DeepSeek V2 style (num_kv_heads=1)."""

    def test_deepseek_v2_rope(self):
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

        for dtype in [torch.bfloat16]:
            with torch.no_grad(), torch.amp.autocast("cpu", enabled=True):
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
                    cos_sin_cache,
                    False,
                )

                atol = rtol = precision[q_pe.dtype]
                torch.testing.assert_close(q_pe, q_pe_clone, atol=atol, rtol=rtol)
                torch.testing.assert_close(k_pe, k_pe_clone, atol=atol, rtol=rtol)


if __name__ == "__main__":
    unittest.main()
