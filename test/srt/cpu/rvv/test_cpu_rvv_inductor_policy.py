import argparse
import gc
import os
import unittest
import weakref
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.compilation.cpu_linear_policy import apply_cpu_linear_policy
from sglang.srt.compilation.cpu_rvv_inductor_policy import (
    _RVV_REGIONAL_COMPILED_BY_SHAPE,
    _regional_bucket_rows,
    _should_use_rvv_inductor_regional_shape,
    install_rvv_inductor_regional_policy,
    invalidate_rvv_inductor_packed_weights,
)


class _TorchNativeLinear(torch.nn.Linear):
    def forward(self, input):
        return apply_cpu_linear_policy(self, input, self.bias)


class _Layer(torch.nn.Module):
    def __init__(
        self,
        *,
        qkv_size: tuple[int, int] = (8, 8),
        o_size: tuple[int, int] = (8, 8),
        gate_up_size: tuple[int, int] = (8, 16),
        down_size: tuple[int, int] = (16, 8),
    ):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.qkv_proj = _TorchNativeLinear(*qkv_size, bias=False)
        self.self_attn.o_proj = _TorchNativeLinear(*o_size, bias=False)
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = _TorchNativeLinear(*gate_up_size, bias=False)
        self.mlp.down_proj = _TorchNativeLinear(*down_size, bias=False)


class TorchNativeLlamaForCausalLM(torch.nn.Module):
    def __init__(self, layer: torch.nn.Module | None = None):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([layer or _Layer()])
        self.model.embed_tokens = torch.nn.Embedding(1024, 1024)
        self.lm_head = self.model.embed_tokens
        self.config = SimpleNamespace(tie_word_embeddings=True)


class UnquantizedLinearMethod:
    def apply(self, layer, input, bias=None):
        policy = getattr(layer, "_sglang_cpu_linear_policy", None)
        if policy is not None:
            return policy.apply(layer, input, bias)
        return torch.nn.functional.linear(input, layer.weight, bias)


class _Qwen2ParallelLinear(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, *, bias: bool = False):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.randn(out_features)) if bias else None
        self.skip_bias_add = False
        self.tp_size = 1
        self.quant_method = UnquantizedLinearMethod()

    def forward(self, input):
        bias = self.bias if not self.skip_bias_add else None
        output = self.quant_method.apply(self, input, bias)
        output_bias = self.bias if self.skip_bias_add else None
        return output, output_bias


class QKVParallelLinear(_Qwen2ParallelLinear):
    pass


class MergedColumnParallelLinear(_Qwen2ParallelLinear):
    pass


class RowParallelLinear(_Qwen2ParallelLinear):
    def forward(self, input, skip_all_reduce=False, forward_batch=None):
        del skip_all_reduce, forward_batch
        return super().forward(input)


class _Qwen2Layer(torch.nn.Module):
    def __init__(self, size: int = 8):
        super().__init__()
        self.self_attn = torch.nn.Module()
        self.self_attn.qkv_proj = QKVParallelLinear(size, size, bias=True)
        self.self_attn.o_proj = RowParallelLinear(size, size)
        self.mlp = torch.nn.Module()
        self.mlp.gate_up_proj = MergedColumnParallelLinear(size, size)
        self.mlp.down_proj = RowParallelLinear(size, size)


class Qwen2ForCausalLM(torch.nn.Module):
    def __init__(self, size: int = 8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Qwen2Layer(size)])
        self.model.embed_tokens = torch.nn.Embedding(size, size)
        self.lm_head = torch.nn.Linear(size, size, bias=False)
        self.config = SimpleNamespace(tie_word_embeddings=False)


class Qwen3ForCausalLM(torch.nn.Module):
    def __init__(self, size: int = 8):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Qwen2Layer(size)])
        self.model.embed_tokens = torch.nn.Embedding(size, size)
        self.lm_head = self.model.embed_tokens
        self.config = SimpleNamespace(tie_word_embeddings=True)


class _OtherModel(TorchNativeLlamaForCausalLM):
    pass


class TestCpuRvvInductorPolicy(unittest.TestCase):
    def _server_args(self, **overrides):
        args = {
            "device": "cpu",
            "dtype": "bfloat16",
            "attention_backend": "torch_native",
            "enable_torch_compile": False,
            "enable_cpu_rvv_inductor": False,
            "cpu_rvv_memory_budget_mib": None,
            "cpu_compile_mode": "regional",
            "cpu_rvv_regional_small_batch": False,
            "cpu_rvv_packed_weight_mode": "off",
            "enable_lora": False,
        }
        args.update(overrides)
        return SimpleNamespace(**args)

    def test_server_args_exposes_simple_rvv_inductor_opt_in(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args = parser.parse_args(["--model-path", "dummy"])
        opted_in_args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--enable-cpu-rvv-inductor",
                "--cpu-rvv-memory-budget-mib",
                "512",
            ]
        )

        self.assertFalse(default_args.enable_cpu_rvv_inductor)
        self.assertTrue(opted_in_args.enable_cpu_rvv_inductor)
        self.assertEqual(opted_in_args.cpu_rvv_memory_budget_mib, 512)

    def test_server_args_exposes_regional_mode_without_cpu_graph(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        args = parser.parse_args(
            ["--model-path", "dummy", "--cpu-compile-mode", "regional"]
        )

        self.assertEqual(args.cpu_compile_mode, "regional")
        self.assertFalse(args.enable_torch_compile)

    def test_server_args_exposes_opt_in_regional_small_batch(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args = parser.parse_args(["--model-path", "dummy"])
        opted_in_args = parser.parse_args(
            ["--model-path", "dummy", "--cpu-rvv-regional-small-batch"]
        )

        self.assertFalse(default_args.cpu_rvv_regional_small_batch)
        self.assertTrue(opted_in_args.cpu_rvv_regional_small_batch)

    def test_explicit_packed_weights_require_an_opt_in(self):
        from sglang.srt.server_args import ServerArgs

        parser = argparse.ArgumentParser()
        ServerArgs.add_cli_args(parser)

        default_args = parser.parse_args(["--model-path", "dummy"])
        explicit_args = parser.parse_args(
            [
                "--model-path",
                "dummy",
                "--cpu-rvv-packed-weight-mode",
                "explicit",
                "--cpu-rvv-packed-weight-max-mib",
                "512",
            ]
        )

        self.assertEqual(default_args.cpu_rvv_packed_weight_mode, "off")
        self.assertEqual(explicit_args.cpu_rvv_packed_weight_mode, "explicit")
        self.assertEqual(explicit_args.cpu_rvv_packed_weight_max_mib, 512)

    def test_regional_mode_does_not_pack_weights_without_explicit_opt_in(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                side_effect=lambda weight: weight.detach().clone(),
            ) as pack,
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                side_effect=torch.nn.functional.linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
                return_value="RVV",
            ),
        ):
            install_rvv_inductor_regional_policy(model, self._server_args())

        pack.assert_not_called()
        self.assertFalse(model._sglang_rvv_explicit_pack_stats["enabled"])

    def test_simple_opt_in_selects_regional_explicit_packing(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
                return_value="RVV",
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                side_effect=lambda weight: weight.detach().clone(),
            ) as pack,
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                side_effect=torch.nn.functional.linear,
            ),
        ):
            wrapped = install_rvv_inductor_regional_policy(
                model,
                self._server_args(
                    enable_cpu_rvv_inductor=True,
                    cpu_compile_mode="off",
                    cpu_rvv_packed_weight_mode="off",
                ),
            )

        self.assertEqual(wrapped, 4)
        self.assertTrue(model._sglang_rvv_explicit_pack_stats["requested"])
        self.assertTrue(model._sglang_rvv_explicit_pack_stats["enabled"])
        self.assertGreater(pack.call_count, 0)

    def test_simple_opt_in_rejects_a_non_rvv_cpu(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
                return_value="DEFAULT",
            ),
            self.assertRaisesRegex(RuntimeError, "capability is DEFAULT, not RVV"),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(
                    enable_cpu_rvv_inductor=True,
                    cpu_compile_mode="off",
                ),
            )

    def test_explicit_packing_respects_the_memory_budget(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)

        def eager_pack(weight):
            block_n = 32
            n, k = weight.shape
            padded_n = (n + block_n - 1) // block_n * block_n
            padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n))
            return (
                padded.reshape(padded_n // block_n, block_n, k)
                .permute(0, 2, 1)
                .contiguous()
            )

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                side_effect=eager_pack,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                side_effect=torch.nn.functional.linear,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(
                    cpu_rvv_packed_weight_mode="explicit",
                    cpu_rvv_packed_weight_max_mib=0.001,
                ),
                force=True,
            )

        stats = model._sglang_rvv_explicit_pack_stats
        self.assertLessEqual(stats["bytes"], stats["max_bytes"])
        self.assertEqual(stats["tensors"], 2)
        self.assertEqual(stats["skipped_tensors"], 3)
        self.assertEqual(stats["eligible_tensors"], 5)
        self.assertEqual(
            stats["eligible_bytes"], stats["bytes"] + stats["skipped_bytes"]
        )

    def test_explicit_packing_uses_available_memory_for_the_default_budget(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                side_effect=lambda weight: weight.detach().clone(),
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                side_effect=torch.nn.functional.linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.psutil.virtual_memory",
                return_value=SimpleNamespace(
                    available=1024 * 1024 * 1024,
                    total=8 * 1024 * 1024 * 1024,
                ),
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                force=True,
            )

        stats = model._sglang_rvv_explicit_pack_stats
        self.assertEqual(stats["budget_source"], "available_memory")
        self.assertEqual(stats["max_bytes"], 0)
        self.assertEqual(stats["tensors"], 0)
        self.assertEqual(stats["skipped_tensors"], stats["eligible_tensors"])

    def test_regional_shape_guard_rejects_unsupported_inputs(self):
        x = torch.randn(16, 1024, dtype=torch.bfloat16)
        weight = torch.randn(1024, 1024, dtype=torch.bfloat16)
        weight_fp32 = torch.randn(1024, 1024, dtype=torch.float32)
        bias = torch.randn(1024, dtype=torch.bfloat16)

        self.assertTrue(
            _should_use_rvv_inductor_regional_shape("qkv_proj", x, weight, None)
        )
        self.assertTrue(
            _should_use_rvv_inductor_regional_shape("qkv_proj", x[:1], weight, None)
        )
        self.assertFalse(
            _should_use_rvv_inductor_regional_shape("qkv_proj", x.float(), weight, None)
        )
        self.assertFalse(
            _should_use_rvv_inductor_regional_shape("qkv_proj", x, weight_fp32, None)
        )
        self.assertFalse(
            _should_use_rvv_inductor_regional_shape("qkv_proj", x, weight, bias)
        )
        self.assertFalse(
            _should_use_rvv_inductor_regional_shape(
                "qkv_proj", x, torch.randn(512, 1024, dtype=torch.bfloat16), None
            )
        )
        self.assertFalse(
            _should_use_rvv_inductor_regional_shape("lm_head", x, weight, None)
        )

    def test_regional_shape_guard_covers_decode_and_lm_head(self):
        decode = torch.randn(1, 1024, dtype=torch.bfloat16)
        projection = torch.randn(1024, 1024, dtype=torch.bfloat16)
        lm_head = torch.randn(65536, 1024, dtype=torch.bfloat16)

        self.assertTrue(
            _should_use_rvv_inductor_regional_shape(
                "qkv_proj", decode, projection, None
            )
        )
        self.assertTrue(
            _should_use_rvv_inductor_regional_shape("lm_head", decode, lm_head, None)
        )

    def test_regional_small_batch_routing_requires_opt_in(self):
        input = torch.randn(2, 1024, dtype=torch.bfloat16)
        weight = torch.randn(1024, 1024, dtype=torch.bfloat16)

        self.assertFalse(
            _should_use_rvv_inductor_regional_shape("qkv_proj", input, weight, None)
        )
        self.assertTrue(
            _should_use_rvv_inductor_regional_shape(
                "qkv_proj",
                input,
                weight,
                None,
                allow_small_batch=True,
            )
        )
        self.assertEqual(_regional_bucket_rows(2, allow_small_batch=True), 2)
        self.assertEqual(_regional_bucket_rows(3, allow_small_batch=True), 4)
        self.assertEqual(_regional_bucket_rows(15, allow_small_batch=True), 16)

    def test_installer_wraps_selected_linear_modules(self):
        model = TorchNativeLlamaForCausalLM()
        qkv_proj = model.model.layers[0].self_attn.qkv_proj
        forward_before = qkv_proj.forward.__func__

        self.assertEqual(install_rvv_inductor_regional_policy(model, force=True), 4)
        self.assertTrue(
            hasattr(
                model.model.layers[0].self_attn.qkv_proj,
                "_sglang_cpu_linear_policy",
            )
        )
        self.assertIs(qkv_proj.forward.__func__, forward_before)
        self.assertEqual(install_rvv_inductor_regional_policy(model, force=True), 0)

    def test_installer_supports_qwen2_with_untied_lm_head(self):
        model = Qwen2ForCausalLM().to(dtype=torch.bfloat16)
        lm_head_weight = model.lm_head.weight

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    model,
                    self._server_args(),
                ),
                4,
            )

        self.assertIs(model.lm_head.weight, lm_head_weight)
        self.assertIsNot(model.lm_head, model.model.embed_tokens)
        self.assertTrue(hasattr(model.lm_head, "_sglang_cpu_linear_policy"))
        self.assertTrue(
            hasattr(
                model.model.layers[0].self_attn.qkv_proj,
                "_sglang_cpu_linear_policy",
            )
        )

    def test_qwen2_wrapper_preserves_tuple_and_qkv_bias(self):
        model = Qwen2ForCausalLM(size=1024).to(dtype=torch.bfloat16)
        qkv_proj = model.model.layers[0].self_attn.qkv_proj
        hidden_states = torch.randn(16, 1024, dtype=torch.bfloat16)
        expected = qkv_proj(hidden_states)

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
            side_effect=lambda fn, *args, **kwargs: fn,
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            self.assertTrue(hasattr(qkv_proj, "_sglang_cpu_linear_policy"))
            actual = qkv_proj(hidden_states)

        self.assertIsInstance(actual, tuple)
        self.assertEqual(len(actual), 2)
        self.assertIsNone(actual[1])
        torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.02)

    def test_qwen2_wrapper_preserves_skip_bias_add(self):
        model = Qwen2ForCausalLM(size=1024).to(dtype=torch.bfloat16)
        qkv_proj = model.model.layers[0].self_attn.qkv_proj
        qkv_proj.skip_bias_add = True
        hidden_states = torch.randn(16, 1024, dtype=torch.bfloat16)

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
            side_effect=lambda fn, *args, **kwargs: fn,
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            actual, output_bias = qkv_proj(hidden_states)

        expected = torch.nn.functional.linear(
            hidden_states,
            qkv_proj.weight,
            None,
        )
        self.assertIs(output_bias, qkv_proj.bias)
        torch.testing.assert_close(actual, expected, rtol=0.02, atol=0.02)

    def test_qwen2_adapter_rejects_tensor_parallel_projection(self):
        model = Qwen2ForCausalLM()
        model.model.layers[0].self_attn.qkv_proj.tp_size = 2

        with self.assertRaisesRegex(RuntimeError, "requires TP=1"):
            install_rvv_inductor_regional_policy(model, force=True)

    def test_installer_supports_qwen3_with_tied_lm_head(self):
        model = Qwen3ForCausalLM().to(dtype=torch.bfloat16)
        lm_head_weight = model.lm_head.weight

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    model,
                    self._server_args(),
                ),
                4,
            )

        self.assertIs(model.lm_head.weight, lm_head_weight)
        self.assertIs(model.lm_head, model.model.embed_tokens)
        self.assertTrue(hasattr(model.lm_head, "_sglang_cpu_linear_policy"))
        self.assertTrue(
            hasattr(
                model.model.layers[0].self_attn.qkv_proj,
                "_sglang_cpu_linear_policy",
            )
        )

    def test_qwen2_down_projection_routes_with_forward_batch(self):
        model = Qwen2ForCausalLM(size=1024).to(dtype=torch.bfloat16)
        down_proj = model.model.layers[0].mlp.down_proj
        hidden_states = torch.randn(16, 1024, dtype=torch.bfloat16)
        expected = down_proj(hidden_states, forward_batch=object())

        with (
            mock.patch.dict(_RVV_REGIONAL_COMPILED_BY_SHAPE, clear=True),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            actual = down_proj(hidden_states, forward_batch=object())
            compiled_projections = {
                shape_key[2] for shape_key in _RVV_REGIONAL_COMPILED_BY_SHAPE
            }

        self.assertIn("down_proj", compiled_projections)
        torch.testing.assert_close(actual[0], expected[0], rtol=0.02, atol=0.02)
        self.assertIsNone(actual[1])

    def test_installer_preserves_tied_lm_head_weight(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)
        tied_module = model.model.embed_tokens
        weight = tied_module.weight
        identity_before = (weight.untyped_storage().data_ptr(), id(weight))

        install_rvv_inductor_regional_policy(model, force=True)

        self.assertIs(model.lm_head, tied_module)
        self.assertIs(model.lm_head.weight, weight)
        self.assertTrue(hasattr(model.lm_head, "_sglang_cpu_linear_policy"))
        identity_after = (weight.untyped_storage().data_ptr(), id(weight))
        self.assertEqual(identity_after, identity_before)

    def test_installer_owns_explicit_packed_side_buffers(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)
        modules = (
            model.model.layers[0].self_attn.qkv_proj,
            model.model.layers[0].self_attn.o_proj,
            model.model.layers[0].mlp.gate_up_proj,
            model.model.layers[0].mlp.down_proj,
            model.lm_head,
        )
        original_weights = tuple(module.weight for module in modules)
        original_tied_module = model.model.embed_tokens
        pack_calls = []

        def eager_pack(weight):
            block_n = 32
            pack_calls.append(weight)
            n, k = weight.shape
            padded_n = (n + block_n - 1) // block_n * block_n
            padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n))
            return (
                padded.reshape(padded_n // block_n, block_n, k)
                .permute(0, 2, 1)
                .contiguous()
            )

        def eager_packed_linear(input, packed_weight, out_features):
            n_blocks, k, block_n = packed_weight.shape
            weight = packed_weight.permute(0, 2, 1).reshape(n_blocks * block_n, k)
            return torch.nn.functional.linear(input, weight[:out_features])

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                eager_pack,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                eager_packed_linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_should_use_rvv_inductor_regional_shape",
                return_value=True,
            ),
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    model,
                    self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                    force=True,
                ),
                4,
            )
            linear = modules[0]
            input = torch.randn(1, linear.in_features, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(input, linear.weight)
            actual = linear(input)

        self.assertEqual(len(pack_calls), len(modules))
        torch.testing.assert_close(actual, expected)
        self.assertFalse(
            any(
                "_sglang_rvv_packed_weight" in name
                for name in model.state_dict().keys()
            )
        )
        self.assertIs(model.lm_head, original_tied_module)
        for module, original_weight in zip(modules, original_weights):
            self.assertIs(module.weight, original_weight)
            self.assertIn("_sglang_rvv_packed_weight", dict(module.named_buffers()))
            packed = module._sglang_rvv_packed_weight
            self.assertNotEqual(
                packed.untyped_storage().data_ptr(),
                original_weight.untyped_storage().data_ptr(),
            )
            self.assertIn(
                "_sglang_rvv_packed_weight", module._non_persistent_buffers_set
            )

    def test_explicit_invalidation_prevents_stale_data_after_weight_mutation(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj

        def eager_pack(weight):
            block_n = 32
            n, k = weight.shape
            padded_n = (n + block_n - 1) // block_n * block_n
            padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n))
            return (
                padded.reshape(padded_n // block_n, block_n, k)
                .permute(0, 2, 1)
                .contiguous()
            )

        def eager_packed_linear(input, packed_weight, out_features):
            n_blocks, k, block_n = packed_weight.shape
            weight = packed_weight.permute(0, 2, 1).reshape(n_blocks * block_n, k)
            return torch.nn.functional.linear(input, weight[:out_features])

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                eager_pack,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                eager_packed_linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                force=True,
            )
            with torch.no_grad():
                linear.weight.add_(1)
            invalidate_rvv_inductor_packed_weights(model)
            input = torch.randn(1, linear.in_features, dtype=torch.bfloat16)
            expected = torch.nn.functional.linear(input, linear.weight)
            actual = linear(input)

        torch.testing.assert_close(actual, expected)

    def test_reloaded_weight_falls_back_instead_of_using_stale_packed_data(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj
        packed_linear_calls = []

        def eager_pack(weight):
            block_n = 32
            n, k = weight.shape
            padded_n = (n + block_n - 1) // block_n * block_n
            padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n))
            return (
                padded.reshape(padded_n // block_n, block_n, k)
                .permute(0, 2, 1)
                .contiguous()
            )

        def eager_packed_linear(input, packed_weight, out_features):
            packed_linear_calls.append(packed_weight)
            n_blocks, k, block_n = packed_weight.shape
            weight = packed_weight.permute(0, 2, 1).reshape(n_blocks * block_n, k)
            return torch.nn.functional.linear(input, weight[:out_features])

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                eager_pack,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                eager_packed_linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                force=True,
            )
            input = torch.randn(1, linear.in_features, dtype=torch.bfloat16)
            linear(input)
            state = model.state_dict()
            state["model.layers.0.self_attn.qkv_proj.weight"] = torch.randn_like(
                linear.weight
            )
            model.load_state_dict(state)
            expected = torch.nn.functional.linear(input, linear.weight)
            actual = linear(input)

        self.assertEqual(len(packed_linear_calls), 1)
        torch.testing.assert_close(actual, expected)

    def test_replaced_or_moved_weight_falls_back_from_explicit_packed_data(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj
        packed_linear_calls = []

        def eager_pack(weight):
            block_n = 32
            n, k = weight.shape
            padded_n = (n + block_n - 1) // block_n * block_n
            padded = torch.nn.functional.pad(weight, (0, 0, 0, padded_n - n))
            return (
                padded.reshape(padded_n // block_n, block_n, k)
                .permute(0, 2, 1)
                .contiguous()
            )

        def eager_packed_linear(input, packed_weight, out_features):
            packed_linear_calls.append(packed_weight)
            n_blocks, k, block_n = packed_weight.shape
            weight = packed_weight.permute(0, 2, 1).reshape(n_blocks * block_n, k)
            return torch.nn.functional.linear(input, weight[:out_features])

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                eager_pack,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                eager_packed_linear,
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                force=True,
            )
            linear.weight = torch.nn.Parameter(torch.randn_like(linear.weight))
            input = torch.randn(1, linear.in_features, dtype=torch.bfloat16)
            torch.testing.assert_close(
                linear(input), torch.nn.functional.linear(input, linear.weight)
            )

            model.to(dtype=torch.float32).to(dtype=torch.bfloat16)
            linear = model.model.layers[0].self_attn.qkv_proj
            input = torch.randn(1, linear.in_features, dtype=torch.bfloat16)
            torch.testing.assert_close(
                linear(input), torch.nn.functional.linear(input, linear.weight)
            )

        self.assertEqual(packed_linear_calls, [])

    def test_explicit_packed_state_does_not_keep_a_destroyed_model_alive(self):
        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)
        weight_ref = weakref.ref(model.model.layers[0].self_attn.qkv_proj.weight)
        model_ref = weakref.ref(model)

        with (
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACK_BF16_WEIGHT_OP",
                side_effect=lambda weight: weight.detach().clone(),
            ),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy."
                "_RVV_PACKED_BF16_LINEAR_OP",
                side_effect=torch.nn.functional.linear,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(cpu_rvv_packed_weight_mode="explicit"),
                force=True,
            )

        del model
        gc.collect()
        self.assertIsNone(model_ref())
        self.assertIsNone(weight_ref())

    def test_installer_rejects_broken_tied_lm_head_alias(self):
        model = TorchNativeLlamaForCausalLM()
        model.lm_head = torch.nn.Embedding(1024, 1024)

        with self.assertRaisesRegex(RuntimeError, "tied lm_head"):
            install_rvv_inductor_regional_policy(model, force=True)

    def test_logits_processor_uses_regional_lm_head_without_mutating_weight(self):
        from sglang.srt.layers.logits_processor import LogitsProcessor

        model = TorchNativeLlamaForCausalLM().to(dtype=torch.bfloat16)
        hidden_states = torch.randn(1, 1024, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(hidden_states, model.lm_head.weight)
        data_ptr_before = model.lm_head.weight.untyped_storage().data_ptr()
        real_compile = torch.compile

        def compile_eager(fn, *args, **kwargs):
            return real_compile(fn, backend="eager", dynamic=False)

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
            side_effect=compile_eager,
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            actual = LogitsProcessor._compute_lm_head(
                SimpleNamespace(use_fp32_lm_head=False),
                hidden_states,
                model.lm_head,
            )

        torch.testing.assert_close(actual, expected)
        self.assertEqual(
            model.lm_head.weight.untyped_storage().data_ptr(), data_ptr_before
        )

    def test_installer_respects_policy_guards(self):
        cases = [
            ("", self._server_args(), 0),
            ("RVV", self._server_args(device="cuda"), 0),
            ("RVV", self._server_args(dtype="float16"), 0),
            ("RVV", self._server_args(attention_backend="flashinfer"), 0),
            ("RVV", self._server_args(cpu_compile_mode="off"), 0),
            ("RVV", self._server_args(enable_lora=True), 0),
        ]
        for capability, server_args, expected in cases:
            with self.subTest(capability=capability, server_args=server_args):
                model = TorchNativeLlamaForCausalLM()
                with mock.patch(
                    "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
                    return_value=capability,
                ):
                    self.assertEqual(
                        install_rvv_inductor_regional_policy(model, server_args),
                        expected,
                    )

    def test_installer_requires_exact_torch_native_model_type(self):
        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    _OtherModel(),
                    self._server_args(),
                ),
                0,
            )

    def test_installer_enables_on_rvv_cpu_policy(self):
        model = TorchNativeLlamaForCausalLM()
        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(model, self._server_args()),
                4,
            )

    def test_regional_mode_is_independent_from_cpu_graph(self):
        regional_model = TorchNativeLlamaForCausalLM()
        graph_model = TorchNativeLlamaForCausalLM()
        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    regional_model,
                    self._server_args(
                        cpu_compile_mode="regional", enable_torch_compile=False
                    ),
                ),
                4,
            )
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    graph_model,
                    self._server_args(
                        cpu_compile_mode="off", enable_torch_compile=True
                    ),
                ),
                0,
            )

    def test_installer_does_not_set_private_pytorch_environment_controls(self):
        model = TorchNativeLlamaForCausalLM()
        with (
            mock.patch.dict("os.environ", {}, clear=True),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
                return_value="RVV",
            ),
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(model, self._server_args()),
                4,
            )
            self.assertFalse(
                any(name.startswith("TORCHINDUCTOR_RVV_BF16_") for name in os.environ)
            )

    def test_installer_accepts_torch_typed_args(self):
        model = TorchNativeLlamaForCausalLM()
        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy._rvv_cpu_capability",
            return_value="RVV",
        ):
            self.assertEqual(
                install_rvv_inductor_regional_policy(
                    model,
                    self._server_args(device=torch.device("cpu"), dtype=torch.bfloat16),
                ),
                4,
            )

    def test_wrapper_matches_linear_with_eager_compile_backend(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj
        x = torch.randn(16, 1024, dtype=torch.bfloat16)
        expected = torch.nn.functional.linear(x, linear.weight, None)
        real_compile = torch.compile

        def compile_eager(fn, *args, **kwargs):
            return real_compile(fn, backend="eager", dynamic=False)

        with mock.patch(
            "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
            side_effect=compile_eager,
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            actual = linear(x)

        torch.testing.assert_close(actual, expected)

    def test_simple_opt_in_uses_m2_regional_buckets(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj
        hidden_states = torch.randn(2, 1024, dtype=torch.bfloat16)
        expected_linear = torch.nn.functional.linear(hidden_states, linear.weight, None)
        expected_logits = torch.nn.functional.linear(
            hidden_states, model.lm_head.weight, None
        )

        with (
            mock.patch.dict(_RVV_REGIONAL_COMPILED_BY_SHAPE, clear=True),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=lambda fn, *args, **kwargs: fn,
            ),
        ):
            install_rvv_inductor_regional_policy(
                model,
                self._server_args(
                    enable_cpu_rvv_inductor=True,
                    cpu_compile_mode="off",
                    cpu_rvv_regional_small_batch=False,
                ),
                force=True,
            )
            actual_linear = linear(hidden_states)
            actual_logits = model.lm_head._sglang_cpu_linear_policy.apply(
                model.lm_head, hidden_states, None
            )
            compiled_targets = {
                (shape_key[2], shape_key[3])
                for shape_key in _RVV_REGIONAL_COMPILED_BY_SHAPE
            }

        torch.testing.assert_close(actual_linear, expected_linear)
        torch.testing.assert_close(actual_logits, expected_logits)
        self.assertIn(("qkv_proj", 2), compiled_targets)
        self.assertIn(("lm_head", 2), compiled_targets)

    def test_regional_targets_do_not_share_one_dynamo_recompile_budget(self):
        layer = _Layer(
            qkv_size=(8, 9),
            o_size=(10, 11),
            gate_up_size=(12, 13),
            down_size=(14, 15),
        )
        model = TorchNativeLlamaForCausalLM(layer).to(dtype=torch.bfloat16)
        model.model.embed_tokens = torch.nn.Embedding(17, 16).to(dtype=torch.bfloat16)
        model.lm_head = model.model.embed_tokens
        real_compile = torch.compile

        def compile_eager(fn, *args, **kwargs):
            return real_compile(fn, backend="eager", dynamic=False, fullgraph=True)

        modules = (
            (layer.self_attn.qkv_proj, 8),
            (layer.self_attn.o_proj, 10),
            (layer.mlp.gate_up_proj, 12),
            (layer.mlp.down_proj, 14),
        )
        torch._dynamo.reset()
        try:
            with (
                mock.patch(
                    "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                    side_effect=compile_eager,
                ),
                mock.patch(
                    "sglang.srt.compilation.cpu_rvv_inductor_policy."
                    "_should_use_rvv_inductor_regional_shape",
                    return_value=True,
                ),
            ):
                install_rvv_inductor_regional_policy(model, force=True)
                for rows in (1, 16):
                    for module, in_features in modules:
                        hidden_states = torch.randn(
                            rows, in_features, dtype=torch.bfloat16
                        )
                        module(hidden_states)
                    model.lm_head._sglang_cpu_linear_policy.apply(
                        model.lm_head,
                        torch.randn(1, 16, dtype=torch.bfloat16),
                        None,
                    )
        finally:
            torch._dynamo.reset()

    def test_prefill_rows_share_bounded_regional_bucket(self):
        model = TorchNativeLlamaForCausalLM(_Layer(qkv_size=(1024, 1024))).to(
            dtype=torch.bfloat16
        )
        linear = model.model.layers[0].self_attn.qkv_proj
        inputs = (
            torch.randn(33, 1024, dtype=torch.bfloat16),
            torch.randn(34, 1024, dtype=torch.bfloat16),
        )
        expected = [torch.nn.functional.linear(x, linear.weight) for x in inputs]
        real_compile = torch.compile

        def compile_eager(fn, *args, **kwargs):
            return real_compile(fn, backend="eager", dynamic=False, fullgraph=True)

        with (
            mock.patch.dict(_RVV_REGIONAL_COMPILED_BY_SHAPE, clear=True),
            mock.patch(
                "sglang.srt.compilation.cpu_rvv_inductor_policy.torch.compile",
                side_effect=compile_eager,
            ) as compile_mock,
        ):
            install_rvv_inductor_regional_policy(model, force=True)
            actual = [linear(x) for x in inputs]

        self.assertEqual(compile_mock.call_count, 1)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference)


if __name__ == "__main__":
    unittest.main()
