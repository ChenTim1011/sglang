import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn

from sglang.srt.model_loader.loader import LayeredModelLoader
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _TiedWeightModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Linear(4, 4, bias=False)
        self.lm_head = self.model.embed_tokens

    def load_weights_to_module(self, fqn, weights):
        module = self.get_submodule(fqn)
        params = dict(module.named_parameters(prefix=fqn, recurse=False))
        for name, loaded_weight in weights:
            if name in params:
                params[name].data.copy_(loaded_weight)


class TestLayeredModelLoader(CustomTestCase):
    def test_shared_module_is_materialized_only_once(self):
        loader = object.__new__(LayeredModelLoader)
        loader.load_config = SimpleNamespace()
        expected = torch.arange(16, dtype=torch.float32).reshape(4, 4)

        with (
            patch(
                "sglang.srt.model_loader.loader._initialize_model",
                side_effect=lambda *args, **kwargs: _TiedWeightModel(),
            ),
            patch(
                "sglang.srt.model_loader.loader._get_quantization_config",
                return_value=None,
            ),
            patch.object(
                loader,
                "_get_all_weights",
                return_value=[("model.embed_tokens.weight", expected)],
            ),
            patch(
                "sglang.srt.runtime_context.get_server_args",
                return_value=SimpleNamespace(torchao_config=None),
            ),
        ):
            model = loader.load_model(
                model_config=SimpleNamespace(
                    dtype=torch.float32,
                    model_path="tied-weight-test",
                ),
                device_config=SimpleNamespace(device="cpu"),
            )

        self.assertIs(model.lm_head, model.model.embed_tokens)
        torch.testing.assert_close(model.model.embed_tokens.weight, expected)


if __name__ == "__main__":
    unittest.main()
