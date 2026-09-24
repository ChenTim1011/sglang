import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.torch_native_llama import TorchNativeLlamaForCausalLM


class TestTorchNativeLlamaWeightLoading(unittest.TestCase):
    @staticmethod
    def _make_model():
        model = TorchNativeLlamaForCausalLM.__new__(TorchNativeLlamaForCausalLM)
        torch.nn.Module.__init__(model)
        model.config = SimpleNamespace(tie_word_embeddings=False)
        model.model = torch.nn.Module()
        model.model.embed_tokens = torch.nn.Embedding(3, 2)
        model.model.other = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            model.model.embed_tokens.weight.zero_()
            model.model.other.weight.zero_()
        return model

    def test_full_model_loader_reaches_nested_parameters(self):
        model = self._make_model()
        loaded_weight = torch.full_like(model.model.embed_tokens.weight, 2.0)

        model.load_weights(iter([("model.embed_tokens.weight", loaded_weight)]))

        torch.testing.assert_close(model.model.embed_tokens.weight, loaded_weight)

    def test_layered_loader_only_reaches_the_requested_module(self):
        model = self._make_model()
        embedding_weight = torch.full_like(model.model.embed_tokens.weight, 2.0)
        other_weight = torch.full_like(model.model.other.weight, 3.0)

        model.load_weights_to_module(
            "model.embed_tokens",
            iter(
                [
                    ("model.embed_tokens.weight", embedding_weight),
                    ("model.other.weight", other_weight),
                ]
            ),
        )

        torch.testing.assert_close(model.model.embed_tokens.weight, embedding_weight)
        torch.testing.assert_close(
            model.model.other.weight, torch.zeros_like(model.model.other.weight)
        )


if __name__ == "__main__":
    unittest.main()
