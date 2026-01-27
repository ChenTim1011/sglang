"""
Usage:
python3 -m unittest test_rvv_benchmark.TestRVVBackend.test_latency_default_model
"""

import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    rvv_benchmark,
)


class TestRVVBackend(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        import os

        if "HF_TOKEN" not in os.environ:
            print("\n" + "=" * 60)
            print("WARNING: HF_TOKEN is not set!")
            print(
                "   If you are testing a Gated Model (e.g., Llama-3.2) and encounter:"
            )
            print("   '401 Client Error: Unauthorized' or 'GatedRepoError'")
            print("   Please run the following command before testing:")
            print("\n       export HF_TOKEN='hf_your_token_here'\n")
            print("=" * 60 + "\n")

    @rvv_benchmark(
        extra_args=[
            "--batch-size",
            "1",
            "--mem-fraction-static",
            "0.6",
            "--trust-remote-code",
            "--disable-radix",
            "--disable-overlap-schedule",
        ],
        min_throughput=0.0,
    )
    def test_latency_default_model(self):
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST


if __name__ == "__main__":
    unittest.main()
