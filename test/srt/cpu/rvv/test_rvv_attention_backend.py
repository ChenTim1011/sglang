"""Unit tests for RVV attention backend runtime behavior.

Usage:
    python3 -m unittest test.srt.cpu.rvv.test_rvv_attention_backend -v
"""

import os
import unittest

from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
    rvv_benchmark,
)


class TestRVVAttentionBackendBenchmark(CustomTestCase):
    """Test suite for RVV attention backend benchmark wiring."""

    @classmethod
    def setUpClass(cls):
        """Prepare environment warnings before benchmark execution."""
        super().setUpClass()

        if "HF_TOKEN" not in os.environ:
            print("\n" + "=" * 60)
            print("WARNING: HF_TOKEN is not set!")
            print("   If you are testing a gated model and encounter:")
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
            "--dtype",
            "float16",
        ],
        min_throughput=0.0,
    )
    def test_case_latency_default_model(self):
        """Case: run the default small model via RVV benchmark decorator."""
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST


if __name__ == "__main__":
    unittest.main()
