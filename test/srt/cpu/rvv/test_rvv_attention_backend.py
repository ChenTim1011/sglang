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

    @rvv_benchmark(
        extra_args=["--batch-size", "1", "--mem-fraction-static", "0.6"],
        min_throughput=0.0,
    )
    def test_latency_default_model(self):
        return DEFAULT_SMALL_MODEL_NAME_FOR_TEST


if __name__ == "__main__":
    unittest.main()
