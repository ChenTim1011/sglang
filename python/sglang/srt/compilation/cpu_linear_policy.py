# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
from torch.nn import functional as F


def apply_cpu_linear_policy(
    layer: torch.nn.Module,
    input: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run an installed CPU linear policy, or preserve eager Linear behavior."""
    policy = getattr(layer, "_sglang_cpu_linear_policy", None)
    if policy is None:
        return F.linear(input, layer.weight, bias)
    return policy.apply(layer, input, bias)
