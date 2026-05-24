"""L1 mask loss matching Spleeter's EstimatorSpecBuilder._build_loss (L1_MASK path).

total_loss = sum_i mean(abs(predictions[i] - targets[i]))
per_stem  = {i: mean(abs(predictions[i] - targets[i])).item()}
"""

from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import Tensor


def l1_mask_loss(
    predictions: Dict[str, Tensor],
    targets: Dict[str, Tensor],
) -> Tuple[Tensor, Dict[str, float]]:
    if set(predictions.keys()) != set(targets.keys()):
        raise ValueError(
            f"prediction/target keys mismatch: {set(predictions)} vs {set(targets)}"
        )
    per_stem_t = {k: (predictions[k] - targets[k]).abs().mean() for k in predictions}
    total = torch.stack(list(per_stem_t.values())).sum()
    per_stem_scalars = {k: v.detach().float().item() for k, v in per_stem_t.items()}
    return total, per_stem_scalars
