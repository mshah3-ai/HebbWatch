from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from hebbwatch.core.metrics import competition_index, sparsity, weight_entropy
from hebbwatch.core.network import NetworkState


@dataclass
class MetricsSnapshot:
    step: int
    mean_abs_weight: float
    max_abs_weight: float
    sparsity: float
    entropy: float
    competition: float


@dataclass
class MetricsTracker:
    history: list[MetricsSnapshot] = field(default_factory=list)

    def update(self, state: NetworkState) -> MetricsSnapshot:
        weights = state.weights
        snap = MetricsSnapshot(
            step=state.step,
            mean_abs_weight=float(np.mean(np.abs(weights))),
            max_abs_weight=float(np.max(np.abs(weights))),
            sparsity=sparsity(weights),
            entropy=weight_entropy(weights),
            competition=competition_index(weights),
        )
        self.history.append(snap)
        return snap
