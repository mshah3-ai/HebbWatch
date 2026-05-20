from __future__ import annotations

import numpy as np


def saturated_rows(weights: np.ndarray, threshold: float = 1.9) -> int:
    return int(np.sum(np.max(np.abs(weights), axis=1) > threshold))


def inactive_rows(weights: np.ndarray, threshold: float = 0.03) -> int:
    return int(np.sum(np.mean(np.abs(weights), axis=1) < threshold))
