from __future__ import annotations

import numpy as np


def sparsity(weights: np.ndarray, threshold: float = 0.05) -> float:
    return float(np.mean(np.abs(weights) < threshold))


def weight_entropy(weights: np.ndarray) -> float:
    vals = np.abs(weights).ravel()
    total = vals.sum()
    if total <= 1e-12:
        return 0.0
    p = vals / total
    p = p[p > 1e-12]
    return float(-(p * np.log2(p)).sum())


def competition_index(weights: np.ndarray) -> float:
    vals = np.sort(np.abs(weights).ravel())
    if vals.size == 0 or vals.sum() <= 1e-12:
        return 0.0
    n = vals.size
    gini = (2 * np.arange(1, n + 1) - n - 1) @ vals / (n * vals.sum())
    return float(np.clip(gini, 0.0, 1.0))


def dead_neurons(weights: np.ndarray, threshold: float = 0.03) -> int:
    row_strength = np.mean(np.abs(weights), axis=1)
    return int(np.sum(row_strength < threshold))


def dominant_connections(weights: np.ndarray, percentile: float = 95.0) -> int:
    cutoff = np.percentile(np.abs(weights), percentile)
    return int(np.sum(np.abs(weights) >= cutoff))
