from __future__ import annotations

import numpy as np


def as_weight_matrix(array) -> np.ndarray:
    matrix = np.asarray(array, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("expected a 2D weight matrix")
    return matrix
