from __future__ import annotations

import numpy as np
from rich.text import Text

from hebbwatch.render.colors import BLOCKS


def downsample(matrix: np.ndarray, max_rows: int = 32, max_cols: int = 64) -> np.ndarray:
    rows, cols = matrix.shape
    row_stride = max(1, int(np.ceil(rows / max_rows)))
    col_stride = max(1, int(np.ceil(cols / max_cols)))
    trimmed = matrix[: rows - rows % row_stride or rows, : cols - cols % col_stride or cols]
    if trimmed.size == 0:
        return matrix
    new_rows = trimmed.shape[0] // row_stride
    new_cols = trimmed.shape[1] // col_stride
    pooled = trimmed.reshape(new_rows, row_stride, new_cols, col_stride).mean(axis=(1, 3))
    return pooled


def render_heatmap(matrix: np.ndarray, width: int = 64, height: int = 28) -> Text:
    view = downsample(matrix, height, width)
    scale = float(np.max(np.abs(view))) if view.size else 1.0
    if scale < 1e-9:
        scale = 1.0
    normalized = np.clip(np.abs(view) / scale, 0.0, 1.0)
    text = Text()
    for row in normalized:
        for val in row:
            idx = min(len(BLOCKS) - 1, int(round(val * (len(BLOCKS) - 1))))
            text.append(BLOCKS[idx])
        text.append("\n")
    return text
