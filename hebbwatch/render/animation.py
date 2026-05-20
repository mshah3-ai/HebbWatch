from __future__ import annotations

from pathlib import Path
from typing import Iterable

from hebbwatch.core.network import NetworkState
from hebbwatch.render.heatmap import render_heatmap


def save_text_frames(states: Iterable[NetworkState], output_dir: str | Path) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for state in states:
        frame = render_heatmap(state.weights)
        (path / f"frame_{state.step:05d}.txt").write_text(frame.plain, encoding="utf-8")
