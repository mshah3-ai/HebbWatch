from __future__ import annotations

from pathlib import Path

from hebbwatch.replay.recorder import load_replay
from hebbwatch.render.heatmap import render_heatmap


def export_ascii(replay_path: str, output_dir: str) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    for idx, weights in enumerate(load_replay(replay_path), start=1):
        out.joinpath(f"frame_{idx:05d}.txt").write_text(render_heatmap(weights).plain, encoding="utf-8")
