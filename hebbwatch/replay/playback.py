from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from hebbwatch.render.heatmap import render_heatmap
from hebbwatch.replay.recorder import load_replay


def play(path: str, fps: int = 24) -> None:
    console = Console()
    frames = load_replay(path)
    delay = 1.0 / max(1, fps)
    with Live(console=console, refresh_per_second=fps) as live:
        for index, weights in enumerate(frames, start=1):
            live.update(Panel(render_heatmap(weights), title=f"hebbwatch replay frame {index}"))
            time.sleep(delay)
