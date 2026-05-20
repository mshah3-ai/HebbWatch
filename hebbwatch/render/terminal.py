from __future__ import annotations

import time

from rich.console import Console
from rich.live import Live

from hebbwatch.core.simulation import SimulationRunner
from hebbwatch.render.layout import build_live_layout


def run_terminal(runner: SimulationRunner, width: int = 64, height: int = 28, record: bool = False):
    console = Console()
    delay = 1.0 / max(1, runner.config.fps)
    last_state = None
    with Live(console=console, refresh_per_second=runner.config.fps, screen=False) as live:
        for state in runner.states():
            last_state = state
            live.update(build_live_layout(state, runner.config.rule, runner.config.input_mode, width, height))
            time.sleep(delay)
    return last_state
