from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from hebbwatch.config import load_config
from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.render.terminal import run_terminal
from hebbwatch.replay.playback import play
from hebbwatch.replay.recorder import ReplayRecorder
from hebbwatch.utils import seconds_to_steps

app = typer.Typer(help="Watch synaptic weights change in real time as a neural network learns.")
console = Console()


@app.command()
def run(
    neurons: int = typer.Option(64, help="Number of post-synaptic neurons."),
    inputs: int = typer.Option(8, help="Number of input dimensions."),
    duration: float = typer.Option(30.0, help="Duration in seconds."),
    fps: int = typer.Option(24, help="Terminal refresh rate."),
    seed: Optional[int] = typer.Option(7, help="Random seed. Use --seed none for random."),
    learning_rate: float = typer.Option(0.015, help="Synaptic learning rate."),
    rule: str = typer.Option("competitive", help="hebbian, oja, or competitive."),
    input_mode: str = typer.Option("correlated", help="random, sparse, correlated, oscillatory, drifting."),
    width: int = typer.Option(64, help="Heatmap width in terminal cells."),
    height: int = typer.Option(28, help="Heatmap height in terminal rows."),
    record: Optional[Path] = typer.Option(None, help="Optional replay output path, e.g. recordings/run.hw."),
) -> None:
    steps = seconds_to_steps(duration, fps)
    config = SimulationConfig(
        neurons=neurons,
        inputs=inputs,
        steps=steps,
        fps=fps,
        seed=seed,
        learning_rate=learning_rate,
        rule=rule,
        input_mode=input_mode,
    )
    runner = SimulationRunner(config)
    if record is None:
        run_terminal(runner, width=width, height=height)
        return

    recorder = ReplayRecorder(record)
    original_states = runner.states

    def recording_states():
        for state in original_states():
            recorder.add(state)
            yield state

    runner.states = recording_states  # type: ignore[method-assign]
    run_terminal(runner, width=width, height=height)
    recorder.save()
    console.print(f"Saved replay to [bold]{record}[/bold]")


@app.command()
def replay(path: Path, fps: int = typer.Option(24, help="Playback FPS.")) -> None:
    play(str(path), fps=fps)


@app.command()
def preset(path: Path, width: int = 64, height: int = 28) -> None:
    config = load_config(path)
    runner = SimulationRunner(config)
    run_terminal(runner, width=width, height=height)


@app.command()
def info() -> None:
    table = Table(title="hebbwatch")
    table.add_column("Command")
    table.add_column("Purpose")
    table.add_row("hebbwatch run", "Start a live Hebbian learning simulation")
    table.add_row("hebbwatch replay <file.hw>", "Replay a saved learning session")
    table.add_row("hebbwatch preset <file.yaml>", "Run a YAML preset")
    console.print(table)


if __name__ == "__main__":
    app()
