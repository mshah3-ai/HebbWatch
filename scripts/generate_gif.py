from pathlib import Path

from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.render.animation import save_text_frames

out = Path("assets/ascii_frames")
runner = SimulationRunner(SimulationConfig(neurons=64, inputs=8, steps=120))
save_text_frames(runner.states(), out)
print(f"Saved ASCII frames to {out}")
