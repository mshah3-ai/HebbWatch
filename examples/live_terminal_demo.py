from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.render.terminal import run_terminal

config = SimulationConfig(neurons=64, inputs=8, steps=360, fps=24)
run_terminal(SimulationRunner(config))
