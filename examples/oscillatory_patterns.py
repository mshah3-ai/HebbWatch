from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.render.terminal import run_terminal

config = SimulationConfig(neurons=48, inputs=12, steps=480, input_mode="oscillatory", rule="oja")
run_terminal(SimulationRunner(config))
