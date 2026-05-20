from hebbwatch.core.simulation import SimulationConfig, SimulationRunner
from hebbwatch.core.metrics import competition_index

config = SimulationConfig(neurons=32, inputs=8, steps=100, rule="competitive")
runner = SimulationRunner(config)

last = None
for last in runner.states():
    pass

print("final competition:", competition_index(last.weights))
