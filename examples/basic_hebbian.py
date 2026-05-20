from hebbwatch.core.simulation import SimulationConfig, SimulationRunner

config = SimulationConfig(neurons=16, inputs=4, steps=10, rule="hebbian", input_mode="correlated")
runner = SimulationRunner(config)

for state in runner.states():
    print(state.step, state.weights.mean())
