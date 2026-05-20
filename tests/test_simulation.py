from hebbwatch.core.simulation import SimulationConfig, SimulationRunner


def test_simulation_runs_requested_steps():
    runner = SimulationRunner(SimulationConfig(neurons=8, inputs=3, steps=5))
    states = list(runner.states())
    assert len(states) == 5
    assert states[-1].weights.shape == (8, 3)
