import cProfile

from hebbwatch.core.simulation import SimulationConfig, SimulationRunner


def main():
    runner = SimulationRunner(SimulationConfig(neurons=256, inputs=64, steps=1000))
    for _ in runner.states():
        pass


cProfile.run("main()", sort="cumtime")
