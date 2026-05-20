from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

from hebbwatch.core.learning_rules import build_learning_rule
from hebbwatch.core.network import HebbianNetwork, NetworkState
from hebbwatch.inputs.generators import build_input_generator


@dataclass(frozen=True)
class SimulationConfig:
    neurons: int = 64
    inputs: int = 8
    steps: int = 900
    fps: int = 24
    seed: int | None = 7
    learning_rate: float = 0.015
    rule: str = "competitive"
    input_mode: str = "correlated"
    noise: float = 0.03


class SimulationRunner:
    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        rule = build_learning_rule(config.rule, config.learning_rate)
        self.network = HebbianNetwork(
            neurons=config.neurons,
            inputs=config.inputs,
            rule=rule,
            seed=config.seed,
            noise=config.noise,
        )
        self.input_generator = build_input_generator(config.input_mode, config.inputs, config.seed)

    def states(self) -> Iterator[NetworkState]:
        for _ in range(self.config.steps):
            yield self.network.step(next(self.input_generator))
