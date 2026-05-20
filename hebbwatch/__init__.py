"""hebbwatch: real-time observability for neural learning dynamics."""

from hebbwatch.core.network import HebbianNetwork
from hebbwatch.core.learning_rules import HebbianRule, OjaRule
from hebbwatch.core.simulation import SimulationConfig, SimulationRunner

__version__ = "0.1.0"

__all__ = [
    "HebbianNetwork",
    "HebbianRule",
    "OjaRule",
    "SimulationConfig",
    "SimulationRunner",
]
