from __future__ import annotations

from pathlib import Path

import yaml

from hebbwatch.core.simulation import SimulationConfig


def load_config(path: str | Path) -> SimulationConfig:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return SimulationConfig(**data)
