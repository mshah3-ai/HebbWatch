from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hebbwatch.core.network import NetworkState


class ReplayRecorder:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.frames: list[dict] = []

    def add(self, state: NetworkState) -> None:
        self.frames.append({"step": state.step, "weights": state.weights.round(6).tolist()})

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({"format": "hebbwatch-replay-v1", "frames": self.frames}), encoding="utf-8")


def load_replay(path: str | Path) -> list[np.ndarray]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [np.array(frame["weights"], dtype=float) for frame in payload["frames"]]
