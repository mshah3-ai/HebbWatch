from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def random_input(inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    while True:
        yield rng.normal(0.0, 1.0, size=inputs)


def sparse_input(inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    active = max(1, inputs // 4)
    while True:
        x = np.zeros(inputs)
        idx = rng.choice(inputs, size=active, replace=False)
        x[idx] = rng.uniform(0.6, 1.2, size=active)
        yield x + rng.normal(0.0, 0.03, size=inputs)


def correlated_input(inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    groups = max(2, min(4, inputs))
    assignments = np.arange(inputs) % groups
    centers = rng.normal(0.0, 0.4, size=(groups, inputs))
    for g in range(groups):
        centers[g, assignments == g] += rng.uniform(0.8, 1.4)
    t = 0
    while True:
        group = (t // 18 + rng.integers(0, groups)) % groups
        x = centers[group] + rng.normal(0.0, 0.12, size=inputs)
        yield np.tanh(x)
        t += 1


def oscillatory_input(inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0, 2 * np.pi, size=inputs)
    freq = rng.uniform(0.015, 0.08, size=inputs)
    t = 0
    while True:
        signal = np.sin(t * freq + phase) + 0.35 * np.sin(t * freq * 2.0 + phase / 2)
        yield signal + rng.normal(0.0, 0.04, size=inputs)
        t += 1


def drifting_input(inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    rng = np.random.default_rng(seed)
    center = rng.normal(0.0, 0.5, size=inputs)
    while True:
        center = 0.995 * center + rng.normal(0.0, 0.025, size=inputs)
        yield np.tanh(center + rng.normal(0.0, 0.10, size=inputs))


def build_input_generator(mode: str, inputs: int, seed: int | None = None) -> Iterator[np.ndarray]:
    key = mode.lower().strip()
    if key == "random":
        return random_input(inputs, seed)
    if key == "sparse":
        return sparse_input(inputs, seed)
    if key == "correlated":
        return correlated_input(inputs, seed)
    if key == "oscillatory":
        return oscillatory_input(inputs, seed)
    if key == "drifting":
        return drifting_input(inputs, seed)
    raise ValueError("Unknown input mode. Choose random, sparse, correlated, oscillatory, or drifting.")
