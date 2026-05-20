from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from hebbwatch.core.learning_rules import LearningRule


@dataclass
class NetworkState:
    step: int
    pre: np.ndarray
    post: np.ndarray
    weights: np.ndarray
    delta: np.ndarray


class HebbianNetwork:
    """Small recurrent-free Hebbian network for live learning visualization."""

    def __init__(
        self,
        neurons: int = 64,
        inputs: int = 8,
        rule: LearningRule | None = None,
        seed: int | None = None,
        activation: str = "tanh",
        noise: float = 0.03,
    ) -> None:
        if neurons <= 0 or inputs <= 0:
            raise ValueError("neurons and inputs must be positive")
        self.neurons = neurons
        self.inputs = inputs
        self.rule = rule
        self.rng = np.random.default_rng(seed)
        self.activation = activation
        self.noise = noise
        self.step_count = 0
        self.weights = self.rng.normal(0.0, 0.08, size=(neurons, inputs)).astype(np.float64)
        self.bias = self.rng.normal(0.0, 0.02, size=neurons).astype(np.float64)

    def _activate(self, values: np.ndarray) -> np.ndarray:
        if self.activation == "sigmoid":
            return 1.0 / (1.0 + np.exp(-values))
        if self.activation == "relu":
            return np.maximum(values, 0.0)
        if self.activation == "linear":
            return values
        return np.tanh(values)

    def step(self, pre: np.ndarray) -> NetworkState:
        if pre.shape != (self.inputs,):
            raise ValueError(f"expected input shape {(self.inputs,)}, got {pre.shape}")
        noise_vec = self.rng.normal(0.0, self.noise, size=self.neurons)
        raw = self.weights @ pre + self.bias + noise_vec
        post = self._activate(raw)
        if self.rule is None:
            delta = np.zeros_like(self.weights)
        else:
            delta = self.rule.delta(self.weights, pre, post)
            self.weights = self.weights + delta
            clip = getattr(self.rule, "clip", None)
            if clip is not None:
                self.weights = np.clip(self.weights, -float(clip), float(clip))
        self.step_count += 1
        return NetworkState(self.step_count, pre.copy(), post.copy(), self.weights.copy(), delta.copy())
