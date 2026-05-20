from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class LearningRule(Protocol):
    name: str

    def delta(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        ...


@dataclass(frozen=True)
class HebbianRule:
    """Classic Hebbian update: weights increase when pre and post co-activate."""

    learning_rate: float = 0.015
    decay: float = 0.001
    clip: float = 2.0
    name: str = "hebbian"

    def delta(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        growth = self.learning_rate * np.outer(post, pre)
        decay = self.decay * weights
        return growth - decay


@dataclass(frozen=True)
class OjaRule:
    """Stable Hebbian rule with normalization pressure."""

    learning_rate: float = 0.012
    decay: float = 0.0005
    clip: float = 2.0
    name: str = "oja"

    def delta(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        hebbian = np.outer(post, pre)
        normalization = (post[:, None] ** 2) * weights
        return self.learning_rate * (hebbian - normalization) - self.decay * weights


@dataclass(frozen=True)
class CompetitiveHebbianRule:
    """Hebbian learning with lateral competition, producing cleaner assemblies."""

    learning_rate: float = 0.018
    decay: float = 0.0015
    winner_boost: float = 1.8
    clip: float = 2.5
    name: str = "competitive"

    def delta(self, weights: np.ndarray, pre: np.ndarray, post: np.ndarray) -> np.ndarray:
        gated = post.copy()
        if gated.size:
            winner = int(np.argmax(gated))
            gated *= 0.35
            gated[winner] *= self.winner_boost
        return self.learning_rate * np.outer(gated, pre) - self.decay * weights


def build_learning_rule(name: str, learning_rate: float) -> LearningRule:
    rule = name.lower().strip()
    if rule == "hebbian":
        return HebbianRule(learning_rate=learning_rate)
    if rule == "oja":
        return OjaRule(learning_rate=learning_rate)
    if rule in {"competitive", "competition"}:
        return CompetitiveHebbianRule(learning_rate=learning_rate)
    raise ValueError(f"Unknown learning rule '{name}'. Choose hebbian, oja, or competitive.")
