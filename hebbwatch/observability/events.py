from __future__ import annotations

from dataclasses import dataclass

from hebbwatch.core.network import NetworkState
from hebbwatch.observability.collapse import inactive_rows, saturated_rows


@dataclass(frozen=True)
class LearningEvent:
    step: int
    kind: str
    message: str


def detect_events(state: NetworkState) -> list[LearningEvent]:
    events: list[LearningEvent] = []
    inactive = inactive_rows(state.weights)
    saturated = saturated_rows(state.weights)
    if inactive:
        events.append(LearningEvent(state.step, "inactive", f"{inactive} low-activity neurons"))
    if saturated:
        events.append(LearningEvent(state.step, "saturated", f"{saturated} saturated neurons"))
    return events
