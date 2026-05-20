from __future__ import annotations


def seconds_to_steps(duration: float, fps: int) -> int:
    return max(1, int(round(duration * fps)))
