from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Any


@dataclass
class WatchHandle:
    remove_callback: Callable[[], None] | None = None

    def remove(self) -> None:
        if self.remove_callback is not None:
            self.remove_callback()
