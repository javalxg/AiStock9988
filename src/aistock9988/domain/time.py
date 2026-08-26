from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True)
class PITWindow:
    """Point-in-time boundary used by every data provider."""

    decision_time: datetime

    def allows(self, available_time: datetime) -> bool:
        return available_time <= self.decision_time

    def require(self, available_time: datetime) -> None:
        if not self.allows(available_time):
            raise ValueError(
                f"PIT violation: available_time={available_time.isoformat()} "
                f"> decision_time={self.decision_time.isoformat()}"
            )
