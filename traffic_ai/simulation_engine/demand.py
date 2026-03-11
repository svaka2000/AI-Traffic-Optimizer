from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal


DemandProfileName = Literal["normal", "rush_hour", "midday_peak"]


@dataclass(slots=True)
class DemandModel:
    profile: DemandProfileName = "normal"
    scale: float = 1.0
    step_seconds: float = 1.0

    def arrival_rate_per_lane(self, step: int, direction: str) -> float:
        # Base rate in vehicles/second/lane.
        base = 0.12
        direction_multiplier = 1.1 if direction in ("N", "S") else 1.0
        hour = self._hour_of_day(step)

        if self.profile == "rush_hour":
            morning = math.exp(-((hour - 8.0) ** 2) / 3.0)
            evening = math.exp(-((hour - 17.5) ** 2) / 3.0)
            peak = 1.0 + 1.6 * max(morning, evening)
        elif self.profile == "midday_peak":
            peak = 1.0 + 1.2 * math.exp(-((hour - 13.0) ** 2) / 4.0)
        else:
            peak = 1.0 + 0.25 * math.sin(2 * math.pi * hour / 24.0)

        return max(0.02, base * direction_multiplier * peak * self.scale)

    def _hour_of_day(self, step: int) -> float:
        elapsed_seconds = step * self.step_seconds
        return (elapsed_seconds / 3600.0) % 24.0

