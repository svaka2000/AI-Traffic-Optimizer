from __future__ import annotations

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class FixedTimingController(BaseController):
    def __init__(
        self,
        cycle_seconds: int = 60,
        green_split_ns: float = 0.55,
        step_seconds: float = 1.0,
    ) -> None:
        super().__init__(name="fixed_timing")
        self.cycle_seconds = cycle_seconds
        self.green_split_ns = green_split_ns
        self.step_seconds = step_seconds

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        cycle_steps = max(1, int(round(self.cycle_seconds / self.step_seconds)))
        ns_steps = max(1, int(round(cycle_steps * self.green_split_ns)))
        phase: SignalPhase = "NS" if (step % cycle_steps) < ns_steps else "EW"
        return {intersection_id: phase for intersection_id in observations}

