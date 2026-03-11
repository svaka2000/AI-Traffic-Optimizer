from __future__ import annotations

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class AdaptiveRuleController(BaseController):
    def __init__(
        self,
        min_green: int = 15,
        max_green: int = 75,
        queue_threshold: float = 6.0,
    ) -> None:
        super().__init__(name="adaptive_rule")
        self.min_green = min_green
        self.max_green = max_green
        self.queue_threshold = queue_threshold
        self.current_phase: dict[int, SignalPhase] = {}
        self.green_elapsed: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self.current_phase = {i: "NS" for i in range(n_intersections)}
        self.green_elapsed = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for intersection_id, obs in observations.items():
            phase = self.current_phase.get(intersection_id, "NS")
            elapsed = self.green_elapsed.get(intersection_id, 0) + 1
            queue_ns = obs.get("queue_ns", 0.0)
            queue_ew = obs.get("queue_ew", 0.0)
            imbalance = queue_ns - queue_ew
            target = "NS" if imbalance >= 0 else "EW"

            if elapsed >= self.max_green:
                phase = "EW" if phase == "NS" else "NS"
                elapsed = 0
            elif elapsed >= self.min_green:
                if target != phase and abs(imbalance) > self.queue_threshold:
                    phase = target
                    elapsed = 0

            self.current_phase[intersection_id] = phase
            self.green_elapsed[intersection_id] = elapsed
            actions[intersection_id] = phase
        return actions

