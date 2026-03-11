from __future__ import annotations

import numpy as np

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class SupervisedMLController(BaseController):
    def __init__(
        self,
        model: object,
        min_green: int = 8,
    ) -> None:
        super().__init__(name=f"ml_{model.__class__.__name__.lower()}")
        self.model = model
        self.min_green = min_green
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
            phase = self.current_phase[intersection_id]
            elapsed = self.green_elapsed[intersection_id] + 1
            if elapsed < self.min_green:
                actions[intersection_id] = phase
                self.green_elapsed[intersection_id] = elapsed
                continue

            feature_vector = np.array(
                [
                    obs.get("queue_ns", 0.0),
                    obs.get("queue_ew", 0.0),
                    obs.get("total_queue", 0.0),
                    obs.get("phase_elapsed", 0.0),
                    obs.get("phase_ns", 0.0),
                    obs.get("phase_ew", 0.0),
                    obs.get("sim_step", 0.0),
                    obs.get("wait_sec", 0.0),
                ],
                dtype=np.float64,
            ).reshape(1, -1)
            pred = int(self.model.predict(feature_vector)[0])
            next_phase: SignalPhase = "NS" if pred == 0 else "EW"
            if next_phase != phase:
                elapsed = 0
            self.current_phase[intersection_id] = next_phase
            self.green_elapsed[intersection_id] = elapsed
            actions[intersection_id] = next_phase
        return actions

