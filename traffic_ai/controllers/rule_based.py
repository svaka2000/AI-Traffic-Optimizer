"""traffic_ai/controllers/rule_based.py

RuleBasedController: adaptive controller that extends the green phase when the
current axis is heavily loaded and switches when the opposite axis has more
waiting vehicles.
"""
from __future__ import annotations

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class RuleBasedController(BaseController):
    """Queue-based adaptive signal controller.

    Logic per intersection at each step:
    1. If the current phase has been green for < min_green steps → keep it.
    2. If the current phase has been green for ≥ max_green steps → force switch.
    3. Otherwise, switch if the opposite axis queue exceeds the current axis
       queue by more than ``threshold`` vehicles.

    Parameters
    ----------
    min_green:
        Minimum steps a phase must stay green.
    max_green:
        Maximum steps a phase can stay green before a forced switch.
    threshold:
        Minimum queue imbalance (vehicles) required to trigger a switch.
    """

    def __init__(
        self,
        min_green: int = 10,
        max_green: int = 60,
        threshold: float = 5.0,
    ) -> None:
        super().__init__(name="rule_based")
        self.min_green = min_green
        self.max_green = max_green
        self.threshold = threshold
        self._current_phase: dict[int, int] = {}  # 0=NS, 1=EW
        self._green_elapsed: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._green_elapsed = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            phase_int = self._decide(iid, obs)
            actions[iid] = "NS" if phase_int == 0 else "EW"
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        return self._decide(0, obs)

    # ------------------------------------------------------------------
    # Core decision logic
    # ------------------------------------------------------------------

    def _decide(self, iid: int, obs: dict[str, float]) -> int:
        if iid not in self._current_phase:
            self._current_phase[iid] = 0
            self._green_elapsed[iid] = 0

        phase = self._current_phase[iid]
        elapsed = self._green_elapsed[iid] + 1

        queue_ns = float(obs.get("queue_ns", 0.0))
        queue_ew = float(obs.get("queue_ew", 0.0))

        # Current axis and opposite axis queues
        current_queue = queue_ns if phase == 0 else queue_ew
        opposite_queue = queue_ew if phase == 0 else queue_ns

        if elapsed >= self.max_green:
            # Force switch
            phase = 1 - phase
            elapsed = 0
        elif elapsed >= self.min_green:
            # Switch if opposite axis has notably more vehicles
            if opposite_queue - current_queue > self.threshold:
                phase = 1 - phase
                elapsed = 0

        self._current_phase[iid] = phase
        self._green_elapsed[iid] = elapsed
        return phase
