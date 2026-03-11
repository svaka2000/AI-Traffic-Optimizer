"""traffic_ai/controllers/fixed.py

FixedTimingController: cycles signal phases on a fixed N-second schedule.
Implements both the legacy compute_actions interface and the new select_action interface.
"""
from __future__ import annotations

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class FixedTimingController(BaseController):
    """Cycle signal phases on a fixed schedule.

    Parameters
    ----------
    cycle_seconds:
        Total cycle duration in seconds. The phase switches every
        ``cycle_seconds / 2`` seconds by default when ``green_split`` is 0.5.
    green_split:
        Fraction of the cycle given to NS direction (0–1, default 0.5).
    step_seconds:
        Real-world seconds per simulation step.
    """

    def __init__(
        self,
        cycle_seconds: int = 30,
        green_split: float = 0.5,
        step_seconds: float = 1.0,
    ) -> None:
        super().__init__(name="fixed_timing")
        self.cycle_seconds = cycle_seconds
        self.green_split = green_split
        self.step_seconds = step_seconds

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        phase = self._phase_at_step(step)
        return {iid: phase for iid in observations}

    def select_action(self, obs: dict[str, float]) -> int:
        step = int(obs.get("step", 0))
        phase = self._phase_at_step(step)
        return 0 if phase == "NS" else 1

    def _phase_at_step(self, step: int) -> SignalPhase:
        cycle_steps = max(1, int(round(self.cycle_seconds / self.step_seconds)))
        ns_steps = max(1, int(round(cycle_steps * self.green_split)))
        return "NS" if (step % cycle_steps) < ns_steps else "EW"
