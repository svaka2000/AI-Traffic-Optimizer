"""MaxPressure signal controller (Varaiya 2013).

At each decision step MaxPressure selects the phase that maximises the sum
of queue lengths on incoming lanes across all movements served by that phase,
minus the sum of queue lengths on the corresponding outgoing lanes.

Because outgoing-link queues are not observable in the local single-intersection
observation space, outgoing queues are treated as zero — the standard
single-intersection simplification used in the literature when downstream
queue state is unavailable.  In a connected multi-intersection network the
controller can be upgraded to subtract downstream queues if/when they are
exposed in the observation dict.

Reference
---------
P. Varaiya, "Max pressure control of a network of signalized intersections,"
Transportation Research Part C: Emerging Technologies, 36:177-195, 2013.
"""

from __future__ import annotations

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase

# Incoming-queue keys for each phase (first matching key wins; allows both
# the canonical 4-phase keys and the legacy 2-phase fallbacks).
_PHASE_INCOMING: dict[str, tuple[tuple[str, ...], ...]] = {
    "NS_THROUGH": (("queue_ns_through", "queue_ns"),),
    "EW_THROUGH": (("queue_ew_through", "queue_ew"),),
    "NS_LEFT":    (("queue_ns_left",),),
    "EW_LEFT":    (("queue_ew_left",),),
}

_PHASES: tuple[str, ...] = ("NS_THROUGH", "EW_THROUGH", "NS_LEFT", "EW_LEFT")


def _phase_pressure(obs: dict[str, float], phase: str) -> float:
    """Return the incoming-queue pressure for *phase* given a single-intersection observation."""
    total = 0.0
    for key_group in _PHASE_INCOMING[phase]:
        for key in key_group:
            if key in obs:
                total += obs[key]
                break
    return total


class MaxPressureController(BaseController):
    """Selects the phase with the highest incoming-queue pressure at each step.

    Parameters
    ----------
    min_green_sec : int
        Minimum number of simulation steps a phase must be held before
        MaxPressure is allowed to switch (HCM 7th ed. default: 7 s).
    """

    def __init__(self, min_green_sec: int = 7) -> None:
        super().__init__(name="max_pressure")
        self.min_green_sec = min_green_sec
        self._current_phase: dict[int, str] = {}
        self._elapsed: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: "NS_THROUGH" for i in range(n_intersections)}
        self._elapsed = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            elapsed = self._elapsed.get(iid, 0) + 1
            current = self._current_phase.get(iid, "NS_THROUGH")

            if elapsed < self.min_green_sec:
                # Enforce minimum green — hold current phase.
                target = current
            else:
                # Select the phase with maximum incoming-queue pressure.
                target = max(_PHASES, key=lambda p: _phase_pressure(obs, p))
                if target != current:
                    elapsed = 0

            self._current_phase[iid] = target
            self._elapsed[iid] = elapsed
            actions[iid] = target  # type: ignore[assignment]
        return actions
