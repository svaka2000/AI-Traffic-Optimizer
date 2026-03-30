"""MaxPressure signal controller (Varaiya 2013).

At each decision step MaxPressure selects the phase that maximises the
pressure differential across all movements served by that phase:

    pressure(phase, i) = Σ incoming_queue(i, phase)
                       − Σ downstream_queue(neighbors, phase)

where ``downstream_queue`` is the total queue at the downstream neighbor
intersections that receive traffic from phase movements (network-aware
formulation from Varaiya 2013, §III).

When a neighbor is absent (boundary intersection) its contribution to the
outgoing sum is treated as zero — the standard boundary condition used in
the literature.

The per-direction neighbor queues (``neighbor_N_queue``, ``neighbor_S_queue``,
``neighbor_E_queue``, ``neighbor_W_queue``) are injected into each
intersection's observation dict by ``TrafficNetworkSimulator._collect_observations``.

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

# Downstream neighbor directions for each phase (Varaiya 2013, §III).
# NS_THROUGH vehicles exit north/south → N and S neighbors absorb flow.
# EW_THROUGH vehicles exit east/west  → E and W neighbors absorb flow.
# NS_LEFT vehicles (from N/S) turn and exit east/west → E and W neighbors.
# EW_LEFT vehicles (from E/W) turn and exit north/south → N and S neighbors.
_PHASE_OUTGOING_DIRS: dict[str, tuple[str, ...]] = {
    "NS_THROUGH": ("N", "S"),
    "EW_THROUGH": ("E", "W"),
    "NS_LEFT":    ("E", "W"),
    "EW_LEFT":    ("N", "S"),
}

_PHASES: tuple[str, ...] = ("NS_THROUGH", "EW_THROUGH", "NS_LEFT", "EW_LEFT")


def _phase_pressure(obs: dict[str, float], phase: str) -> float:
    """Compute Varaiya pressure for *phase* given a single-intersection observation.

    pressure = Σ incoming_queue − Σ downstream_neighbor_queue
    """
    # Incoming: local queued vehicles served by this phase
    incoming = 0.0
    for key_group in _PHASE_INCOMING[phase]:
        for key in key_group:
            if key in obs:
                incoming += obs[key]
                break

    # Outgoing: congestion at downstream neighbor intersections (Varaiya 2013)
    outgoing_dirs = _PHASE_OUTGOING_DIRS[phase]
    outgoing = sum(obs.get(f"neighbor_{d}_queue", 0.0) for d in outgoing_dirs)

    return incoming - outgoing


class MaxPressureController(BaseController):
    """Selects the phase with the highest network pressure at each step.

    Uses the full Varaiya (2013) pressure formula:
        pressure = incoming_queue − downstream_neighbor_queue

    When directional neighbor queue data is available in the observation
    dict (4-intersection network), the controller accounts for downstream
    congestion when selecting phases.  At a single isolated intersection,
    all neighbor queues default to 0.0 and the formula reduces to greedy
    incoming-queue maximisation.

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
                # Select the phase with maximum Varaiya pressure.
                target = max(_PHASES, key=lambda p: _phase_pressure(obs, p))
                if target != current:
                    elapsed = 0

            self._current_phase[iid] = target
            self._elapsed[iid] = elapsed
            actions[iid] = target  # type: ignore[assignment]
        return actions
