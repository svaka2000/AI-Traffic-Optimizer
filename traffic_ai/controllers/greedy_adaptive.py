"""traffic_ai/controllers/greedy_adaptive.py

Greedy adaptive controller — models the InSync algorithm (Rhythm Engineering).

InSync is deployed in San Diego on:
  - Mira Mesa Boulevard
  - Rosecrans Street, Hancock to Nimitz (12 signals)
    Verified results (San Diego Mayor Faulconer, 2017):
      • 25 % travel time reduction during rush hour
      • 53 % stop reduction

Algorithm (from Rhythm Engineering technical whitepaper)
---------------------------------------------------------
"Imagine a greedy person with a basket of cookies.  They must give one
 cookie per vehicle waiting AND one cookie per 5 seconds a vehicle has
 been waiting.  They will change the signal to conserve cookies — minimise
 total cost."

Cost function:
    cost(phase) = Σ_vehicles (count_waiting + wait_time / 5)
               ≡ volume_weight × queue_volume
                + delay_weight × accumulated_delay

Key differences from Webster-based systems
-------------------------------------------
- No fixed cycle length — each phase runs as long as productive.
- Real-time: re-evaluates every simulation step.
- Greedy: optimises the current moment; no model, no look-ahead.
- Coordination: considers downstream neighbour queues to prevent
  platoons hitting consecutive red lights (corridor "time tunnel").

References
----------
Rhythm Engineering, "How InSync Works", Technical Whitepaper, 2017.
San Diego Mayor's Office, "Rosecrans Street Traffic Signal Improvements",
  Press Release, 2017.
"""

from __future__ import annotations

from typing import Any

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class GreedyAdaptiveController(BaseController):
    """Greedy volume×delay cost-minimising adaptive controller.

    Models the InSync approach deployed on Mira Mesa Boulevard and
    Rosecrans Street in San Diego.

    Parameters
    ----------
    min_green_steps : int
        Minimum phase duration (steps) before a switch is allowed.
        Default 7 steps = 7 s at 1 s/step (MUTCD minimum).
    volume_weight : float
        Weight on the queued-vehicle-count component of cost.
        Per the InSync cost function: 1.0 cookie per waiting vehicle.
    delay_weight : float
        Weight on the accumulated-delay component of cost.
        Per InSync: 1 cookie per 5 s of waiting → 0.2 cookies/s.
    coordination_weight : float
        Weight applied to downstream neighbour queue pressure.
        Positive value penalises serving a phase whose downstream
        neighbour already has high queue (avoids platoon injection).
    hysteresis : float
        Fractional premium the alternative phase must exceed before
        triggering a switch.  Default 0.10 = 10 % hysteresis prevents
        rapid oscillation on balanced demand.
    """

    def __init__(
        self,
        min_green_steps: int        = 7,
        volume_weight: float        = 1.0,
        delay_weight: float         = 0.2,    # 1/5 per InSync cost function
        coordination_weight: float  = 0.3,
        hysteresis: float           = 0.10,
    ) -> None:
        super().__init__(name="greedy_adaptive")
        self.min_green_steps        = min_green_steps
        self.volume_weight          = volume_weight
        self.delay_weight           = delay_weight
        self.coordination_weight    = coordination_weight
        self.hysteresis             = hysteresis

        # Per-intersection state
        self._phase:        dict[int, int]               = {}  # 0=NS, 1=EW
        self._steps_in:     dict[int, int]               = {}
        self._delay:        dict[int, dict[int, float]]  = {}  # accumulated delay per phase
        self._neighbors:    dict[int, list[int]]         = {}

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._phase     = {i: 0 for i in range(n_intersections)}
        self._steps_in  = {i: 0 for i in range(n_intersections)}
        self._delay     = {i: {0: 0.0, 1: 0.0} for i in range(n_intersections)}
        # Simple linear neighbour map (works for both grid and corridor topologies)
        self._neighbors = {
            i: [j for j in [i - 1, i + 1] if 0 <= j < n_intersections]
            for i in range(n_intersections)
        }

    def compute_actions(
        self,
        observations: dict[int, dict[str, float]],
        step: int,
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}

        for iid, obs in observations.items():
            current_phase = self._phase[iid]
            steps_in      = self._steps_in[iid]
            waiting_phase = 1 - current_phase

            # Accumulate delay for the phase NOT currently being served.
            # Every step of red increases the cost of continuing to ignore it.
            wait_vol = self._queue_for_phase(waiting_phase, obs)
            self._delay[iid][waiting_phase] = (
                self._delay[iid].get(waiting_phase, 0.0) + wait_vol
            )

            # Enforce minimum green before considering a switch
            if steps_in < self.min_green_steps:
                actions[iid] = self._phase_to_signal(current_phase)
                self._steps_in[iid] = steps_in + 1
                continue

            # Neighbour observations for coordination
            nbr_obs = [
                observations.get(nbr, obs)
                for nbr in self._neighbors.get(iid, [])
            ]

            # Greedy cost: how urgently does each phase need service?
            cost_current = self._compute_cost(current_phase, iid, obs, nbr_obs)
            cost_switch  = self._compute_cost(waiting_phase, iid, obs, nbr_obs)

            # Switch when the alternative phase has meaningfully higher need
            # (10 % hysteresis prevents oscillation on balanced demand)
            if cost_switch > cost_current * (1.0 + self.hysteresis):
                self._phase[iid]    = waiting_phase
                self._steps_in[iid] = 0
                # Reset accumulated delay for the newly served phase
                self._delay[iid][waiting_phase] = 0.0
            else:
                self._steps_in[iid] = steps_in + 1

            actions[iid] = self._phase_to_signal(self._phase[iid])

        return actions  # type: ignore[return-value]

    def select_action(self, observation: dict[str, float]) -> int:
        return self._phase.get(0, 0)

    def update(self, obs: Any, action: Any, reward: Any, next_obs: Any, done: bool = False) -> None:
        pass  # Greedy does not learn

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _queue_for_phase(self, phase: int, obs: dict[str, float]) -> float:
        """Extract queue volume relevant to the given phase index."""
        if phase == 0:
            return obs.get("queue_ns_through", obs.get("queue_ns", 0.0))
        return obs.get("queue_ew_through", obs.get("queue_ew", 0.0))

    def _compute_cost(
        self,
        phase: int,
        iid: int,
        obs: dict[str, float],
        nbr_obs: list[dict[str, float]],
    ) -> float:
        """Greedy cost for serving *phase* at intersection *iid*.

        Cost = volume_weight × queued_vehicles
             + delay_weight  × accumulated_delay
             − coordination_weight × downstream_queue   (downstream congestion
               penalty discourages sending traffic into congested neighbours)

        A higher cost means this phase is more urgently needed.
        """
        volume = self._queue_for_phase(phase, obs)
        delay  = self._delay[iid].get(phase, 0.0)

        # Downstream pressure: if neighbours already have high queue on the
        # same axis, sending more traffic towards them is counterproductive.
        downstream_pressure = 0.0
        if nbr_obs:
            for nbr in nbr_obs:
                downstream_pressure += self._queue_for_phase(phase, nbr)
            downstream_pressure /= len(nbr_obs)

        return (
            self.volume_weight      * volume
            + self.delay_weight     * delay
            - self.coordination_weight * downstream_pressure
        )

    @staticmethod
    def _phase_to_signal(phase: int) -> SignalPhase:
        return "NS_THROUGH" if phase == 0 else "EW_THROUGH"  # type: ignore[return-value]
