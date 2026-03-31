"""traffic_ai/controllers/webster.py

Webster (1958) optimal cycle-length controller.

This is the industry-standard signal timing algorithm deployed in:
  - Econolite Centracs Edaptive (≥3 000 intersections)
  - SCATS — Sydney Coordinated Adaptive Traffic System (TransCore)
  - MAXTIME Adaptive (Q-Free)
  - SynchroGreen (Cubic Transportation Systems)
  - Kadence (Kimley-Horn)

Algorithm
---------
Webster (1958) derived the delay-minimising cycle-length formula:

    C_opt = (1.5 × L + 5) / (1 − Y)

where:
    L = total lost time per cycle
      = n_phases × (yellow_sec + all_red_sec)   [default: 2 × 6 = 12 s]
    Y = Σ y_i = Σ (q_i / s_i)
      q_i = demand flow rate on critical approach (veh/s equivalent)
      s_i = saturation flow rate = 1 800 veh/hr/lane ÷ 3 600 = 0.5 veh/s/lane
    Y must be < 1.0 (oversaturation → Webster breaks down → use max_cycle)

Green time allocation:
    g_i = (C_opt − L) × (y_i / Y)   [proportional to critical demand]

Key limitation vs. Greedy/RL
-----------------------------
Webster recalculates every `recalc_interval` steps (default 300 = 5 min
at 1 s/step).  It cannot respond to second-by-second demand fluctuations.
This is the central advantage of InSync and AITO-DQN over Webster-based
systems in irregular-demand conditions.

Reference
---------
F. V. Webster, "Traffic Signal Settings", Road Research Technical Paper
No. 39, H.M. Stationery Office, London, 1958.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase


class WebsterController(BaseController):
    """Webster (1958) optimal cycle-length adaptive controller.

    Parameters
    ----------
    recalc_interval : int
        Steps between timing plan recalculations.
        Default 300 = 5 minutes at 1 s/step (industry standard).
    saturation_flow : float
        Saturation flow rate in vehicles per second per lane.
        Default 0.5 veh/s = 1 800 veh/hr/lane (HCM 7th edition default).
    lost_time_per_phase : float
        Clearance loss per phase change in seconds = yellow + all-red.
        Default 6.0 s (4 s yellow + 2 s all-red, HCM 7th ed. §19.6).
    min_cycle : float
        Minimum cycle length in seconds. Webster lower bound = 60 s.
    max_cycle : float
        Maximum cycle length in seconds. Webster upper bound = 180 s.
    step_seconds : float
        Simulation step duration in seconds.  Default 1.0.
    """

    def __init__(
        self,
        recalc_interval: int    = 300,
        saturation_flow: float  = 0.5,      # veh/s/lane = 1 800 veh/hr/lane
        lost_time_per_phase: float = 6.0,   # yellow(4) + all-red(2)
        min_cycle: float        = 60.0,
        max_cycle: float        = 180.0,
        step_seconds: float     = 1.0,
    ) -> None:
        super().__init__(name="webster")
        self.recalc_interval    = recalc_interval
        self.saturation_flow    = saturation_flow
        self.lost_time_per_phase = lost_time_per_phase
        self.min_cycle          = min_cycle
        self.max_cycle          = max_cycle
        self.step_seconds       = step_seconds

        # Per-intersection state
        self._obs_buffer:      dict[int, list[dict[str, float]]] = {}
        self._current_timing:  dict[int, dict[str, Any]]         = {}
        self._phase_step:      dict[int, int]                    = {}
        self._steps_since_recalc: int = 0

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._obs_buffer = {i: [] for i in range(n_intersections)}
        self._phase_step = {i: 0 for i in range(n_intersections)}
        self._steps_since_recalc = 0
        # Initialise with balanced 60-second timing (Webster default lower bound)
        default = {"ns_green": 25, "ew_green": 25, "cycle": 60}
        self._current_timing = {i: dict(default) for i in range(n_intersections)}

    def compute_actions(
        self,
        observations: dict[int, dict[str, float]],
        step: int,
    ) -> dict[int, SignalPhase]:
        # Buffer observations for demand estimation
        for iid, obs in observations.items():
            buf = self._obs_buffer.setdefault(iid, [])
            buf.append(obs)
            if len(buf) > self.recalc_interval:
                buf.pop(0)

        self._steps_since_recalc += 1
        if self._steps_since_recalc >= self.recalc_interval:
            self._recalculate_timing()
            self._steps_since_recalc = 0

        # Execute current timing plan for each intersection
        actions: dict[int, SignalPhase] = {}
        for iid in range(self.n_intersections):
            timing = self._current_timing.get(iid, {"ns_green": 25, "cycle": 60})
            cycle   = max(timing["cycle"], 1)
            ns_green = timing["ns_green"]
            pos = self._phase_step[iid] % cycle
            actions[iid] = "NS_THROUGH" if pos < ns_green else "EW_THROUGH"
            self._phase_step[iid] = (self._phase_step[iid] + 1) % cycle

        return actions  # type: ignore[return-value]

    def select_action(self, observation: dict[str, float]) -> int:
        timing = self._current_timing.get(0, {"ns_green": 25, "cycle": 60})
        cycle   = max(timing["cycle"], 1)
        pos = self._phase_step.get(0, 0) % cycle
        return 0 if pos < timing["ns_green"] else 1

    def update(self, obs: Any, action: Any, reward: Any, next_obs: Any, done: bool = False) -> None:
        pass  # Webster does not learn

    # ------------------------------------------------------------------
    # Webster formula internals
    # ------------------------------------------------------------------

    def _recalculate_timing(self) -> None:
        """Recompute optimal cycle length and green splits (Webster 1958).

        Uses the buffered observation window as a proxy for the 5-minute
        turning-movement count that field engineers measure in the field.
        """
        n_phases = 2  # NS and EW through phases
        L = n_phases * self.lost_time_per_phase  # Total lost time per cycle (s)

        for iid, obs_list in self._obs_buffer.items():
            if not obs_list:
                continue

            # Average observed queue lengths over the recalculation window.
            # Queue length is used as a proxy for demand: higher queue →
            # higher unmet demand → higher y_i.
            keys = obs_list[0].keys()
            avg: dict[str, float] = {
                k: float(np.mean([o.get(k, 0.0) for o in obs_list]))
                for k in keys
            }

            # Critical-movement queue lengths
            # Use both aliased keys for robustness (4-phase and 2-phase obs)
            q_ns = avg.get("queue_ns_through", avg.get("queue_ns", 0.0))
            q_ew = avg.get("queue_ew_through", avg.get("queue_ew", 0.0))

            # Flow-to-saturation ratios: q_i / s_i
            # saturation_flow is in veh/s; recalc_interval steps at step_seconds
            # converts queue to an effective veh/s rate.
            denom = self.saturation_flow * self.recalc_interval * self.step_seconds
            y_ns = min(q_ns / max(denom, 1e-6), 0.95)
            y_ew = min(q_ew / max(denom, 1e-6), 0.95)
            Y = y_ns + y_ew

            if Y >= 0.99:
                # Oversaturation: Webster formula is undefined.
                # Fallback: maximum cycle, proportional split.
                cycle    = int(self.max_cycle)
                ns_green = max(7, int((cycle - L) * 0.5))
            else:
                # Webster (1958) Equation 1: optimal cycle length
                c_opt = (1.5 * L + 5.0) / (1.0 - Y)
                cycle = int(np.clip(c_opt, self.min_cycle, self.max_cycle))

                # Proportional green allocation (Webster 1958, Eq. 2)
                effective_green = cycle - L
                if Y > 1e-9:
                    ns_green = max(7, int(effective_green * (y_ns / Y)))
                else:
                    ns_green = max(7, int(effective_green * 0.5))

            ew_green = max(7, cycle - ns_green - int(L))
            self._current_timing[iid] = {
                "ns_green":  ns_green,
                "ew_green":  ew_green,
                "cycle":     cycle,
                "Y":         round(Y, 4),
                "L":         L,
            }
