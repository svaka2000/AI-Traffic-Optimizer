"""traffic_ai/simulation_engine/interconnect.py

Intersection interconnect and clock-drift model.

Based on Steve Celniker's description (Senior Traffic Engineer, City of San Diego):
  "Copper/fiber interconnect is cut by construction workers.  Without interconnect,
   controller clocks drift apart — coordination breaks.  GPS can fix drift but
   antennas are vandalism targets.  Wireless: no trenching needed but unreliable,
   line-of-sight issues."

Clock Drift Impact
------------------
Rate ≈ 0.5 s/hr without synchronisation.
After 48 hr without sync: 24 s of offset error → corridor coordination fails.

When an interconnect link is down:
- Real-time neighbour observation is unavailable.
- Controllers receive stale historical average instead.
- Clock drift accumulates on the isolated intersection(s).

When a wireless link is degraded:
- Packets are lost with probability `wireless_packet_loss`.
- On packet loss: stale observation is substituted (same as full link failure).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class InterconnectType(Enum):
    FIBER    = "fiber"     # Most reliable; cut by construction
    COPPER   = "copper"    # Legacy hardwire; also cut by construction
    WIRELESS = "wireless"  # No trenching; unreliable, line-of-sight required
    NONE     = "none"      # No interconnect — clocks always drift


@dataclass(slots=True)
class InterconnectConfig:
    """Configuration for corridor interconnect reliability.

    Attributes
    ----------
    type : InterconnectType
        Physical medium.  Affects failure mode.
    failure_rate_per_hour : float
        Per-link failure probability per hour.  Fiber/copper cut by
        construction; wireless affected by obstructions.
        Default 0.0005 /hr ≈ 4 outages per year per link.
    repair_time_hours : float
        Mean repair time.  Fibre/copper requires trenching: ~48 hr.
    wireless_packet_loss : float
        Fraction of observations lost on wireless links when operational.
        Default 0.05 = 5 % packet loss.
    clock_drift_sec_per_hour : float
        Seconds of offset that accumulate per hour without GPS sync.
        Celniker estimate: 0.5 s/hr.
    gps_sync_enabled : bool
        If True, isolated intersections are resynchronised when link repairs.
    enabled : bool
        Master switch.  When False, all links behave as perfectly reliable.
    """
    type: InterconnectType              = InterconnectType.FIBER
    failure_rate_per_hour: float        = 0.0005
    repair_time_hours: float            = 48.0
    wireless_packet_loss: float         = 0.05
    clock_drift_sec_per_hour: float     = 0.5
    gps_sync_enabled: bool              = True
    enabled: bool                       = False


class InterconnectLink:
    """Bidirectional communication link between two adjacent intersections.

    Tracks operational state, failure/repair timing, and accumulated clock
    drift for the pair of intersections.
    """

    def __init__(
        self,
        config: InterconnectConfig,
        intersection_a: int,
        intersection_b: int,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.intersection_a = intersection_a
        self.intersection_b = intersection_b
        self.rng = rng

        self._is_down: bool = False
        self._down_since_step: Optional[int] = None
        self._repair_step: Optional[int] = None
        self._clock_drift_sec: float = 0.0

    def update(self, step: int, step_seconds: float) -> None:
        """Advance link state by one simulation step."""
        if not self.config.enabled:
            return
        if self.config.type == InterconnectType.NONE:
            # No interconnect — clocks always drift
            self._is_down = True
            self._clock_drift_sec += self.config.clock_drift_sec_per_hour / (3600.0 / step_seconds)
            return

        steps_per_hour = 3600.0 / step_seconds

        if self._is_down:
            self._clock_drift_sec += self.config.clock_drift_sec_per_hour / steps_per_hour
            if self._repair_step is not None and step >= self._repair_step:
                self._is_down = False
                self._down_since_step = None
                if self.config.gps_sync_enabled:
                    self._clock_drift_sec = 0.0  # Resynchronise on repair
        else:
            fail_prob = self.config.failure_rate_per_hour / steps_per_hour
            if self.rng.random() < fail_prob:
                self._is_down = True
                self._down_since_step = step
                repair_steps = int(self.config.repair_time_hours * steps_per_hour)
                self._repair_step = step + repair_steps

    @property
    def is_down(self) -> bool:
        if not self.config.enabled:
            return False
        return self._is_down

    @property
    def clock_drift_sec(self) -> float:
        return self._clock_drift_sec

    def filter_observation(
        self,
        obs: dict[str, float],
        historical_avg: dict[str, float],
    ) -> dict[str, float]:
        """Return stale historical data when interconnect is unavailable."""
        if not self.config.enabled:
            return obs

        if self._is_down:
            return historical_avg

        if self.config.type == InterconnectType.WIRELESS:
            if self.rng.random() < self.config.wireless_packet_loss:
                return historical_avg  # Packet dropped

        return obs


class InterconnectNetwork:
    """Manages all inter-intersection communication links in the grid.

    Maintains a dict of InterconnectLink objects keyed by canonical
    (min, max) intersection-ID pairs.  Provides observation filtering
    for the engine's _collect_observations() method.
    """

    def __init__(
        self,
        n_intersections: int,
        neighbors: dict[int, dict[str, Optional[int]]],
        config: InterconnectConfig,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.links: dict[tuple[int, int], InterconnectLink] = {}

        # Historical average store: per-intersection running mean observation
        self._history: dict[int, dict[str, float]] = {
            i: {} for i in range(n_intersections)
        }
        self._history_count: dict[int, int] = {i: 0 for i in range(n_intersections)}

        for node, nbr_map in neighbors.items():
            for nbr_id in nbr_map.values():
                if nbr_id is None:
                    continue
                key = (min(node, nbr_id), max(node, nbr_id))
                if key not in self.links:
                    self.links[key] = InterconnectLink(config, node, nbr_id, rng)

    def update(self, step: int, step_seconds: float) -> None:
        for link in self.links.values():
            link.update(step, step_seconds)

    def update_history(self, intersection_id: int, obs: dict[str, float]) -> None:
        """Update rolling mean observation for an intersection."""
        hist = self._history[intersection_id]
        n = self._history_count[intersection_id]
        for k, v in obs.items():
            if isinstance(v, (int, float)):
                hist[k] = (hist.get(k, 0.0) * n + v) / (n + 1)
        self._history_count[intersection_id] = n + 1

    def get_neighbor_obs(
        self,
        from_node: int,
        to_node: int,
        live_obs: dict[str, float],
    ) -> dict[str, float]:
        """Return (possibly filtered) observation of *to_node* from *from_node*'s perspective."""
        key = (min(from_node, to_node), max(from_node, to_node))
        link = self.links.get(key)
        if link is None:
            return live_obs
        historical = self._history.get(to_node, live_obs)
        return link.filter_observation(live_obs, historical)
