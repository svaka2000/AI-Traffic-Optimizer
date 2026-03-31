"""traffic_ai/simulation_engine/priority.py

Priority event system: emergency vehicle preemption, bus Transit Signal
Priority (TSP), and Leading Pedestrian Intervals (LPI).

Based on Steve Celniker's description (Senior Traffic Engineer, City of San Diego):
  "Emergency vehicles, bus priority, and leading pedestrian intervals delay
   everybody.  These are non-negotiable operational requirements."

Event Hierarchy
---------------
1. Emergency preemption — overrides ALL controller outputs immediately.
   Green is forced in the emergency vehicle's approach direction for
   15–30 s.  No controller can prevent this.

2. Bus Transit Signal Priority (TSP) — extends green up to 10 s or
   truncates red by up to 10 s.  Less disruptive than preemption; the
   controller still runs but the phase duration is modified.  Not yet
   modelled as a hard override — tracked as metadata for future work.

3. Leading Pedestrian Interval (LPI) — 3–7 s head-start for pedestrians
   before vehicle green.  Reduces effective vehicle green by the LPI
   duration.  Applied stochastically at phase changes.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np


class PriorityEventType(Enum):
    EMERGENCY_VEHICLE   = "emergency"
    BUS_PRIORITY        = "bus"
    LEADING_PED_INTERVAL = "lpi"


@dataclass(slots=True)
class PriorityConfig:
    """Configuration for stochastic priority event generation.

    Attributes
    ----------
    emergency_enabled : bool
        Enable emergency vehicle preemption (default True — legally required).
    emergency_rate_per_hour_per_intersection : float
        Average preemption events per intersection per hour.
        ~0.15 /hr → ~4 events/hr across 4 intersections.
    emergency_duration_sec_min, _max : float
        Preemption lasts uniformly in [min, max] seconds.
    bus_priority_enabled : bool
        Track bus TSP events (informational in this implementation).
    bus_rate_per_hour_per_intersection : float
        Average bus priority calls per intersection per hour.
    lpi_enabled : bool
        Apply Leading Pedestrian Intervals at phase changes.
    lpi_duration_sec : float
        LPI duration in seconds (typically 3–7 s per NACTO guidance).
    lpi_probability_per_phase_change : float
        Probability that an LPI is triggered at each phase change.
    enabled : bool
        Master switch.  When False, no priority events are generated.
    """
    emergency_enabled: bool                             = True
    emergency_rate_per_hour_per_intersection: float     = 0.15
    emergency_duration_sec_min: float                   = 15.0
    emergency_duration_sec_max: float                   = 30.0
    bus_priority_enabled: bool                          = True
    bus_rate_per_hour_per_intersection: float           = 2.0
    bus_max_extend_sec: float                           = 10.0
    lpi_enabled: bool                                   = True
    lpi_duration_sec: float                             = 5.0
    lpi_probability_per_phase_change: float             = 0.30
    enabled: bool                                       = False


@dataclass
class PriorityEvent:
    """Single priority event record."""
    event_type: PriorityEventType
    intersection_id: int
    direction: str         # "NS" or "EW" (approach direction for emergency/bus)
    arrival_step: int
    duration_steps: int


class PriorityEventSystem:
    """Generates and applies priority events across all intersections.

    Usage (inside engine per-step loop)
    ------------------------------------
    1. Call ``generate_events(step)`` to create new events this step.
    2. Call ``apply_preemptions(actions, step)`` to override actions where
       emergency preemption is active.
    3. Metrics are accumulated in ``total_preemption_events``,
       ``total_bus_events``, and ``total_preemption_delay_steps``.
    """

    def __init__(
        self,
        config: PriorityConfig,
        n_intersections: int,
        step_seconds: float,
        rng: np.random.Generator,
    ) -> None:
        self.config = config
        self.n_intersections = n_intersections
        self.step_seconds = step_seconds
        self.rng = rng

        self._active_preemptions: dict[int, PriorityEvent] = {}
        self.total_preemption_events: int = 0
        self.total_bus_events: int = 0
        self.total_preemption_delay_steps: int = 0

    # ------------------------------------------------------------------
    # Event generation
    # ------------------------------------------------------------------

    def generate_events(self, step: int) -> list[PriorityEvent]:
        """Stochastically generate priority events for this simulation step."""
        if not self.config.enabled:
            return []

        events: list[PriorityEvent] = []
        steps_per_hour = 3600.0 / self.step_seconds

        for iid in range(self.n_intersections):
            # --- Emergency vehicle preemption ---
            if self.config.emergency_enabled and iid not in self._active_preemptions:
                p = self.config.emergency_rate_per_hour_per_intersection / steps_per_hour
                if self.rng.random() < p:
                    direction = str(self.rng.choice(["NS", "EW"]))
                    duration_sec = float(self.rng.uniform(
                        self.config.emergency_duration_sec_min,
                        self.config.emergency_duration_sec_max,
                    ))
                    evt = PriorityEvent(
                        event_type=PriorityEventType.EMERGENCY_VEHICLE,
                        intersection_id=iid,
                        direction=direction,
                        arrival_step=step,
                        duration_steps=int(duration_sec / self.step_seconds),
                    )
                    self._active_preemptions[iid] = evt
                    events.append(evt)
                    self.total_preemption_events += 1

            # --- Bus Transit Signal Priority (informational) ---
            if self.config.bus_priority_enabled:
                p = self.config.bus_rate_per_hour_per_intersection / steps_per_hour
                if self.rng.random() < p:
                    events.append(PriorityEvent(
                        event_type=PriorityEventType.BUS_PRIORITY,
                        intersection_id=iid,
                        direction=str(self.rng.choice(["NS", "EW"])),
                        arrival_step=step,
                        duration_steps=int(self.config.bus_max_extend_sec / self.step_seconds),
                    ))
                    self.total_bus_events += 1

        return events

    # ------------------------------------------------------------------
    # Action override
    # ------------------------------------------------------------------

    def apply_preemptions(
        self,
        actions: dict[int, str],
        step: int,
    ) -> dict[int, str]:
        """Override controller actions with active emergency preemptions.

        Emergency preemption is non-negotiable — it always wins regardless
        of the controller's requested phase.
        """
        if not self.config.enabled:
            return actions

        modified = dict(actions)
        expired: list[int] = []

        for iid, evt in self._active_preemptions.items():
            steps_elapsed = step - evt.arrival_step
            if steps_elapsed < evt.duration_steps:
                # Force the emergency vehicle's approach direction
                phase = "NS_THROUGH" if evt.direction == "NS" else "EW_THROUGH"
                modified[iid] = phase
                self.total_preemption_delay_steps += 1
            else:
                expired.append(iid)

        for iid in expired:
            del self._active_preemptions[iid]

        return modified

    # ------------------------------------------------------------------
    # LPI application
    # ------------------------------------------------------------------

    def lpi_green_reduction_steps(self) -> int:
        """Return steps of effective green lost to LPI at a phase change."""
        if not self.config.enabled or not self.config.lpi_enabled:
            return 0
        if self.rng.random() < self.config.lpi_probability_per_phase_change:
            return int(self.config.lpi_duration_sec / self.step_seconds)
        return 0

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    @property
    def active_preemption_count(self) -> int:
        return len(self._active_preemptions)
