"""traffic_ai/simulation_engine/demand.py

Stochastic vehicle arrival demand model with time-varying profiles.

Each profile models a different real-world traffic scenario, from typical
rush-hour commuting to developing-world high-density flows. The model returns
per-lane arrival rates (vehicles/second) for a given simulation step and
direction. The engine applies Poisson sampling on top of this rate.

Profiles
--------
normal          : Mild sinusoidal variation around a 0.12 veh/s base.
rush_hour       : Twin Gaussian peaks at 8 AM and 5:30 PM (1.6× scaling).
midday_peak     : Single gentle lunchtime peak at 1 PM.
weekend         : Reduced base volume, single mid-day hump (no commute spikes).
school_zone     : Sharp narrow spikes at 7:45 AM and 3:00 PM (NS-only).
event_surge     : Pre/post-event traffic surges (e.g. stadium, concert).
construction    : East-West capacity reduced; arrivals elevated to simulate backup.
emergency_priority : Random emergency vehicle events every ~500 steps.
high_density_developing : High base rate (3× normal) with non-compliant vehicles.
incident_response : Capacity loss at step 300 on one direction for 200 steps.
weather_degraded : Higher arrivals, lower service (rain).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Literal


DemandProfileName = Literal[
    "normal",
    "rush_hour",
    "midday_peak",
    "weekend",
    "school_zone",
    "event_surge",
    "construction",
    "emergency_priority",
    "high_density_developing",
    "incident_response",
    "weather_degraded",
]

ALL_DEMAND_PROFILES: list[str] = [
    "normal",
    "rush_hour",
    "midday_peak",
    "weekend",
    "school_zone",
    "event_surge",
    "construction",
    "emergency_priority",
    "high_density_developing",
    "incident_response",
    "weather_degraded",
]


@dataclass(slots=True)
class EmergencyEvent:
    """Active emergency vehicle event."""
    direction: str
    steps_remaining: int
    intersection_id: int


@dataclass
class DemandModel:
    """Time-varying arrival rate generator for the traffic simulation engine.

    Parameters
    ----------
    profile:
        Named demand scenario. See module docstring for descriptions.
    scale:
        Global arrival-rate multiplier applied after profile computation.
    step_seconds:
        Real-world seconds represented by each simulation step.
    seed:
        Random seed for stochastic demand events (emergency, incident).
    """

    profile: DemandProfileName = "normal"
    scale: float = 1.0
    step_seconds: float = 1.0
    seed: int = 42
    # Internal mutable state (not slots to allow dataclass flexibility)
    _rng: random.Random = field(default_factory=lambda: random.Random(42), init=False, repr=False, compare=False)
    _emergency_events: list[EmergencyEvent] = field(default_factory=list, init=False, repr=False, compare=False)
    _incident_active: bool = field(default=False, init=False, repr=False, compare=False)
    _incident_direction: str = field(default="E", init=False, repr=False, compare=False)
    _incident_capacity_fraction: float = field(default=1.0, init=False, repr=False, compare=False)
    # Phase 9C: real PeMS hourly calibration profile (hour → arrival_rate_per_sec)
    _pems_profile: dict[int, float] | None = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)

    # ------------------------------------------------------------------
    # Phase 9C: PeMS calibration
    # ------------------------------------------------------------------

    def calibrate_from_pems_profile(
        self, hourly_profile: dict[int, float]
    ) -> None:
        """Calibrate demand from a real PeMS hourly arrival-rate profile.

        When a PeMS profile is loaded, ``arrival_rate_per_lane`` returns the
        real data rate for the current hour instead of the synthetic Gaussian
        peaks.  The global ``scale`` multiplier is still applied on top.

        Rush-hour scaling (``_rush_hour`` etc.) is bypassed in PeMS mode;
        the calibrated rates already encode the real demand shape.

        Parameters
        ----------
        hourly_profile:
            Dict mapping hour (0–23) → mean arrival_rate_per_sec per lane.
            Produced by ``PeMSConnector.compute_hourly_demand_profile()``.

        Key formula (from PeMS documentation)::

            PeMS gives: total_flow_per_5min (vehicles per 5 minutes, all lanes)
            AITO needs: arrival_rate_per_sec per lane

            flow_per_lane_per_5min = total_flow_per_5min / n_lanes
            arrival_rate_per_sec   = flow_per_lane_per_5min / 300.0

            Example — Station 400456 (I-5, ~6 lanes), 7am rush:
              PeMS total flow: ~360 veh/5min
              Per lane: 360 / 6 = 60 veh/5min
              Per sec:  60 / 300 = 0.20 veh/sec/lane  ← realistic rush-hour rate
        """
        self._pems_profile = dict(hourly_profile)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def arrival_rate_per_lane(self, step: int, direction: str) -> float:
        """Return the Poisson arrival rate (veh/s/lane) for a step and direction.

        Parameters
        ----------
        step:
            Simulation step index (0-based).
        direction:
            One of "N", "S", "E", "W".

        Returns
        -------
        float
            Non-negative arrival rate in vehicles per second per lane.
        """
        hour = self._hour_of_day(step)
        base = 0.12

        # Phase 9C: when calibrated from real PeMS data, use the real hourly
        # rate directly (already captures the demand shape; skip synthetic peaks).
        if self._pems_profile is not None:
            hour_int = int(hour) % 24
            rate = self._pems_profile.get(hour_int, base)
            return max(0.02, rate * self.scale)

        profile = self.profile
        if profile == "rush_hour":
            rate = self._rush_hour(base, hour, direction)
        elif profile == "midday_peak":
            rate = self._midday_peak(base, hour)
        elif profile == "weekend":
            rate = self._weekend(base, hour)
        elif profile == "school_zone":
            rate = self._school_zone(base, hour, direction)
        elif profile == "event_surge":
            rate = self._event_surge(base, hour)
        elif profile == "construction":
            rate = self._construction(base, hour, direction)
        elif profile == "emergency_priority":
            rate = self._emergency_priority_rate(base, hour, direction, step)
        elif profile == "high_density_developing":
            rate = self._high_density_developing(base, hour, direction)
        elif profile == "incident_response":
            rate = self._incident_response(base, hour, direction, step)
        elif profile == "weather_degraded":
            rate = self._weather_degraded(base, hour)
        else:
            # normal
            rate = self._normal(base, hour)

        return max(0.02, rate * self.scale)

    def service_rate_multiplier(self, direction: str) -> float:
        """Return a service-rate multiplier for the engine's saturation flow.

        Used by the engine to adjust per-direction departure capacity beyond
        the simple arrival model. Returns 1.0 for all directions in most
        profiles except ``construction`` (0.5 for E/W) and
        ``incident_response`` (0.25 for affected direction at incident time).

        Parameters
        ----------
        direction:
            One of "N", "S", "E", "W".

        Returns
        -------
        float
            Multiplier in (0, 1]. Engine multiplies saturation rate by this.
        """
        if self.profile == "construction" and direction in ("E", "W"):
            return 0.5
        if self.profile == "weather_degraded":
            return 0.85
        if self.profile == "incident_response" and self._incident_active:
            if direction == self._incident_direction:
                return max(0.25, self._incident_capacity_fraction)
        return 1.0

    def pop_emergency_events(self) -> list[EmergencyEvent]:
        """Return and clear active emergency events for this step."""
        events = list(self._emergency_events)
        self._emergency_events = []
        return events

    def tick_emergency(self, step: int) -> None:
        """Progress emergency vehicle simulation for one step.

        Called by the engine each step when profile == 'emergency_priority'.
        Uses a Poisson process with mean 1 event per 500 steps.
        """
        if self.profile != "emergency_priority":
            return
        # Generate new emergency event with probability 1/500 per step
        if self._rng.random() < (1.0 / 500.0):
            directions = ["N", "S", "E", "W"]
            direction = self._rng.choice(directions)
            self._emergency_events.append(EmergencyEvent(
                direction=direction,
                steps_remaining=10,
                intersection_id=0,  # engine assigns to random intersection
            ))

    def tick_incident(self, step: int) -> None:
        """Update incident state for the incident_response profile."""
        if self.profile != "incident_response":
            return
        if step == 300:
            self._incident_active = True
            self._incident_direction = self._rng.choice(["E", "W"])
            self._incident_capacity_fraction = 0.25
        elif 300 < step < 500:
            self._incident_active = True
        elif 500 <= step < 600:
            # Gradual recovery over 100 steps
            recovery = (step - 500) / 100.0
            self._incident_capacity_fraction = 0.25 + 0.75 * recovery
        elif step >= 600:
            self._incident_active = False
            self._incident_capacity_fraction = 1.0

    # ------------------------------------------------------------------
    # Profile implementations
    # ------------------------------------------------------------------

    def _rush_hour(self, base: float, hour: float, direction: str) -> float:
        direction_multiplier = 1.1 if direction in ("N", "S") else 1.0
        morning = math.exp(-((hour - 8.0) ** 2) / 3.0)
        evening = math.exp(-((hour - 17.5) ** 2) / 3.0)
        peak = 1.0 + 1.6 * max(morning, evening)
        return base * direction_multiplier * peak

    def _midday_peak(self, base: float, hour: float) -> float:
        peak = 1.0 + 1.2 * math.exp(-((hour - 13.0) ** 2) / 4.0)
        return base * peak

    def _normal(self, base: float, hour: float) -> float:
        peak = 1.0 + 0.25 * math.sin(2 * math.pi * hour / 24.0)
        return base * peak

    def _weekend(self, base: float, hour: float) -> float:
        """Lower volume, single mid-day hump, no commute spikes."""
        mid_day = math.exp(-((hour - 12.5) ** 2) / (2 * 3.0 ** 2))
        peak = 1.0 + 0.8 * mid_day
        return base * 0.7 * peak

    def _school_zone(self, base: float, hour: float, direction: str) -> float:
        """Sharp morning and afternoon school spikes on N/S only."""
        sigma = 0.3
        morning_spike = math.exp(-((hour - 7.75) ** 2) / (2 * sigma ** 2))
        afternoon_spike = math.exp(-((hour - 15.0) ** 2) / (2 * sigma ** 2))
        ns_extra = 2.0 * max(morning_spike, afternoon_spike)
        if direction in ("N", "S"):
            return base * (1.0 + ns_extra)
        # E/W normal background
        return base * (1.0 + 0.2 * math.sin(2 * math.pi * hour / 24.0))

    def _event_surge(self, base: float, hour: float) -> float:
        """Stadium/concert event: pre-event surge, quiet during, exodus after."""
        if hour < 17.0:
            return base * self._normal(1.0, hour)
        elif 17.0 <= hour < 19.0:
            # Pre-event: 4× normal
            return base * 4.0
        elif 19.0 <= hour < 22.0:
            # During event: mostly parked, 0.5×
            return base * 0.5
        elif 22.0 <= hour < 23.5:
            # Post-event exodus: 3.5×
            return base * 3.5
        else:
            return base * 1.0

    def _construction(self, base: float, hour: float, direction: str) -> float:
        """E/W lane closure: higher E/W arrivals (backup), normal N/S.

        Service rate is reduced separately via service_rate_multiplier().
        """
        morning = math.exp(-((hour - 8.0) ** 2) / 3.0)
        evening = math.exp(-((hour - 17.5) ** 2) / 3.0)
        peak = 1.0 + 1.6 * max(morning, evening)
        if direction in ("E", "W"):
            # Backup on E/W: arrivals increase because reduced throughput
            return base * peak * 1.5
        return base * peak

    def _emergency_priority_rate(
        self, base: float, hour: float, direction: str, step: int
    ) -> float:
        """Normal traffic rates; emergency events are injected via tick_emergency()."""
        return self._rush_hour(base, hour, direction)

    def _high_density_developing(self, base: float, hour: float, direction: str) -> float:
        """Lagos/Mumbai/Dhaka-style: 3× base rate, extended peak hours.

        30% signal non-compliance is handled by the engine reading
        service_rate_multiplier() for red-direction departures.
        Peak hours: 7-10 AM and 4-8 PM.
        """
        # Extended twin peaks
        morning = math.exp(-((hour - 8.5) ** 2) / (2 * 1.5 ** 2))
        evening = math.exp(-((hour - 18.0) ** 2) / (2 * 2.0 ** 2))
        rush = 1.0 + 3.0 * max(morning, evening)
        # Direction multiplier (motorcycles lighter on E/W corridors)
        dir_factor = 1.05 if direction in ("N", "S") else 0.95
        return base * 3.0 * rush * dir_factor

    def _incident_response(self, base: float, hour: float, direction: str, step: int) -> float:
        """Normal traffic until step 300 capacity loss on one direction."""
        rate = self._rush_hour(base, hour, direction)
        if self._incident_active and direction == self._incident_direction:
            # Arrivals build up upstream
            rate *= 1.8
        return rate

    def _weather_degraded(self, base: float, hour: float) -> float:
        """Rain: 1.3× arrivals (fewer pedestrians/cyclists → more drivers)."""
        peak = 1.0 + 0.25 * math.sin(2 * math.pi * hour / 24.0)
        return base * 1.3 * peak

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def _hour_of_day(self, step: int) -> float:
        elapsed_seconds = step * self.step_seconds
        return (elapsed_seconds / 3600.0) % 24.0

    def noncompliance_rate(self) -> float:
        """Fraction of vehicles that depart on red (high-density profile only)."""
        if self.profile == "high_density_developing":
            return 0.30
        return 0.0
