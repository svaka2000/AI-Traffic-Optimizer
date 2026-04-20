"""aito/optimization/spillback_predictor.py

GF12: Queue Spillback Prediction & Prevention.

Predicts when upstream queue discharge will block downstream intersection
green phases, causing spillback-induced gridlock.  Provides automatic
cycle and split adjustments to prevent spillback before it occurs.

Algorithm:
  1. Estimate queue length at each intersection from probe travel times
     and stop-bar occupancy inference.
  2. Compare queue length to available storage capacity (approach bay length).
  3. Predict spillback time = (storage_remaining / discharge_rate)
  4. If spillback_time < warning_horizon_s → trigger alert
  5. Compute corrective split adjustment to accelerate queue discharge.

Physics model:
  - Shockwave theory (LWR model, Lighthill-Whitham-Richards 1955)
  - Saturation flow = 1800 veh/hr/ln (HCM default)
  - Jam density = 0.125 veh/m (8m headway at standstill)
  - Queue discharge rate = effective_green / cycle × saturation_flow

References:
  Lighthill, M. J. & Whitham, G. B. (1955). "On kinematic waves: II.
  A theory of traffic flow on long crowded roads."
  Proc. Royal Society London A, 229, 317–345.

  HCM 7th Edition, Chapter 31 (Queue Storage Estimation).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SAT_FLOW_VEH_HR_LN = 1800.0     # HCM saturation flow rate (veh/hr/lane)
JAM_DENSITY_VEH_M = 0.125       # vehicles per meter (8m avg headway at jam)
FREE_FLOW_DENSITY_VEH_M = 0.02  # 50m headway at free flow
AVG_VEHICLE_LENGTH_M = 6.5      # including headway at cruise


# ---------------------------------------------------------------------------
# Queue estimation
# ---------------------------------------------------------------------------

def estimate_queue_length_m(
    volume_veh_hr: float,
    capacity_veh_hr: float,
    cycle_s: float,
    green_s: float,
    n_lanes: int = 1,
) -> float:
    """Estimate maximum queue length (meters) using D/D/1 deterministic model.

    For oversaturated conditions (v/c > 1), queue grows without bound;
    we cap at 300m (length of two standard approach bays).

    Returns queue length per lane.
    """
    if capacity_veh_hr <= 0 or n_lanes <= 0:
        return 0.0

    v_c = volume_veh_hr / max(capacity_veh_hr, 1.0)
    red_time = cycle_s - green_s

    if v_c >= 1.0:
        # Oversaturated: queue grows each cycle
        overflow_per_cycle = (volume_veh_hr / 3600.0) * cycle_s * (v_c - 1.0)
        q_veh = max(0, red_time * volume_veh_hr / 3600.0 + overflow_per_cycle)
    else:
        # Webster uniform delay model queue
        arrival_rate = volume_veh_hr / 3600.0 / max(n_lanes, 1)
        q_veh = arrival_rate * red_time  # peak queue at end of red

    # Convert vehicles to distance
    q_m = q_veh * (1.0 / JAM_DENSITY_VEH_M)  # meters at jam density
    return min(q_m, 300.0)


def estimate_discharge_rate_veh_s(
    green_s: float,
    cycle_s: float,
    n_lanes: int = 1,
) -> float:
    """Vehicles discharged per second during effective green."""
    sat_veh_s = SAT_FLOW_VEH_HR_LN / 3600.0 * n_lanes
    # Net discharge rate (arrivals during green continue; only net is relevant)
    return sat_veh_s * (green_s / max(cycle_s, 1.0))


# ---------------------------------------------------------------------------
# Spillback prediction
# ---------------------------------------------------------------------------

class SpillbackRisk(str, Enum):
    NONE     = "none"       # queue well within storage
    LOW      = "low"        # queue at 60–80% of storage
    MODERATE = "moderate"   # queue at 80–95% of storage
    HIGH     = "high"       # queue >95% of storage, spillback imminent
    ACTIVE   = "active"     # queue exceeds storage, spillback occurring


@dataclass
class IntersectionStorage:
    """Available storage on one approach of an intersection."""
    intersection_id: str
    approach_direction: str         # N, S, E, W
    storage_length_m: float         # distance from stop bar to upstream intersection
    n_lanes: int = 1
    # Optionally reduced by parking, driveways, etc.
    effective_storage_m: Optional[float] = None

    @property
    def capacity_veh(self) -> float:
        eff = self.effective_storage_m or self.storage_length_m
        return eff * JAM_DENSITY_VEH_M * self.n_lanes

    @property
    def usable_m(self) -> float:
        return self.effective_storage_m or self.storage_length_m


@dataclass
class SpillbackForecast:
    """Spillback risk forecast for one approach."""
    timestamp: datetime
    intersection_id: str
    approach_direction: str
    queue_length_m: float
    storage_length_m: float
    v_c_ratio: float
    risk: SpillbackRisk
    time_to_spillback_s: Optional[float]    # None if no spillback predicted
    recommended_green_increase_s: float     # seconds to add to prevent spillback
    description: str


def forecast_spillback(
    storage: IntersectionStorage,
    volume_veh_hr: float,
    capacity_veh_hr: float,
    cycle_s: float,
    green_s: float,
    now: datetime,
    warning_horizon_s: float = 300.0,
) -> SpillbackForecast:
    """Predict spillback risk for one approach.

    Parameters
    ----------
    storage : IntersectionStorage
    volume_veh_hr : float
    capacity_veh_hr : float
    cycle_s : float
    green_s : float
    now : datetime
    warning_horizon_s : float
        Alert horizon — if spillback predicted within this window, flag HIGH risk.
    """
    q_m = estimate_queue_length_m(volume_veh_hr, capacity_veh_hr, cycle_s, green_s, storage.n_lanes)
    v_c = volume_veh_hr / max(capacity_veh_hr, 1.0)
    storage_remaining = max(0.0, storage.usable_m - q_m)

    # Time to spillback: remaining_storage / queue_growth_rate
    if v_c >= 1.0:
        # Queue grows each cycle
        overflow_veh_per_s = (volume_veh_hr / 3600.0) * (v_c - 1.0)
        overflow_m_per_s = overflow_veh_per_s / JAM_DENSITY_VEH_M
        if overflow_m_per_s > 0:
            time_to_spill = storage_remaining / overflow_m_per_s
        else:
            time_to_spill = None
    elif q_m >= storage.usable_m:
        time_to_spill = 0.0
    else:
        # Under-saturated: no spillback
        time_to_spill = None

    # Risk classification
    fill_ratio = q_m / max(storage.usable_m, 1.0)
    if q_m > storage.usable_m:
        risk = SpillbackRisk.ACTIVE
    elif fill_ratio > 0.95:
        risk = SpillbackRisk.HIGH
    elif fill_ratio > 0.80:
        risk = SpillbackRisk.MODERATE
    elif fill_ratio > 0.60:
        risk = SpillbackRisk.LOW
    else:
        risk = SpillbackRisk.NONE

    # Override to HIGH if spillback predicted within warning horizon
    if time_to_spill is not None and time_to_spill < warning_horizon_s:
        if risk not in (SpillbackRisk.ACTIVE, SpillbackRisk.HIGH):
            risk = SpillbackRisk.HIGH

    # Recommended green increase to reduce v/c below 0.90
    target_vc = 0.90
    if v_c > target_vc:
        # Additional green needed: G_new / C × SAT × n_lanes = V × target_vc
        target_cap = volume_veh_hr / target_vc
        target_green_ratio = target_cap / (SAT_FLOW_VEH_HR_LN * storage.n_lanes)
        current_green_ratio = green_s / max(cycle_s, 1.0)
        green_increase = max(0.0, (target_green_ratio - current_green_ratio) * cycle_s)
    else:
        green_increase = 0.0

    desc_parts = [
        f"Queue={q_m:.0f}m / Storage={storage.usable_m:.0f}m ({fill_ratio * 100:.0f}%)",
        f"v/c={v_c:.2f}",
    ]
    if time_to_spill is not None:
        desc_parts.append(f"Spillback in {time_to_spill:.0f}s")
    if green_increase > 0:
        desc_parts.append(f"Add {green_increase:.1f}s green")
    desc = " | ".join(desc_parts)

    return SpillbackForecast(
        timestamp=now,
        intersection_id=storage.intersection_id,
        approach_direction=storage.approach_direction,
        queue_length_m=round(q_m, 1),
        storage_length_m=storage.usable_m,
        v_c_ratio=round(v_c, 3),
        risk=risk,
        time_to_spillback_s=round(time_to_spill, 0) if time_to_spill is not None else None,
        recommended_green_increase_s=round(green_increase, 1),
        description=desc,
    )


# ---------------------------------------------------------------------------
# Corridor-level spillback scan
# ---------------------------------------------------------------------------

@dataclass
class CorridorSpillbackReport:
    """Spillback assessment for all approaches on a corridor."""
    corridor_id: str
    timestamp: datetime
    forecasts: list[SpillbackForecast]
    active_spillbacks: list[SpillbackForecast]
    high_risk_approaches: list[SpillbackForecast]

    @property
    def worst_risk(self) -> SpillbackRisk:
        order = [SpillbackRisk.NONE, SpillbackRisk.LOW, SpillbackRisk.MODERATE,
                 SpillbackRisk.HIGH, SpillbackRisk.ACTIVE]
        if not self.forecasts:
            return SpillbackRisk.NONE
        return max(self.forecasts, key=lambda f: order.index(f.risk)).risk

    @property
    def requires_immediate_action(self) -> bool:
        return len(self.active_spillbacks) > 0

    def adjustment_summary(self) -> dict[str, float]:
        """Per-intersection recommended green increases (seconds)."""
        result: dict[str, float] = {}
        for f in self.forecasts:
            if f.recommended_green_increase_s > 0:
                key = f"{f.intersection_id}_{f.approach_direction}"
                result[key] = max(result.get(key, 0.0), f.recommended_green_increase_s)
        return result


class SpillbackPredictor:
    """Corridor-wide spillback prediction and prevention.

    Usage:
        predictor = SpillbackPredictor(corridor_id, storages)
        report = predictor.scan(volumes_veh_hr, capacities, timing_plan)
        if report.requires_immediate_action:
            adjustments = report.adjustment_summary()
    """

    def __init__(
        self,
        corridor_id: str,
        storages: list[IntersectionStorage],
        warning_horizon_s: float = 300.0,
    ) -> None:
        self.corridor_id = corridor_id
        self.storages = storages
        self.warning_horizon_s = warning_horizon_s

    def scan(
        self,
        volumes_veh_hr: list[float],
        capacities_veh_hr: list[float],
        cycle_s: float,
        green_ratios: list[float],
        now: Optional[datetime] = None,
    ) -> CorridorSpillbackReport:
        """Run spillback scan for all approaches.

        Parameters
        ----------
        volumes_veh_hr : list[float]
            One per storage in the same order as self.storages.
        capacities_veh_hr : list[float]
            Phase capacity per storage.
        cycle_s : float
            Common cycle length.
        green_ratios : list[float]
            Effective green ratio per storage.
        """
        if now is None:
            now = datetime.utcnow()

        assert len(volumes_veh_hr) == len(self.storages)
        assert len(capacities_veh_hr) == len(self.storages)

        forecasts: list[SpillbackForecast] = []
        for storage, vol, cap, gr in zip(
            self.storages, volumes_veh_hr, capacities_veh_hr, green_ratios
        ):
            forecast = forecast_spillback(
                storage=storage,
                volume_veh_hr=vol,
                capacity_veh_hr=cap,
                cycle_s=cycle_s,
                green_s=gr * cycle_s,
                now=now,
                warning_horizon_s=self.warning_horizon_s,
            )
            forecasts.append(forecast)

        active = [f for f in forecasts if f.risk == SpillbackRisk.ACTIVE]
        high_risk = [f for f in forecasts if f.risk == SpillbackRisk.HIGH]

        return CorridorSpillbackReport(
            corridor_id=self.corridor_id,
            timestamp=now,
            forecasts=forecasts,
            active_spillbacks=active,
            high_risk_approaches=high_risk,
        )

    @staticmethod
    def from_corridor(corridor, distances_m: Optional[list[float]] = None) -> "SpillbackPredictor":
        """Build a predictor from a Corridor model.

        Uses inter-signal spacing as storage length proxy.
        """
        distances = distances_m or getattr(corridor, "distances_m", [])
        storages: list[IntersectionStorage] = []

        for i, ix in enumerate(corridor.intersections):
            # Upstream storage = distance to upstream intersection
            if i > 0:
                storage_m = distances[i - 1] * 0.85  # 85% usable (driveways, parking)
            else:
                storage_m = 200.0  # terminal intersection default

            for approach in ["N", "S", "E", "W"]:
                storages.append(IntersectionStorage(
                    intersection_id=ix.id,
                    approach_direction=approach,
                    storage_length_m=storage_m,
                    n_lanes=2,  # typical arterial
                ))

        return SpillbackPredictor(
            corridor_id=corridor.id,
            storages=storages,
        )
