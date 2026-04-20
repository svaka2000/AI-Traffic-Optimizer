"""aito/data/turn_movement_estimator.py

GF5: Turn-Movement Estimation from Connected Vehicle Trajectories.

Derives NEMA turn-movement counts (LT / TH / RT per approach) from
raw GPS trajectories — enabling full Webster/NSGA-III optimization with
zero loop detectors or manual counts.

Algorithm:
  1. Classify each trajectory passing through an intersection bbox into
     an approach direction (N/S/E/W) based on entry heading.
  2. Classify the exit movement (LT/TH/RT) from the heading change
     between approach and departure vectors.
  3. Aggregate counts over the observation window.
  4. Apply Bayesian correction for CV penetration rate to extrapolate
     true demand.
  5. Return DemandProfile compatible with AITO optimization.

Heading convention: bearing in [0, 360) degrees, clockwise from North.

Reference:
  Zheng et al. (2024), Nature Communications 15, 4821.
  GM OnStar Birmingham MI pilot — 28% CV penetration, <5% count error.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

from aito.models import DemandProfile


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------

def bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute geodetic bearing from point 1 to point 2 (degrees, 0=N, CW)."""
    lat1_r = math.radians(lat1)
    lat2_r = math.radians(lat2)
    d_lon = math.radians(lon2 - lon1)
    x = math.sin(d_lon) * math.cos(lat2_r)
    y = (math.cos(lat1_r) * math.sin(lat2_r)
         - math.sin(lat1_r) * math.cos(lat2_r) * math.cos(d_lon))
    brg = math.degrees(math.atan2(x, y)) % 360.0
    return brg


def heading_change(hdg_in: float, hdg_out: float) -> float:
    """Signed heading change in [-180, 180] from approach to departure."""
    delta = (hdg_out - hdg_in + 540.0) % 360.0 - 180.0
    return delta


class Approach(str, Enum):
    NORTH = "N"   # vehicle approaching from North (travelling South)
    SOUTH = "S"
    EAST  = "E"
    WEST  = "W"
    UNKNOWN = "?"


class Movement(str, Enum):
    LEFT_TURN   = "LT"
    THROUGH     = "TH"
    RIGHT_TURN  = "RT"
    U_TURN      = "UT"
    UNKNOWN     = "?"


def classify_approach(entry_bearing: float) -> Approach:
    """Map entry bearing to NEMA approach direction.

    A vehicle bearing ~180° is heading south → it's on the NB approach.
    """
    # Normalize: bearing of the vehicle's direction of travel
    b = entry_bearing % 360.0
    if 315 <= b or b < 45:
        return Approach.NORTH   # heading north → SB approach (but called NB thru)
    if 45 <= b < 135:
        return Approach.EAST
    if 135 <= b < 225:
        return Approach.SOUTH
    return Approach.WEST


def classify_movement(delta_heading: float) -> Movement:
    """Classify turn movement from signed heading change.

    Thresholds per MUTCD geometric definitions:
      Right turn:  delta ∈ [+45, +135]
      Left turn:   delta ∈ [-135, -45]
      Through:     |delta| < 45
      U-turn:      |delta| > 135
    """
    if -45 < delta_heading < 45:
        return Movement.THROUGH
    if 45 <= delta_heading <= 135:
        return Movement.RIGHT_TURN
    if -135 <= delta_heading <= -45:
        return Movement.LEFT_TURN
    return Movement.U_TURN


# ---------------------------------------------------------------------------
# Trajectory data model
# ---------------------------------------------------------------------------

@dataclass
class TrajectoryPoint:
    """Single GPS fix from a connected vehicle."""
    vehicle_id: str
    timestamp: datetime
    latitude: float
    longitude: float
    speed_mph: float = 0.0
    heading: Optional[float] = None   # degrees; computed if None


@dataclass
class VehicleTrajectory:
    """Ordered sequence of GPS fixes for one vehicle pass-through."""
    vehicle_id: str
    points: list[TrajectoryPoint] = field(default_factory=list)

    def add_point(self, pt: TrajectoryPoint) -> None:
        self.points.append(pt)
        self.points.sort(key=lambda p: p.timestamp)

    @property
    def duration_s(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return (self.points[-1].timestamp - self.points[0].timestamp).total_seconds()

    def entry_bearing(self) -> Optional[float]:
        """Bearing of approach vector using first two points."""
        if len(self.points) < 2:
            return None
        p0, p1 = self.points[0], self.points[1]
        return bearing(p0.latitude, p0.longitude, p1.latitude, p1.longitude)

    def exit_bearing(self) -> Optional[float]:
        """Bearing of departure vector using last two points."""
        if len(self.points) < 2:
            return None
        pm1, pm2 = self.points[-2], self.points[-1]
        return bearing(pm1.latitude, pm1.longitude, pm2.latitude, pm2.longitude)


# ---------------------------------------------------------------------------
# Intersection bounding box
# ---------------------------------------------------------------------------

@dataclass
class IntersectionBBox:
    """Geographic bounding box for filtering trajectories at an intersection."""
    center_lat: float
    center_lon: float
    radius_m: float = 75.0   # capture zone radius

    def contains(self, lat: float, lon: float) -> bool:
        """Check if a point is within radius_m of intersection center."""
        dlat = math.radians(lat - self.center_lat)
        dlon = math.radians(lon - self.center_lon)
        a = (math.sin(dlat / 2) ** 2
             + math.cos(math.radians(self.center_lat))
             * math.cos(math.radians(lat))
             * math.sin(dlon / 2) ** 2)
        dist_m = 6371000.0 * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return dist_m <= self.radius_m


# ---------------------------------------------------------------------------
# Classified movement record
# ---------------------------------------------------------------------------

@dataclass
class TurnMovementRecord:
    """One classified vehicle turn movement through an intersection."""
    vehicle_id: str
    timestamp: datetime
    approach: Approach
    movement: Movement
    entry_bearing_deg: float
    exit_bearing_deg: float
    delta_heading_deg: float
    travel_time_s: float
    confidence: float = 1.0


# ---------------------------------------------------------------------------
# Count aggregation
# ---------------------------------------------------------------------------

@dataclass
class TurnMovementCounts:
    """Raw (un-expanded) turn movement counts per approach × movement."""
    intersection_id: str
    period_start: datetime
    period_end: datetime
    # probe count per approach per movement (before penetration expansion)
    N_LT: int = 0; N_TH: int = 0; N_RT: int = 0
    S_LT: int = 0; S_TH: int = 0; S_RT: int = 0
    E_LT: int = 0; E_TH: int = 0; E_RT: int = 0
    W_LT: int = 0; W_TH: int = 0; W_RT: int = 0
    total_observed: int = 0
    penetration_rate: float = 0.28   # default Birmingham MI baseline

    def expand(self) -> "TurnMovementCounts":
        """Apply penetration rate expansion to get true demand counts."""
        factor = 1.0 / max(self.penetration_rate, 0.01)
        expanded = TurnMovementCounts(
            intersection_id=self.intersection_id,
            period_start=self.period_start,
            period_end=self.period_end,
            N_LT=round(self.N_LT * factor), N_TH=round(self.N_TH * factor), N_RT=round(self.N_RT * factor),
            S_LT=round(self.S_LT * factor), S_TH=round(self.S_TH * factor), S_RT=round(self.S_RT * factor),
            E_LT=round(self.E_LT * factor), E_TH=round(self.E_TH * factor), E_RT=round(self.E_RT * factor),
            W_LT=round(self.W_LT * factor), W_TH=round(self.W_TH * factor), W_RT=round(self.W_RT * factor),
            total_observed=self.total_observed,
            penetration_rate=self.penetration_rate,
        )
        return expanded

    def to_demand_profile(self, period_minutes: int = 15) -> DemandProfile:
        """Convert to AITO DemandProfile (vehicles/hour).

        Probe counts are for the observation period; scale to veh/hr.
        """
        expanded = self.expand()
        scale = 60.0 / max(period_minutes, 1)  # convert period counts → veh/hr

        return DemandProfile(
            intersection_id=self.intersection_id,
            period_minutes=period_minutes,
            # NEMA phase mapping:
            # Phase 2 = NB thru, Phase 4 = SB thru, Phase 6 = EB thru, Phase 8 = WB thru
            # Phase 1 = NB left, Phase 3 = SB left, Phase 5 = EB left, Phase 7 = WB left
            north_thru=expanded.N_TH * scale,
            north_left=expanded.N_LT * scale,
            north_right=expanded.N_RT * scale,
            south_thru=expanded.S_TH * scale,
            south_left=expanded.S_LT * scale,
            south_right=expanded.S_RT * scale,
            east_thru=expanded.E_TH * scale,
            east_left=expanded.E_LT * scale,
            east_right=expanded.E_RT * scale,
            west_thru=expanded.W_TH * scale,
            west_left=expanded.W_LT * scale,
            west_right=expanded.W_RT * scale,
        )


# ---------------------------------------------------------------------------
# TrajectoryTurnMovementEstimator — main class
# ---------------------------------------------------------------------------

class TrajectoryTurnMovementEstimator:
    """Estimates turn movements from CV trajectory data.

    Parameters
    ----------
    intersection_id : str
    bbox : IntersectionBBox
        Geographic capture zone around intersection.
    min_points_in_zone : int
        Minimum GPS fixes inside the bbox to classify a trajectory.
    min_trajectory_duration_s : float
        Filter out stationary / parked vehicles.
    penetration_rate : float
        Estimated CV market penetration [0, 1].
    """

    def __init__(
        self,
        intersection_id: str,
        bbox: IntersectionBBox,
        min_points_in_zone: int = 2,
        min_trajectory_duration_s: float = 2.0,
        penetration_rate: float = 0.28,
    ) -> None:
        self.intersection_id = intersection_id
        self.bbox = bbox
        self.min_points_in_zone = min_points_in_zone
        self.min_trajectory_duration_s = min_trajectory_duration_s
        self.penetration_rate = penetration_rate
        self._records: list[TurnMovementRecord] = []

    def process_trajectory(self, traj: VehicleTrajectory) -> Optional[TurnMovementRecord]:
        """Classify one vehicle trajectory into a turn movement.

        Returns None if the trajectory does not pass through the intersection
        or cannot be reliably classified.
        """
        # Filter points inside the intersection bbox
        zone_pts = [p for p in traj.points
                    if self.bbox.contains(p.latitude, p.longitude)]
        if len(zone_pts) < self.min_points_in_zone:
            return None
        if traj.duration_s < self.min_trajectory_duration_s:
            return None

        entry_brg = traj.entry_bearing()
        exit_brg = traj.exit_bearing()
        if entry_brg is None or exit_brg is None:
            return None

        delta = heading_change(entry_brg, exit_brg)
        approach = classify_approach(entry_brg)
        movement = classify_movement(delta)

        if approach == Approach.UNKNOWN or movement == Movement.UNKNOWN:
            return None

        # Confidence: degrade for very short traces or high heading uncertainty
        n_pts = len(zone_pts)
        conf = min(1.0, 0.6 + 0.1 * n_pts)

        record = TurnMovementRecord(
            vehicle_id=traj.vehicle_id,
            timestamp=zone_pts[0].timestamp,
            approach=approach,
            movement=movement,
            entry_bearing_deg=round(entry_brg, 1),
            exit_bearing_deg=round(exit_brg, 1),
            delta_heading_deg=round(delta, 1),
            travel_time_s=round(traj.duration_s, 1),
            confidence=round(conf, 2),
        )
        self._records.append(record)
        return record

    def process_batch(self, trajectories: list[VehicleTrajectory]) -> list[TurnMovementRecord]:
        """Process a batch of trajectories."""
        results = []
        for traj in trajectories:
            rec = self.process_trajectory(traj)
            if rec is not None:
                results.append(rec)
        return results

    def aggregate(
        self,
        period_start: datetime,
        period_end: datetime,
        min_confidence: float = 0.6,
    ) -> TurnMovementCounts:
        """Aggregate classified records into turn movement counts.

        Parameters
        ----------
        period_start, period_end : datetime
            Time window to aggregate.
        min_confidence : float
            Only include records with confidence >= threshold.
        """
        counts = TurnMovementCounts(
            intersection_id=self.intersection_id,
            period_start=period_start,
            period_end=period_end,
            penetration_rate=self.penetration_rate,
        )

        in_window = [
            r for r in self._records
            if period_start <= r.timestamp < period_end
            and r.confidence >= min_confidence
        ]
        counts.total_observed = len(in_window)

        _attr_map: dict[tuple[Approach, Movement], str] = {
            (Approach.NORTH, Movement.LEFT_TURN):  "N_LT",
            (Approach.NORTH, Movement.THROUGH):    "N_TH",
            (Approach.NORTH, Movement.RIGHT_TURN): "N_RT",
            (Approach.SOUTH, Movement.LEFT_TURN):  "S_LT",
            (Approach.SOUTH, Movement.THROUGH):    "S_TH",
            (Approach.SOUTH, Movement.RIGHT_TURN): "S_RT",
            (Approach.EAST,  Movement.LEFT_TURN):  "E_LT",
            (Approach.EAST,  Movement.THROUGH):    "E_TH",
            (Approach.EAST,  Movement.RIGHT_TURN): "E_RT",
            (Approach.WEST,  Movement.LEFT_TURN):  "W_LT",
            (Approach.WEST,  Movement.THROUGH):    "W_TH",
            (Approach.WEST,  Movement.RIGHT_TURN): "W_RT",
        }
        for rec in in_window:
            attr = _attr_map.get((rec.approach, rec.movement))
            if attr:
                setattr(counts, attr, getattr(counts, attr) + 1)

        return counts

    def estimate_demand_profile(
        self,
        period_start: datetime,
        period_end: Optional[datetime] = None,
        period_minutes: int = 15,
    ) -> DemandProfile:
        """Full pipeline: aggregate → expand → return DemandProfile."""
        if period_end is None:
            period_end = period_start + timedelta(minutes=period_minutes)
        counts = self.aggregate(period_start, period_end)
        return counts.to_demand_profile(period_minutes=period_minutes)

    def clear(self) -> None:
        """Reset buffered records."""
        self._records.clear()

    @property
    def record_count(self) -> int:
        return len(self._records)


# ---------------------------------------------------------------------------
# Synthetic trajectory generator (for testing / demo)
# ---------------------------------------------------------------------------

def generate_synthetic_trajectories(
    intersection_lat: float,
    intersection_lon: float,
    n_vehicles: int = 200,
    period_start: Optional[datetime] = None,
    period_minutes: int = 15,
    penetration_rate: float = 0.28,
    demand_mix: Optional[dict[str, float]] = None,
    seed: int = 42,
) -> list[VehicleTrajectory]:
    """Generate synthetic CV trajectories through an intersection.

    demand_mix: fraction of total demand per approach+movement.
    Defaults to a typical urban arterial distribution.
    """
    import random
    rng = random.Random(seed)

    if period_start is None:
        period_start = datetime(2025, 6, 1, 7, 0)

    # Default arterial demand mix (fraction of all vehicles)
    if demand_mix is None:
        demand_mix = {
            "N_TH": 0.28, "N_LT": 0.06, "N_RT": 0.04,
            "S_TH": 0.22, "S_LT": 0.05, "S_RT": 0.04,
            "E_TH": 0.12, "E_LT": 0.05, "E_RT": 0.03,
            "W_TH": 0.06, "W_LT": 0.03, "W_RT": 0.02,
        }

    # Approach entry bearings (vehicle heading into intersection)
    approach_bearings: dict[str, float] = {
        "N": 180.0, "S": 0.0, "E": 270.0, "W": 90.0
    }
    # Turn delta headings (degrees change)
    movement_deltas: dict[str, float] = {
        "LT": -90.0, "TH": 0.0, "RT": 90.0
    }

    trajectories: list[VehicleTrajectory] = []
    # Only a penetration_rate fraction of vehicles are CVs
    n_cv = max(1, round(n_vehicles * penetration_rate))

    # Build cumulative distribution for sampling
    keys = list(demand_mix.keys())
    cumulative: list[float] = []
    acc = 0.0
    for k in keys:
        acc += demand_mix[k]
        cumulative.append(acc)
    total = acc

    for vid in range(n_cv):
        r = rng.uniform(0, total)
        chosen = keys[0]
        for k, cum in zip(keys, cumulative):
            if r <= cum:
                chosen = k
                break
        approach_str, movement_str = chosen.split("_")
        entry_brg = approach_bearings[approach_str] + rng.gauss(0, 8)
        delta = movement_deltas[movement_str] + rng.gauss(0, 10)
        exit_brg = (entry_brg + delta) % 360.0

        # Generate 4–6 GPS points spanning the intersection
        t_start = period_start + timedelta(seconds=rng.uniform(0, period_minutes * 60))
        n_pts = rng.randint(4, 7)
        travel_time = rng.uniform(8.0, 25.0)
        dt = travel_time / (n_pts - 1)

        pts: list[TrajectoryPoint] = []
        offset_m = 60.0  # approach distance
        cos_lat = math.cos(math.radians(intersection_lat))
        for j in range(n_pts):
            frac = j / (n_pts - 1)
            if frac < 0.5:
                # Approaching: vehicle coming FROM the opposite of entry_brg direction.
                # Negate so point starts on the far side and moves toward center.
                brg_r = math.radians(entry_brg)
                dlat = -math.cos(brg_r) * offset_m * (0.5 - frac) / 111111
                dlon = -math.sin(brg_r) * offset_m * (0.5 - frac) / (111111 * cos_lat)
            else:
                # Departing: move in exit_brg direction away from center
                brg_r = math.radians(exit_brg)
                dlat = math.cos(brg_r) * offset_m * (frac - 0.5) / 111111
                dlon = math.sin(brg_r) * offset_m * (frac - 0.5) / (111111 * cos_lat)

            lat = intersection_lat + dlat + rng.gauss(0, 3e-6)
            lon = intersection_lon + dlon + rng.gauss(0, 3e-6)
            pts.append(TrajectoryPoint(
                vehicle_id=f"v{vid:04d}",
                timestamp=t_start + timedelta(seconds=dt * j),
                latitude=lat,
                longitude=lon,
                speed_mph=rng.uniform(10, 35),
                heading=(entry_brg if frac < 0.5 else exit_brg),
            ))

        traj = VehicleTrajectory(vehicle_id=f"v{vid:04d}", points=pts)
        trajectories.append(traj)

    return trajectories


# ---------------------------------------------------------------------------
# Multi-intersection pipeline
# ---------------------------------------------------------------------------

def estimate_corridor_demands(
    intersections: list,  # list[Intersection]
    trajectories_per_intersection: list[list[VehicleTrajectory]],
    period_start: datetime,
    period_minutes: int = 15,
    penetration_rate: float = 0.28,
) -> list[DemandProfile]:
    """Estimate demand profiles for all intersections in a corridor.

    Parameters
    ----------
    intersections : list[Intersection]
        Corridor intersections in order.
    trajectories_per_intersection : list[list[VehicleTrajectory]]
        One trajectory list per intersection.
    """
    assert len(intersections) == len(trajectories_per_intersection)

    profiles: list[DemandProfile] = []
    for ix, trajs in zip(intersections, trajectories_per_intersection):
        bbox = IntersectionBBox(
            center_lat=ix.latitude,
            center_lon=ix.longitude,
            radius_m=80.0,
        )
        estimator = TrajectoryTurnMovementEstimator(
            intersection_id=ix.id,
            bbox=bbox,
            penetration_rate=penetration_rate,
        )
        estimator.process_batch(trajs)
        profile = estimator.estimate_demand_profile(period_start, period_minutes=period_minutes)
        profiles.append(profile)

    return profiles
