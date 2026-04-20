"""aito/models.py

Core Pydantic models for the AITO platform.

These are the authoritative data types for all timing plans, findings,
analysis results, and reports.  No raw dicts cross module boundaries.
"""
from __future__ import annotations

import hashlib
from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import uuid4

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class ConfidenceLevel(str, Enum):
    CONFIRMED = "confirmed"     # 2+ independent evidence sources
    PROBABLE = "probable"       # 1 strong evidence source
    POSSIBLE = "possible"       # circumstantial evidence
    UNVERIFIED = "unverified"   # single weak source


class OptimizationObjective(str, Enum):
    DELAY = "delay"
    EMISSIONS = "emissions"
    STOPS = "stops"
    SAFETY = "safety"
    EQUITY = "equity"


class AnalysisPhase(str, Enum):
    TRIAGE = "triage"
    OPTIMIZATION = "optimization"
    VALIDATION = "validation"
    REPORTING = "reporting"


class DeploymentStatus(str, Enum):
    PENDING = "pending"
    STAGED = "staged"
    ACTIVE = "active"
    FAILED = "failed"
    ROLLED_BACK = "rolled_back"


# ---------------------------------------------------------------------------
# Signal / Intersection geometry
# ---------------------------------------------------------------------------

class DetectorConfig(BaseModel):
    id: str
    phase: int
    distance_from_stop_bar: float  # feet
    type: str = "loop"  # loop, video, radar, radar_advance
    operational: bool = True


class PreemptionConfig(BaseModel):
    railroad_enabled: bool = False
    evp_enabled: bool = True
    evp_phases: list[int] = Field(default_factory=lambda: [2, 6])


class RingBarrierConfig(BaseModel):
    """NEMA TS-2 ring-barrier-channel configuration."""
    rings: int = 2
    barriers: int = 2
    # phases[ring][position] — standard 8-phase NEMA layout
    phases: list[list[int]] = Field(
        default_factory=lambda: [[1, 2, 3, 4], [5, 6, 7, 8]]
    )
    barrier_positions: list[int] = Field(default_factory=lambda: [2, 6])


class PhaseConfig(BaseModel):
    id: int
    min_green: float = 7.0          # seconds (MUTCD minimum)
    max_green: float = 60.0
    yellow: float = 4.0
    all_red: float = 2.0
    ped_walk: Optional[float] = None    # None = no pedestrian
    ped_clearance: Optional[float] = None
    is_protected_left: bool = False
    concurrent_phases: list[int] = Field(default_factory=list)


class Intersection(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    latitude: float
    longitude: float
    controller_type: str = "MAXTIME"  # MAXTIME, Cobalt, 2070, ATC
    num_phases: int = 8
    ring_barrier_config: RingBarrierConfig = Field(default_factory=RingBarrierConfig)
    phase_configs: list[PhaseConfig] = Field(default_factory=list)
    detector_config: list[DetectorConfig] = Field(default_factory=list)
    pedestrian_phases: list[int] = Field(default_factory=lambda: [2, 4, 6, 8])
    preemption_config: PreemptionConfig = Field(default_factory=PreemptionConfig)
    ntcip_address: Optional[str] = None   # IP:port for SNMP
    ntcip_community: str = "public"
    # Geometric / demand data
    crossing_distance_ft: float = 60.0    # longest ped crossing in feet
    approach_speed_mph: float = 35.0      # posted speed limit
    aadt: int = 20000                     # annual average daily traffic

    @property
    def crossing_clearance(self) -> float:
        """Minimum pedestrian clearance per MUTCD (§4E.08) at 3.5 ft/s."""
        return self.crossing_distance_ft / 3.5


class Corridor(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    description: str = ""
    intersections: list[Intersection]
    distances_m: list[float]   # meters between consecutive intersections
    speed_limits_mph: list[float]   # mph per segment
    aadt: int = 30000
    functional_class: str = "principal_arterial"
    city: str = "San Diego"
    state: str = "CA"

    @field_validator("distances_m")
    @classmethod
    def distances_match(cls, v: list[float], info) -> list[float]:
        intersections = info.data.get("intersections", [])
        if intersections and len(v) != len(intersections) - 1:
            raise ValueError(
                f"distances_m length {len(v)} must equal len(intersections)-1 "
                f"= {len(intersections)-1}"
            )
        return v


# ---------------------------------------------------------------------------
# Timing plans
# ---------------------------------------------------------------------------

class PhaseTiming(BaseModel):
    phase_id: int
    min_green: float
    max_green: float
    split: float           # allocated green time in seconds
    yellow: float
    all_red: float
    ped_walk: Optional[float] = None
    ped_clearance: Optional[float] = None


class TimingPlan(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    intersection_id: str
    plan_number: int = 1
    cycle_length: float       # seconds
    offset: float = 0.0       # seconds relative to system reference
    phases: list[PhaseTiming]
    reference_phase: int = 2  # phase used for coordination
    created_at: datetime = Field(default_factory=datetime.utcnow)
    source: str = "aito"      # aito, existing, manual

    @property
    def total_green(self) -> float:
        return sum(p.split for p in self.phases)

    @property
    def total_lost_time(self) -> float:
        return sum(p.yellow + p.all_red for p in self.phases)


class CorridorPlan(BaseModel):
    """Complete set of timing plans for a coordinated corridor."""
    id: str = Field(default_factory=lambda: str(uuid4()))
    corridor_id: str
    plan_name: str = "AM Peak"
    timing_plans: list[TimingPlan]   # one per intersection, ordered
    cycle_length: float              # common cycle length (seconds)
    offsets: list[float]             # offset per intersection (seconds)
    bandwidth_inbound: float = 0.0   # computed green wave bandwidth (s)
    bandwidth_outbound: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Optimization I/O
# ---------------------------------------------------------------------------

class DemandProfile(BaseModel):
    """Volume by approach and movement for a single intersection."""
    intersection_id: str
    period_minutes: int = 15
    # volume in veh/hr per approach direction (N, S, E, W)
    north_thru: float = 0
    north_left: float = 0
    north_right: float = 0
    south_thru: float = 0
    south_left: float = 0
    south_right: float = 0
    east_thru: float = 0
    east_left: float = 0
    east_right: float = 0
    west_thru: float = 0
    west_left: float = 0
    west_right: float = 0


class OptimizationRequest(BaseModel):
    corridor_id: str
    demand_profiles: list[DemandProfile]
    objectives: list[OptimizationObjective] = Field(
        default_factory=lambda: [
            OptimizationObjective.DELAY,
            OptimizationObjective.EMISSIONS,
        ]
    )
    min_cycle: float = 60.0
    max_cycle: float = 180.0
    force_common_cycle: bool = True


class ParetoSolution(BaseModel):
    """One point on the Pareto frontier."""
    plan: CorridorPlan
    delay_score: float          # avg intersection delay (s/veh), lower = better
    emissions_score: float      # CO2 kg/hour, lower = better
    stops_score: float          # stops per vehicle, lower = better
    safety_score: float         # conflict index, lower = better
    equity_score: float         # standard deviation of delay across approaches, lower = better
    description: str = ""       # human-readable tradeoff summary


class OptimizationResult(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    corridor_id: str
    pareto_solutions: list[ParetoSolution]
    recommended_solution: ParetoSolution
    computation_seconds: float
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Performance analytics
# ---------------------------------------------------------------------------

class PerformanceMetrics(BaseModel):
    """Measured performance for a corridor or intersection."""
    period_start: datetime
    period_end: datetime
    avg_delay_sec: float          # average intersection delay (s/veh)
    avg_travel_time_sec: float    # end-to-end corridor travel time (s)
    arrival_on_green_pct: float   # % arrivals during green
    split_failure_pct: float      # % phases ending in split failure
    stops_per_veh: float
    co2_kg_hr: float
    throughput_veh_hr: float


class BeforeAfterResult(BaseModel):
    corridor_id: str
    before: PerformanceMetrics
    after: PerformanceMetrics
    travel_time_improvement_pct: float
    delay_improvement_pct: float
    co2_reduction_pct: float
    stops_reduction_pct: float
    annual_veh_hours_saved: float
    annual_fuel_saved_gallons: float
    annual_co2_reduction_tonnes: float


# ---------------------------------------------------------------------------
# Deployment
# ---------------------------------------------------------------------------

class DeploymentRecord(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    plan_id: str
    intersection_id: str
    status: DeploymentStatus = DeploymentStatus.PENDING
    ntcip_address: str
    deployed_at: Optional[datetime] = None
    deployed_by: str = "aito-system"
    validation_passed: bool = False
    raw_response: Optional[str] = None
    error: Optional[str] = None


# ---------------------------------------------------------------------------
# Evidence / audit (for future MCP integration)
# ---------------------------------------------------------------------------

class Evidence(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    path: str
    sha256: Optional[str] = None
    type: str  # disk_image, memory_capture, log_file

    def compute_hash(self) -> str:
        h = hashlib.sha256()
        with open(self.path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        self.sha256 = h.hexdigest()
        return self.sha256
