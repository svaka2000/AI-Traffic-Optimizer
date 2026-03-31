from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Direction = Literal["N", "S", "E", "W"]

# 4-phase signal model (AITO Phase 3):
# NS and EW are kept as legacy aliases for backward compatibility with all
# non-RL controllers. New RL controllers use the 4-phase values.
# NS → equivalent to NS_THROUGH (backward compat)
# EW → equivalent to EW_THROUGH (backward compat)
SignalPhase = Literal[
    "NS",           # Legacy alias: northbound + southbound through (backward compat)
    "EW",           # Legacy alias: eastbound + westbound through (backward compat)
    "NS_THROUGH",   # PHASE_0: Northbound and southbound through traffic + right turns
    "EW_THROUGH",   # PHASE_1: Eastbound and westbound through traffic + right turns
    "NS_LEFT",      # PHASE_2: Protected NS left turns only (1200 veh/hr/lane, HCM)
    "EW_LEFT",      # PHASE_3: Protected EW left turns only (1200 veh/hr/lane, HCM)
]

# Which green directions (in queue_matrix) are served for each phase
# NS / NS_THROUGH serve through lanes for N and S
# EW / EW_THROUGH serve through lanes for E and W
# NS_LEFT / EW_LEFT serve left_queue_matrix for their respective axes
_PHASE_THROUGH_DIRS: dict[str, list[str]] = {
    "NS":         ["N", "S"],
    "EW":         ["E", "W"],
    "NS_THROUGH": ["N", "S"],
    "EW_THROUGH": ["E", "W"],
    "NS_LEFT":    [],  # left-turn lanes only — no through service
    "EW_LEFT":    [],
}

_PHASE_LEFT_DIRS: dict[str, list[str]] = {
    "NS":         [],  # legacy: no explicit left phase
    "EW":         [],
    "NS_THROUGH": [],
    "EW_THROUGH": [],
    "NS_LEFT":    ["N", "S"],
    "EW_LEFT":    ["E", "W"],
}

# Map phase string → integer index (for observation encoding)
PHASE_TO_IDX: dict[str, int] = {
    "NS":         0,
    "NS_THROUGH": 0,
    "EW":         1,
    "EW_THROUGH": 1,
    "NS_LEFT":    2,
    "EW_LEFT":    3,
}

# Map integer index → canonical 4-phase string
IDX_TO_PHASE: dict[int, str] = {
    0: "NS_THROUGH",
    1: "EW_THROUGH",
    2: "NS_LEFT",
    3: "EW_LEFT",
}


@dataclass(slots=True)
class IntersectionState:
    intersection_id: int
    lanes_per_direction: int
    max_queue_per_lane: int
    queue_matrix: dict[Direction, np.ndarray]           # through-lane queues
    left_queue_matrix: dict[Direction, np.ndarray] = field(default_factory=dict)  # left-turn queues
    current_phase: SignalPhase = "NS"
    phase_elapsed: int = 0
    total_arrivals: int = 0
    total_departures: int = 0
    cumulative_wait_sec: float = 0.0
    cumulative_stopped_vehicles: float = 0.0
    phase_changes: int = 0
    pending_inflow: dict[Direction, float] = field(default_factory=dict)
    left_pending_inflow: dict[Direction, float] = field(default_factory=dict)
    # Emergency vehicle tracking
    emergency_active: bool = False
    emergency_direction: str = ""
    emergency_steps_remaining: int = 0
    # HCM signal timing: yellow + all-red clearance interval tracking
    transition_steps_remaining: int = 0
    target_phase: SignalPhase = "NS"
    # Left-turn starvation tracking (for multi-objective reward)
    left_ns_last_served: int = 0   # step at which NS_LEFT last had service
    left_ew_last_served: int = 0   # step at which EW_LEFT last had service

    @property
    def queue_ns(self) -> float:
        """Total through-lane queue for NS axis."""
        return float(self.queue_matrix["N"].sum() + self.queue_matrix["S"].sum())

    @property
    def queue_ew(self) -> float:
        """Total through-lane queue for EW axis."""
        return float(self.queue_matrix["E"].sum() + self.queue_matrix["W"].sum())

    @property
    def total_queue(self) -> float:
        """All vehicles queued (through + left)."""
        left = self.queue_ns_left + self.queue_ew_left
        return self.queue_ns + self.queue_ew + left

    @property
    def queue_ns_left(self) -> float:
        """Left-turn queue for NS axis."""
        if not self.left_queue_matrix:
            return 0.0
        return float(
            self.left_queue_matrix.get("N", np.zeros(1)).sum()
            + self.left_queue_matrix.get("S", np.zeros(1)).sum()
        )

    @property
    def queue_ew_left(self) -> float:
        """Left-turn queue for EW axis."""
        if not self.left_queue_matrix:
            return 0.0
        return float(
            self.left_queue_matrix.get("E", np.zeros(1)).sum()
            + self.left_queue_matrix.get("W", np.zeros(1)).sum()
        )

    def as_observation(self, sim_step: int, upstream_queue: float = 0.0) -> dict[str, float]:
        phase_idx = float(PHASE_TO_IDX.get(self.current_phase, 0))
        return {
            "intersection_id": float(self.intersection_id),
            "sim_step": float(sim_step),
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "total_queue": self.total_queue,
            "phase_ns": 1.0 if self.current_phase in ("NS", "NS_THROUGH") else 0.0,
            "phase_ew": 1.0 if self.current_phase in ("EW", "EW_THROUGH") else 0.0,
            "phase_elapsed": float(self.phase_elapsed),
            "arrivals": float(self.total_arrivals),
            "departures": float(self.total_departures),
            "wait_sec": float(self.cumulative_wait_sec),
            "emergency_active": 1.0 if self.emergency_active else 0.0,
            # 4-phase observation fields (AITO Phase 3)
            "queue_ns_through": self.queue_ns,
            "queue_ew_through": self.queue_ew,
            "queue_ns_left": self.queue_ns_left,
            "queue_ew_left": self.queue_ew_left,
            "current_phase_idx": phase_idx,
            # Expanded RL observation features
            "upstream_queue": upstream_queue,
            "time_of_day_normalized": (sim_step / 3600.0 % 24.0) / 24.0,
            "in_transition": 1.0 if self.transition_steps_remaining > 0 else 0.0,
        }


@dataclass(slots=True)
class StepMetrics:
    step: int
    total_queue: float
    avg_wait_sec: float
    throughput: float
    emissions_proxy: float   # kept for backward compat; equals emissions_co2_kg
    fuel_proxy: float
    fairness: float
    efficiency_score: float
    delay_reduction_pct: float
    fuel_gallons: float = 0.0
    co2_kg: float = 0.0
    # First-class EPA MOVES2014b emission metric
    emissions_co2_kg: float = 0.0
    # Phase 8: signal physics, detection, and priority event metrics
    clearance_loss_sec: float = 0.0       # seconds lost to yellow+all-red this step
    detector_fallback_steps: int = 0      # intersections in fixed-timing fallback
    preemption_events: int = 0            # active emergency preemptions this step


@dataclass(slots=True)
class SimulationResult:
    controller_name: str
    step_metrics: list[StepMetrics]
    intersection_summaries: list[dict[str, float]]
    aggregate: dict[str, float]
