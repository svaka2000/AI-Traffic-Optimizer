from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np


Direction = Literal["N", "S", "E", "W"]
SignalPhase = Literal["NS", "EW"]


@dataclass(slots=True)
class IntersectionState:
    intersection_id: int
    lanes_per_direction: int
    max_queue_per_lane: int
    queue_matrix: dict[Direction, np.ndarray]
    current_phase: SignalPhase = "NS"
    phase_elapsed: int = 0
    total_arrivals: int = 0
    total_departures: int = 0
    cumulative_wait_sec: float = 0.0
    cumulative_stopped_vehicles: float = 0.0
    phase_changes: int = 0
    pending_inflow: dict[Direction, float] = field(default_factory=dict)
    # Emergency vehicle tracking
    emergency_active: bool = False
    emergency_direction: str = ""
    emergency_steps_remaining: int = 0

    @property
    def queue_ns(self) -> float:
        return float(self.queue_matrix["N"].sum() + self.queue_matrix["S"].sum())

    @property
    def queue_ew(self) -> float:
        return float(self.queue_matrix["E"].sum() + self.queue_matrix["W"].sum())

    @property
    def total_queue(self) -> float:
        return self.queue_ns + self.queue_ew

    def as_observation(self, sim_step: int) -> dict[str, float]:
        return {
            "intersection_id": float(self.intersection_id),
            "sim_step": float(sim_step),
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "total_queue": self.total_queue,
            "phase_ns": 1.0 if self.current_phase == "NS" else 0.0,
            "phase_ew": 1.0 if self.current_phase == "EW" else 0.0,
            "phase_elapsed": float(self.phase_elapsed),
            "arrivals": float(self.total_arrivals),
            "departures": float(self.total_departures),
            "wait_sec": float(self.cumulative_wait_sec),
            "emergency_active": 1.0 if self.emergency_active else 0.0,
        }


@dataclass(slots=True)
class StepMetrics:
    step: int
    total_queue: float
    avg_wait_sec: float
    throughput: float
    emissions_proxy: float
    fuel_proxy: float
    fairness: float
    efficiency_score: float
    delay_reduction_pct: float
    fuel_gallons: float = 0.0
    co2_kg: float = 0.0


@dataclass(slots=True)
class SimulationResult:
    controller_name: str
    step_metrics: list[StepMetrics]
    intersection_summaries: list[dict[str, float]]
    aggregate: dict[str, float]
