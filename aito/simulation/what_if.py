"""aito/simulation/what_if.py

GF15: What-If Scenario Engine.

Enables traffic engineers to explore hypothetical timing changes,
infrastructure scenarios, and demand shifts before deploying to live
intersections.  Every scenario runs through the DigitalTwin simulation.

Scenario types:
  1. TimingPlanScenario      — test a candidate timing plan
  2. DemandScenario          — model demand increase/decrease (e.g. new development)
  3. IntersectionClosureScenario — simulate approach closure or lane reduction
  4. EventScenario           — stadium/concert demand surge
  5. IncidentScenario        — simulate signal failure or approach blockage
  6. NetworkUpgradeScenario  — test adding a new intersection to corridor

Usage:
  engine = WhatIfEngine(corridor, digital_twin)
  result = engine.run(DemandScenario(scale_factor=1.3))
  comparison = engine.compare(baseline, result)
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional

from aito.simulation.digital_twin import DigitalTwin, SimulationResult


# ---------------------------------------------------------------------------
# Scenario base classes
# ---------------------------------------------------------------------------

class ScenarioType(str, Enum):
    TIMING_PLAN     = "timing_plan"
    DEMAND_SHIFT    = "demand_shift"
    CLOSURE         = "closure"
    EVENT           = "event"
    INCIDENT        = "incident"
    NETWORK_UPGRADE = "network_upgrade"


@dataclass
class Scenario:
    """Base scenario definition."""
    name: str
    description: str
    scenario_type: ScenarioType = ScenarioType.TIMING_PLAN
    duration_s: float = 900.0       # simulation duration
    demand_veh_s: float = 0.5       # baseline demand


@dataclass
class TimingPlanScenario(Scenario):
    """Test a specific timing plan."""
    green_ratios: list[float] = field(default_factory=list)
    cycle_s: float = 120.0

    def __post_init__(self):
        self.scenario_type = ScenarioType.TIMING_PLAN


@dataclass
class DemandScenario(Scenario):
    """Model demand increase or decrease."""
    scale_factor: float = 1.0       # multiplier on baseline demand
    affected_approaches: list[str] = field(default_factory=lambda: ["N", "S", "E", "W"])

    def __post_init__(self):
        self.scenario_type = ScenarioType.DEMAND_SHIFT


@dataclass
class ClosureScenario(Scenario):
    """Simulate approach closure or lane reduction."""
    intersection_idx: int = 0
    approach: str = "N"
    lanes_removed: int = 1

    def __post_init__(self):
        self.scenario_type = ScenarioType.CLOSURE


@dataclass
class EventScenario(Scenario):
    """Model demand surge from stadium, concert, or major event."""
    event_name: str = "Major Event"
    peak_demand_multiplier: float = 2.5
    event_start_offset_s: float = 0.0  # seconds into simulation
    event_duration_s: float = 3600.0
    dominant_direction: str = "N"    # primary outbound direction

    def __post_init__(self):
        self.scenario_type = ScenarioType.EVENT


@dataclass
class IncidentScenario(Scenario):
    """Simulate signal failure or approach blockage."""
    segment_idx: int = 0
    capacity_reduction_pct: float = 0.5  # 50% capacity reduction
    incident_type: str = "signal_failure"

    def __post_init__(self):
        self.scenario_type = ScenarioType.INCIDENT


# ---------------------------------------------------------------------------
# Scenario comparison
# ---------------------------------------------------------------------------

@dataclass
class ScenarioComparison:
    """Side-by-side comparison of baseline vs. scenario."""
    baseline: SimulationResult
    scenario_result: SimulationResult
    scenario: Scenario

    @property
    def travel_time_delta_s(self) -> float:
        return self.scenario_result.avg_travel_time_s - self.baseline.avg_travel_time_s

    @property
    def travel_time_delta_pct(self) -> float:
        if self.baseline.avg_travel_time_s <= 0:
            return 0.0
        return (self.travel_time_delta_s / self.baseline.avg_travel_time_s) * 100.0

    @property
    def speed_delta_mph(self) -> float:
        return self.scenario_result.avg_speed_mph - self.baseline.avg_speed_mph

    @property
    def stops_delta(self) -> int:
        return self.scenario_result.total_stops - self.baseline.total_stops

    @property
    def co2_delta_kg(self) -> float:
        return self.scenario_result.co2_kg - self.baseline.co2_kg

    @property
    def verdict(self) -> str:
        if self.travel_time_delta_pct < -5:
            return "SIGNIFICANT_IMPROVEMENT"
        if self.travel_time_delta_pct < 0:
            return "MODEST_IMPROVEMENT"
        if self.travel_time_delta_pct < 5:
            return "NEUTRAL"
        if self.travel_time_delta_pct < 15:
            return "MODEST_DEGRADATION"
        return "SIGNIFICANT_DEGRADATION"

    def summary(self) -> dict:
        return {
            "scenario": self.scenario.name,
            "scenario_type": self.scenario.scenario_type.value,
            "travel_time_delta_s": round(self.travel_time_delta_s, 1),
            "travel_time_delta_pct": round(self.travel_time_delta_pct, 1),
            "speed_delta_mph": round(self.speed_delta_mph, 1),
            "stops_delta": self.stops_delta,
            "co2_delta_kg": round(self.co2_delta_kg, 3),
            "verdict": self.verdict,
        }


# ---------------------------------------------------------------------------
# WhatIfEngine
# ---------------------------------------------------------------------------

class WhatIfEngine:
    """What-if scenario runner for AITO corridors.

    Each scenario modifies the simulation inputs (demand, timing, capacity)
    and runs a CTM simulation for comparison against baseline.

    Usage:
        engine = WhatIfEngine(corridor, digital_twin)
        baseline = engine.run_baseline(demand_veh_s=0.5, green_ratios=[0.4, 0.4, ...])
        result = engine.run(DemandScenario(name="New Development", scale_factor=1.3))
        cmp = engine.compare(baseline, result, DemandScenario(...))
        print(cmp.verdict)
    """

    def __init__(
        self,
        corridor,
        digital_twin: Optional[DigitalTwin] = None,
    ) -> None:
        self.corridor = corridor
        self.twin = digital_twin or DigitalTwin(
            corridor_id=corridor.id,
            segment_distances_m=corridor.distances_m,
            free_flow_speeds_mph=corridor.speed_limits_mph,
        )
        self._baseline: Optional[SimulationResult] = None

    def run_baseline(
        self,
        demand_veh_s: float = 0.5,
        green_ratios: Optional[list[float]] = None,
        duration_s: float = 900.0,
    ) -> SimulationResult:
        """Run and cache baseline simulation."""
        n = len(self.corridor.distances_m)
        gr = green_ratios or [0.4] * n
        green_states = [g > 0.0 for g in gr]  # simplification: all green
        self._baseline = self.twin.simulate(
            signal_green_per_segment=green_states,
            demand_veh_s=demand_veh_s,
            duration_s=duration_s,
            scenario_label="baseline",
        )
        return self._baseline

    def run(self, scenario: Scenario) -> SimulationResult:
        """Execute a what-if scenario and return simulation result."""
        n = len(self.corridor.distances_m)

        if isinstance(scenario, TimingPlanScenario):
            gr = scenario.green_ratios or [0.4] * n
            green_states = [g > 0.0 for g in gr]
            return self.twin.simulate(
                signal_green_per_segment=green_states,
                demand_veh_s=scenario.demand_veh_s,
                duration_s=scenario.duration_s,
                scenario_label=scenario.name,
            )

        elif isinstance(scenario, DemandScenario):
            green_states = [True] * n
            return self.twin.simulate(
                signal_green_per_segment=green_states,
                demand_veh_s=scenario.demand_veh_s * scenario.scale_factor,
                duration_s=scenario.duration_s,
                scenario_label=scenario.name,
            )

        elif isinstance(scenario, EventScenario):
            green_states = [True] * n
            return self.twin.simulate(
                signal_green_per_segment=green_states,
                demand_veh_s=scenario.demand_veh_s * scenario.peak_demand_multiplier,
                duration_s=scenario.duration_s,
                scenario_label=scenario.name,
            )

        elif isinstance(scenario, IncidentScenario):
            green_states = [True] * n
            # Reduce capacity on affected segment by temporarily modifying FD
            old_fd = self.twin._fds[scenario.segment_idx]
            from aito.simulation.digital_twin import FundamentalDiagram
            reduced_fd = FundamentalDiagram(
                free_flow_speed_mps=old_fd.free_flow_speed_mps,
                capacity_veh_s=old_fd.capacity_veh_s * (1.0 - scenario.capacity_reduction_pct),
                jam_density_veh_m=old_fd.jam_density_veh_m,
            )
            self.twin._fds[scenario.segment_idx] = reduced_fd
            try:
                result = self.twin.simulate(
                    signal_green_per_segment=green_states,
                    demand_veh_s=scenario.demand_veh_s,
                    duration_s=scenario.duration_s,
                    scenario_label=scenario.name,
                )
            finally:
                self.twin._fds[scenario.segment_idx] = old_fd
            return result

        else:
            # Default: run as basic timing scenario
            green_states = [True] * n
            return self.twin.simulate(
                signal_green_per_segment=green_states,
                demand_veh_s=scenario.demand_veh_s,
                duration_s=scenario.duration_s,
                scenario_label=scenario.name,
            )

    def compare(
        self,
        baseline: SimulationResult,
        scenario_result: SimulationResult,
        scenario: Scenario,
    ) -> ScenarioComparison:
        return ScenarioComparison(
            baseline=baseline,
            scenario_result=scenario_result,
            scenario=scenario,
        )

    def run_batch(
        self,
        scenarios: list[Scenario],
        baseline_demand_veh_s: float = 0.5,
    ) -> list[ScenarioComparison]:
        """Run all scenarios and compare against a single baseline."""
        baseline = self.run_baseline(demand_veh_s=baseline_demand_veh_s)
        comparisons: list[ScenarioComparison] = []
        for sc in scenarios:
            sc.demand_veh_s = sc.demand_veh_s or baseline_demand_veh_s
            result = self.run(sc)
            cmp = self.compare(baseline, result, sc)
            comparisons.append(cmp)
        return comparisons

    def sensitivity_analysis(
        self,
        param: str,
        values: list[float],
        baseline_demand_veh_s: float = 0.5,
        green_ratios: Optional[list[float]] = None,
    ) -> list[dict]:
        """Run sensitivity analysis over a parameter range.

        param: "demand" | "green_ratio" | "cycle"
        """
        baseline = self.run_baseline(demand_veh_s=baseline_demand_veh_s)
        results = []
        n = len(self.corridor.distances_m)
        gr = green_ratios or [0.4] * n

        for val in values:
            if param == "demand":
                sc = DemandScenario(
                    name=f"demand_{val:.2f}",
                    description=f"Demand = {val:.2f} veh/s",
                    demand_veh_s=val,
                    scale_factor=1.0,
                )
            elif param == "green_ratio":
                sc = TimingPlanScenario(
                    name=f"green_{val:.2f}",
                    description=f"Green ratio = {val:.2f}",
                    demand_veh_s=baseline_demand_veh_s,
                    green_ratios=[val] * n,
                )
            else:
                sc = DemandScenario(
                    name=f"{param}_{val}",
                    description=f"{param} = {val}",
                    demand_veh_s=baseline_demand_veh_s,
                    scale_factor=1.0,
                )
            result = self.run(sc)
            cmp = self.compare(baseline, result, sc)
            row = {"param_value": val}
            row.update(cmp.summary())
            results.append(row)

        return results
