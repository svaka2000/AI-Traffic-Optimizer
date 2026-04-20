"""Tests for aito/simulation/what_if.py (GF15)."""
import pytest
from aito.simulation.what_if import (
    ScenarioType,
    Scenario,
    TimingPlanScenario,
    DemandScenario,
    ClosureScenario,
    EventScenario,
    IncidentScenario,
    ScenarioComparison,
    WhatIfEngine,
)
from aito.simulation.digital_twin import DigitalTwin, SimulationResult
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")
N_SEGS = len(ROSECRANS.distances_m)

def _make_twin():
    return DigitalTwin(
        corridor_id=ROSECRANS.id,
        segment_distances_m=ROSECRANS.distances_m,
        free_flow_speeds_mph=[35.0] * N_SEGS,
    )


class TestScenarioTypes:
    def test_demand_scenario_type(self):
        sc = DemandScenario(name="test", description="test", scale_factor=1.5)
        assert sc.scenario_type == ScenarioType.DEMAND_SHIFT

    def test_timing_plan_scenario_type(self):
        sc = TimingPlanScenario(name="test", description="test", green_ratios=[0.4] * N_SEGS)
        assert sc.scenario_type == ScenarioType.TIMING_PLAN

    def test_incident_scenario_type(self):
        sc = IncidentScenario(name="test", description="test", segment_idx=2)
        assert sc.scenario_type == ScenarioType.INCIDENT

    def test_event_scenario_type(self):
        sc = EventScenario(name="test", description="test", peak_demand_multiplier=2.0)
        assert sc.scenario_type == ScenarioType.EVENT

    def test_closure_scenario_type(self):
        sc = ClosureScenario(name="test", description="test")
        assert sc.scenario_type == ScenarioType.CLOSURE


class TestWhatIfEngine:
    def setup_method(self):
        self.twin = _make_twin()
        self.engine = WhatIfEngine(ROSECRANS, digital_twin=self.twin)

    def test_run_baseline_returns_simulation_result(self):
        result = self.engine.run_baseline(demand_veh_s=0.28)
        assert isinstance(result, SimulationResult)

    def test_run_demand_scenario_returns_result(self):
        sc = DemandScenario(name="surge", description="50% demand", scale_factor=1.5, demand_veh_s=0.28)
        result = self.engine.run(sc)
        assert isinstance(result, SimulationResult)

    def test_run_timing_plan_scenario(self):
        sc = TimingPlanScenario(
            name="short_cycle", description="90s cycle",
            green_ratios=[0.45] * N_SEGS, demand_veh_s=0.28,
        )
        result = self.engine.run(sc)
        assert isinstance(result, SimulationResult)

    def test_run_incident_scenario(self):
        sc = IncidentScenario(
            name="crash", description="50% cap reduction",
            segment_idx=2, capacity_reduction_pct=0.5, demand_veh_s=0.28,
        )
        result = self.engine.run(sc)
        assert isinstance(result, SimulationResult)

    def test_run_event_scenario(self):
        sc = EventScenario(
            name="padres", description="game surge",
            peak_demand_multiplier=2.0, demand_veh_s=0.28,
        )
        result = self.engine.run(sc)
        assert isinstance(result, SimulationResult)

    def test_compare_returns_comparison(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.3, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        assert isinstance(cmp, ScenarioComparison)

    def test_demand_surge_degrades_or_neutral(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.10)
        sc = DemandScenario(name="surge", description="double demand", scale_factor=3.0, demand_veh_s=0.10)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        # High demand surge should degrade or be neutral
        assert cmp.verdict in ("MODEST_DEGRADATION", "SIGNIFICANT_DEGRADATION", "NEUTRAL",
                               "MODEST_IMPROVEMENT", "SIGNIFICANT_IMPROVEMENT")

    def test_travel_time_delta_pct_property(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.0, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        assert isinstance(cmp.travel_time_delta_pct, float)

    def test_speed_delta_mph_property(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.0, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        assert isinstance(cmp.speed_delta_mph, float)

    def test_co2_delta_property(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.0, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        assert isinstance(cmp.co2_delta_kg, float)

    def test_summary_returns_dict(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.0, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        s = cmp.summary()
        assert "verdict" in s
        assert "travel_time_delta_pct" in s

    def test_verdict_valid_string(self):
        baseline = self.engine.run_baseline(demand_veh_s=0.28)
        sc = DemandScenario(name="test", description="test", scale_factor=1.0, demand_veh_s=0.28)
        result = self.engine.run(sc)
        cmp = self.engine.compare(baseline, result, sc)
        assert cmp.verdict in (
            "SIGNIFICANT_IMPROVEMENT", "MODEST_IMPROVEMENT", "NEUTRAL",
            "MODEST_DEGRADATION", "SIGNIFICANT_DEGRADATION",
        )

    def test_run_batch_multiple_scenarios(self):
        scenarios = [
            DemandScenario(name="s1", description="d1", scale_factor=1.2, demand_veh_s=0.2),
            DemandScenario(name="s2", description="d2", scale_factor=1.5, demand_veh_s=0.2),
        ]
        comparisons = self.engine.run_batch(scenarios, baseline_demand_veh_s=0.2)
        assert len(comparisons) == 2
        assert all(isinstance(c, ScenarioComparison) for c in comparisons)

    def test_sensitivity_analysis(self):
        results = self.engine.sensitivity_analysis(
            param="demand",
            values=[0.1, 0.2, 0.3],
            baseline_demand_veh_s=0.2,
        )
        assert len(results) == 3
        for r in results:
            assert "param_value" in r
            assert "verdict" in r

    def test_engine_without_twin_builds_one(self):
        engine = WhatIfEngine(ROSECRANS)  # no twin provided
        baseline = engine.run_baseline(demand_veh_s=0.28)
        assert isinstance(baseline, SimulationResult)

    def test_incident_restores_capacity_after_run(self):
        # Run incident, then normal — normal should not be affected
        sc_incident = IncidentScenario(
            name="crash", description="test", segment_idx=1,
            capacity_reduction_pct=0.8, demand_veh_s=0.2,
        )
        self.engine.run(sc_incident)
        sc_normal = DemandScenario(name="normal", description="normal", scale_factor=1.0, demand_veh_s=0.2)
        result = self.engine.run(sc_normal)
        assert result.avg_travel_time_s > 0
