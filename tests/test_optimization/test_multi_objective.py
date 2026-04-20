"""Tests for the multi-objective Pareto optimizer."""
import pytest
from aito.data.san_diego_inventory import get_corridor
from aito.models import DemandProfile, OptimizationObjective, OptimizationRequest
from aito.optimization.multi_objective import (
    MultiObjectiveOptimizer,
    score_corridor_plan,
    pareto_front,
    PlanScores,
    dominates,
    hcm_delay,
    compute_emissions_kg_hr,
)


@pytest.fixture
def downtown_corridor():
    return get_corridor("downtown")


@pytest.fixture
def demands(downtown_corridor):
    return [
        DemandProfile(
            intersection_id=ix.id,
            north_thru=500, south_thru=400, east_thru=300, west_thru=250,
        )
        for ix in downtown_corridor.intersections
    ]


@pytest.fixture
def opt_request(downtown_corridor, demands):
    return OptimizationRequest(
        corridor_id=downtown_corridor.id,
        demand_profiles=demands,
        objectives=[
            OptimizationObjective.DELAY,
            OptimizationObjective.EMISSIONS,
        ],
        min_cycle=70.0,
        max_cycle=140.0,
    )


def test_pareto_solutions_returned(downtown_corridor, opt_request):
    optimizer = MultiObjectiveOptimizer(downtown_corridor)
    result = optimizer.optimize(opt_request)
    assert len(result.pareto_solutions) >= 1


def test_recommended_solution_in_pareto(downtown_corridor, opt_request):
    optimizer = MultiObjectiveOptimizer(downtown_corridor)
    result = optimizer.optimize(opt_request)
    rec = result.recommended_solution
    plan_ids = {sol.plan.id for sol in result.pareto_solutions}
    assert rec.plan.id in plan_ids


def test_all_pareto_plans_valid(downtown_corridor, opt_request):
    from aito.optimization.constraints import validate_corridor_plan
    optimizer = MultiObjectiveOptimizer(downtown_corridor)
    result = optimizer.optimize(opt_request)
    for sol in result.pareto_solutions:
        for plan, ix in zip(sol.plan.timing_plans, downtown_corridor.intersections):
            plan.intersection_id = ix.id
        vr = validate_corridor_plan(downtown_corridor, sol.plan.timing_plans)
        for iid, v in vr.items():
            assert v.valid, f"Plan invalid for {iid}: {v.errors}"


def test_scores_non_negative(downtown_corridor, opt_request):
    optimizer = MultiObjectiveOptimizer(downtown_corridor)
    result = optimizer.optimize(opt_request)
    for sol in result.pareto_solutions:
        assert sol.delay_score >= 0
        assert sol.emissions_score >= 0
        assert sol.stops_score >= 0
        assert sol.safety_score >= 0
        assert sol.equity_score >= 0


def test_dominance_relation():
    a = PlanScores(delay_s_veh=20, emissions_kg_hr=10, stops_per_veh=0.3, safety_index=5, equity_std=2)
    b = PlanScores(delay_s_veh=30, emissions_kg_hr=15, stops_per_veh=0.5, safety_index=8, equity_std=4)
    assert dominates(a, b)
    assert not dominates(b, a)


def test_dominance_equal():
    a = PlanScores(delay_s_veh=20, emissions_kg_hr=10, stops_per_veh=0.3, safety_index=5, equity_std=2)
    assert not dominates(a, a)


def test_pareto_front_removes_dominated():
    from aito.models import CorridorPlan, TimingPlan
    plans = [
        (CorridorPlan(corridor_id="x", plan_name="A", timing_plans=[], cycle_length=90, offsets=[]),
         PlanScores(delay_s_veh=20, emissions_kg_hr=10, stops_per_veh=0.3, safety_index=5, equity_std=2)),
        (CorridorPlan(corridor_id="x", plan_name="B", timing_plans=[], cycle_length=100, offsets=[]),
         PlanScores(delay_s_veh=30, emissions_kg_hr=15, stops_per_veh=0.5, safety_index=8, equity_std=4)),
    ]
    front = pareto_front(plans)
    assert 0 in front  # A is non-dominated
    assert 1 not in front  # B is dominated by A


def test_hcm_delay_reasonable():
    d = hcm_delay(cycle=120, green=40, flow_ratio=0.6, capacity_veh_hr=900, volume_veh_hr=500)
    assert 5 <= d <= 120  # reasonable range for signalised intersection


def test_hcm_delay_increases_with_saturation():
    d_low = hcm_delay(cycle=120, green=40, flow_ratio=0.3, capacity_veh_hr=900, volume_veh_hr=200)
    d_high = hcm_delay(cycle=120, green=40, flow_ratio=0.3, capacity_veh_hr=900, volume_veh_hr=800)
    assert d_high >= d_low


def test_emissions_positive():
    e = compute_emissions_kg_hr(avg_queue_veh=5.0, stops_per_hr=300.0)
    assert e > 0
