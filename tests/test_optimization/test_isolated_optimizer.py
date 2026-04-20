"""Tests for the isolated intersection optimizer."""
import pytest
from aito.models import DemandProfile, Intersection
from aito.optimization.isolated_optimizer import (
    IsolatedIntersectionOptimizer,
    TOD_DEMAND_FACTORS,
    webster_cycle,
    allocate_greens,
)
from aito.optimization.constraints import TimingPlanValidator


@pytest.fixture
def intersection():
    return Intersection(
        name="Test @ Cross",
        latitude=32.7, longitude=-117.1,
        approach_speed_mph=35.0,
        crossing_distance_ft=60.0,
        pedestrian_phases=[2, 4, 6, 8],
        num_phases=8,
    )


@pytest.fixture
def base_demand(intersection):
    return DemandProfile(
        intersection_id=intersection.id,
        north_thru=600, north_left=200,
        south_thru=500, south_left=150,
        east_thru=400, east_left=180,
        west_thru=350, west_left=120,
    )


def test_optimize_am_peak_produces_valid_plan(intersection, base_demand):
    optimizer = IsolatedIntersectionOptimizer(intersection)
    result = optimizer.optimize(base_demand, period="AM Peak")
    assert result.plan is not None
    assert result.validation.valid, f"Plan invalid: {result.validation.errors}"
    assert 60.0 <= result.cycle_length <= 180.0


def test_all_tod_periods(intersection, base_demand):
    optimizer = IsolatedIntersectionOptimizer(intersection)
    results = optimizer.optimize_all_periods(base_demand)
    assert len(results) == len(TOD_DEMAND_FACTORS)
    for r in results:
        assert r.plan is not None
        assert 60.0 <= r.cycle_length <= 180.0


def test_overnight_has_shorter_cycle_than_am_peak(intersection, base_demand):
    optimizer = IsolatedIntersectionOptimizer(intersection)
    results = {r.period: r for r in optimizer.optimize_all_periods(base_demand)}
    assert results["Overnight"].cycle_length <= results["AM Peak"].cycle_length


def test_low_demand_produces_minimum_cycle(intersection):
    low_demand = DemandProfile(
        intersection_id=intersection.id,
        north_thru=50, south_thru=50, east_thru=30, west_thru=30,
    )
    optimizer = IsolatedIntersectionOptimizer(intersection)
    result = optimizer.optimize(low_demand)
    assert result.cycle_length >= 60.0


def test_oversaturated_demand_uses_max_cycle(intersection):
    """Oversaturated intersection should get max cycle."""
    high_demand = DemandProfile(
        intersection_id=intersection.id,
        north_thru=1800, south_thru=1600, east_thru=1500, west_thru=1400,
    )
    optimizer = IsolatedIntersectionOptimizer(intersection)
    result = optimizer.optimize(high_demand)
    assert result.cycle_length == 180.0 or result.degree_of_saturation >= 0.99


def test_ped_splits_satisfy_clearance(intersection, base_demand):
    optimizer = IsolatedIntersectionOptimizer(intersection)
    result = optimizer.optimize(base_demand)
    for pt in result.plan.phases:
        if pt.phase_id in intersection.pedestrian_phases:
            assert pt.ped_walk is not None
            assert pt.ped_clearance is not None
            assert pt.ped_walk >= 7.0
            assert pt.ped_clearance >= intersection.crossing_clearance - 0.5


def test_webster_cycle_formula():
    """Webster C_opt = (1.5L+5)/(1-Y)."""
    y_vals = [0.3, 0.2, 0.25]
    c = webster_cycle(y_vals, lost_time_per_phase=6.0)
    Y = sum(y_vals)
    L = len(y_vals) * 6.0
    expected = (1.5 * L + 5.0) / (1.0 - Y)
    assert abs(c - min(180, max(60, expected))) < 1.0


def test_allocate_greens_respects_min_green():
    greens = allocate_greens(cycle=120, critical_flow_ratios=[0.3, 0.2, 0.25],
                              lost_time_per_phase=6.0, min_green=7.0)
    assert all(g >= 7.0 for g in greens)


def test_allocate_greens_proportional():
    """Higher flow ratio → more green."""
    greens = allocate_greens(cycle=120, critical_flow_ratios=[0.4, 0.1],
                              lost_time_per_phase=6.0, min_green=7.0)
    assert greens[0] > greens[1]
