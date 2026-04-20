"""Tests for the corridor optimizer (MAXBAND-enhanced)."""
import pytest
from aito.data.san_diego_inventory import get_corridor
from aito.models import DemandProfile
from aito.optimization.corridor_optimizer import (
    CorridorOptimizer,
    SegmentTravelData,
    compute_bandwidth,
    optimize_offsets_maxband,
)


@pytest.fixture
def downtown_corridor():
    return get_corridor("downtown")  # 4-intersection for fast tests


@pytest.fixture
def demands(downtown_corridor):
    return [
        DemandProfile(
            intersection_id=ix.id,
            north_thru=500, south_thru=400, east_thru=300, west_thru=250,
            north_left=150, east_left=120,
        )
        for ix in downtown_corridor.intersections
    ]


def test_optimize_produces_valid_plans(downtown_corridor, demands):
    opt = CorridorOptimizer(downtown_corridor)
    result = opt.optimize(demands)
    assert result.corridor_plan is not None
    assert len(result.corridor_plan.timing_plans) == len(downtown_corridor.intersections)
    assert result.all_valid, f"Validation errors: {result.validation_results}"


def test_common_cycle_all_intersections(downtown_corridor, demands):
    opt = CorridorOptimizer(downtown_corridor)
    result = opt.optimize(demands)
    cycles = {p.cycle_length for p in result.corridor_plan.timing_plans}
    assert len(cycles) == 1, "All intersections must share common cycle"


def test_offsets_within_cycle(downtown_corridor, demands):
    opt = CorridorOptimizer(downtown_corridor)
    result = opt.optimize(demands)
    for plan in result.corridor_plan.timing_plans:
        assert 0 <= plan.offset < result.cycle_length


def test_bandwidth_positive(downtown_corridor, demands):
    opt = CorridorOptimizer(downtown_corridor)
    result = opt.optimize(demands)
    assert result.bandwidth_outbound_pct >= 0
    assert result.bandwidth_inbound_pct >= 0


def test_probe_data_override(downtown_corridor, demands):
    """Custom travel data should not crash optimizer."""
    travel_data = [
        SegmentTravelData(inbound_travel_time_s=30.0, outbound_travel_time_s=25.0, distance_m=300)
        for _ in range(len(downtown_corridor.distances_m))
    ]
    opt = CorridorOptimizer(downtown_corridor)
    result = opt.optimize(demands, travel_data=travel_data)
    assert result.corridor_plan is not None


def test_compute_bandwidth_zero_for_bad_offsets():
    # If offset diff = 0 but travel time = cycle, bandwidth should be near 0
    bw = compute_bandwidth(
        offsets=[0, 0], green_splits=[30, 30],
        travel_times_s=[100], cycle=120, direction="outbound"
    )
    assert 0 <= bw <= 1.0


def test_compute_bandwidth_ideal_offsets():
    """Perfect offset = travel time → maximum bandwidth."""
    travel_time = 30.0
    cycle = 120.0
    offsets = [0.0, travel_time]
    green = 40.0
    bw = compute_bandwidth(offsets, [green, green], [travel_time], cycle, "outbound")
    assert bw > 0.0


def test_optimize_all_periods(downtown_corridor, demands):
    opt = CorridorOptimizer(downtown_corridor)
    results = opt.optimize_all_periods(demands)
    assert len(results) == 6  # 6 TOD periods
    for r in results:
        assert r.corridor_plan is not None
