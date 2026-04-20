"""Tests for the constraints engine (MUTCD/NEMA/HCM)."""
import pytest
from aito.models import Intersection, PhaseTiming, RingBarrierConfig, TimingPlan
from aito.optimization.constraints import (
    TimingPlanValidator,
    ValidationResult,
    compute_yellow,
    compute_all_red,
    compute_ped_clearance,
)


@pytest.fixture
def basic_intersection():
    return Intersection(
        name="Test @ Main",
        latitude=32.7, longitude=-117.1,
        approach_speed_mph=35.0,
        crossing_distance_ft=60.0,
        pedestrian_phases=[2, 6],
        num_phases=8,
    )


@pytest.fixture
def valid_plan(basic_intersection):
    phases = [
        PhaseTiming(phase_id=2, min_green=7, max_green=45, split=25, yellow=4, all_red=2,
                    ped_walk=7, ped_clearance=18),
        PhaseTiming(phase_id=4, min_green=7, max_green=45, split=20, yellow=4, all_red=2),
        PhaseTiming(phase_id=6, min_green=7, max_green=45, split=25, yellow=4, all_red=2,
                    ped_walk=7, ped_clearance=18),
        PhaseTiming(phase_id=8, min_green=7, max_green=45, split=20, yellow=4, all_red=2),
    ]
    return TimingPlan(
        intersection_id=basic_intersection.id,
        cycle_length=120,
        phases=phases,
    )


def test_valid_plan_passes(valid_plan, basic_intersection):
    validator = TimingPlanValidator()
    result = validator.validate(valid_plan, basic_intersection)
    assert result.valid, f"Should pass: {result.errors}"


def test_min_green_violation():
    ix = Intersection(name="X", latitude=0, longitude=0, approach_speed_mph=35,
                      crossing_distance_ft=50, pedestrian_phases=[2])
    plan = TimingPlan(
        intersection_id=ix.id,
        cycle_length=100,
        phases=[
            PhaseTiming(phase_id=2, min_green=7, max_green=40, split=5.0,  # <7s VIOLATION
                        yellow=4, all_red=2, ped_walk=7, ped_clearance=15),
            PhaseTiming(phase_id=6, min_green=7, max_green=40, split=25,
                        yellow=4, all_red=2),
        ],
    )
    result = TimingPlanValidator().validate(plan, ix)
    assert not result.valid
    assert any("min green" in e.lower() or "split" in e.lower() for e in result.errors)


def test_ped_timing_missing_triggers_error(basic_intersection):
    """Plan with pedestrian phase but no ped_walk should fail."""
    plan = TimingPlan(
        intersection_id=basic_intersection.id,
        cycle_length=120,
        phases=[
            PhaseTiming(phase_id=2, min_green=7, max_green=45, split=25,
                        yellow=4, all_red=2, ped_walk=None, ped_clearance=None),
            PhaseTiming(phase_id=4, min_green=7, max_green=45, split=20,
                        yellow=4, all_red=2),
            PhaseTiming(phase_id=6, min_green=7, max_green=45, split=25,
                        yellow=4, all_red=2, ped_walk=7, ped_clearance=18),
            PhaseTiming(phase_id=8, min_green=7, max_green=45, split=20,
                        yellow=4, all_red=2),
        ],
    )
    result = TimingPlanValidator().validate(plan, basic_intersection)
    assert not result.valid
    errors_lower = [e.lower() for e in result.errors]
    assert any("walk" in e or "clearance" in e or "pedestrian" in e for e in errors_lower)


def test_cycle_too_short():
    ix = Intersection(name="X", latitude=0, longitude=0, approach_speed_mph=35,
                      crossing_distance_ft=50, pedestrian_phases=[])
    plan = TimingPlan(
        intersection_id=ix.id,
        cycle_length=30,  # <60s HCM minimum
        phases=[
            PhaseTiming(phase_id=2, min_green=7, max_green=20, split=10, yellow=4, all_red=2),
            PhaseTiming(phase_id=6, min_green=7, max_green=20, split=10, yellow=4, all_red=2),
        ],
    )
    result = TimingPlanValidator().validate(plan, ix)
    assert not result.valid
    assert any("60" in e or "hcm" in e.lower() or "minimum" in e.lower() for e in result.errors)


def test_compute_yellow_formula():
    """ITE yellow formula: t + v/(2*(a+G*g))."""
    y = compute_yellow(approach_speed_mph=35.0)
    assert 3.5 <= y <= 5.5  # Reasonable range for 35 mph


def test_compute_all_red_formula():
    ar = compute_all_red(intersection_width_ft=60.0, approach_speed_mph=35.0)
    assert 1.0 <= ar <= 4.0  # Reasonable range


def test_ped_clearance_formula():
    cl = compute_ped_clearance(crossing_distance_ft=60.0)
    assert abs(cl - 60.0 / 3.5) < 0.01


def test_nema_conflict_detection(basic_intersection):
    """A valid 8-phase NEMA plan with 2+6 concurrent should NOT be flagged."""
    plan = TimingPlan(
        intersection_id=basic_intersection.id,
        cycle_length=120,
        phases=[
            PhaseTiming(phase_id=2, min_green=7, max_green=45, split=25, yellow=4, all_red=2,
                        ped_walk=7, ped_clearance=18),
            PhaseTiming(phase_id=4, min_green=7, max_green=45, split=20, yellow=4, all_red=2),
            PhaseTiming(phase_id=6, min_green=7, max_green=45, split=25, yellow=4, all_red=2,
                        ped_walk=7, ped_clearance=18),
            PhaseTiming(phase_id=8, min_green=7, max_green=45, split=20, yellow=4, all_red=2),
        ],
    )
    result = TimingPlanValidator().validate(plan, basic_intersection)
    # Standard 4-phase plan (phases 2,4,6,8) is valid — no ring-barrier conflicts
    assert result.valid, f"Standard 4-phase plan should be valid: {result.errors}"


def test_evp_phase_missing_triggers_error():
    ix = Intersection(
        name="X", latitude=0, longitude=0, approach_speed_mph=35,
        crossing_distance_ft=50, pedestrian_phases=[],
        preemption_config=__import__("aito.models", fromlist=["PreemptionConfig"]).PreemptionConfig(
            evp_enabled=True, evp_phases=[2]
        ),
    )
    plan = TimingPlan(
        intersection_id=ix.id,
        cycle_length=100,
        phases=[
            PhaseTiming(phase_id=6, min_green=7, max_green=40, split=30,
                        yellow=4, all_red=2),  # Phase 2 missing!
            PhaseTiming(phase_id=8, min_green=7, max_green=40, split=20,
                        yellow=4, all_red=2),
        ],
    )
    result = TimingPlanValidator().validate(plan, ix)
    assert not result.valid
    assert any("preemption" in e.lower() or "evp" in e.lower() for e in result.errors)
