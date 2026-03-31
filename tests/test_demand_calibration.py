"""tests/test_demand_calibration.py

Tests for Phase 9C: DemandModel.calibrate_from_pems_profile.
"""
from __future__ import annotations

import pytest

from traffic_ai.simulation_engine.demand import DemandModel


# ---------------------------------------------------------------------------
# Test 1: calibrate_from_pems_profile overrides synthetic rates
# ---------------------------------------------------------------------------

def test_calibrate_from_pems_overrides_synthetic() -> None:
    """DemandModel uses PeMS rates when calibrated."""
    model = DemandModel(profile="rush_hour", scale=1.0, step_seconds=1.0, seed=42)

    # Synthetic rate at hour 8 (rush hour)
    # step=8*3600 = 28800 → hour_of_day = 8.0
    step_at_8am = 8 * 3600
    synthetic_rate = model.arrival_rate_per_lane(step_at_8am, "N")

    # Now calibrate with a flat, known rate for all hours
    flat_rate = 0.42
    profile = {h: flat_rate for h in range(24)}
    model.calibrate_from_pems_profile(profile)

    calibrated_rate = model.arrival_rate_per_lane(step_at_8am, "N")

    # Should return exactly flat_rate * scale (scale=1.0)
    assert abs(calibrated_rate - flat_rate) < 1e-6, (
        f"PeMS calibration should return {flat_rate}, got {calibrated_rate}"
    )
    # Should differ from the synthetic rush-hour rate
    assert abs(calibrated_rate - synthetic_rate) > 0.01, (
        "PeMS rate should override synthetic rate"
    )


# ---------------------------------------------------------------------------
# Test 2: synthetic mode when no calibration
# ---------------------------------------------------------------------------

def test_synthetic_mode_when_no_calibration() -> None:
    """DemandModel uses default Poisson rates without calibration."""
    model = DemandModel(profile="rush_hour", scale=1.0, step_seconds=1.0, seed=42)

    # No calibration called — should use synthetic rush-hour formula
    rate = model.arrival_rate_per_lane(step=0, direction="N")

    # At step=0 (hour≈0), rush-hour rate is near base (0.12 * scale ≈ 0.13 for N/S)
    assert rate >= 0.02, "Rate must be non-negative"
    assert rate < 1.0,   "Rate at hour=0 should be below 1.0 veh/sec"


# ---------------------------------------------------------------------------
# Test 3: arrival rate at rush hour is higher than off-peak
# ---------------------------------------------------------------------------

def test_arrival_rate_at_rush_hour_is_higher() -> None:
    """Hour 7-9 arrival rate exceeds hour 2-4 in both synthetic and PeMS modes."""
    model = DemandModel(profile="rush_hour", scale=1.0, step_seconds=1.0, seed=42)

    # Synthetic mode
    rate_rush_synth    = model.arrival_rate_per_lane(8 * 3600, "N")    # hour 8
    rate_offpeak_synth = model.arrival_rate_per_lane(3 * 3600, "N")    # hour 3

    assert rate_rush_synth > rate_offpeak_synth, (
        f"Synthetic rush-hour rate ({rate_rush_synth:.4f}) must exceed "
        f"off-peak rate ({rate_offpeak_synth:.4f})"
    )

    # PeMS-calibrated mode with a realistic rush-hour profile
    profile = {h: (0.30 if 7 <= h <= 9 else 0.08) for h in range(24)}
    model.calibrate_from_pems_profile(profile)

    rate_rush_pems    = model.arrival_rate_per_lane(8 * 3600, "N")   # hour 8
    rate_offpeak_pems = model.arrival_rate_per_lane(3 * 3600, "N")   # hour 3

    assert rate_rush_pems > rate_offpeak_pems, (
        f"PeMS rush-hour rate ({rate_rush_pems:.4f}) must exceed "
        f"off-peak rate ({rate_offpeak_pems:.4f})"
    )


# ---------------------------------------------------------------------------
# Test 4: scale multiplier still applies in PeMS mode
# ---------------------------------------------------------------------------

def test_scale_applies_in_pems_mode() -> None:
    """Global scale multiplier is applied on top of PeMS calibration."""
    model_1x = DemandModel(profile="rush_hour", scale=1.0, step_seconds=1.0, seed=42)
    model_2x = DemandModel(profile="rush_hour", scale=2.0, step_seconds=1.0, seed=42)

    profile = {h: 0.20 for h in range(24)}
    model_1x.calibrate_from_pems_profile(profile)
    model_2x.calibrate_from_pems_profile(profile)

    rate_1x = model_1x.arrival_rate_per_lane(0, "N")
    rate_2x = model_2x.arrival_rate_per_lane(0, "N")

    assert abs(rate_2x - 2 * rate_1x) < 1e-6, (
        f"scale=2 should double the rate: {rate_2x:.4f} vs 2×{rate_1x:.4f}"
    )


# ---------------------------------------------------------------------------
# Test 5: calibrate replaces previous calibration on second call
# ---------------------------------------------------------------------------

def test_calibration_is_replaceable() -> None:
    """Calling calibrate_from_pems_profile twice replaces the first profile."""
    model = DemandModel(profile="rush_hour", scale=1.0, step_seconds=1.0, seed=42)

    profile_a = {h: 0.10 for h in range(24)}
    profile_b = {h: 0.50 for h in range(24)}

    model.calibrate_from_pems_profile(profile_a)
    model.calibrate_from_pems_profile(profile_b)

    rate = model.arrival_rate_per_lane(0, "N")
    assert abs(rate - 0.50) < 1e-6, (
        f"Second calibration should replace first; expected 0.50, got {rate}"
    )
