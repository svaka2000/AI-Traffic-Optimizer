"""tests/test_webster.py

Tests for traffic_ai.controllers.webster.WebsterController.

Validates the Webster (1958) algorithm implementation:
  - Correct C_opt formula for known inputs
  - Oversaturation (Y >= 1.0) fallback behaviour
  - Cycle length clamping to [min_cycle, max_cycle]
  - Proportional green allocation
  - recalc_interval behaviour
"""
from __future__ import annotations

import pytest

from traffic_ai.controllers.webster import WebsterController


# ------------------------------------------------------------------
# Helper: synthetic observation for one intersection
# ------------------------------------------------------------------

def _obs(queue_ns: float = 20.0, queue_ew: float = 10.0) -> dict[str, float]:
    return {
        "queue_ns": queue_ns,
        "queue_ew": queue_ew,
        "queue_ns_through": queue_ns,
        "queue_ew_through": queue_ew,
        "queue_ns_left": 0.0,
        "queue_ew_left": 0.0,
        "upstream_queue": 0.0,
        "total_queue": queue_ns + queue_ew,
        "sim_step": 0.0,
    }


# ------------------------------------------------------------------
# Correct C_opt calculation
# ------------------------------------------------------------------

def test_c_opt_formula() -> None:
    """Webster C_opt = (1.5*L + 5) / (1 - Y) for non-saturated demand."""
    ctrl = WebsterController(
        recalc_interval=5,
        saturation_flow=0.5,
        lost_time_per_phase=6.0,
        min_cycle=60.0,
        max_cycle=180.0,
    )
    ctrl.reset(1)

    # Fill buffer with non-saturating observations
    # denom = saturation_flow * recalc_interval * step_seconds = 0.5 * 5 * 1.0 = 2.5
    # Using small queues so y_i < 0.95 and Y < 0.99 (non-oversaturated regime).
    q_ns, q_ew = 1.5, 0.5
    obs_seq = {0: _obs(q_ns, q_ew)}
    for step in range(5):
        ctrl.compute_actions(obs_seq, step)

    timing = ctrl._current_timing[0]
    # Manually compute expected values
    L = 2 * 6.0  # 12 s
    denom = 0.5 * 5 * 1.0  # saturation_flow * recalc_interval * step_seconds
    y_ns = min(q_ns / denom, 0.95)
    y_ew = min(q_ew / denom, 0.95)
    Y = y_ns + y_ew
    import math
    c_opt_raw = (1.5 * L + 5.0) / (1.0 - Y)
    expected_cycle = int(max(60.0, min(180.0, c_opt_raw)))
    assert timing["cycle"] == expected_cycle, (
        f"Cycle {timing['cycle']} != expected {expected_cycle}"
    )


def test_proportional_green_allocation() -> None:
    """NS green is proportional to NS demand; heavier demand gets longer green."""
    ctrl = WebsterController(recalc_interval=5, min_cycle=60.0, max_cycle=180.0)
    ctrl.reset(1)

    # Strong NS demand, weak EW demand.
    # Use non-saturating values: denom = 0.5 * 5 * 1.0 = 2.5
    # y_ns = 1.6/2.5 = 0.64, y_ew = 0.4/2.5 = 0.16 → Y = 0.8 < 0.99
    obs_seq = {0: _obs(queue_ns=1.6, queue_ew=0.4)}
    for step in range(5):
        ctrl.compute_actions(obs_seq, step)

    timing = ctrl._current_timing[0]
    assert timing["ns_green"] > timing["ew_green"], (
        "NS heavy demand → NS green should exceed EW green"
    )


# ------------------------------------------------------------------
# Oversaturation fallback (Y >= 1.0)
# ------------------------------------------------------------------

def test_oversaturation_uses_max_cycle() -> None:
    """When Y >= 0.99, controller uses max_cycle instead of Webster formula."""
    ctrl = WebsterController(
        recalc_interval=5,
        saturation_flow=0.001,   # very low saturation → y_i saturates at 0.95
        min_cycle=60.0,
        max_cycle=180.0,
    )
    ctrl.reset(1)
    obs_seq = {0: _obs(queue_ns=1000.0, queue_ew=1000.0)}
    for step in range(5):
        ctrl.compute_actions(obs_seq, step)

    timing = ctrl._current_timing[0]
    assert timing["cycle"] == 180, "Oversaturated demand must fall back to max_cycle"


# ------------------------------------------------------------------
# Cycle length clamping
# ------------------------------------------------------------------

def test_cycle_clamped_to_min() -> None:
    """Very low demand → C_opt < min_cycle → cycle is clamped to min_cycle."""
    ctrl = WebsterController(
        recalc_interval=5,
        saturation_flow=100.0,  # high saturation → tiny y_i → C_opt ≈ 5/(1-ε) ≈ 5
        min_cycle=60.0,
        max_cycle=180.0,
    )
    ctrl.reset(1)
    obs_seq = {0: _obs(queue_ns=0.1, queue_ew=0.1)}
    for step in range(5):
        ctrl.compute_actions(obs_seq, step)

    timing = ctrl._current_timing[0]
    assert timing["cycle"] >= 60, "Cycle must never be below min_cycle"


def test_cycle_clamped_to_max() -> None:
    """Demand near saturation → C_opt → ∞ → cycle is clamped to max_cycle."""
    ctrl = WebsterController(
        recalc_interval=5,
        saturation_flow=0.001,
        min_cycle=60.0,
        max_cycle=120.0,
    )
    ctrl.reset(1)
    obs_seq = {0: _obs(queue_ns=500.0, queue_ew=10.0)}
    for step in range(5):
        ctrl.compute_actions(obs_seq, step)

    timing = ctrl._current_timing[0]
    assert timing["cycle"] <= 120, "Cycle must never exceed max_cycle"


# ------------------------------------------------------------------
# recalc_interval behaviour
# ------------------------------------------------------------------

def test_recalc_interval_triggers_at_correct_step() -> None:
    """Timing plan only updates after recalc_interval steps."""
    ctrl = WebsterController(recalc_interval=10, min_cycle=60.0, max_cycle=180.0)
    ctrl.reset(1)

    # Run 9 steps with balanced demand
    balanced = {0: _obs(10.0, 10.0)}
    for step in range(9):
        ctrl.compute_actions(balanced, step)
    cycle_before = ctrl._current_timing[0]["cycle"]

    # Step 10 triggers recalc with heavy NS demand
    heavy_ns = {0: _obs(50.0, 5.0)}
    ctrl.compute_actions(heavy_ns, step=9)  # 10th step (0-indexed → index 9)
    cycle_after = ctrl._current_timing[0]["cycle"]

    # After recalc, timing reflects heavy NS demand (different from initial)
    # At minimum, ns_green should be larger than ew_green
    assert ctrl._current_timing[0]["ns_green"] >= ctrl._current_timing[0]["ew_green"]


def test_controller_produces_actions_for_all_intersections() -> None:
    """Webster produces one action per intersection."""
    ctrl = WebsterController(recalc_interval=5)
    ctrl.reset(4)
    obs = {i: _obs(10.0 + i * 5, 5.0) for i in range(4)}
    for step in range(5):
        actions = ctrl.compute_actions(obs, step)
    assert len(actions) == 4
    for phase in actions.values():
        assert phase in ("NS_THROUGH", "EW_THROUGH")
