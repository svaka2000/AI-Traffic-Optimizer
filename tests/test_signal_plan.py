"""tests/test_signal_plan.py

Tests for traffic_ai.simulation_engine.signal_plan:
  PhaseConstraints, MovementType, DetailedSignalState.
"""
from __future__ import annotations

import pytest

from traffic_ai.simulation_engine.signal_plan import (
    DetailedSignalState,
    MovementType,
    PhaseConstraints,
)


def test_phase_constraints_defaults() -> None:
    """Default PhaseConstraints match HCM 7th ed. and MUTCD requirements."""
    pc = PhaseConstraints()
    assert pc.min_green_sec == 7.0,    "MUTCD Table 4D-2 minimum green = 7 s"
    assert pc.yellow_sec == 4.0,       "ITE clearance standard = 4 s"
    assert pc.all_red_sec == 2.0,      "HCM 7th ed. all-red = 2 s"
    assert pc.min_cycle_sec == 60.0,   "Webster minimum cycle = 60 s"
    assert pc.max_cycle_sec == 180.0,  "Webster maximum cycle = 180 s"


def test_phase_constraints_clearance_sec() -> None:
    """clearance_sec = yellow + all_red."""
    pc = PhaseConstraints(yellow_sec=4.0, all_red_sec=2.0)
    assert pc.clearance_sec == 6.0


def test_phase_constraints_lost_time() -> None:
    """lost_time_per_phase equals clearance_sec."""
    pc = PhaseConstraints(yellow_sec=3.0, all_red_sec=1.0)
    assert pc.lost_time_per_phase == 4.0


def test_movement_type_enum_values() -> None:
    """All expected movement types are present."""
    expected = {"ns_through", "ns_left", "ew_through", "ew_left", "ns_ped", "ew_ped"}
    actual = {m.value for m in MovementType}
    assert expected == actual


def test_detailed_signal_state_defaults() -> None:
    """DetailedSignalState initialises with safe defaults."""
    ds = DetailedSignalState()
    assert ds.current_movement == MovementType.NS_THROUGH
    assert ds.steps_in_phase == 0
    assert ds.clearance_steps_remaining == 0
    assert ds.is_in_clearance is False
    assert ds.ped_call_ns is False
    assert ds.ped_call_ew is False
    assert ds.ped_wait_steps_ns == 0
    assert ds.ped_wait_steps_ew == 0
    assert ds.preempt_active is False
    assert ds.preempt_steps_remaining == 0
