"""tests/test_priority.py

Tests for traffic_ai.simulation_engine.priority:
  PriorityConfig, PriorityEventSystem, PriorityEventType.
"""
from __future__ import annotations

import numpy as np
import pytest

from traffic_ai.simulation_engine.priority import (
    PriorityConfig,
    PriorityEvent,
    PriorityEventSystem,
    PriorityEventType,
)


def _make_system(enabled: bool = True, **kwargs) -> PriorityEventSystem:
    cfg = PriorityConfig(enabled=enabled, **kwargs)
    rng = np.random.default_rng(42)
    return PriorityEventSystem(cfg, n_intersections=4, step_seconds=1.0, rng=rng)


# ------------------------------------------------------------------
# Disabled system is a no-op
# ------------------------------------------------------------------

def test_disabled_system_no_events() -> None:
    """When enabled=False, generate_events returns empty list."""
    sys = _make_system(enabled=False)
    events = sys.generate_events(step=0)
    assert events == []


def test_disabled_system_no_override() -> None:
    """When enabled=False, apply_preemptions returns actions unchanged."""
    sys = _make_system(enabled=False)
    actions = {0: "NS_THROUGH", 1: "EW_THROUGH"}
    result = sys.apply_preemptions(actions, step=0)
    assert result == actions


# ------------------------------------------------------------------
# Emergency preemption overrides controller output
# ------------------------------------------------------------------

def test_emergency_preemption_overrides_action() -> None:
    """An active emergency preemption forces the intersection to the EV direction."""
    cfg = PriorityConfig(
        enabled=True,
        emergency_enabled=True,
        emergency_rate_per_hour_per_intersection=9999.0,  # guaranteed
        emergency_duration_sec_min=20.0,
        emergency_duration_sec_max=20.0,
        bus_priority_enabled=False,
        lpi_enabled=False,
    )
    rng = np.random.default_rng(0)
    sys = PriorityEventSystem(cfg, n_intersections=1, step_seconds=1.0, rng=rng)
    sys.generate_events(step=0)  # Trigger preemption at intersection 0

    # Now override the controller's action
    actions = {0: "EW_THROUGH"}
    result = sys.apply_preemptions(actions, step=0)

    # Phase forced to emergency vehicle direction (NS_THROUGH or EW_THROUGH)
    assert result[0] in ("NS_THROUGH", "EW_THROUGH")
    assert sys.total_preemption_events >= 1


def test_emergency_preemption_expires() -> None:
    """Preemption expires after duration_steps and controller regains control."""
    cfg = PriorityConfig(
        enabled=True,
        emergency_enabled=True,
        emergency_rate_per_hour_per_intersection=9999.0,
        emergency_duration_sec_min=5.0,
        emergency_duration_sec_max=5.0,
        bus_priority_enabled=False,
        lpi_enabled=False,
    )
    rng = np.random.default_rng(0)
    sys = PriorityEventSystem(cfg, n_intersections=1, step_seconds=1.0, rng=rng)
    sys.generate_events(step=0)

    # Override for 5 steps (duration), then should stop overriding
    actions = {0: "EW_THROUGH"}
    for step in range(5):
        sys.apply_preemptions(actions, step=step)

    # After expiry: controller's original action should pass through
    result = sys.apply_preemptions(actions, step=5)
    assert result[0] == "EW_THROUGH", "Expired preemption must release control"


# ------------------------------------------------------------------
# Bus priority does NOT override like emergency preemption
# ------------------------------------------------------------------

def test_bus_priority_tracked_but_no_hard_override() -> None:
    """Bus priority events are counted but do not force-override controller."""
    cfg = PriorityConfig(
        enabled=True,
        emergency_enabled=False,
        bus_priority_enabled=True,
        bus_rate_per_hour_per_intersection=9999.0,
        lpi_enabled=False,
    )
    rng = np.random.default_rng(42)
    sys = PriorityEventSystem(cfg, n_intersections=1, step_seconds=1.0, rng=rng)
    sys.generate_events(step=0)
    assert sys.total_bus_events >= 1

    # Bus events do not register active preemptions
    actions = {0: "NS_THROUGH"}
    result = sys.apply_preemptions(actions, step=0)
    assert result[0] == "NS_THROUGH", "Bus priority must not hard-override like emergency"


# ------------------------------------------------------------------
# LPI green reduction
# ------------------------------------------------------------------

def test_lpi_reduces_effective_green() -> None:
    """LPI at 100% probability returns non-zero step reduction."""
    cfg = PriorityConfig(
        enabled=True,
        lpi_enabled=True,
        lpi_duration_sec=5.0,
        lpi_probability_per_phase_change=1.0,  # always trigger
    )
    rng = np.random.default_rng(0)
    sys = PriorityEventSystem(cfg, n_intersections=1, step_seconds=1.0, rng=rng)
    reduction = sys.lpi_green_reduction_steps()
    assert reduction == 5, "LPI at p=1.0 must return lpi_duration_sec / step_seconds"
