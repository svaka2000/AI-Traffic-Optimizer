"""tests/test_scenarios.py

Tests for traffic_ai.scenarios.san_diego.SanDiegoScenario.

Validates that all San Diego corridor scenarios:
  - Load without error and return a valid SimulatorConfig
  - Have the correct number of intersections
  - Have the correct demand scales
  - Raise ValueError on unknown scenario name
"""
from __future__ import annotations

import pytest

from traffic_ai.scenarios.san_diego import SanDiegoScenario
from traffic_ai.simulation_engine.engine import SimulatorConfig


# ------------------------------------------------------------------
# All scenarios load cleanly
# ------------------------------------------------------------------

def test_all_scenarios_load() -> None:
    """Every named scenario returns a SimulatorConfig without raising."""
    for name in SanDiegoScenario.list_scenarios():
        cfg = SanDiegoScenario.get_scenario(name)
        assert isinstance(cfg, SimulatorConfig), (
            f"Scenario '{name}' did not return a SimulatorConfig"
        )


# ------------------------------------------------------------------
# Intersection counts
# ------------------------------------------------------------------

def test_rosecrans_corridor_has_12_intersections() -> None:
    """rosecrans_corridor models the real 12-signal corridor exactly."""
    cfg = SanDiegoScenario.rosecrans_corridor()
    assert cfg.intersections == 12, (
        f"rosecrans_corridor should have 12 intersections, got {cfg.intersections}"
    )


def test_downtown_grid_has_16_intersections() -> None:
    """downtown_grid models 4×4 = 16 intersections."""
    cfg = SanDiegoScenario.downtown_grid()
    assert cfg.intersections == 16, (
        f"downtown_grid should have 16 intersections, got {cfg.intersections}"
    )


# ------------------------------------------------------------------
# Demand scales
# ------------------------------------------------------------------

def test_downtown_grid_demand_scale() -> None:
    """downtown_grid demand_scale=1.8 (dense urban core)."""
    cfg = SanDiegoScenario.downtown_grid()
    assert cfg.demand_scale == pytest.approx(1.8), (
        f"downtown_grid demand_scale should be 1.8, got {cfg.demand_scale}"
    )


def test_mira_mesa_corridor_demand_scale() -> None:
    """mira_mesa_corridor demand_scale=2.2 (heavy commuter corridor)."""
    cfg = SanDiegoScenario.mira_mesa_corridor()
    assert cfg.demand_scale == pytest.approx(2.2), (
        f"mira_mesa_corridor demand_scale should be 2.2, got {cfg.demand_scale}"
    )


# ------------------------------------------------------------------
# Unknown scenario raises ValueError
# ------------------------------------------------------------------

def test_get_scenario_raises_on_unknown_name() -> None:
    """get_scenario() raises ValueError for unrecognised scenario names."""
    with pytest.raises(ValueError, match="Unknown scenario"):
        SanDiegoScenario.get_scenario("nonexistent_corridor")
