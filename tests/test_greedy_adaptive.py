"""tests/test_greedy_adaptive.py

Tests for traffic_ai.controllers.greedy_adaptive.GreedyAdaptiveController.

Validates the InSync-modelled greedy cost-minimising algorithm:
  - Minimum green enforcement before phase switch
  - Phase switching driven by demand imbalance
  - Coordination weight discourages sending traffic into congested neighbours
  - Delay accumulation eventually drives a phase switch
  - Hysteresis prevents oscillation on nearly-balanced demand
"""
from __future__ import annotations

import pytest

from traffic_ai.controllers.greedy_adaptive import GreedyAdaptiveController


# ------------------------------------------------------------------
# Helper: synthetic observations
# ------------------------------------------------------------------

def _obs(
    queue_ns: float = 10.0,
    queue_ew: float = 10.0,
    upstream_queue: float = 0.0,
) -> dict[str, float]:
    return {
        "queue_ns": queue_ns,
        "queue_ew": queue_ew,
        "queue_ns_through": queue_ns,
        "queue_ew_through": queue_ew,
        "queue_ns_left": 0.0,
        "queue_ew_left": 0.0,
        "upstream_queue": upstream_queue,
        "total_queue": queue_ns + queue_ew,
        "sim_step": 0.0,
    }


# ------------------------------------------------------------------
# Minimum green enforcement
# ------------------------------------------------------------------

def test_min_green_enforced() -> None:
    """Controller must not switch phase within min_green_steps even under extreme demand."""
    ctrl = GreedyAdaptiveController(min_green_steps=10)
    ctrl.reset(1)

    # EW has massive demand — controller would want to switch immediately
    obs_seq = {0: _obs(queue_ns=1.0, queue_ew=1000.0)}
    phases_seen = []
    for step in range(9):
        actions = ctrl.compute_actions(obs_seq, step)
        phases_seen.append(actions[0])

    # All 9 steps must remain NS_THROUGH (phase 0 is the starting phase)
    assert all(p == "NS_THROUGH" for p in phases_seen), (
        f"Should not switch before min_green_steps=10, got phases: {phases_seen}"
    )


# ------------------------------------------------------------------
# Phase switching on demand imbalance
# ------------------------------------------------------------------

def test_phase_switches_on_heavy_ew_demand() -> None:
    """Controller switches to EW_THROUGH when EW queue greatly exceeds NS queue."""
    ctrl = GreedyAdaptiveController(
        min_green_steps=3,
        volume_weight=1.0,
        delay_weight=0.2,
        coordination_weight=0.0,  # disable coordination to isolate volume effect
        hysteresis=0.05,
    )
    ctrl.reset(1)

    # Balanced demand first (to pass min_green), then strong EW demand
    balanced = {0: _obs(queue_ns=10.0, queue_ew=10.0)}
    heavy_ew = {0: _obs(queue_ns=1.0, queue_ew=100.0)}

    # Prime for min_green
    for step in range(3):
        ctrl.compute_actions(balanced, step)

    # Apply heavy EW demand; switch should happen within a few more steps
    ew_through_seen = False
    for step in range(3, 20):
        actions = ctrl.compute_actions(heavy_ew, step)
        if actions[0] == "EW_THROUGH":
            ew_through_seen = True
            break

    assert ew_through_seen, "Should switch to EW_THROUGH under heavy EW demand"


# ------------------------------------------------------------------
# Coordination weight discourages injecting into congested neighbours
# ------------------------------------------------------------------

def test_coordination_weight_reduces_cost_when_neighbour_congested() -> None:
    """With high coordination_weight, congested downstream neighbour reduces cost of that phase."""
    ctrl_no_coord = GreedyAdaptiveController(
        min_green_steps=3,
        coordination_weight=0.0,
        hysteresis=0.0,
    )
    ctrl_coord = GreedyAdaptiveController(
        min_green_steps=3,
        coordination_weight=10.0,  # strong coordination penalty
        hysteresis=0.0,
    )
    ctrl_no_coord.reset(2)
    ctrl_coord.reset(2)

    # Intersection 0: balanced demand; intersection 1 (neighbour): heavy EW queue
    obs = {
        0: _obs(queue_ns=20.0, queue_ew=20.0),
        1: _obs(queue_ns=5.0,  queue_ew=200.0),  # congested EW downstream
    }

    # Run past min_green
    for step in range(3):
        ctrl_no_coord.compute_actions(obs, step)
        ctrl_coord.compute_actions(obs, step)

    # Run a few more steps and compare EW selection frequency
    ew_no_coord = 0
    ew_coord = 0
    for step in range(3, 30):
        a_nc = ctrl_no_coord.compute_actions(obs, step)
        a_c  = ctrl_coord.compute_actions(obs, step)
        if a_nc[0] == "EW_THROUGH":
            ew_no_coord += 1
        if a_c[0] == "EW_THROUGH":
            ew_coord += 1

    # Coordination penalty should cause fewer or equal EW selections
    assert ew_coord <= ew_no_coord, (
        f"Coordination weight should reduce EW serving into congested neighbour "
        f"(coord={ew_coord}, no_coord={ew_no_coord})"
    )


# ------------------------------------------------------------------
# Delay accumulation drives phase switch
# ------------------------------------------------------------------

def test_delay_accumulation_drives_switch() -> None:
    """Even with balanced volume, accumulated delay eventually forces a phase switch."""
    ctrl = GreedyAdaptiveController(
        min_green_steps=3,
        volume_weight=1.0,
        delay_weight=1.0,   # high delay weight to ensure delay triggers switch
        coordination_weight=0.0,
        hysteresis=0.05,
    )
    ctrl.reset(1)

    # Equal queues — volume alone won't trigger a switch
    balanced = {0: _obs(queue_ns=10.0, queue_ew=10.0)}

    switch_step = None
    for step in range(200):
        actions = ctrl.compute_actions(balanced, step)
        if actions[0] == "EW_THROUGH":
            switch_step = step
            break

    assert switch_step is not None, (
        "Delay accumulation should eventually trigger a phase switch even with balanced volume"
    )
    assert switch_step >= 3, "Switch must not happen before min_green_steps"


# ------------------------------------------------------------------
# Hysteresis prevents oscillation
# ------------------------------------------------------------------

def test_hysteresis_prevents_rapid_oscillation() -> None:
    """With high hysteresis, controller does not oscillate on nearly-balanced demand."""
    ctrl = GreedyAdaptiveController(
        min_green_steps=3,
        volume_weight=1.0,
        delay_weight=0.0,   # no delay accumulation — pure volume
        coordination_weight=0.0,
        hysteresis=0.50,    # 50 % premium required to switch
    )
    ctrl.reset(1)

    # Slightly imbalanced (not enough to overcome 50 % hysteresis)
    obs_seq = {0: _obs(queue_ns=10.0, queue_ew=12.0)}

    phases = []
    for step in range(50):
        actions = ctrl.compute_actions(obs_seq, step)
        phases.append(actions[0])

    # Count phase changes
    changes = sum(1 for a, b in zip(phases, phases[1:]) if a != b)
    assert changes <= 2, (
        f"High hysteresis should prevent oscillation, but saw {changes} phase changes"
    )
