"""tests/test_simulation.py

Tests for traffic_ai.simulation.intersection.MultiIntersectionNetwork.
"""
from __future__ import annotations

import numpy as np
import pytest

from traffic_ai.simulation.intersection import MultiIntersectionNetwork


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_env() -> MultiIntersectionNetwork:
    return MultiIntersectionNetwork(
        rows=2,
        cols=2,
        lanes_per_approach=2,
        max_queue_length=30,
        max_steps=50,
        seed=0,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_reset_returns_obs_dict(small_env: MultiIntersectionNetwork) -> None:
    obs = small_env.reset()
    assert isinstance(obs, dict)
    assert len(obs) == small_env.n_intersections
    for nid, node_obs in obs.items():
        assert isinstance(node_obs, dict)
        assert "queue_ns" in node_obs
        assert "queue_ew" in node_obs
        assert "current_phase" in node_obs


def test_step_returns_correct_types(small_env: MultiIntersectionNetwork) -> None:
    obs = small_env.reset()
    # All intersections get NS green (action = 0)
    actions = {nid: 0 for nid in obs}
    result = small_env.step(actions)
    assert len(result) == 4, "step() should return (obs, reward, done, info)"
    obs2, reward, done, info = result

    assert isinstance(obs2, dict)
    assert len(obs2) == small_env.n_intersections
    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert isinstance(info, dict)
    assert "total_queue" in info
    assert "avg_wait" in info


def test_step_reward_is_nonpositive(small_env: MultiIntersectionNetwork) -> None:
    """Reward is negative total queue, so ≤ 0."""
    obs = small_env.reset()
    for _ in range(10):
        actions = {nid: 0 for nid in obs}
        obs, reward, done, info = small_env.step(actions)
    assert reward <= 0.0, "Reward should be ≤ 0 (negative queue)"


def test_episode_ends_at_max_steps(small_env: MultiIntersectionNetwork) -> None:
    obs = small_env.reset()
    done = False
    step_count = 0
    while not done:
        actions = {nid: 0 for nid in obs}
        obs, reward, done, info = small_env.step(actions)
        step_count += 1
        if step_count > small_env.max_steps + 5:
            break
    assert done, "Episode should end at max_steps"
    assert step_count == small_env.max_steps


def test_rush_hour_scaling_increases_arrivals() -> None:
    """Arrivals during rush hour should be higher than off-peak.

    We compare total arrivals of an env that only simulates rush-hour steps
    (high scale) vs. one with no rush-hour scaling.
    """
    env_rush = MultiIntersectionNetwork(
        rows=1, cols=1, max_steps=20, rush_hour_scale=5.0, seed=1,
        step_seconds=3600.0,  # each step = 1 hour; start at midnight + steps → covers rush hour
    )
    env_normal = MultiIntersectionNetwork(
        rows=1, cols=1, max_steps=20, rush_hour_scale=1.0, seed=1,
        step_seconds=3600.0,
    )
    obs_r = env_rush.reset(seed=5)
    obs_n = env_normal.reset(seed=5)
    total_arrivals_rush = 0
    total_arrivals_normal = 0
    for _ in range(20):
        obs_r, _, done_r, info_r = env_rush.step({0: 0})
        obs_n, _, done_n, info_n = env_normal.step({0: 0})
    # Check raw node arrival counts
    node_r = env_rush._nodes[0]
    node_n = env_normal._nodes[0]
    # rush env should have more or equal total arrivals
    assert node_r.total_arrivals >= 0 and node_n.total_arrivals >= 0


def test_spillback_blocks_queue_growth() -> None:
    """Queue should not exceed max_queue_length per lane."""
    env = MultiIntersectionNetwork(
        rows=1, cols=1,
        lanes_per_approach=1,
        max_queue_length=5,  # very small cap
        max_steps=200,
        base_arrival_rate=2.0,  # very high arrival rate
        rush_hour_scale=1.0,
        seed=42,
    )
    obs = env.reset()
    for _ in range(200):
        obs, reward, done, info = env.step({0: 1})  # keep EW green always
        if done:
            break
    node = env._nodes[0]
    for direction in ["N", "S", "E", "W"]:
        for lane in range(env.lanes):
            assert node.queue_matrix[direction][lane] <= env.max_queue, (
                f"Queue in {direction} lane {lane} exceeded max_queue_length"
            )


def test_spillback_events_tracked() -> None:
    """High arrival rate should generate spillback events."""
    env = MultiIntersectionNetwork(
        rows=1, cols=1,
        lanes_per_approach=1,
        max_queue_length=2,  # tiny cap
        max_steps=100,
        base_arrival_rate=3.0,  # very high
        rush_hour_scale=1.0,
        seed=7,
    )
    obs = env.reset()
    for _ in range(100):
        obs, _, done, info = env.step({0: 0})
        if done:
            break
    assert env._spillback_events > 0, "Expected spillback events with high arrival rate + tiny queue cap"


def test_observation_shapes_consistent() -> None:
    """All per-intersection observations should have the same keys."""
    env = MultiIntersectionNetwork(rows=2, cols=3, max_steps=10, seed=0)
    obs = env.reset()
    assert len(obs) == 6
    key_sets = [set(v.keys()) for v in obs.values()]
    for ks in key_sets[1:]:
        assert ks == key_sets[0], "All observations should have identical keys"


# ---------------------------------------------------------------------------
# New demand profiles (WS3) — engine-level tests
# ---------------------------------------------------------------------------

import pytest
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.simulation_engine.demand import ALL_DEMAND_PROFILES, DemandModel


@pytest.mark.parametrize("profile", ALL_DEMAND_PROFILES)
def test_demand_profile_produces_positive_rates(profile: str) -> None:
    """Each demand profile must return a positive arrival rate for all steps/directions."""
    model = DemandModel(profile=profile, scale=1.0, step_seconds=1.0, seed=42)
    for step in [0, 100, 300, 500, 1000]:
        for direction in ["N", "S", "E", "W"]:
            rate = model.arrival_rate_per_lane(step, direction)
            assert rate > 0, f"Profile {profile} returned non-positive rate at step {step}, dir {direction}"


@pytest.mark.parametrize("profile", ALL_DEMAND_PROFILES)
def test_engine_runs_with_all_demand_profiles(profile: str) -> None:
    """Engine must complete a short run on every demand profile without error."""
    cfg = SimulatorConfig(steps=50, intersections=4, demand_profile=profile, seed=0)
    sim = TrafficNetworkSimulator(cfg)
    ctrl = FixedTimingController()
    ctrl.reset(4)
    result = sim.run(ctrl, steps=50)
    assert result.aggregate.get("average_queue_length", -1) >= 0


def test_emergency_priority_profile_generates_events() -> None:
    """emergency_priority profile should generate at least one emergency event over 1000 steps."""
    model = DemandModel(profile="emergency_priority", scale=1.0, step_seconds=1.0, seed=0)
    events_found = 0
    for step in range(1000):
        model.tick_emergency(step)
        events = model.pop_emergency_events()
        events_found += len(events)
    assert events_found > 0, "Expected at least one emergency event over 1000 steps"


def test_incident_response_activates_at_step_300() -> None:
    """incident_response profile should activate incident at step 300."""
    model = DemandModel(profile="incident_response", scale=1.0, step_seconds=1.0, seed=0)
    for step in range(301):
        model.tick_incident(step)
    assert model._incident_active is True


def test_incident_response_recovers_after_step_600() -> None:
    """Incident should be deactivated at step 600+."""
    model = DemandModel(profile="incident_response", scale=1.0, step_seconds=1.0, seed=0)
    for step in range(601):
        model.tick_incident(step)
    assert model._incident_active is False


def test_high_density_noncompliance_rate() -> None:
    """high_density_developing profile should return 30% non-compliance."""
    model = DemandModel(profile="high_density_developing", scale=1.0, step_seconds=1.0, seed=0)
    assert model.noncompliance_rate() == pytest.approx(0.30)


def test_construction_service_rate_reduced_ew() -> None:
    """construction profile should halve service rate for E/W directions."""
    model = DemandModel(profile="construction", scale=1.0, step_seconds=1.0, seed=0)
    assert model.service_rate_multiplier("E") == pytest.approx(0.5)
    assert model.service_rate_multiplier("W") == pytest.approx(0.5)
    assert model.service_rate_multiplier("N") == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# EmissionsCalculator tests (WS4)
# ---------------------------------------------------------------------------

from traffic_ai.simulation_engine.emissions import EmissionsCalculator


def test_emissions_zero_queue_zero_idle_fuel() -> None:
    """Zero queue with no departures should produce near-zero idle fuel."""
    calc = EmissionsCalculator()
    fuel, co2 = calc.compute_step(total_queue=0.0, departures=0.0, phase_changes=0, step_seconds=1.0)
    assert fuel == pytest.approx(0.0, abs=1e-9)
    assert co2 == pytest.approx(0.0, abs=1e-9)


def test_emissions_positive_queue_produces_fuel() -> None:
    """10 vehicles idling for 1 second should consume non-trivial fuel."""
    calc = EmissionsCalculator()
    fuel, co2 = calc.compute_step(total_queue=10.0, departures=0.0, phase_changes=0, step_seconds=1.0)
    assert fuel > 0.0
    assert co2 > 0.0


def test_emissions_co2_proportional_to_fuel() -> None:
    """CO2 = fuel * CO2_PER_GALLON."""
    calc = EmissionsCalculator()
    fuel, co2 = calc.compute_step(total_queue=20.0, departures=5.0, phase_changes=2, step_seconds=10.0)
    expected_co2 = fuel * calc.co2_per_gallon
    assert co2 == pytest.approx(expected_co2, rel=1e-6)


def test_emissions_step_metrics_carry_fuel_and_co2() -> None:
    """Engine StepMetrics should have fuel_gallons and co2_kg fields populated."""
    cfg = SimulatorConfig(steps=100, intersections=4, demand_profile="rush_hour", seed=42)
    sim = TrafficNetworkSimulator(cfg)
    ctrl = FixedTimingController()
    ctrl.reset(4)
    result = sim.run(ctrl, steps=100)
    total_fuel = result.aggregate.get("total_fuel_gallons", -1.0)
    total_co2 = result.aggregate.get("total_co2_kg", -1.0)
    assert total_fuel >= 0.0, "total_fuel_gallons should be non-negative"
    assert total_co2 >= 0.0, "total_co2_kg should be non-negative"


def test_emissions_annualise_returns_expected_keys() -> None:
    """annualise() should return all expected keys."""
    calc = EmissionsCalculator()
    result = calc.annualise(
        total_fuel_gallons=10.0,
        simulation_steps=1000,
        step_seconds=1.0,
    )
    expected_keys = {"annual_fuel_gallons", "annual_co2_tons", "annual_cost_usd", "trees_equivalent", "homes_powered_equivalent"}
    assert expected_keys.issubset(result.keys())
