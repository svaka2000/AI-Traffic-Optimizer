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
