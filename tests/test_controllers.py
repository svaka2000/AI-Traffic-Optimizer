"""tests/test_controllers.py

Tests for new-style controllers in traffic_ai.controllers.
"""
from __future__ import annotations

import numpy as np
import pytest

from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.rule_based import RuleBasedController
from traffic_ai.controllers.ml_controllers import (
    GradientBoostingController,
    MLPController,
    RandomForestController,
    XGBoostController,
    ImitationLearningController,
    LSTMForecastController,
)
from traffic_ai.controllers.rl_controllers import (
    DQNController,
    PPOController,
    QLearningController,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_obs(queue_ns: float = 5.0, queue_ew: float = 3.0, step: int = 0) -> dict[str, float]:
    return {
        "queue_ns": queue_ns,
        "queue_ew": queue_ew,
        "total_queue": queue_ns + queue_ew,
        "current_phase": 0.0,
        "phase_elapsed": float(step % 30),
        "avg_speed": 30.0,
        "lane_occupancy": 0.4,
        "step": float(step),
        "day_of_week": 2.0,
    }


ALL_CONTROLLERS = [
    FixedTimingController,
    RuleBasedController,
    RandomForestController,
    XGBoostController,
    GradientBoostingController,
    MLPController,
    LSTMForecastController,
    ImitationLearningController,
    QLearningController,
    DQNController,
    PPOController,
]


# ---------------------------------------------------------------------------
# Generic controller tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("CtrlCls", ALL_CONTROLLERS)
def test_select_action_returns_valid_action(CtrlCls) -> None:
    """select_action must return 0 (NS) or 1 (EW)."""
    ctrl = CtrlCls()
    for step in range(5):
        obs = _make_obs(step=step * 10)
        action = ctrl.select_action(obs)
        assert action in (0, 1), f"{CtrlCls.__name__}.select_action returned {action!r}"


@pytest.mark.parametrize("CtrlCls", ALL_CONTROLLERS)
def test_compute_actions_returns_valid_phases(CtrlCls) -> None:
    """compute_actions must return dict mapping int → 'NS'|'EW'."""
    ctrl = CtrlCls()
    n = 4
    ctrl.reset(n)
    observations = {i: _make_obs(step=0) for i in range(n)}
    actions = ctrl.compute_actions(observations, step=0)
    assert isinstance(actions, dict)
    assert len(actions) == n
    for iid, phase in actions.items():
        assert phase in ("NS", "EW"), f"Invalid phase {phase!r} from {CtrlCls.__name__}"


@pytest.mark.parametrize("CtrlCls", ALL_CONTROLLERS)
def test_update_does_not_crash(CtrlCls) -> None:
    """update() must not raise."""
    ctrl = CtrlCls()
    obs = _make_obs()
    next_obs = _make_obs(queue_ns=4.0, queue_ew=6.0)
    ctrl.update(obs, action=0, reward=-5.0, next_obs=next_obs)


# ---------------------------------------------------------------------------
# FixedTimingController specific tests
# ---------------------------------------------------------------------------

def test_fixed_timing_switches_phase_at_correct_step() -> None:
    """FixedTimingController should switch phase every cycle_seconds/2 steps."""
    ctrl = FixedTimingController(cycle_seconds=20, green_split=0.5, step_seconds=1.0)
    phases = [ctrl.select_action(_make_obs(step=s)) for s in range(25)]
    # Steps 0-9 → NS (0), Steps 10-19 → EW (1), Step 20 → NS (0) again
    assert all(a == 0 for a in phases[:10]), "First 10 steps should be NS green"
    assert all(a == 1 for a in phases[10:20]), "Steps 10-19 should be EW green"
    assert phases[20] == 0, "Step 20 should wrap back to NS green"


def test_fixed_timing_default_30s_cycle() -> None:
    """Default cycle is 30 seconds."""
    ctrl = FixedTimingController()
    assert ctrl.cycle_seconds == 30


# ---------------------------------------------------------------------------
# RuleBasedController specific tests
# ---------------------------------------------------------------------------

def test_rule_based_prefers_higher_queue_axis() -> None:
    """After min_green steps, RuleBasedController should switch to the axis
    with more waiting vehicles."""
    ctrl = RuleBasedController(min_green=2, max_green=100, threshold=2.0)
    ctrl.reset(1)

    # Run for min_green steps on NS (step 0-1), then trigger EW preference
    for step in range(2):
        obs = _make_obs(queue_ns=3.0, queue_ew=3.0, step=step)
        ctrl.select_action(obs)

    # NS has 2 vehicles, EW has 20 → should switch to EW
    obs = _make_obs(queue_ns=2.0, queue_ew=20.0, step=2)
    action = ctrl.select_action(obs)
    assert action == 1, "Should switch to EW green when EW queue is much larger"


def test_rule_based_forces_switch_at_max_green() -> None:
    """RuleBasedController must switch phase when max_green is reached."""
    ctrl = RuleBasedController(min_green=2, max_green=5, threshold=100.0)
    ctrl.reset(1)
    # Keep NS queue always higher so it won't switch early
    actions = []
    for step in range(10):
        obs = _make_obs(queue_ns=50.0, queue_ew=0.0, step=step)
        actions.append(ctrl.select_action(obs))
    # After 5 steps of NS green, should flip to EW at step 5+
    assert 1 in actions[5:], "Should have switched to EW after max_green steps"


# ---------------------------------------------------------------------------
# RL controller specific tests
# ---------------------------------------------------------------------------

def test_q_learning_action_in_valid_range() -> None:
    ctrl = QLearningController(epsilon=0.0, seed=0)
    obs = _make_obs()
    for _ in range(20):
        action = ctrl.select_action(obs)
        assert action in (0, 1)


def test_q_learning_updates_q_table() -> None:
    ctrl = QLearningController(seed=0)
    q_before = ctrl._q.copy()
    obs = _make_obs()
    next_obs = _make_obs(queue_ns=1.0, queue_ew=10.0)
    ctrl.update(obs, action=0, reward=-5.0, next_obs=next_obs)
    assert not np.array_equal(ctrl._q, q_before), "Q-table should be updated after update()"


def test_dqn_action_valid() -> None:
    ctrl = DQNController(seed=0)
    obs = _make_obs()
    for _ in range(5):
        action = ctrl.select_action(obs)
        assert action in (0, 1)


def test_ppo_action_valid() -> None:
    ctrl = PPOController(seed=0)
    obs = _make_obs()
    for _ in range(5):
        action = ctrl.select_action(obs)
        assert action in (0, 1)


# ---------------------------------------------------------------------------
# ML controllers – untrained fallback
# ---------------------------------------------------------------------------

def test_rf_untrained_falls_back_to_queue_heuristic() -> None:
    ctrl = RandomForestController()
    # Higher NS queue → should pick NS (0)
    obs = _make_obs(queue_ns=20.0, queue_ew=1.0)
    action = ctrl.select_action(obs)
    assert action == 0

    # Higher EW queue → should pick EW (1)
    obs2 = _make_obs(queue_ns=1.0, queue_ew=20.0)
    action2 = ctrl.select_action(obs2)
    assert action2 == 1


def test_ml_controllers_fit_and_predict() -> None:
    """RF and GB can be trained on small synthetic data."""
    rng = np.random.default_rng(0)
    n = 200
    X = rng.standard_normal((n, 7)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "NS", "EW")

    for CtrlCls in [RandomForestController, GradientBoostingController]:
        ctrl = CtrlCls()
        metrics = ctrl.fit(X, y, cv_folds=2)
        assert "cv_mean" in metrics
        assert 0.0 <= metrics["cv_mean"] <= 1.0
        obs = _make_obs()
        action = ctrl.select_action(obs)
        assert action in (0, 1)


def test_mlp_fit_and_predict() -> None:
    rng = np.random.default_rng(1)
    n = 100
    X = rng.standard_normal((n, 7)).astype(np.float32)
    y = np.where(X[:, 0] > 0, "NS", "EW")

    ctrl = MLPController(seed=0)
    metrics = ctrl.fit(X, y, epochs=3, batch_size=32)
    assert "final_loss" in metrics
    obs = _make_obs()
    action = ctrl.select_action(obs)
    assert action in (0, 1)


def test_imitation_learning_fit_and_predict() -> None:
    rng = np.random.default_rng(2)
    n = 100
    X = rng.standard_normal((n, 7)).astype(np.float32)
    y = rng.integers(0, 2, n)

    ctrl = ImitationLearningController(seed=0)
    metrics = ctrl.fit(X, y, epochs=3)
    assert "final_loss" in metrics
    action = ctrl.select_action(_make_obs())
    assert action in (0, 1)
