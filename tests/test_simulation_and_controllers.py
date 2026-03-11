from __future__ import annotations

import numpy as np

from traffic_ai.controllers import AdaptiveRuleController, FixedTimingController
from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv
from traffic_ai.rl_models.q_learning import train_q_learning
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator


def test_simulator_with_baselines() -> None:
    sim = TrafficNetworkSimulator(
        SimulatorConfig(steps=120, intersections=2, lanes_per_direction=2, seed=13)
    )
    controllers = [FixedTimingController(), AdaptiveRuleController()]
    for controller in controllers:
        result = sim.run(controller, steps=120)
        assert "average_wait_time" in result.aggregate
        assert result.aggregate["average_wait_time"] >= 0
        assert len(result.step_metrics) == 120


def test_q_learning_policy_training() -> None:
    env = SignalControlEnv(EnvConfig(step_limit=40, seed=11))
    policy, rewards = train_q_learning(env, episodes=15)
    assert len(rewards) == 15
    action = policy.act(np.array([10, 8, 18, 4, 0], dtype=np.float32))
    assert action in (0, 1)

