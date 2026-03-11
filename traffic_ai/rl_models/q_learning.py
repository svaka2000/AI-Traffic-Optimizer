from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import DefaultDict

import numpy as np

from traffic_ai.rl_models.environment import SignalControlEnv


@dataclass(slots=True)
class QLearningPolicy:
    q_table: dict[tuple[int, int, int], np.ndarray]

    def act(self, features: np.ndarray) -> int:
        key = discretize_state(features)
        values = self.q_table.get(key)
        if values is None:
            return int(features[0] < features[1])
        return int(np.argmax(values))


def discretize_state(state: np.ndarray) -> tuple[int, int, int]:
    queue_ns = int(min(12, state[0] // 8))
    queue_ew = int(min(12, state[1] // 8))
    phase_elapsed = int(min(8, state[3] // 10))
    return queue_ns, queue_ew, phase_elapsed


def train_q_learning(
    env: SignalControlEnv,
    episodes: int = 300,
    alpha: float = 0.15,
    gamma: float = 0.95,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 42,
) -> tuple[QLearningPolicy, list[float]]:
    rng = np.random.default_rng(seed)
    q_table: DefaultDict[tuple[int, int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(2, dtype=np.float32)
    )
    rewards: list[float] = []
    epsilon = epsilon_start
    decay = (epsilon_start - epsilon_end) / max(episodes, 1)

    for _ in range(episodes):
        state = env.reset()
        done = False
        episode_reward = 0.0
        while not done:
            key = discretize_state(state)
            if rng.random() < epsilon:
                action = int(rng.integers(0, 2))
            else:
                action = int(np.argmax(q_table[key]))
            next_state, reward, done, _ = env.step(action)
            next_key = discretize_state(next_state)
            td_target = reward + gamma * float(np.max(q_table[next_key])) * (1.0 - float(done))
            td_error = td_target - float(q_table[key][action])
            q_table[key][action] += alpha * td_error
            state = next_state
            episode_reward += reward
        epsilon = max(epsilon_end, epsilon - decay)
        rewards.append(float(episode_reward))

    policy = QLearningPolicy(q_table={k: v.copy() for k, v in q_table.items()})
    return policy, rewards

