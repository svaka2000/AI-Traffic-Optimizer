from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque

import numpy as np

from traffic_ai.rl_models.environment import MultiIntersectionSignalEnv, SignalControlEnv
from traffic_ai.rl_models.q_learning import QLearningPolicy, train_q_learning

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim

    TORCH_AVAILABLE = True
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    optim = None  # type: ignore[assignment]
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class DQNetwork(nn.Module):
        def __init__(self, input_dim: int = 8, output_dim: int = 4) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(64, output_dim),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


@dataclass(slots=True)
class DQNPolicy:
    network: Any

    def act(self, features: np.ndarray) -> int:
        if TORCH_AVAILABLE and self.network is not None:
            obs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                q = self.network(obs)
            return int(torch.argmax(q, dim=-1).item())
        return int(features[2] < features[3])  # queue_ns_norm < queue_ew_norm → EW


def train_dqn(
    env: SignalControlEnv,
    episodes: int = 220,
    gamma: float = 0.99,
    lr: float = 5e-4,
    batch_size: int = 128,
    seed: int = 42,
    warmup: int = 1_000,
    reward_scale: float = 3.0,
    n_train_seeds: int = 1,
) -> tuple[DQNPolicy, list[float], Any]:
    rng = np.random.default_rng(seed)
    if not TORCH_AVAILABLE:
        fallback_policy, rewards = train_q_learning(env, episodes=max(20, episodes // 2), seed=seed)
        return DQNPolicy(network=fallback_policy), rewards, None

    torch.manual_seed(seed)
    device = torch.device("cpu")
    obs_dim = env.observation_dim
    n_act = env.n_actions
    policy_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    # Cosine annealing: decays lr from lr → lr/100 over the full training run.
    # Prevents late-stage gradient updates from destabilising the converged policy.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(episodes, 1), eta_min=lr * 0.01)
    memory: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=100_000)
    rewards: list[float] = []

    epsilon = 1.0
    epsilon_min = 0.05
    # Always reach epsilon_min within the first 500 episodes (or fewer if training
    # is shorter). For long runs (e.g. 2000 ep) this leaves the bulk of training
    # as near-greedy exploitation so Q-values stabilise rather than chasing noise.
    epsilon_decay = (epsilon - epsilon_min) / min(episodes, 500)

    for episode in range(episodes):
        # Cycle through demand seeds so the policy generalises across the same
        # range of seeds used in cross-validation (seed, seed+1, … seed+n-1).
        if n_train_seeds > 1:
            env._engine.config.seed = seed + (episode % n_train_seeds)
        state = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            if rng.random() < epsilon:
                action = int(rng.integers(0, n_act))
            else:
                with torch.no_grad():
                    q_values = policy_net(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
                    action = int(torch.argmax(q_values, dim=1).item())

            next_state, reward, done, _ = env.step(action)
            # Normalize reward to prevent multi-objective reward scaling from destabilising Q-targets.
            memory.append((state, action, reward / reward_scale, next_state, done))
            state = next_state
            total_reward += reward

            if len(memory) >= warmup:
                batch_idx = rng.choice(len(memory), size=batch_size, replace=False)
                batch = [memory[i] for i in batch_idx]
                states = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                actions = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
                rewards_t = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                dones = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_values = policy_net(states).gather(1, actions).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states).max(dim=1).values
                    target = rewards_t + gamma * next_q * (1 - dones)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

        # Step the LR schedule once per episode (not per batch step) so cosine
        # annealing decays smoothly from lr → lr/100 over the full training run.
        scheduler.step()
        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        rewards.append(float(total_reward))
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return DQNPolicy(network=policy_net.eval()), rewards, policy_net.eval()


def train_dqn_multi(
    env: MultiIntersectionSignalEnv,
    episodes: int = 2000,
    gamma: float = 0.99,
    lr: float = 5e-4,
    batch_size: int = 128,
    seed: int = 42,
    warmup: int = 1_000,
    reward_scale: float = 3.0,
    n_train_seeds: int = 1,
) -> tuple[DQNPolicy, list[float], Any]:
    """Independent DQN (IDQN) for multi-intersection networks.

    A single shared DQN policy is trained across all intersections simultaneously.
    Each intersection's transition ``(s_i, a_i, r_i, s'_i)`` is added to the
    shared replay buffer, so the agent learns from N times as many experiences
    per environment step as single-intersection training.

    Same hyperparameters as ``train_dqn``:
        lr=5e-4, cosine annealing, warmup=1000, gradient clip max_norm=1.0.
    """
    rng = np.random.default_rng(seed)
    if not TORCH_AVAILABLE:
        # Fallback: train a Q-learning policy on a simple single-intersection env
        from traffic_ai.rl_models.environment import SignalControlEnv, EnvConfig
        fallback_env = SignalControlEnv(EnvConfig(seed=seed))
        fallback_policy, fb_rewards = train_q_learning(fallback_env, episodes=max(20, episodes // 2), seed=seed)
        return DQNPolicy(network=fallback_policy), fb_rewards, None

    torch.manual_seed(seed)
    device = torch.device("cpu")
    obs_dim = env.observation_dim  # 8
    n_act = env.n_actions          # 4
    n_int = env.n_intersections

    policy_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net = DQNetwork(input_dim=obs_dim, output_dim=n_act).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(episodes, 1), eta_min=lr * 0.01
    )
    memory: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque(maxlen=100_000)

    epsilon = 1.0
    epsilon_min = 0.05
    epsilon_decay = (epsilon - epsilon_min) / min(episodes, 500)

    ep_rewards: list[float] = []

    for episode in range(episodes):
        if n_train_seeds > 1:
            env._engine.config.seed = seed + (episode % n_train_seeds)
        obs_dict = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            # Choose actions for all intersections using shared policy
            actions: dict[int, int] = {}
            for i, obs_i in obs_dict.items():
                if rng.random() < epsilon:
                    actions[i] = int(rng.integers(0, n_act))
                else:
                    with torch.no_grad():
                        q_vals = policy_net(
                            torch.tensor(obs_i, dtype=torch.float32).unsqueeze(0)
                        )
                        actions[i] = int(torch.argmax(q_vals, dim=1).item())

            next_obs_dict, reward_dict, done, _ = env.step(actions)

            # Deposit one transition per intersection into the shared buffer
            for i in range(n_int):
                r_scaled = reward_dict.get(i, 0.0) / reward_scale
                memory.append((obs_dict[i], actions[i], r_scaled, next_obs_dict[i], done))
                total_reward += reward_dict.get(i, 0.0)

            obs_dict = next_obs_dict

            if len(memory) >= warmup:
                batch_idx = rng.choice(len(memory), size=batch_size, replace=False)
                batch = [memory[j] for j in batch_idx]
                states_t = torch.tensor(np.array([b[0] for b in batch]), dtype=torch.float32)
                actions_t = torch.tensor([b[1] for b in batch], dtype=torch.long).unsqueeze(1)
                rewards_t = torch.tensor([b[2] for b in batch], dtype=torch.float32)
                next_states_t = torch.tensor(np.array([b[3] for b in batch]), dtype=torch.float32)
                dones_t = torch.tensor([b[4] for b in batch], dtype=torch.float32)

                q_values = policy_net(states_t).gather(1, actions_t).squeeze(1)
                with torch.no_grad():
                    next_q = target_net(next_states_t).max(dim=1).values
                    target = rewards_t + gamma * next_q * (1 - dones_t)
                loss = nn.functional.mse_loss(q_values, target)
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)
                optimizer.step()

        scheduler.step()
        epsilon = max(epsilon_min, epsilon - epsilon_decay)
        ep_rewards.append(float(total_reward))
        if episode % 10 == 0:
            target_net.load_state_dict(policy_net.state_dict())

    return DQNPolicy(network=policy_net.eval()), ep_rewards, policy_net.eval()
