"""traffic_ai/controllers/rl_controllers.py

Reinforcement learning controllers for traffic signal optimization.

Controllers
-----------
- QLearningController: tabular Q-learning with discretized state space
- DQNController: PyTorch DQN with replay buffer and target network
- PPOController: PyTorch PPO with actor-critic architecture
"""
from __future__ import annotations

import collections
import logging
import random
from pathlib import Path
from typing import Any, Deque

import numpy as np

from traffic_ai.controllers.base import BaseController
from traffic_ai.simulation_engine.types import SignalPhase

logger = logging.getLogger(__name__)

N_ACTIONS = 2  # 0 = NS green, 1 = EW green


def _obs_to_vec(obs: dict[str, float]) -> np.ndarray:
    return np.array(
        [
            obs.get("queue_ns", 0.0),
            obs.get("queue_ew", 0.0),
            obs.get("avg_speed", 30.0),
            obs.get("lane_occupancy", 0.5),
            float(obs.get("current_phase", 0.0)),
            float(obs.get("step", 0.0)) % 86400.0 / 86400.0,
        ],
        dtype=np.float32,
    )


# ---------------------------------------------------------------------------
# Tabular Q-Learning Controller
# ---------------------------------------------------------------------------

class QLearningController(BaseController):
    """Tabular Q-learning with discretized state space.

    State: (queue_ns_bucket, queue_ew_bucket, current_phase)
    Action: 0 (NS green) or 1 (EW green)
    """

    QUEUE_BUCKETS: int = 6
    QUEUE_MAX: float = 60.0

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int = 42,
    ) -> None:
        super().__init__(name="q_learning")
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self._rng = np.random.default_rng(seed)
        # Q-table: shape (queue_buckets, queue_buckets, 2_phases, 2_actions)
        self._q: np.ndarray = np.zeros(
            (self.QUEUE_BUCKETS, self.QUEUE_BUCKETS, 2, N_ACTIONS), dtype=np.float64
        )
        self._prev_state: dict[int, tuple[int, int, int]] | None = None
        self._prev_action: dict[int, int] | None = None
        self._current_phase: dict[int, int] = {}

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._prev_state = None
        self._prev_action = None

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            a = self._act(iid, obs)
            actions[iid] = "NS" if a == 0 else "EW"
            self._current_phase[iid] = a
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        return self._act(0, obs)

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        s = self._discretize(obs)
        s_next = self._discretize(next_obs)
        q_next = 0.0 if done else float(np.max(self._q[s_next]))
        td_target = reward + self.gamma * q_next
        self._q[s][action] += self.alpha * (td_target - self._q[s][action])
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def _act(self, iid: int, obs: dict[str, float]) -> int:
        if float(self._rng.random()) < self.epsilon:
            return int(self._rng.integers(0, N_ACTIONS))
        s = self._discretize(obs)
        return int(np.argmax(self._q[s]))

    def _discretize(self, obs: dict[str, float]) -> tuple[int, int, int]:
        def bucket(val: float) -> int:
            idx = int(val / self.QUEUE_MAX * self.QUEUE_BUCKETS)
            return min(idx, self.QUEUE_BUCKETS - 1)

        return (
            bucket(obs.get("queue_ns", 0.0)),
            bucket(obs.get("queue_ew", 0.0)),
            int(obs.get("current_phase", 0.0)) % 2,
        )

    def save(self, path: Path) -> None:
        import joblib
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"q": self._q, "epsilon": self.epsilon}, str(path))

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "QLearningController":
        import joblib
        ctrl = cls(**kwargs)
        state = joblib.load(str(path))
        ctrl._q = state["q"]
        ctrl.epsilon = state["epsilon"]
        return ctrl


# ---------------------------------------------------------------------------
# DQN Controller
# ---------------------------------------------------------------------------

class DQNController(BaseController):
    """PyTorch DQN with experience replay and target network.

    State vector: [queue_NS, queue_EW, avg_speed, lane_occupancy,
                   current_phase, hour_of_day_normalised]
    Actions: 0 (NS green) or 1 (EW green)
    """

    STATE_DIM: int = 6

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        gamma: float = 0.99,
        lr: float = 1e-3,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        buffer_size: int = 10_000,
        batch_size: int = 64,
        target_update_freq: int = 100,
        seed: int = 42,
    ) -> None:
        super().__init__(name="dqn")
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self._step_count = 0
        self._current_phase: dict[int, int] = {}

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)
        random.seed(seed)

        def make_net() -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = self.STATE_DIM
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.ReLU()]
                prev = h
            layers.append(nn.Linear(prev, N_ACTIONS))
            return nn.Sequential(*layers)

        self._online = make_net()
        self._target = make_net()
        self._target.load_state_dict(self._online.state_dict())
        self._target.eval()
        self._optim = torch.optim.Adam(self._online.parameters(), lr=lr)
        self._loss_fn = nn.SmoothL1Loss()
        self._buffer: Deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = (
            collections.deque(maxlen=buffer_size)
        )

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            a = self._act(_obs_to_vec(obs))
            actions[iid] = "NS" if a == 0 else "EW"
            self._current_phase[iid] = a
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        return self._act(_obs_to_vec(obs))

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        self._buffer.append((_obs_to_vec(obs), action, reward, _obs_to_vec(next_obs), done))
        if len(self._buffer) >= self.batch_size:
            self._learn()
        self._step_count += 1
        if self._step_count % self.target_update_freq == 0:
            self._target.load_state_dict(self._online.state_dict())

    def pretrain_from_demonstrations(
        self, states: np.ndarray, actions: np.ndarray, epochs: int = 1
    ) -> None:
        """Pre-train the online network on imitation data for 1 epoch."""
        import torch
        import torch.nn.functional as F
        X_t = torch.tensor(states, dtype=torch.float32)
        y_t = torch.tensor(actions, dtype=torch.long)
        self._online.train()
        for start in range(0, len(X_t), 256):
            end = start + 256
            logits = self._online(X_t[start:end])
            loss = F.cross_entropy(logits, y_t[start:end])
            self._optim.zero_grad()
            loss.backward()
            self._optim.step()
        self._target.load_state_dict(self._online.state_dict())

    def _act(self, state: np.ndarray) -> int:
        if random.random() < self.epsilon:
            return random.randint(0, N_ACTIONS - 1)
        import torch
        self._online.eval()
        with torch.no_grad():
            q = self._online(torch.tensor(state, dtype=torch.float32).unsqueeze(0))
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        return int(torch.argmax(q).item())

    def _learn(self) -> None:
        import torch
        batch = random.sample(list(self._buffer), self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        s = torch.tensor(np.stack(states), dtype=torch.float32)
        a = torch.tensor(actions, dtype=torch.long)
        r = torch.tensor(rewards, dtype=torch.float32)
        ns = torch.tensor(np.stack(next_states), dtype=torch.float32)
        d = torch.tensor(dones, dtype=torch.float32)

        self._online.train()
        q_vals = self._online(s).gather(1, a.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad():
            q_next = self._target(ns).max(1).values
        target = r + self.gamma * q_next * (1 - d)
        loss = self._loss_fn(q_vals, target)
        self._optim.zero_grad()
        loss.backward()
        self._optim.step()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"online": self._online.state_dict(), "target": self._target.state_dict(), "epsilon": self.epsilon},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "DQNController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._online.load_state_dict(state["online"])
        ctrl._target.load_state_dict(state["target"])
        ctrl.epsilon = state["epsilon"]
        return ctrl


# ---------------------------------------------------------------------------
# PPO Controller
# ---------------------------------------------------------------------------

class PPOController(BaseController):
    """PyTorch PPO with actor-critic architecture.

    Same state/action space as DQNController.
    Uses clipped surrogate objective with entropy bonus.
    """

    STATE_DIM: int = 6

    def __init__(
        self,
        hidden: tuple[int, ...] = (128, 64),
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        lr: float = 3e-4,
        update_epochs: int = 4,
        rollout_len: int = 128,
        entropy_coef: float = 0.01,
        seed: int = 42,
    ) -> None:
        super().__init__(name="ppo")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_eps = clip_eps
        self.update_epochs = update_epochs
        self.rollout_len = rollout_len
        self.entropy_coef = entropy_coef
        self._current_phase: dict[int, int] = {}

        import torch
        import torch.nn as nn
        torch.manual_seed(seed)

        def mlp(in_dim: int, out_dim: int) -> nn.Sequential:
            layers: list[nn.Module] = []
            prev = in_dim
            for h in hidden:
                layers += [nn.Linear(prev, h), nn.Tanh()]
                prev = h
            layers.append(nn.Linear(prev, out_dim))
            return nn.Sequential(*layers)

        self._actor = mlp(self.STATE_DIM, N_ACTIONS)
        self._critic = mlp(self.STATE_DIM, 1)
        params = list(self._actor.parameters()) + list(self._critic.parameters())
        self._optim = torch.optim.Adam(params, lr=lr)

        # Rollout buffer
        self._rollout: list[dict[str, Any]] = []
        self._last_obs: np.ndarray | None = None
        self._last_action: int = 0
        self._last_log_prob: float = 0.0

    def reset(self, n_intersections: int) -> None:
        super().reset(n_intersections)
        self._current_phase = {i: 0 for i in range(n_intersections)}
        self._rollout = []

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]:
        actions: dict[int, SignalPhase] = {}
        for iid, obs in observations.items():
            vec = _obs_to_vec(obs)
            a, log_prob = self._sample_action(vec)
            actions[iid] = "NS" if a == 0 else "EW"
            self._current_phase[iid] = a
        return actions

    def select_action(self, obs: dict[str, float]) -> int:
        vec = _obs_to_vec(obs)
        self._last_obs = vec
        a, log_prob = self._sample_action(vec)
        self._last_action = a
        self._last_log_prob = log_prob
        return a

    def update(
        self,
        obs: dict[str, float],
        action: int,
        reward: float,
        next_obs: dict[str, float],
        done: bool = False,
    ) -> None:
        vec = _obs_to_vec(obs)
        next_vec = _obs_to_vec(next_obs)
        _, log_prob = self._sample_action(vec)
        self._rollout.append(
            {
                "obs": vec,
                "action": action,
                "reward": reward,
                "next_obs": next_vec,
                "done": done,
                "log_prob": log_prob,
            }
        )
        if len(self._rollout) >= self.rollout_len:
            self._ppo_update()
            self._rollout = []

    def _sample_action(self, state: np.ndarray) -> tuple[int, float]:
        import torch
        import torch.nn.functional as F
        s = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        self._actor.eval()
        with torch.no_grad():
            logits = self._actor(s)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item())

    def _ppo_update(self) -> None:
        import torch
        import torch.nn.functional as F
        if not self._rollout:
            return
        obs_t = torch.tensor(np.stack([r["obs"] for r in self._rollout]), dtype=torch.float32)
        acts_t = torch.tensor([r["action"] for r in self._rollout], dtype=torch.long)
        rews_t = torch.tensor([r["reward"] for r in self._rollout], dtype=torch.float32)
        next_obs_t = torch.tensor(np.stack([r["next_obs"] for r in self._rollout]), dtype=torch.float32)
        dones_t = torch.tensor([r["done"] for r in self._rollout], dtype=torch.float32)
        old_log_probs_t = torch.tensor([r["log_prob"] for r in self._rollout], dtype=torch.float32)

        # Compute returns and advantages using GAE
        with torch.no_grad():
            values = self._critic(obs_t).squeeze(-1)
            next_values = self._critic(next_obs_t).squeeze(-1)
        advantages = torch.zeros_like(rews_t)
        gae = 0.0
        for t in reversed(range(len(self._rollout))):
            delta = rews_t[t] + self.gamma * next_values[t] * (1 - dones_t[t]) - values[t]
            gae = float(delta.item()) + self.gamma * self.gae_lambda * gae * (1 - float(dones_t[t].item()))
            advantages[t] = gae
        returns = advantages + values

        # Normalise advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.update_epochs):
            self._actor.train()
            self._critic.train()
            logits = self._actor(obs_t)
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            log_probs = dist.log_prob(acts_t)
            entropy = dist.entropy().mean()

            ratio = torch.exp(log_probs - old_log_probs_t)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages
            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            critic_vals = self._critic(obs_t).squeeze(-1)
            critic_loss = F.mse_loss(critic_vals, returns)

            loss = actor_loss + 0.5 * critic_loss
            self._optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._actor.parameters(), 0.5)
            torch.nn.utils.clip_grad_norm_(self._critic.parameters(), 0.5)
            self._optim.step()

    def save(self, path: Path) -> None:
        import torch
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"actor": self._actor.state_dict(), "critic": self._critic.state_dict()},
            str(path),
        )

    @classmethod
    def load(cls, path: Path, **kwargs: Any) -> "PPOController":
        import torch
        ctrl = cls(**kwargs)
        state = torch.load(str(path), map_location="cpu")
        ctrl._actor.load_state_dict(state["actor"])
        ctrl._critic.load_state_dict(state["critic"])
        return ctrl
