"""traffic_ai/controllers/maddpg_controller.py

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) controller.

KEY INNOVATION: Each intersection is an independent agent with its own actor
network. However, during training the critic sees the observations AND actions
of ALL neighboring intersections (centralized training, decentralized execution).

Architecture
------------
Actor  : local_obs (augmented with neighbor queues) → 128 → 64 → 2 logits
         Discrete actions via Gumbel-Softmax during training; argmax at inference.
Critic : concat([all_obs, all_actions]) → 256 → 128 → 1  (Q-value)

The neighbor observation extension adds queue_ns and queue_ew for each
neighbor (up to 4 directions). Missing neighbors are zero-padded.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from traffic_ai.simulation_engine.types import SignalPhase


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------
OBS_DIM = 12          # base observation keys from IntersectionState.as_observation()
NEIGHBOR_FEATS = 2    # queue_ns, queue_ew per neighbor
MAX_NEIGHBORS = 4     # N, S, E, W
ACTOR_INPUT_DIM = OBS_DIM + MAX_NEIGHBORS * NEIGHBOR_FEATS  # 20
N_ACTIONS = 2         # NS or EW
CRITIC_HIDDEN = 256
ACTOR_HIDDEN = 128
REPLAY_CAPACITY = 50_000
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.01            # soft target update
ACTOR_LR = 1e-4
CRITIC_LR = 3e-4
GUMBEL_TEMP = 1.0
GUMBEL_TEMP_MIN = 0.3
GUMBEL_ANNEAL = 0.9995
UPDATE_EVERY = 4
WARMUP_STEPS = 2_000


# ---------------------------------------------------------------------------
# Network modules
# ---------------------------------------------------------------------------

class _Actor(nn.Module):
    def __init__(self, input_dim: int = ACTOR_INPUT_DIM) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, ACTOR_HIDDEN),
            nn.LayerNorm(ACTOR_HIDDEN),
            nn.ReLU(),
            nn.Linear(ACTOR_HIDDEN, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)  # logits

    def gumbel_action(self, obs: torch.Tensor, tau: float = GUMBEL_TEMP) -> torch.Tensor:
        logits = self.forward(obs)
        return F.gumbel_softmax(logits, tau=tau, hard=False)

    def greedy_action(self, obs: torch.Tensor) -> int:
        with torch.no_grad():
            logits = self.forward(obs)
            return int(logits.argmax(dim=-1).item())


class _Critic(nn.Module):
    """Centralized critic. Input = concat of all agent obs + all agent actions."""

    def __init__(self, n_agents: int) -> None:
        super().__init__()
        input_dim = n_agents * ACTOR_INPUT_DIM + n_agents * N_ACTIONS
        self.net = nn.Sequential(
            nn.Linear(input_dim, CRITIC_HIDDEN),
            nn.LayerNorm(CRITIC_HIDDEN),
            nn.ReLU(),
            nn.Linear(CRITIC_HIDDEN, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


# ---------------------------------------------------------------------------
# Simple uniform-replay buffer for MADDPG
# ---------------------------------------------------------------------------

class _MADDPGReplayBuffer:
    """Stores joint (obs, actions, rewards, next_obs, dones) tuples."""

    def __init__(self, capacity: int, n_agents: int) -> None:
        self._cap = capacity
        self._n = n_agents
        self._obs: List[np.ndarray] = []
        self._actions: List[np.ndarray] = []
        self._rewards: List[np.ndarray] = []
        self._next_obs: List[np.ndarray] = []
        self._dones: List[np.ndarray] = []
        self._ptr = 0

    def push(
        self,
        obs: np.ndarray,        # (n_agents, obs_dim)
        actions: np.ndarray,    # (n_agents,) int
        rewards: np.ndarray,    # (n_agents,)
        next_obs: np.ndarray,   # (n_agents, obs_dim)
        dones: np.ndarray,      # (n_agents,) bool
    ) -> None:
        if len(self._obs) < self._cap:
            self._obs.append(obs)
            self._actions.append(actions)
            self._rewards.append(rewards)
            self._next_obs.append(next_obs)
            self._dones.append(dones)
        else:
            self._obs[self._ptr] = obs
            self._actions[self._ptr] = actions
            self._rewards[self._ptr] = rewards
            self._next_obs[self._ptr] = next_obs
            self._dones[self._ptr] = dones
        self._ptr = (self._ptr + 1) % self._cap

    def sample(self, batch_size: int, rng: np.random.Generator) -> Tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        idx = rng.integers(0, len(self._obs), size=batch_size)
        obs = np.stack([self._obs[i] for i in idx])           # (B, n, obs_dim)
        actions = np.stack([self._actions[i] for i in idx])   # (B, n)
        rewards = np.stack([self._rewards[i] for i in idx])   # (B, n)
        next_obs = np.stack([self._next_obs[i] for i in idx]) # (B, n, obs_dim)
        dones = np.stack([self._dones[i] for i in idx])       # (B, n)
        return obs, actions, rewards, next_obs, dones

    def __len__(self) -> int:
        return len(self._obs)


# ---------------------------------------------------------------------------
# MADDPGController
# ---------------------------------------------------------------------------

class MADDPGController:
    """Multi-agent controller. One actor per intersection, shared centralized critic."""

    name = "MADDPG"

    def __init__(
        self,
        n_intersections: int = 4,
        neighbors: Optional[Dict[int, Dict[str, Optional[int]]]] = None,
        update_steps: int = 10_000,
        device: str = "cpu",
    ) -> None:
        self.n = n_intersections
        self._neighbors: Dict[int, Dict[str, Optional[int]]] = neighbors or {}
        self.update_steps = update_steps
        self.device = torch.device(device)

        self._actors: List[_Actor] = []
        self._target_actors: List[_Actor] = []
        self._critics: List[_Critic] = []
        self._target_critics: List[_Critic] = []
        self._actor_opts: List[optim.Adam] = []
        self._critic_opts: List[optim.Adam] = []

        self._replay = _MADDPGReplayBuffer(REPLAY_CAPACITY, n_intersections)
        self._rng = np.random.default_rng(42)
        self._step = 0
        self._gumbel_temp = GUMBEL_TEMP
        self._prev_obs: Optional[np.ndarray] = None
        self._prev_actions: Optional[np.ndarray] = None

        self._initialized = False

    def reset(self, n_intersections: int) -> None:
        self.n = n_intersections
        self._build_networks()
        self._replay = _MADDPGReplayBuffer(REPLAY_CAPACITY, n_intersections)
        self._step = 0
        self._gumbel_temp = GUMBEL_TEMP
        self._prev_obs = None
        self._prev_actions = None
        self._initialized = True

    def _build_networks(self) -> None:
        self._actors = []
        self._target_actors = []
        self._critics = []
        self._target_critics = []
        self._actor_opts = []
        self._critic_opts = []

        for _ in range(self.n):
            actor = _Actor().to(self.device)
            target_actor = _Actor().to(self.device)
            target_actor.load_state_dict(actor.state_dict())

            critic = _Critic(self.n).to(self.device)
            target_critic = _Critic(self.n).to(self.device)
            target_critic.load_state_dict(critic.state_dict())

            self._actors.append(actor)
            self._target_actors.append(target_actor)
            self._critics.append(critic)
            self._target_critics.append(target_critic)
            self._actor_opts.append(optim.Adam(actor.parameters(), lr=ACTOR_LR))
            self._critic_opts.append(optim.Adam(critic.parameters(), lr=CRITIC_LR))

    def _augment_obs(
        self, raw_obs: Dict[int, Dict[str, float]]
    ) -> np.ndarray:
        """Build (n_agents, ACTOR_INPUT_DIM) observation array.

        For each agent, base obs (12 features) + neighbor queues (2 * 4 features).
        """
        # Order keys to ensure consistent 12-dim vector
        BASE_KEYS = [
            "intersection_id", "sim_step", "queue_ns", "queue_ew", "total_queue",
            "phase_ns", "phase_ew", "phase_elapsed", "arrivals", "departures",
            "wait_sec", "emergency_active",
        ]
        agents_obs = np.zeros((self.n, ACTOR_INPUT_DIM), dtype=np.float32)
        for iid in range(self.n):
            obs = raw_obs.get(iid, {})
            base = np.array([obs.get(k, 0.0) for k in BASE_KEYS], dtype=np.float32)
            # Normalize a few features for numerical stability
            base[4] /= 100.0   # total_queue
            base[7] /= 100.0   # phase_elapsed
            base[8] /= 1000.0  # arrivals
            base[9] /= 1000.0  # departures
            base[10] /= 1e5    # wait_sec

            # Neighbor features: up to 4 (N, S, E, W)
            neighbor_feats = np.zeros(MAX_NEIGHBORS * NEIGHBOR_FEATS, dtype=np.float32)
            dirs = ["N", "S", "E", "W"]
            neighbor_map = self._neighbors.get(iid, {})
            for slot, d in enumerate(dirs):
                nid = neighbor_map.get(d)
                if nid is not None and nid in raw_obs:
                    nobs = raw_obs[nid]
                    neighbor_feats[slot * 2] = nobs.get("queue_ns", 0.0) / 100.0
                    neighbor_feats[slot * 2 + 1] = nobs.get("queue_ew", 0.0) / 100.0

            agents_obs[iid] = np.concatenate([base, neighbor_feats])
        return agents_obs

    def _reward_from_obs(self, obs: Dict[int, Dict[str, float]]) -> np.ndarray:
        """Simple local reward: negative of normalized total_queue per agent."""
        rewards = np.zeros(self.n, dtype=np.float32)
        for iid in range(self.n):
            q = obs.get(iid, {}).get("total_queue", 0.0)
            rewards[iid] = -q / 100.0
        return rewards

    def compute_actions(
        self, observations: Dict[int, Dict[str, float]], step: int
    ) -> Dict[int, SignalPhase]:
        if not self._initialized:
            self.reset(len(observations))

        current_obs = self._augment_obs(observations)

        # Store transition from previous step
        if self._prev_obs is not None and self._prev_actions is not None:
            rewards = self._reward_from_obs(observations)
            dones = np.zeros(self.n, dtype=bool)
            self._replay.push(
                self._prev_obs,
                self._prev_actions,
                rewards,
                current_obs,
                dones,
            )

        # Choose actions
        actions_int = np.zeros(self.n, dtype=np.int64)
        for iid in range(self.n):
            obs_t = torch.tensor(current_obs[iid], dtype=torch.float32, device=self.device).unsqueeze(0)
            if self._step < WARMUP_STEPS:
                actions_int[iid] = int(self._rng.integers(0, N_ACTIONS))
            else:
                actions_int[iid] = self._actors[iid].greedy_action(obs_t)

        self._prev_obs = current_obs
        self._prev_actions = actions_int

        # Update networks
        if (
            len(self._replay) >= BATCH_SIZE
            and self._step >= WARMUP_STEPS
            and self._step % UPDATE_EVERY == 0
        ):
            self._update()

        self._step += 1
        self._gumbel_temp = max(GUMBEL_TEMP_MIN, self._gumbel_temp * GUMBEL_ANNEAL)

        phase_map: Dict[int, SignalPhase] = {
            iid: ("NS" if actions_int[iid] == 0 else "EW")
            for iid in range(self.n)
        }
        return phase_map

    def _update(self) -> None:
        obs_b, act_b, rew_b, next_obs_b, done_b = self._replay.sample(BATCH_SIZE, self._rng)
        # obs_b: (B, n, obs_dim),  act_b: (B, n) int

        obs_t = torch.tensor(obs_b, dtype=torch.float32, device=self.device)        # (B,n,d)
        next_obs_t = torch.tensor(next_obs_b, dtype=torch.float32, device=self.device)
        act_t = torch.tensor(act_b, dtype=torch.long, device=self.device)            # (B,n)
        rew_t = torch.tensor(rew_b, dtype=torch.float32, device=self.device)         # (B,n)
        done_t = torch.tensor(done_b, dtype=torch.float32, device=self.device)       # (B,n)

        # One-hot encode current actions for critic input
        act_onehot = F.one_hot(act_t, num_classes=N_ACTIONS).float()                 # (B,n,2)
        all_obs_flat = obs_t.reshape(obs_t.shape[0], -1)                             # (B, n*d)
        all_act_flat = act_onehot.reshape(act_onehot.shape[0], -1)                   # (B, n*2)

        # Target actions (next step)
        with torch.no_grad():
            next_act_onehot_list = []
            for i in range(self.n):
                logits = self._target_actors[i](next_obs_t[:, i, :])
                next_act_onehot_list.append(F.softmax(logits, dim=-1))
            next_act_onehot = torch.stack(next_act_onehot_list, dim=1)              # (B,n,2)
            next_all_obs_flat = next_obs_t.reshape(next_obs_t.shape[0], -1)
            next_all_act_flat = next_act_onehot.reshape(next_act_onehot.shape[0], -1)

        for i in range(self.n):
            # --- Critic update ---
            with torch.no_grad():
                target_q = self._target_critics[i](next_all_obs_flat, next_all_act_flat)  # (B,1)
                y = rew_t[:, i:i+1] + GAMMA * (1.0 - done_t[:, i:i+1]) * target_q

            current_q = self._critics[i](all_obs_flat, all_act_flat)                 # (B,1)
            critic_loss = F.mse_loss(current_q, y)

            self._critic_opts[i].zero_grad()
            critic_loss.backward()
            nn.utils.clip_grad_norm_(self._critics[i].parameters(), max_norm=1.0)
            self._critic_opts[i].step()

            # --- Actor update (policy gradient via Gumbel-Softmax) ---
            gumbel_acts = []
            for j in range(self.n):
                if j == i:
                    act_soft = self._actors[i].gumbel_action(obs_t[:, i, :], tau=self._gumbel_temp)
                else:
                    with torch.no_grad():
                        act_soft = F.softmax(self._actors[j](obs_t[:, j, :]), dim=-1)
                gumbel_acts.append(act_soft)
            gumbel_all = torch.stack(gumbel_acts, dim=1).reshape(obs_t.shape[0], -1)
            actor_loss = -self._critics[i](all_obs_flat, gumbel_all).mean()

            self._actor_opts[i].zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self._actors[i].parameters(), max_norm=1.0)
            self._actor_opts[i].step()

            # --- Soft target update ---
            self._soft_update(self._actors[i], self._target_actors[i])
            self._soft_update(self._critics[i], self._target_critics[i])

    @staticmethod
    def _soft_update(online: nn.Module, target: nn.Module) -> None:
        for tp, op in zip(target.parameters(), online.parameters()):
            tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)
