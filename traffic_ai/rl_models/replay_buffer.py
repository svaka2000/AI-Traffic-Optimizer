"""traffic_ai/rl_models/replay_buffer.py

Prioritized Experience Replay (PER) buffer for DQN training.

Unlike a uniform replay buffer, PER samples transitions proportional to their
TD-error magnitude, focusing training on surprising or high-error experiences.
Importance-sampling weights correct the bias introduced by non-uniform sampling.

References
----------
Schaul et al. (2016) "Prioritized Experience Replay", ICLR 2016.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


@dataclass(slots=True)
class Transition:
    """Single stored experience."""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class PrioritizedReplayBuffer:
    """Proportional-priority replay buffer with importance-sampling correction.

    Transitions are stored in a circular buffer. Each transition has a priority
    p_i = |TD_error| + epsilon. Sampling probability is p_i^alpha / sum(p_j^alpha).
    IS weights = (N * P(i))^(-beta) are returned to correct the gradient bias.

    Parameters
    ----------
    capacity:
        Maximum number of transitions to store.
    alpha:
        Priority exponent (0 = uniform, 1 = full prioritization).
    beta_start:
        Initial IS-weight exponent (annealed toward 1.0 over training).
    beta_frames:
        Number of frames over which beta anneals from beta_start to 1.0.
    epsilon:
        Small constant added to |TD error| to prevent zero priority.
    """

    def __init__(
        self,
        capacity: int = 30_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 100_000,
        epsilon: float = 0.01,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.epsilon = epsilon

        self._buffer: List[Transition] = []
        self._priorities: np.ndarray = np.zeros(capacity, dtype=np.float64)
        self._pos: int = 0
        self._frame: int = 0
        self._max_priority: float = 1.0

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a new transition with maximum current priority."""
        t = Transition(state=state.copy(), action=action, reward=reward,
                       next_state=next_state.copy(), done=done)
        if len(self._buffer) < self.capacity:
            self._buffer.append(t)
        else:
            self._buffer[self._pos] = t
        self._priorities[self._pos] = self._max_priority
        self._pos = (self._pos + 1) % self.capacity

    def sample(
        self, batch_size: int, rng: np.random.Generator
    ) -> Tuple[List[Transition], np.ndarray, np.ndarray]:
        """Sample a batch proportional to priority.

        Returns
        -------
        transitions:
            List of sampled Transition objects.
        indices:
            Indices into the buffer for priority updates.
        weights:
            Importance-sampling weights (shape: [batch_size]).
        """
        n = len(self._buffer)
        priorities = self._priorities[:n] ** self.alpha
        probs = priorities / priorities.sum()

        indices = rng.choice(n, size=batch_size, replace=False, p=probs)
        transitions = [self._buffer[i] for i in indices]

        beta = self._current_beta()
        weights = (n * probs[indices]) ** (-beta)
        weights = (weights / weights.max()).astype(np.float32)

        self._frame += 1
        return transitions, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        """Update priorities after computing new TD errors.

        Parameters
        ----------
        indices:
            Indices returned by the last ``sample()`` call.
        td_errors:
            Absolute TD errors for each transition.
        """
        new_priorities = (np.abs(td_errors) + self.epsilon).astype(np.float64)
        self._priorities[indices] = new_priorities
        self._max_priority = max(float(self._priorities[:len(self._buffer)].max()), self._max_priority)

    def __len__(self) -> int:
        return len(self._buffer)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _current_beta(self) -> float:
        """Linearly anneal beta from beta_start toward 1.0."""
        fraction = min(self._frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)
