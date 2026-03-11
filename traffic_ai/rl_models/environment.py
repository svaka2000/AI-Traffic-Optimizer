from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class EnvConfig:
    max_queue: float = 120.0
    arrival_rate_ns: float = 1.7
    arrival_rate_ew: float = 1.4
    service_rate_green: float = 2.4
    switch_penalty: float = 2.0
    step_limit: int = 220
    seed: int = 42


class SignalControlEnv:
    """Single-intersection MDP for RL pretraining."""

    def __init__(self, config: EnvConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.state = np.zeros(5, dtype=np.float32)
        self.step_idx = 0
        self.current_phase = 0

    def reset(self) -> np.ndarray:
        queue_ns = float(self.rng.uniform(2, 18))
        queue_ew = float(self.rng.uniform(2, 18))
        self.current_phase = int(self.rng.integers(0, 2))
        self.step_idx = 0
        self.state = np.array(
            [queue_ns, queue_ew, queue_ns + queue_ew, 0.0, float(self.current_phase)],
            dtype=np.float32,
        )
        return self.state.copy()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        self.step_idx += 1
        queue_ns, queue_ew, _, phase_elapsed, current_phase = self.state
        switch_cost = 0.0
        if int(action) != int(current_phase):
            switch_cost = self.config.switch_penalty
            phase_elapsed = 0.0
            current_phase = float(action)
        else:
            phase_elapsed += 1.0

        arrivals_ns = float(self.rng.poisson(self.config.arrival_rate_ns))
        arrivals_ew = float(self.rng.poisson(self.config.arrival_rate_ew))
        queue_ns = min(self.config.max_queue, queue_ns + arrivals_ns)
        queue_ew = min(self.config.max_queue, queue_ew + arrivals_ew)

        if int(action) == 0:
            departed = min(queue_ns, float(self.rng.poisson(self.config.service_rate_green)))
            queue_ns -= departed
        else:
            departed = min(queue_ew, float(self.rng.poisson(self.config.service_rate_green)))
            queue_ew -= departed

        total_queue = queue_ns + queue_ew
        reward = -0.12 * total_queue - 0.05 * abs(queue_ns - queue_ew) - switch_cost
        done = self.step_idx >= self.config.step_limit
        self.state = np.array(
            [queue_ns, queue_ew, total_queue, phase_elapsed, current_phase],
            dtype=np.float32,
        )
        info = {
            "queue_ns": queue_ns,
            "queue_ew": queue_ew,
            "total_queue": total_queue,
            "departed": departed,
        }
        return self.state.copy(), float(reward), done, info

