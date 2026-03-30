"""traffic_ai/rl_models/environment.py

Single-intersection MDP for RL pretraining, built on the canonical
TrafficNetworkSimulator.

4-Phase Action Space (AITO Phase 3)
-------------------------------------
Action ∈ {0, 1, 2, 3}:
    0 = NS_THROUGH  (northbound + southbound through traffic)
    1 = EW_THROUGH  (eastbound + westbound through traffic)
    2 = NS_LEFT     (protected NS left turns — HCM 7th ed.)
    3 = EW_LEFT     (protected EW left turns — HCM 7th ed.)

Observation space (8 floats, STATE_DIM=8)
------------------------------------------
    [phase_elapsed_norm, current_phase_idx_norm,
     queue_ns_through_norm, queue_ew_through_norm,
     queue_ns_left_norm, queue_ew_left_norm,
     time_of_day_norm, upstream_queue_norm]

Multi-Objective Reward (AITO Phase 4)
---------------------------------------
Six components with configurable weights (default_config.yaml: rl.reward_weights):

    reward = w1 * (-avg_delay_component)
           + w2 * (-pedestrian_wait_component)
           + w3 * (-emissions_co2_component)
           + w4 * (-switch_penalty_component)
           + w5 * (+throughput_component)
           + w6 * (-left_starvation_component)

Default weights sourced from:
    Mannion et al. (2016) "An Experimental Review of Reinforcement Learning
    Algorithms for Adaptive Traffic Signal Control" — AAMAS workshop.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
from traffic_ai.simulation_engine.types import IDX_TO_PHASE, SignalPhase


# 4-phase model constants — in sync with rl_controllers.py
N_PHASES = 4
N_DURATIONS = 1     # duration selection removed; phase-only action space
N_ACTIONS = N_PHASES  # 4


@dataclass(slots=True)
class RewardWeights:
    """Configurable weights for the 6-component multi-objective reward.

    Sources
    -------
    - avg_delay     : primary minimisation target (Mannion et al., 2016)
    - ped_wait      : fairness proxy (pedestrian/minor-road wait time)
    - emissions_co2 : EPA MOVES2014b proxy — total_queue × stop_factor
    - switch_penalty: phase-change cost (HCM 7th ed. — yellow + all-red = 4 s)
    - throughput    : throughput reward (Liang et al., 2019, ITSC)
    - left_starve   : left-turn starvation penalty (Mannion et al., 2016)
    """
    avg_delay: float = 0.12       # weight on negative normalized queue
    ped_wait: float = 0.05        # weight on queue imbalance (fairness)
    emissions_co2: float = 0.03   # weight on CO2 proxy
    switch_penalty: float = 0.1   # flat penalty per phase change; scaled to match other dimensionless weights
    throughput: float = 0.08      # weight on normalized throughput
    left_starve: float = 0.0      # disabled: starvation penalty causes left-turn over-service with long training


@dataclass(slots=True)
class EnvConfig:
    max_queue: float = 120.0
    step_limit: int = 220
    seed: int = 42
    demand_profile: str = "rush_hour"
    reward_weights: RewardWeights = field(default_factory=RewardWeights)


class SignalControlEnv:
    """Single-intersection MDP wrapping the canonical simulation engine.

    Uses ``TrafficNetworkSimulator`` with one intersection so all physics
    (HCM signal timing, EPA emissions, demand profiles) are identical to
    the benchmarking engine used by the experiment runner.

    Action encoding
    ---------------
    ``action`` ∈ [0, N_ACTIONS):
        0 → NS_THROUGH, 1 → EW_THROUGH, 2 → NS_LEFT, 3 → EW_LEFT
    """

    def __init__(self, config: EnvConfig | None = None) -> None:
        if config is None:
            config = EnvConfig()
        self.config = config

        engine_cfg = SimulatorConfig(
            steps=config.step_limit,
            intersections=1,
            lanes_per_direction=2,
            step_seconds=1.0,
            max_queue_per_lane=int(config.max_queue // 2),
            demand_profile=config.demand_profile,
            demand_scale=1.0,
            seed=config.seed,
        )
        self._engine = TrafficNetworkSimulator(engine_cfg)
        self.step_idx: int = 0
        self._current_phase: int = 0   # 0-3
        self._last_obs: dict[str, float] = {}
        self._step_since_ns_left: int = 0
        self._step_since_ew_left: int = 0

    @property
    def observation_dim(self) -> int:
        return 8  # STATE_DIM

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def reset(self) -> np.ndarray:
        raw = self._engine.reset_env()
        self.step_idx = 0
        obs0 = raw.get(0, {})
        self._current_phase = 0
        self._last_obs = obs0
        self._step_since_ns_left = 0
        self._step_since_ew_left = 0
        return self._encode_obs(obs0)

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, dict[str, float]]:
        """Advance one simulation step.

        Parameters
        ----------
        action:
            Integer in [0, N_ACTIONS) mapping to a 4-phase signal.
        """
        self.step_idx += 1
        w = self.config.reward_weights

        new_phase = int(action) % N_PHASES
        phase_str: SignalPhase = IDX_TO_PHASE[new_phase]

        # Track left-turn starvation counters
        if new_phase == 2:
            self._step_since_ns_left = 0
        else:
            self._step_since_ns_left += 1
        if new_phase == 3:
            self._step_since_ew_left = 0
        else:
            self._step_since_ew_left += 1

        # Phase-change switch cost
        switch_cost = w.switch_penalty if new_phase != self._current_phase else 0.0
        self._current_phase = new_phase

        # Apply to engine
        engine_actions = {0: phase_str}
        raw_obs, _, engine_done, _ = self._engine.step_env(engine_actions)

        obs = raw_obs.get(0, {})
        self._last_obs = obs

        queue_ns = obs.get("queue_ns", 0.0)
        queue_ew = obs.get("queue_ew", 0.0)
        queue_ns_left = obs.get("queue_ns_left", 0.0)
        queue_ew_left = obs.get("queue_ew_left", 0.0)
        total_queue = queue_ns + queue_ew + queue_ns_left + queue_ew_left
        max_q = max(self.config.max_queue, 1.0)

        # Component 1: avg_delay — normalized queue
        avg_delay = total_queue / max_q

        # Component 2: pedestrian_wait — queue imbalance between NS and EW axes
        ped_wait = abs(queue_ns - queue_ew) / max_q

        # Component 3: emissions_co2 — stopped-vehicle proxy (EPA MOVES2014b:
        #   idling produces ~431 g CO2/hr per vehicle; normalized to [0,1] here)
        emissions_co2 = total_queue / max_q  # same proxy as avg_delay (simplified)

        # Component 4: throughput — departures as a positive signal
        throughput = obs.get("departures", 0.0) / max(obs.get("arrivals", 1.0), 1.0)

        # Component 5: left_starvation — penalise long waits without left service
        STARVATION_THRESHOLD = 60  # steps (~60 s) before penalty kicks in
        left_starve = 0.0
        if self._step_since_ns_left > STARVATION_THRESHOLD and queue_ns_left > 0:
            left_starve += (self._step_since_ns_left - STARVATION_THRESHOLD) / 60.0
        if self._step_since_ew_left > STARVATION_THRESHOLD and queue_ew_left > 0:
            left_starve += (self._step_since_ew_left - STARVATION_THRESHOLD) / 60.0

        reward = (
            -w.avg_delay * avg_delay
            - w.ped_wait * ped_wait
            - w.emissions_co2 * emissions_co2
            - switch_cost
            + w.throughput * throughput
            - w.left_starve * left_starve
        )

        done = self.step_idx >= self.config.step_limit or engine_done
        info = {
            "queue_ns": queue_ns,
            "queue_ew": queue_ew,
            "queue_ns_left": queue_ns_left,
            "queue_ew_left": queue_ew_left,
            "total_queue": total_queue,
            "phase": float(new_phase),
            "throughput": throughput,
            "left_starve": left_starve,
        }
        return self._encode_obs(obs), float(reward), done, info

    def _encode_obs(self, obs: dict[str, float]) -> np.ndarray:
        """Encode engine observation dict to 8-float RL state vector (STATE_DIM=8).

        Features
        --------
        0  phase_elapsed_norm      : phase_elapsed / 60
        1  current_phase_idx_norm  : current_phase_idx / 3
        2  queue_ns_through_norm   : queue_ns / max_queue
        3  queue_ew_through_norm   : queue_ew / max_queue
        4  queue_ns_left_norm      : queue_ns_left / 120
        5  queue_ew_left_norm      : queue_ew_left / 120
        6  time_of_day_norm        : time_of_day in [0, 1]
        7  upstream_queue_norm     : upstream_queue / max_queue
        """
        max_q = max(self.config.max_queue, 1.0)
        return np.array(
            [
                obs.get("phase_elapsed", 0.0) / 60.0,
                float(self._current_phase) / 3.0,
                obs.get("queue_ns", 0.0) / max_q,
                obs.get("queue_ew", 0.0) / max_q,
                obs.get("queue_ns_left", 0.0) / 120.0,
                obs.get("queue_ew_left", 0.0) / 120.0,
                obs.get("time_of_day_normalized", 0.0),
                obs.get("upstream_queue", 0.0) / max_q,
            ],
            dtype=np.float32,
        )


class MultiIntersectionSignalEnv:
    """N-intersection Independent DQN (IDQN) training environment.

    Wraps ``TrafficNetworkSimulator`` with ``n_intersections`` nodes (default 4,
    arranged as a 2×2 grid).  At each step every intersection receives an
    independent action from the shared policy; all experiences are pushed to
    a shared replay buffer so the agent learns across the whole network.

    Observation space
    -----------------
    Same 8-float feature vector as ``SignalControlEnv`` (STATE_DIM=8), with
    feature[7] = upstream_queue / max_queue where upstream_queue is the mean
    total queue of the intersection's immediate neighbours.  This is non-zero
    in a multi-intersection network and allows the agent to learn basic
    network-coordination behaviour.

    Reward
    ------
    Local reward per intersection:
        r_i = -(local_queue / max_queue) - switch_penalty * phase_changed
    """

    def __init__(
        self,
        config: EnvConfig | None = None,
        n_intersections: int = 4,
    ) -> None:
        if config is None:
            config = EnvConfig()
        self.config = config
        self.n_intersections = n_intersections

        engine_cfg = SimulatorConfig(
            steps=config.step_limit,
            intersections=n_intersections,
            lanes_per_direction=2,
            step_seconds=1.0,
            max_queue_per_lane=int(config.max_queue // 2),
            demand_profile=config.demand_profile,
            demand_scale=1.0,
            seed=config.seed,
        )
        self._engine = TrafficNetworkSimulator(engine_cfg)
        self.step_idx: int = 0
        self._current_phases: dict[int, int] = {i: 0 for i in range(n_intersections)}

    @property
    def observation_dim(self) -> int:
        return 8  # STATE_DIM — identical to SignalControlEnv

    @property
    def n_actions(self) -> int:
        return N_ACTIONS

    def reset(self) -> dict[int, np.ndarray]:
        """Reset environment; returns per-intersection encoded observations."""
        raw = self._engine.reset_env()
        self.step_idx = 0
        self._current_phases = {i: 0 for i in range(self.n_intersections)}
        return {i: self._encode_obs(raw.get(i, {}), i) for i in range(self.n_intersections)}

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, np.ndarray], dict[int, float], bool, dict]:
        """Advance one step with per-intersection actions.

        Returns
        -------
        (obs_dict, reward_dict, done, info)
        """
        self.step_idx += 1
        w = self.config.reward_weights

        engine_actions: dict[int, SignalPhase] = {}
        switch_costs: dict[int, float] = {}
        for i, action in actions.items():
            new_phase = int(action) % N_PHASES
            switch_costs[i] = w.switch_penalty if new_phase != self._current_phases[i] else 0.0
            self._current_phases[i] = new_phase
            engine_actions[i] = IDX_TO_PHASE[new_phase]

        raw_obs, _, engine_done, _ = self._engine.step_env(engine_actions)
        done = self.step_idx >= self.config.step_limit or engine_done

        obs_dict = {i: self._encode_obs(raw_obs.get(i, {}), i) for i in range(self.n_intersections)}

        max_q = max(self.config.max_queue, 1.0)
        reward_dict: dict[int, float] = {}
        for i in range(self.n_intersections):
            obs_i = raw_obs.get(i, {})
            local_queue = obs_i.get("total_queue", 0.0)
            throughput = obs_i.get("departures", 0.0) / max(obs_i.get("arrivals", 1.0), 1.0)
            reward_dict[i] = float(
                -w.avg_delay * local_queue / max_q
                - switch_costs.get(i, 0.0)
                + w.throughput * throughput
            )

        return obs_dict, reward_dict, done, {}

    def _encode_obs(self, obs: dict[str, float], iid: int) -> np.ndarray:
        """Same 8-float encoding as SignalControlEnv; feature[7] uses actual upstream_queue."""
        max_q = max(self.config.max_queue, 1.0)
        return np.array(
            [
                obs.get("phase_elapsed", 0.0) / 60.0,
                float(self._current_phases.get(iid, 0)) / 3.0,
                obs.get("queue_ns_through", obs.get("queue_ns", 0.0)) / max_q,
                obs.get("queue_ew_through", obs.get("queue_ew", 0.0)) / max_q,
                obs.get("queue_ns_left", 0.0) / 120.0,
                obs.get("queue_ew_left", 0.0) / 120.0,
                obs.get("time_of_day_normalized", 0.0),
                obs.get("upstream_queue", 0.0) / max_q,  # mean of neighbour queues
            ],
            dtype=np.float32,
        )
