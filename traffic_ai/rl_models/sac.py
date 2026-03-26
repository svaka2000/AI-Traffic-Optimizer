"""traffic_ai/rl_models/sac.py

Soft Actor-Critic (discrete) training function.

Uses the SACController from rl_controllers.py, which implements:
  - Twin Q-networks with target networks (Polyak update)
  - Learnable entropy temperature α
  - Off-policy replay buffer (50k transitions)
"""
from __future__ import annotations

from traffic_ai.controllers.rl_controllers import SACController
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator


def train_sac(
    n_intersections: int = 4,
    demand_profile: str = "rush_hour",
    steps_per_episode: int = 500,
    n_episodes: int = 100,
    update_steps: int = 50_000,
    seed: int = 42,
    verbose: bool = False,
) -> SACController:
    """Train a SAC controller (discrete actions) on the traffic simulation.

    Parameters
    ----------
    n_intersections:
        Number of intersections in the simulation grid.
    demand_profile:
        Demand scenario name (see demand.py for valid profiles).
    steps_per_episode:
        Number of simulation steps per training episode.
    n_episodes:
        Total training episodes.
    update_steps:
        Total expected update steps (used for temperature annealing schedule).
    seed:
        Random seed for reproducibility.
    verbose:
        Print per-episode metrics if True.

    Returns
    -------
    SACController
        Trained controller ready for evaluation.
    """
    config = SimulatorConfig(
        steps=steps_per_episode,
        intersections=n_intersections,
        demand_profile=demand_profile,
        seed=seed,
    )
    sim = TrafficNetworkSimulator(config)
    controller = SACController(
        n_intersections=n_intersections,
        update_steps=update_steps,
    )

    for episode in range(n_episodes):
        result = sim.run(controller, steps=steps_per_episode)
        if verbose:
            avg_q = result.aggregate.get("average_queue_length", 0.0)
            avg_w = result.aggregate.get("average_wait_time", 0.0)
            thr = result.aggregate.get("average_throughput", 0.0)
            print(
                f"[SAC] Episode {episode + 1}/{n_episodes} | "
                f"avg_queue={avg_q:.2f} | avg_wait={avg_w:.2f}s | "
                f"throughput={thr:.3f}"
            )

    return controller
