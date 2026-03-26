"""traffic_ai/rl_models/a2c.py

Advantage Actor-Critic (A2C) training function.

Uses the A2CController from rl_controllers.py to train a policy on the
traffic simulation environment via GAE-lambda advantage estimation.
"""
from __future__ import annotations

from typing import Optional

from traffic_ai.controllers.rl_controllers import A2CController
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator


def train_a2c(
    n_intersections: int = 4,
    demand_profile: str = "rush_hour",
    steps_per_episode: int = 500,
    n_episodes: int = 80,
    update_steps: int = 40_000,
    seed: int = 42,
    verbose: bool = False,
) -> A2CController:
    """Train an A2C controller on the traffic simulation.

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
        Total expected steps used to schedule entropy/lr annealing.
    seed:
        Random seed for reproducibility.
    verbose:
        Print per-episode metrics if True.

    Returns
    -------
    A2CController
        Trained controller ready for evaluation.
    """
    config = SimulatorConfig(
        steps=steps_per_episode,
        intersections=n_intersections,
        demand_profile=demand_profile,
        seed=seed,
    )
    sim = TrafficNetworkSimulator(config)
    controller = A2CController(
        n_intersections=n_intersections,
        update_steps=update_steps,
    )

    total_rewards = []
    for episode in range(n_episodes):
        result = sim.run(controller, steps=steps_per_episode)
        ep_reward = -result.aggregate.get("average_queue_length", 0.0)
        total_rewards.append(ep_reward)
        if verbose:
            avg_q = result.aggregate.get("average_queue_length", 0.0)
            avg_w = result.aggregate.get("average_wait_time", 0.0)
            print(
                f"[A2C] Episode {episode + 1}/{n_episodes} | "
                f"avg_queue={avg_q:.2f} | avg_wait={avg_w:.2f}s | "
                f"reward={ep_reward:.3f}"
            )

    return controller
