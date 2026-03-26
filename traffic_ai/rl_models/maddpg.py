"""traffic_ai/rl_models/maddpg.py

Multi-Agent Deep Deterministic Policy Gradient (MADDPG) training function.

Uses the MADDPGController from maddpg_controller.py.

Key distinction from single-agent methods:
  - Passes `sim.neighbors` to the controller so actors can build neighbor-
    augmented observations during training.
  - Uses centralized critics that see all agents' observations and actions.
"""
from __future__ import annotations

from traffic_ai.controllers.maddpg_controller import MADDPGController
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator


def train_maddpg(
    n_intersections: int = 4,
    demand_profile: str = "rush_hour",
    steps_per_episode: int = 500,
    n_episodes: int = 120,
    update_steps: int = 60_000,
    seed: int = 42,
    verbose: bool = False,
) -> MADDPGController:
    """Train a MADDPG controller on the traffic simulation.

    Parameters
    ----------
    n_intersections:
        Number of intersections in the simulation grid.
    demand_profile:
        Demand scenario name (see demand.py for valid profiles).
    steps_per_episode:
        Number of simulation steps per training episode.
    n_episodes:
        Total training episodes. MADDPG needs more episodes than single-agent
        methods due to the larger joint observation/action space.
    update_steps:
        Expected total update steps (used for Gumbel-Softmax temperature schedule).
    seed:
        Random seed for reproducibility.
    verbose:
        Print per-episode metrics if True.

    Returns
    -------
    MADDPGController
        Trained controller ready for evaluation.
    """
    config = SimulatorConfig(
        steps=steps_per_episode,
        intersections=n_intersections,
        demand_profile=demand_profile,
        seed=seed,
    )
    sim = TrafficNetworkSimulator(config)

    # Pass the neighbor topology so each actor can augment its observation
    controller = MADDPGController(
        n_intersections=n_intersections,
        neighbors=sim.neighbors,
        update_steps=update_steps,
    )

    best_avg_queue = float("inf")
    best_state: dict | None = None

    for episode in range(n_episodes):
        result = sim.run(controller, steps=steps_per_episode)
        avg_q = result.aggregate.get("average_queue_length", 0.0)

        # Track best model weights per agent
        if avg_q < best_avg_queue:
            best_avg_queue = avg_q
            best_state = [
                {
                    "actor": actor.state_dict(),
                    "critic": critic.state_dict(),
                }
                for actor, critic in zip(controller._actors, controller._critics)
            ]

        if verbose:
            avg_w = result.aggregate.get("average_wait_time", 0.0)
            temp = controller._gumbel_temp
            print(
                f"[MADDPG] Episode {episode + 1}/{n_episodes} | "
                f"avg_queue={avg_q:.2f} | avg_wait={avg_w:.2f}s | "
                f"gumbel_temp={temp:.4f}"
            )

    # Restore best weights
    if best_state is not None:
        for i, state in enumerate(best_state):
            controller._actors[i].load_state_dict(state["actor"])
            controller._critics[i].load_state_dict(state["critic"])

    return controller
