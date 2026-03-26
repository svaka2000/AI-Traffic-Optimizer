"""traffic_ai/rl_models/recurrent_ppo.py

Recurrent PPO (LSTM actor-critic) training function.

Uses the RecurrentPPOController from rl_controllers.py, which implements:
  - LSTM actor and critic with hidden_size=64
  - SEQ_LEN=16 truncated BPTT
  - Per-intersection hidden state management
  - PPO clipping (eps=0.2), entropy bonus
"""
from __future__ import annotations

from traffic_ai.controllers.rl_controllers import RecurrentPPOController
from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator


def train_recurrent_ppo(
    n_intersections: int = 4,
    demand_profile: str = "rush_hour",
    steps_per_episode: int = 500,
    n_episodes: int = 80,
    update_steps: int = 40_000,
    seed: int = 42,
    verbose: bool = False,
) -> RecurrentPPOController:
    """Train a Recurrent PPO controller on the traffic simulation.

    The LSTM architecture enables the controller to reason about temporal
    dependencies across steps — useful for dynamic demand profiles such as
    rush_hour, event_surge, and incident_response where queue build-up is
    predictable from recent history.

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
        Total expected update steps for learning rate / entropy scheduling.
    seed:
        Random seed for reproducibility.
    verbose:
        Print per-episode metrics if True.

    Returns
    -------
    RecurrentPPOController
        Trained controller ready for evaluation.
    """
    config = SimulatorConfig(
        steps=steps_per_episode,
        intersections=n_intersections,
        demand_profile=demand_profile,
        seed=seed,
    )
    sim = TrafficNetworkSimulator(config)
    controller = RecurrentPPOController(
        n_intersections=n_intersections,
        update_steps=update_steps,
    )

    for episode in range(n_episodes):
        result = sim.run(controller, steps=steps_per_episode)
        if verbose:
            avg_q = result.aggregate.get("average_queue_length", 0.0)
            avg_w = result.aggregate.get("average_wait_time", 0.0)
            eff = result.aggregate.get("average_efficiency_score", 0.0)
            print(
                f"[RecurrentPPO] Episode {episode + 1}/{n_episodes} | "
                f"avg_queue={avg_q:.2f} | avg_wait={avg_w:.2f}s | "
                f"efficiency={eff:.4f}"
            )

    return controller
