from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from traffic_ai.rl_models.dqn import DQNPolicy, train_dqn, train_dqn_multi
from traffic_ai.rl_models.environment import EnvConfig, MultiIntersectionSignalEnv, SignalControlEnv
from traffic_ai.rl_models.policy_gradient import PolicyGradientPolicy, train_policy_gradient
from traffic_ai.rl_models.q_learning import train_q_learning
from traffic_ai.utils.io_utils import load_model, save_model, write_dataframe

try:
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RLTrainingResult:
    policies: dict[str, Any]
    reward_history: pd.DataFrame


def _try_load_dqn(path: Path) -> DQNPolicy | None:
    """Load a pre-trained DQN policy from a PyTorch state-dict file, or return None."""
    if not path.exists() or torch is None:
        return None
    try:
        from traffic_ai.rl_models.dqn import DQNetwork
        net = DQNetwork(input_dim=8, output_dim=4)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        logger.info("Loaded pre-trained DQN weights from %s", path)
        return DQNPolicy(network=net)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load DQN weights from %s: %s — retraining.", path, exc)
        return None


def _try_load_pg(path: Path) -> PolicyGradientPolicy | None:
    """Load a pre-trained policy-gradient policy, or return None."""
    if not path.exists() or torch is None:
        return None
    try:
        from traffic_ai.rl_models.policy_gradient import PolicyNet
        net = PolicyNet(input_dim=8, output_dim=4)
        net.load_state_dict(torch.load(path, map_location="cpu"))
        net.eval()
        logger.info("Loaded pre-trained PG weights from %s", path)
        return PolicyGradientPolicy(network=net)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not load PG weights from %s: %s — retraining.", path, exc)
        return None


def train_rl_policy_suite(
    output_dir: Path,
    seed: int = 42,
    quick_run: bool = True,
    full_run: bool = False,
) -> RLTrainingResult:
    if full_run:
        episodes = 2000
    elif quick_run:
        episodes = 35
    else:
        episodes = 260

    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    env = SignalControlEnv(EnvConfig(seed=seed))
    policies: dict[str, Any] = {}
    rows: list[dict[str, float | str]] = []

    # -------------------------------------------------------------------------
    # Q-Learning
    # -------------------------------------------------------------------------
    q_model_path = model_dir / "q_learning_policy.joblib"
    if q_model_path.exists():
        q_policy = load_model(q_model_path)
        logger.info("Loaded pre-trained Q-learning policy from %s", q_model_path)
    else:
        q_policy, q_rewards = train_q_learning(env, episodes=episodes)
        rows.extend({"algorithm": "q_learning", "episode": i, "reward": r} for i, r in enumerate(q_rewards))
        save_model(q_policy, q_model_path)
    policies["q_learning"] = q_policy

    # -------------------------------------------------------------------------
    # DQN — trained on the 4-intersection/2000-step network for full_run so the
    # policy distribution matches the benchmark evaluation environment exactly.
    # Quick/medium runs fall back to the lighter single-intersection env.
    # Pre-trained weights are loaded from disk when available to skip retraining.
    # -------------------------------------------------------------------------
    dqn_model_path = model_dir / "dqn_policy.pt"
    dqn_policy = _try_load_dqn(dqn_model_path)
    if dqn_policy is None:
        n_seeds = int(full_run) * 4 + 1  # 5 for full_run, 1 otherwise
        if full_run:
            # 4-intersection IDQN: trains on the same 2×2 grid used in benchmarking
            dqn_train_env: MultiIntersectionSignalEnv | SignalControlEnv = MultiIntersectionSignalEnv(
                EnvConfig(seed=seed, step_limit=2000),
                n_intersections=4,
            )
            dqn_policy, dqn_rewards, dqn_network = train_dqn_multi(
                dqn_train_env,  # type: ignore[arg-type]
                episodes=episodes,
                n_train_seeds=n_seeds,
            )
        else:
            dqn_train_env = SignalControlEnv(EnvConfig(seed=seed, step_limit=500))
            dqn_policy, dqn_rewards, dqn_network = train_dqn(
                dqn_train_env,  # type: ignore[arg-type]
                episodes=episodes,
                n_train_seeds=n_seeds,
            )
        rows.extend({"algorithm": "dqn", "episode": i, "reward": r} for i, r in enumerate(dqn_rewards))
        if torch is not None and dqn_network is not None:
            torch.save(dqn_network.state_dict(), dqn_model_path)
    policies["dqn"] = dqn_policy

    # -------------------------------------------------------------------------
    # Policy Gradient
    # -------------------------------------------------------------------------
    pg_model_path = model_dir / "policy_gradient.pt"
    pg_policy = _try_load_pg(pg_model_path)
    if pg_policy is None:
        pg_policy, pg_rewards, pg_network = train_policy_gradient(env, episodes=episodes)
        rows.extend({"algorithm": "policy_gradient", "episode": i, "reward": r} for i, r in enumerate(pg_rewards))
        if torch is not None and pg_network is not None:
            torch.save(pg_network.state_dict(), pg_model_path)
    policies["policy_gradient"] = pg_policy

    history = pd.DataFrame(rows)
    history_path = output_dir / "results" / "rl_training_history.csv"
    if not history.empty:
        write_dataframe(history, history_path)
    elif history_path.exists():
        # All models were loaded from disk — reuse the saved training history for plots.
        try:
            history = pd.read_csv(history_path)
        except Exception:  # noqa: BLE001
            pass
    return RLTrainingResult(policies=policies, reward_history=history)
