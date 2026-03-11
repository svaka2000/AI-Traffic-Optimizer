from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

try:
    from xgboost import XGBClassifier  # type: ignore
except Exception:  # pragma: no cover
    XGBClassifier = None


@dataclass(slots=True)
class ControllerTrainingData:
    X: np.ndarray
    y: np.ndarray


def generate_controller_training_data(seed: int = 42, n_samples: int = 6_000) -> ControllerTrainingData:
    rng = np.random.default_rng(seed)
    queue_ns = rng.gamma(shape=2.2, scale=7.0, size=n_samples)
    queue_ew = rng.gamma(shape=2.1, scale=6.5, size=n_samples)
    total_queue = queue_ns + queue_ew
    phase_elapsed = rng.integers(0, 80, size=n_samples)
    phase_ns = rng.integers(0, 2, size=n_samples)
    phase_ew = 1 - phase_ns
    sim_step = rng.integers(0, 12_000, size=n_samples)
    wait_sec = np.clip(total_queue * rng.normal(4.2, 0.9, size=n_samples), 0, 600)

    signal_pressure = queue_ns - queue_ew + 0.08 * phase_elapsed + rng.normal(0, 2.2, n_samples)
    y = (signal_pressure < 0).astype(int)
    X = np.column_stack(
        [queue_ns, queue_ew, total_queue, phase_elapsed, phase_ns, phase_ew, sim_step, wait_sec]
    ).astype(np.float32)
    return ControllerTrainingData(X=X, y=y)


def train_supervised_controller_models(seed: int = 42) -> dict[str, Any]:
    data = generate_controller_training_data(seed=seed)
    X, y = data.X, data.y

    models: dict[str, Any] = {
        "random_forest": RandomForestClassifier(
            n_estimators=90, max_depth=10, random_state=seed
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=80, learning_rate=0.08, max_depth=2, random_state=seed
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(32, 32),
            max_iter=120,
            random_state=seed,
            early_stopping=True,
        ),
        "timeseries": LogisticRegression(max_iter=1200),
    }
    if XGBClassifier is not None:
        models["xgboost"] = XGBClassifier(
            n_estimators=90,
            max_depth=4,
            learning_rate=0.08,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=seed,
            eval_metric="logloss",
        )
    else:
        models["xgboost"] = GradientBoostingClassifier(random_state=seed)

    for model in models.values():
        model.fit(X, y)
    return models
