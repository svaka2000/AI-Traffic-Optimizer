from __future__ import annotations

from pathlib import Path

from traffic_ai.config.settings import ensure_runtime_dirs, load_settings
from traffic_ai.experiments import ExperimentRunner


def _temp_settings(tmp_path: Path):
    settings = load_settings("traffic_ai/config/default_config.yaml")
    settings.payload["project"]["output_dir"] = str(tmp_path / "artifacts")
    settings.payload["data"]["raw_dir"] = str(tmp_path / "raw")
    settings.payload["data"]["processed_dir"] = str(tmp_path / "processed")
    settings.payload["simulation"]["steps"] = 120
    ensure_runtime_dirs(settings)
    return settings


def test_runner_ingest_only(tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    runner = ExperimentRunner(settings=settings, quick_run=True)
    artifacts = runner.run(
        ingest_only=True,
        include_kaggle=False,
        include_public=False,
    )
    assert artifacts.summary_csv.exists()
    assert artifacts.step_metrics_csv.exists()


def test_runner_pretrain_only(tmp_path: Path) -> None:
    """--pretrain-only trains RL models and exits without running CV benchmark."""
    settings = _temp_settings(tmp_path)
    runner = ExperimentRunner(settings=settings, quick_run=True)
    artifacts = runner.run(
        pretrain_only=True,
        include_kaggle=False,
        include_public=False,
    )
    # Should exit early — only the stub artifact is written
    assert artifacts.trained_model_dir.exists()


def test_rl_controller_uses_upstream_queue() -> None:
    """RLPolicyController passes upstream_queue as feature[7], normalised by 120.

    The DQN is now trained on the 4-intersection network where upstream_queue
    is the mean of neighbouring intersection queues (non-zero).  The feature
    vector must forward the actual value so training and evaluation are
    on-distribution.
    """
    import numpy as np
    from traffic_ai.controllers.rl_controller import RLPolicyController

    captured: list[np.ndarray] = []

    def mock_policy(features: np.ndarray) -> int:
        captured.append(features.copy())
        return 0

    ctrl = RLPolicyController(policy=mock_policy, name="test_rl", min_green=1)
    ctrl.reset(1)
    obs = {0: {
        "phase_elapsed": 5.0,
        "current_phase_idx": 0.0,
        "queue_ns_through": 20.0,
        "queue_ew_through": 10.0,
        "queue_ns_left": 3.0,
        "queue_ew_left": 2.0,
        "time_of_day_normalized": 0.5,
        "upstream_queue": 60.0,
    }}
    ctrl.compute_actions(obs, step=10)
    assert len(captured) == 1
    assert abs(captured[0][7] - 60.0 / 120.0) < 1e-6, (
        "feature[7] must be upstream_queue / 120 (4-intersection training distribution)"
    )

