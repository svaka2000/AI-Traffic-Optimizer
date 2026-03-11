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

