from __future__ import annotations

from pathlib import Path

from traffic_ai.config.settings import ensure_runtime_dirs, load_settings
from traffic_ai.data_pipeline import TrafficDataPipeline


def _temp_settings(tmp_path: Path):
    settings = load_settings("traffic_ai/config/default_config.yaml")
    settings.payload["project"]["output_dir"] = str(tmp_path / "artifacts")
    settings.payload["data"]["raw_dir"] = str(tmp_path / "raw")
    settings.payload["data"]["processed_dir"] = str(tmp_path / "processed")
    ensure_runtime_dirs(settings)
    return settings


def test_data_pipeline_end_to_end(tmp_path: Path) -> None:
    settings = _temp_settings(tmp_path)
    pipeline = TrafficDataPipeline(settings)
    result = pipeline.run(
        include_kaggle=False,
        include_public=False,
        include_local_csv=False,
    )
    assert result.modeling_table_path.exists()
    assert len(result.cleaned_files) >= 1
    assert result.prepared_dataset.X_train.shape[0] > 0
    assert result.prepared_dataset.X_test.shape[0] > 0

