from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(slots=True)
class Settings:
    config_path: Path
    payload: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.payload["project"]["seed"])

    @property
    def output_dir(self) -> Path:
        return Path(self.payload["project"]["output_dir"])

    @property
    def raw_data_dir(self) -> Path:
        return Path(self.payload["data"]["raw_dir"])

    @property
    def processed_data_dir(self) -> Path:
        return Path(self.payload["data"]["processed_dir"])

    def get(self, key: str, default: Any = None) -> Any:
        cursor: Any = self.payload
        for part in key.split("."):
            if not isinstance(cursor, dict) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor


def load_settings(config_path: str | Path | None = None) -> Settings:
    target = (
        Path(config_path)
        if config_path
        else Path("traffic_ai/config/default_config.yaml")
    )
    with target.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream)
    settings = Settings(config_path=target, payload=payload)
    ensure_runtime_dirs(settings)
    return settings


def ensure_runtime_dirs(settings: Settings) -> None:
    dirs = [
        settings.output_dir,
        settings.output_dir / "results",
        settings.output_dir / "models",
        settings.output_dir / "plots",
        settings.output_dir / "cache",
        settings.raw_data_dir,
        settings.processed_data_dir,
    ]
    for directory in dirs:
        directory.mkdir(parents=True, exist_ok=True)

