"""traffic_ai/data/ingestion.py

Ingest traffic datasets from Kaggle, UCI ML Repository, and Mendeley Data.
Raw files are saved to settings.data_dir/raw/ without overwriting existing files.
"""
from __future__ import annotations

import io
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Kaggle dataset slugs to ingest
# ---------------------------------------------------------------------------
KAGGLE_DATASETS: list[str] = [
    "fedesoriano/traffic-prediction-dataset",
    "hasaanansari/traffic-dataset",
    "denkuznets81/dynamic-traffic-signal-sensor-fusion-dataset",
    "alistairking/urban-traffic-light-control-dataset",
]

# ---------------------------------------------------------------------------
# Public dataset sources (UCI ML Repository / Mendeley)
# ---------------------------------------------------------------------------
PUBLIC_SOURCES: list[dict[str, str]] = [
    {
        "name": "metro_interstate_traffic_volume",
        "url": (
            "https://archive.ics.uci.edu/ml/machine-learning-databases/"
            "00492/Metro_Interstate_Traffic_Volume.csv.gz"
        ),
        "filename": "metro_interstate_traffic_volume.csv",
    },
]


@dataclass
class IngestionResult:
    """Metadata for a single ingested dataset."""

    name: str
    path: Path
    rows: int
    columns: list[str]
    source: str = "unknown"
    extra: dict[str, Any] = field(default_factory=dict)


class DataIngestion:
    """Download and persist raw traffic datasets.

    Parameters
    ----------
    raw_dir:
        Directory where raw files are saved (defaults to ``data/raw``).
    """

    def __init__(self, raw_dir: str | Path = "data/raw") -> None:
        self.raw_dir = Path(raw_dir)
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_all(
        self,
        include_kaggle: bool = True,
        include_public: bool = True,
    ) -> list[IngestionResult]:
        """Run full ingestion pipeline and return metadata for each dataset."""
        results: list[IngestionResult] = []
        if include_public:
            results.extend(self._ingest_public_sources())
        if include_kaggle:
            results.extend(self._ingest_kaggle_datasets())
        if not results:
            results.append(self._generate_synthetic_fallback())
        total_rows = sum(r.rows for r in results)
        if total_rows < 3_000:
            results.append(self._generate_synthetic_fallback())
        return results

    # ------------------------------------------------------------------
    # Kaggle ingestion
    # ------------------------------------------------------------------

    def _ingest_kaggle_datasets(self) -> list[IngestionResult]:
        try:
            from kaggle.api.kaggle_api_extended import KaggleApi  # type: ignore

            api = KaggleApi()
            api.authenticate()
        except Exception as exc:
            logger.warning("Kaggle API unavailable (%s); skipping Kaggle ingestion.", exc)
            return []

        outputs: list[IngestionResult] = []
        for slug in KAGGLE_DATASETS:
            outputs.extend(self._download_kaggle_dataset(api, slug))
        return outputs

    def _download_kaggle_dataset(self, api: Any, slug: str) -> list[IngestionResult]:
        safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
        dest_dir = self.raw_dir / f"kaggle_{safe}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            api.dataset_download_files(
                dataset=slug,
                path=str(dest_dir),
                unzip=True,
                quiet=True,
            )
        except Exception as exc:
            logger.warning("Failed to download Kaggle dataset %s: %s", slug, exc)
            return []

        results: list[IngestionResult] = []
        for csv_path in dest_dir.rglob("*.csv"):
            canonical = self.raw_dir / f"{safe}_{csv_path.name}"
            if not canonical.exists():
                shutil.copy2(csv_path, canonical)
            df = self._safe_read_csv(canonical)
            if df is None:
                continue
            results.append(
                IngestionResult(
                    name=f"kaggle_{slug}_{csv_path.stem}",
                    path=canonical,
                    rows=len(df),
                    columns=list(df.columns),
                    source="kaggle",
                )
            )
        return results

    # ------------------------------------------------------------------
    # Public source ingestion
    # ------------------------------------------------------------------

    def _ingest_public_sources(self) -> list[IngestionResult]:
        results: list[IngestionResult] = []
        for src in PUBLIC_SOURCES:
            result = self._download_public_source(src)
            if result is not None:
                results.append(result)
        return results

    def _download_public_source(self, src: dict[str, str]) -> IngestionResult | None:
        filename = src.get("filename", src["name"] + ".csv")
        target = self.raw_dir / filename
        if target.exists():
            df = self._safe_read_csv(target)
            if df is not None:
                return IngestionResult(
                    name=src["name"],
                    path=target,
                    rows=len(df),
                    columns=list(df.columns),
                    source="public",
                )
        try:
            resp = requests.get(src["url"], timeout=60)
            resp.raise_for_status()
            content = resp.content
            # Handle gzip-compressed CSV
            if src["url"].endswith(".gz"):
                import gzip
                content = gzip.decompress(content)
            target.write_bytes(content)
        except Exception as exc:
            logger.warning("Failed to download %s: %s", src["name"], exc)
            return None

        df = self._safe_read_csv(target)
        if df is None:
            return None
        return IngestionResult(
            name=src["name"],
            path=target,
            rows=len(df),
            columns=list(df.columns),
            source="public",
        )

    # ------------------------------------------------------------------
    # Synthetic fallback
    # ------------------------------------------------------------------

    def _generate_synthetic_fallback(self) -> IngestionResult:
        """Generate a synthetic traffic dataset when real data is unavailable."""
        rng = np.random.default_rng(42)
        n = 8_000
        timestamps = pd.date_range("2023-01-01", periods=n, freq="15min")
        junction_ids = rng.integers(1, 5, size=n)
        hours = timestamps.hour
        is_rush = ((hours >= 7) & (hours <= 9)) | ((hours >= 16) & (hours <= 19))
        base_count = rng.poisson(20, size=n).astype(float)
        base_count[is_rush] *= 2.5
        df = pd.DataFrame(
            {
                "timestamp": timestamps,
                "junction_id": junction_ids,
                "vehicle_count": base_count.round().astype(int),
                "avg_speed": np.clip(rng.normal(35, 10, n), 5, 80),
                "lane_occupancy": np.clip(rng.beta(2, 5, n), 0, 1),
                "queue_length": rng.poisson(5, n),
                "wait_time": np.clip(rng.exponential(30, n), 0, 180),
                "phase": rng.choice(["NS", "EW"], n),
            }
        )
        target = self.raw_dir / "synthetic_traffic_fallback.csv"
        if not target.exists():
            df.to_csv(target, index=False)
        return IngestionResult(
            name="synthetic_traffic_fallback",
            path=target,
            rows=len(df),
            columns=list(df.columns),
            source="synthetic",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _safe_read_csv(path: Path) -> pd.DataFrame | None:
        try:
            return pd.read_csv(path, low_memory=False)
        except Exception as exc:
            logger.debug("Could not read %s: %s", path, exc)
            return None
