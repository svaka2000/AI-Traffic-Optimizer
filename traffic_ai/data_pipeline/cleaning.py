from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


COLUMN_ALIASES = {
    "date_time": "timestamp",
    "datetime": "timestamp",
    "date": "timestamp",
    "time": "timestamp",
    "station": "location_id",
    "site_id": "location_id",
    "intersection_id": "location_id",
    "direction_name": "direction",
    "lane_direction": "direction",
    "volume": "vehicle_count",
    "count": "vehicle_count",
    "flow": "vehicle_count",
    "speed": "speed_kph",
    "avg_speed": "speed_kph",
    "occupancy_rate": "occupancy",
    "phase": "signal_phase",
    "signal": "signal_phase",
    "queue": "queue_length",
    "avg_wait": "avg_wait_sec",
    "wait_time": "avg_wait_sec",
}


class DataCleaner:
    canonical_columns = [
        "timestamp",
        "location_id",
        "direction",
        "vehicle_count",
        "speed_kph",
        "occupancy",
        "signal_phase",
        "queue_length",
        "avg_wait_sec",
    ]

    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        normalized = self._normalize_columns(df)
        normalized = self._ensure_columns(normalized)
        normalized = self._coerce_types(normalized)
        normalized = self._clip_outliers(normalized)
        normalized = self._fill_missing(normalized)
        normalized = normalized.drop_duplicates(
            subset=["timestamp", "location_id", "direction"], keep="last"
        ).sort_values("timestamp")
        return normalized.reset_index(drop=True)

    def clean_files(self, sources: Iterable[Path], output_dir: Path) -> list[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        outputs: list[Path] = []
        for source in sources:
            try:
                frame = pd.read_csv(source, low_memory=False)
            except Exception:
                continue
            cleaned = self.clean_dataframe(frame)
            target = output_dir / f"cleaned_{source.stem}.csv"
            cleaned.to_csv(target, index=False)
            outputs.append(target)
        return outputs

    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        renamed = {
            col: COLUMN_ALIASES.get(col.strip().lower(), col.strip().lower())
            for col in df.columns
        }
        return df.rename(columns=renamed)

    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        copy = df.copy()
        if "timestamp" not in copy.columns:
            copy["timestamp"] = pd.date_range("2025-01-01", periods=len(copy), freq="5min")
        if "location_id" not in copy.columns:
            copy["location_id"] = 0
        if "direction" not in copy.columns:
            copy["direction"] = "N"
        if "vehicle_count" not in copy.columns:
            copy["vehicle_count"] = 0
        if "speed_kph" not in copy.columns:
            copy["speed_kph"] = np.nan
        if "occupancy" not in copy.columns:
            copy["occupancy"] = np.nan
        if "signal_phase" not in copy.columns:
            copy["signal_phase"] = "NS"
        if "queue_length" not in copy.columns:
            copy["queue_length"] = np.nan
        if "avg_wait_sec" not in copy.columns:
            copy["avg_wait_sec"] = np.nan
        return copy[self.canonical_columns]

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        coerced = df.copy()
        coerced["timestamp"] = pd.to_datetime(coerced["timestamp"], errors="coerce")
        coerced["timestamp"] = coerced["timestamp"].ffill().fillna(pd.Timestamp("2025-01-01"))
        coerced["location_id"] = coerced["location_id"].astype(str)
        coerced["direction"] = coerced["direction"].astype(str).str.upper().str[:1]
        coerced.loc[~coerced["direction"].isin(["N", "S", "E", "W"]), "direction"] = "N"
        for col in ["vehicle_count", "speed_kph", "occupancy", "queue_length", "avg_wait_sec"]:
            coerced[col] = pd.to_numeric(coerced[col], errors="coerce")
        coerced["signal_phase"] = coerced["signal_phase"].astype(str).str.upper()
        return coerced

    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        clipped = df.copy()
        for col in ["vehicle_count", "speed_kph", "queue_length", "avg_wait_sec"]:
            series = clipped[col].dropna()
            if series.empty:
                continue
            lower = series.quantile(0.01)
            upper = series.quantile(0.99)
            clipped[col] = clipped[col].clip(lower, upper)
        clipped["occupancy"] = clipped["occupancy"].clip(0, 1)
        return clipped

    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        filled = df.copy()
        grouped = filled.groupby("location_id", group_keys=False)
        for col in ["vehicle_count", "speed_kph", "occupancy", "queue_length", "avg_wait_sec"]:
            filled[col] = grouped[col].apply(lambda s: s.ffill().bfill())
            filled[col] = filled[col].fillna(filled[col].median() if not filled[col].dropna().empty else 0.0)
        return filled
