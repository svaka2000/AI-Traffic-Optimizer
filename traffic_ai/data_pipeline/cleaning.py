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

    # -------------------------------------------------------------------------
    # Orchestrator: runs each cleaning step in the required order.
    # -------------------------------------------------------------------------
    def clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Applies the full cleaning pipeline: normalise → ensure → coerce → clip → fill → dedup."""
        normalized = self._normalize_columns(df)    # align heterogeneous column names
        normalized = self._ensure_columns(normalized)  # add missing canonical columns
        normalized = self._coerce_types(normalized)    # enforce correct dtypes
        normalized = self._clip_outliers(normalized)   # remove statistical extremes
        normalized = self._fill_missing(normalized)    # impute nulls per location group
        normalized = normalized.drop_duplicates(       # keep last record per unique event
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

    # -------------------------------------------------------------------------
    # SRP: Maps heterogeneous source column names to the canonical schema.
    # -------------------------------------------------------------------------
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Lowercases and strips column names, then applies COLUMN_ALIASES remapping."""
        renamed = {
            col: COLUMN_ALIASES.get(col.strip().lower(), col.strip().lower())
            for col in df.columns
        }
        return df.rename(columns=renamed)

    # -------------------------------------------------------------------------
    # SRP: Adds any missing canonical columns with safe default values.
    # -------------------------------------------------------------------------
    def _ensure_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Guarantees every required column exists before type coercion downstream."""
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

    # -------------------------------------------------------------------------
    # SRP: Enforces correct Python/Pandas dtypes for every canonical column.
    # -------------------------------------------------------------------------
    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Parses timestamps, uppercases direction codes, and converts numerics safely."""
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

    # -------------------------------------------------------------------------
    # SRP: Removes statistical extremes by clipping at the 1st and 99th percentiles.
    # -------------------------------------------------------------------------
    def _clip_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clips numeric sensor columns to the [P1, P99] range; pins occupancy to [0, 1]."""
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

    # -------------------------------------------------------------------------
    # SRP: Imputes null values using forward/backward fill within each location group.
    # -------------------------------------------------------------------------
    def _fill_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """Forward-fills then back-fills per location; falls back to column median."""
        filled = df.copy()
        grouped = filled.groupby("location_id", group_keys=False)
        for col in ["vehicle_count", "speed_kph", "occupancy", "queue_length", "avg_wait_sec"]:
            filled[col] = grouped[col].apply(lambda s: s.ffill().bfill())
            filled[col] = filled[col].fillna(filled[col].median() if not filled[col].dropna().empty else 0.0)
        return filled
