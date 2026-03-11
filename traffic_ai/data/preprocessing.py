"""traffic_ai/data/preprocessing.py

Normalize all ingested datasets into a unified schema, engineer features,
and produce stratified train/val/test splits saved to data/processed/.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from traffic_ai.data.ingestion import DataIngestion, IngestionResult

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Unified schema columns
# ---------------------------------------------------------------------------
UNIFIED_COLUMNS: list[str] = [
    "timestamp",
    "junction_id",
    "vehicle_count",
    "avg_speed",
    "lane_occupancy",
    "queue_length",
    "wait_time",
    "phase",
    "weather_temp",
    "weather_rain",
    "weather_snow",
    "is_holiday",
    "hour_of_day",
    "day_of_week",
    "is_rush_hour",
]


@dataclass
class ProcessedSplits:
    """Container for train/val/test DataFrames and split paths."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_path: Path
    val_path: Path
    test_path: Path
    feature_columns: list[str] = field(default_factory=list)
    target_column: str = "phase"


class DataPreprocessor:
    """Preprocess raw traffic datasets into a unified schema.

    Parameters
    ----------
    raw_dir:
        Directory containing raw CSV files.
    processed_dir:
        Directory where processed splits are written.
    """

    def __init__(
        self,
        raw_dir: str | Path = "data/raw",
        processed_dir: str | Path = "data/processed",
    ) -> None:
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        ingestion_results: list[IngestionResult] | None = None,
        include_kaggle: bool = False,
        include_public: bool = True,
    ) -> ProcessedSplits:
        """Full preprocessing pipeline.

        1. Ingest raw data if not provided.
        2. Normalize each dataset to the unified schema.
        3. Concatenate and de-duplicate.
        4. Engineer features.
        5. Stratified train/val/test split.
        6. Save to processed_dir.
        """
        if ingestion_results is None:
            ingestor = DataIngestion(raw_dir=self.raw_dir)
            ingestion_results = ingestor.ingest_all(
                include_kaggle=include_kaggle,
                include_public=include_public,
            )

        frames: list[pd.DataFrame] = []
        for result in ingestion_results:
            df = self._load_and_normalize(result)
            if df is not None and len(df) > 0:
                frames.append(df)

        if not frames:
            frames.append(self._synthetic_unified())

        combined = pd.concat(frames, ignore_index=True)
        combined = self._deduplicate(combined)
        combined = self._engineer_features(combined)
        combined = combined.sort_values("timestamp").reset_index(drop=True)

        splits = self._split(combined)
        return splits

    # ------------------------------------------------------------------
    # Normalization
    # ------------------------------------------------------------------

    def _load_and_normalize(self, result: IngestionResult) -> pd.DataFrame | None:
        try:
            raw = pd.read_csv(result.path, low_memory=False)
        except Exception as exc:
            logger.warning("Could not load %s: %s", result.path, exc)
            return None

        cols = {c.lower().strip().replace(" ", "_") for c in raw.columns}
        name = result.name.lower()

        # Dispatch to specialized normalizer based on source name
        if "traffic_prediction" in name or "fedesoriano" in name:
            return self._normalize_junction_counts(raw)
        if "hasaan" in name or "traffic_dataset" in name:
            return self._normalize_hasaan(raw)
        if "sensor_fusion" in name or "denkuznets" in name:
            return self._normalize_sensor_fusion(raw)
        if "urban_traffic_light" in name or "alistair" in name:
            return self._normalize_rl_state_action(raw)
        if "metro" in name or "interstate" in name:
            return self._normalize_metro_interstate(raw)
        if "synthetic" in name:
            return self._normalize_synthetic(raw)

        # Generic fallback: try to map known column names
        return self._normalize_generic(raw)

    def _normalize_junction_counts(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles fedesoriano/traffic-prediction-dataset (DateTime, Junction, Vehicles)."""
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        out = pd.DataFrame(index=df.index)
        out["timestamp"] = pd.to_datetime(df.get("datetime", df.get("date", None)), errors="coerce")
        out["junction_id"] = df.get("junction", df.get("junction_id", 1)).fillna(1).astype(int)
        out["vehicle_count"] = pd.to_numeric(df.get("vehicles", df.get("vehicle_count", 0)), errors="coerce").fillna(0)
        out["avg_speed"] = np.nan
        out["lane_occupancy"] = np.nan
        out["queue_length"] = np.nan
        out["wait_time"] = np.nan
        out["phase"] = "NS"
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    def _normalize_hasaan(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles hasaanansari/traffic-dataset."""
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame(index=df.index)
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        out["timestamp"] = (
            pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.Timestamp.now()
        )
        out["junction_id"] = df.get("junction_id", df.get("location_id", 1)).fillna(1).astype(int)
        out["vehicle_count"] = pd.to_numeric(
            df.get("vehicle_count", df.get("vehicles", df.get("count", 0))), errors="coerce"
        ).fillna(0)
        out["avg_speed"] = pd.to_numeric(df.get("avg_speed", df.get("speed", np.nan)), errors="coerce")
        out["lane_occupancy"] = pd.to_numeric(df.get("lane_occupancy", df.get("occupancy", np.nan)), errors="coerce")
        out["queue_length"] = pd.to_numeric(df.get("queue_length", df.get("queue", np.nan)), errors="coerce")
        out["wait_time"] = pd.to_numeric(df.get("wait_time", df.get("avg_wait", np.nan)), errors="coerce")
        out["phase"] = df.get("phase", "NS").fillna("NS").apply(lambda x: "NS" if "N" in str(x).upper() or "S" in str(x).upper() else "EW")
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    def _normalize_sensor_fusion(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles denkuznets81 IR/radar sensor fusion dataset."""
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame(index=df.index)
        ts_col = next((c for c in df.columns if "time" in c or "date" in c), None)
        out["timestamp"] = (
            pd.to_datetime(df[ts_col], errors="coerce") if ts_col else pd.Timestamp.now()
        )
        out["junction_id"] = df.get("intersection_id", df.get("junction_id", 1)).fillna(1).astype(int)
        out["vehicle_count"] = pd.to_numeric(df.get("vehicle_count", df.get("count", 0)), errors="coerce").fillna(0)
        out["avg_speed"] = pd.to_numeric(df.get("speed", df.get("avg_speed", np.nan)), errors="coerce")
        out["lane_occupancy"] = pd.to_numeric(df.get("occupancy", df.get("lane_occupancy", np.nan)), errors="coerce")
        out["queue_length"] = pd.to_numeric(df.get("queue_length", df.get("queue", np.nan)), errors="coerce")
        out["wait_time"] = pd.to_numeric(df.get("wait_time", np.nan), errors="coerce")
        out["phase"] = "NS"
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    def _normalize_rl_state_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles alistairking Urban Traffic Light Control RL dataset."""
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame(index=df.index)
        ts_col = next((c for c in df.columns if "time" in c or "step" in c), None)
        out["timestamp"] = (
            pd.to_datetime(df[ts_col], errors="coerce")
            if ts_col and df[ts_col].dtype != "int64"
            else pd.date_range("2023-01-01", periods=len(df), freq="30s")
        )
        out["junction_id"] = df.get("intersection_id", df.get("junction", 1)).fillna(1).astype(int)
        out["vehicle_count"] = pd.to_numeric(df.get("vehicle_count", df.get("count", 0)), errors="coerce").fillna(0)
        out["avg_speed"] = pd.to_numeric(df.get("avg_speed", df.get("speed", np.nan)), errors="coerce")
        out["lane_occupancy"] = pd.to_numeric(df.get("lane_occupancy", df.get("occupancy", np.nan)), errors="coerce")
        out["queue_length"] = pd.to_numeric(
            df.get("queue_ns", df.get("queue_length", df.get("queue", np.nan))), errors="coerce"
        )
        out["wait_time"] = pd.to_numeric(df.get("wait_time", df.get("avg_wait", np.nan)), errors="coerce")
        # Action column: map to NS/EW phase
        action_col = next((c for c in df.columns if "action" in c or "phase" in c), None)
        if action_col:
            out["phase"] = df[action_col].apply(
                lambda x: "NS" if int(x) in (0, 2) else "EW" if int(x) in (1, 3) else "NS"
            )
        else:
            out["phase"] = "NS"
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    def _normalize_metro_interstate(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles UCI Metro Interstate Traffic Volume dataset."""
        df = df.copy()
        df.columns = [c.lower().strip() for c in df.columns]
        out = pd.DataFrame(index=df.index)
        out["timestamp"] = pd.to_datetime(df.get("date_time", df.get("datetime", None)), errors="coerce")
        out["junction_id"] = 1
        out["vehicle_count"] = pd.to_numeric(df.get("traffic_volume", 0), errors="coerce").fillna(0)
        out["avg_speed"] = np.nan
        out["lane_occupancy"] = np.nan
        out["queue_length"] = np.nan
        out["wait_time"] = np.nan
        out["phase"] = "NS"
        out["weather_temp"] = pd.to_numeric(df.get("temp", np.nan), errors="coerce") - 273.15
        out["weather_rain"] = pd.to_numeric(df.get("rain_1h", 0), errors="coerce").fillna(0)
        out["weather_snow"] = pd.to_numeric(df.get("snow_1h", 0), errors="coerce").fillna(0)
        out["is_holiday"] = df.get("holiday", "None").apply(lambda x: 0 if str(x).strip() in ("None", "") else 1)
        return out.dropna(subset=["timestamp"])

    def _normalize_synthetic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handles our own synthetic dataset."""
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        out = pd.DataFrame(index=df.index)
        out["timestamp"] = pd.to_datetime(df.get("timestamp", None), errors="coerce")
        out["junction_id"] = df.get("junction_id", df.get("location_id", 1)).fillna(1).astype(int)
        out["vehicle_count"] = pd.to_numeric(df.get("vehicle_count", 0), errors="coerce").fillna(0)
        out["avg_speed"] = pd.to_numeric(df.get("avg_speed", df.get("speed_kph", np.nan)), errors="coerce")
        out["lane_occupancy"] = pd.to_numeric(df.get("lane_occupancy", df.get("occupancy", np.nan)), errors="coerce")
        out["queue_length"] = pd.to_numeric(df.get("queue_length", np.nan), errors="coerce")
        out["wait_time"] = pd.to_numeric(df.get("wait_time", df.get("avg_wait_sec", np.nan)), errors="coerce")
        out["phase"] = df.get("phase", df.get("signal_phase", "NS")).fillna("NS")
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    def _normalize_generic(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generic normalizer for unknown schemas."""
        df = df.copy()
        df.columns = [c.lower().strip().replace(" ", "_") for c in df.columns]
        # Try to find timestamp
        ts_candidates = [c for c in df.columns if any(kw in c for kw in ("time", "date", "ts", "stamp"))]
        ts_col = ts_candidates[0] if ts_candidates else None
        out = pd.DataFrame(index=df.index)
        out["timestamp"] = (
            pd.to_datetime(df[ts_col], errors="coerce")
            if ts_col
            else pd.date_range("2023-01-01", periods=len(df), freq="1min")
        )
        out["junction_id"] = 1
        count_col = next((c for c in df.columns if "count" in c or "volume" in c or "vehicle" in c), None)
        out["vehicle_count"] = pd.to_numeric(df[count_col], errors="coerce").fillna(0) if count_col else 0
        out["avg_speed"] = np.nan
        out["lane_occupancy"] = np.nan
        out["queue_length"] = np.nan
        out["wait_time"] = np.nan
        out["phase"] = "NS"
        self._fill_weather_holiday(out)
        return out.dropna(subset=["timestamp"])

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------

    def _engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df = df.dropna(subset=["timestamp"])
        df["hour_of_day"] = df["timestamp"].dt.hour.astype(float)
        df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(float)
        df["is_rush_hour"] = (
            ((df["hour_of_day"] >= 7) & (df["hour_of_day"] < 9))
            | ((df["hour_of_day"] >= 16) & (df["hour_of_day"] < 19))
        ).astype(float)

        # Rolling means (sorted by timestamp per junction)
        df = df.sort_values(["junction_id", "timestamp"]).reset_index(drop=True)
        df["rolling_mean_vehicle_count_15min"] = (
            df.groupby("junction_id")["vehicle_count"]
            .transform(lambda s: s.rolling(3, min_periods=1).mean())
        )
        df["rolling_mean_vehicle_count_60min"] = (
            df.groupby("junction_id")["vehicle_count"]
            .transform(lambda s: s.rolling(12, min_periods=1).mean())
        )
        return df

    # ------------------------------------------------------------------
    # Train/val/test split
    # ------------------------------------------------------------------

    def _split(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
    ) -> ProcessedSplits:
        # Stratification key: junction_id × hour_of_day bucket
        df = df.copy()
        df["_strat"] = (
            df["junction_id"].astype(str)
            + "_"
            + (df["hour_of_day"].fillna(0).astype(int) // 3).astype(str)
        )
        strat = df["_strat"]
        # Only stratify if each class has ≥2 samples
        vc = strat.value_counts()
        valid_strat = strat.map(vc).ge(2)
        strat_arg = strat if valid_strat.all() else None

        try:
            trainval, test = train_test_split(
                df,
                test_size=1 - train_ratio - val_ratio,
                stratify=strat_arg,
                random_state=42,
            )
            val_frac = val_ratio / (train_ratio + val_ratio)
            strat_tv = trainval["_strat"] if strat_arg is not None else None
            vc2 = strat_tv.value_counts() if strat_tv is not None else None
            if vc2 is not None and (vc2 < 2).any():
                strat_tv = None
            train, val = train_test_split(
                trainval,
                test_size=val_frac,
                stratify=strat_tv,
                random_state=42,
            )
        except Exception:
            n = len(df)
            n_train = int(n * train_ratio)
            n_val = int(n * val_ratio)
            train = df.iloc[:n_train]
            val = df.iloc[n_train : n_train + n_val]
            test = df.iloc[n_train + n_val :]

        for frame in (train, val, test):
            frame.drop(columns=["_strat"], errors="ignore", inplace=True)

        train_path = self.processed_dir / "train.csv"
        val_path = self.processed_dir / "val.csv"
        test_path = self.processed_dir / "test.csv"
        train.to_csv(train_path, index=False)
        val.to_csv(val_path, index=False)
        test.to_csv(test_path, index=False)

        feature_cols = [
            c
            for c in train.columns
            if c not in ("timestamp", "phase", "_strat")
        ]
        return ProcessedSplits(
            train=train,
            val=val,
            test=test,
            train_path=train_path,
            val_path=val_path,
            test_path=test_path,
            feature_columns=feature_cols,
            target_column="phase",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fill_weather_holiday(df: pd.DataFrame) -> None:
        for col in ("weather_temp", "weather_rain", "weather_snow", "is_holiday"):
            if col not in df.columns:
                df[col] = np.nan

    def _deduplicate(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates(subset=["timestamp", "junction_id"]).reset_index(drop=True)

    def _synthetic_unified(self) -> pd.DataFrame:
        """Return a small synthetic unified dataset when no real data is available."""
        rng = np.random.default_rng(0)
        n = 2_000
        timestamps = pd.date_range("2023-01-01", periods=n, freq="15min")
        hours = timestamps.hour
        is_rush = ((hours >= 7) & (hours <= 9)) | ((hours >= 16) & (hours <= 19))
        vc = rng.poisson(15, n).astype(float)
        vc[is_rush] *= 2.5
        return pd.DataFrame(
            {
                "timestamp": timestamps,
                "junction_id": rng.integers(1, 5, n),
                "vehicle_count": vc.round().astype(int),
                "avg_speed": np.clip(rng.normal(35, 10, n), 5, 80),
                "lane_occupancy": np.clip(rng.beta(2, 5, n), 0, 1),
                "queue_length": rng.poisson(4, n),
                "wait_time": np.clip(rng.exponential(25, n), 0, 180),
                "phase": rng.choice(["NS", "EW"], n),
                "weather_temp": np.nan,
                "weather_rain": np.nan,
                "weather_snow": np.nan,
                "is_holiday": 0,
            }
        )
