from __future__ import annotations

import numpy as np
import pandas as pd


class FeatureEngineer:
    def __init__(self, aggregation_minutes: int = 5) -> None:
        self.aggregation_minutes = aggregation_minutes

    def build_modeling_table(self, cleaned_df: pd.DataFrame) -> pd.DataFrame:
        work = cleaned_df.copy()
        work["timestamp"] = pd.to_datetime(work["timestamp"], errors="coerce")
        work = work.dropna(subset=["timestamp"])
        work["bucket_time"] = work["timestamp"].dt.floor(f"{self.aggregation_minutes}min")

        grouped = (
            work.groupby(["bucket_time", "location_id", "direction"], as_index=False)
            .agg(
                vehicle_count=("vehicle_count", "sum"),
                speed_kph=("speed_kph", "mean"),
                occupancy=("occupancy", "mean"),
                queue_length=("queue_length", "mean"),
                avg_wait_sec=("avg_wait_sec", "mean"),
                signal_phase=("signal_phase", "last"),
            )
            .sort_values("bucket_time")
        )
        pivot = self._pivot_directional(grouped)
        featured = self._add_time_features(pivot)
        featured = self._add_history_features(featured)
        featured = self._add_state_history(featured)
        featured = featured.dropna().reset_index(drop=True)
        return featured

    def _pivot_directional(self, grouped: pd.DataFrame) -> pd.DataFrame:
        metrics = ["vehicle_count", "speed_kph", "occupancy", "queue_length", "avg_wait_sec"]
        base_cols = ["bucket_time", "location_id"]
        tables: list[pd.DataFrame] = []
        for metric in metrics:
            table = (
                grouped.pivot_table(
                    index=base_cols,
                    columns="direction",
                    values=metric,
                    aggfunc="mean",
                    fill_value=0.0,
                )
                .rename(columns=lambda d: f"{metric}_{d}")
                .reset_index()
            )
            tables.append(table)

        merged = tables[0]
        for table in tables[1:]:
            merged = merged.merge(table, on=base_cols, how="inner")

        for side in ["N", "S", "E", "W"]:
            if f"vehicle_count_{side}" not in merged.columns:
                merged[f"vehicle_count_{side}"] = 0.0
            if f"queue_length_{side}" not in merged.columns:
                merged[f"queue_length_{side}"] = 0.0

        merged["arrival_rate_ns"] = merged["vehicle_count_N"] + merged["vehicle_count_S"]
        merged["arrival_rate_ew"] = merged["vehicle_count_E"] + merged["vehicle_count_W"]
        merged["queue_ns"] = merged["queue_length_N"] + merged["queue_length_S"]
        merged["queue_ew"] = merged["queue_length_E"] + merged["queue_length_W"]
        merged["density"] = (
            merged.get("occupancy_N", 0)
            + merged.get("occupancy_S", 0)
            + merged.get("occupancy_E", 0)
            + merged.get("occupancy_W", 0)
        ) / 4.0

        merged["optimal_phase"] = np.where(merged["queue_ns"] >= merged["queue_ew"], 0, 1)
        merged["timestamp"] = merged["bucket_time"]
        merged.drop(columns=["bucket_time"], inplace=True)
        return merged

    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["hour"] = out["timestamp"].dt.hour
        out["minute"] = out["timestamp"].dt.minute
        out["day_of_week"] = out["timestamp"].dt.dayofweek
        out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)

        out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
        out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)
        out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7.0)
        out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7.0)
        return out

    def _add_history_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.sort_values(["location_id", "timestamp"]).copy()
        group = out.groupby("location_id", group_keys=False)

        lag_cols = ["queue_ns", "queue_ew", "arrival_rate_ns", "arrival_rate_ew", "density"]
        for col in lag_cols:
            for lag in [1, 2, 3]:
                out[f"{col}_lag_{lag}"] = group[col].shift(lag)

        roll_cols = ["queue_ns", "queue_ew", "arrival_rate_ns", "arrival_rate_ew", "density"]
        for col in roll_cols:
            out[f"{col}_roll_mean_3"] = group[col].transform(
                lambda s: s.rolling(3, min_periods=1).mean()
            )
            out[f"{col}_roll_std_6"] = group[col].transform(
                lambda s: s.rolling(6, min_periods=2).std().fillna(0.0)
            )
        return out

    def _add_state_history(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        out["signal_state_prev"] = out.groupby("location_id")["optimal_phase"].shift(1)
        out["signal_state_prev2"] = out.groupby("location_id")["optimal_phase"].shift(2)
        out["wait_mean_proxy"] = (
            out.get("avg_wait_sec_N", 0)
            + out.get("avg_wait_sec_S", 0)
            + out.get("avg_wait_sec_E", 0)
            + out.get("avg_wait_sec_W", 0)
        ) / 4.0
        out["queue_imbalance"] = out["queue_ns"] - out["queue_ew"]
        return out
