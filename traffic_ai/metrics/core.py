from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd

from traffic_ai.simulation_engine.types import SimulationResult


def simulation_result_to_step_dataframe(result: SimulationResult) -> pd.DataFrame:
    rows = [asdict(item) for item in result.step_metrics]
    frame = pd.DataFrame(rows)
    frame["controller"] = result.controller_name
    return frame


def simulation_result_to_summary_row(result: SimulationResult) -> dict[str, float | str]:
    row: dict[str, float | str] = {"controller": result.controller_name}
    row.update(result.aggregate)
    return row


def aggregate_experiment_rows(rows: list[dict[str, float | str]]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows)
    numeric_cols = [c for c in frame.columns if c != "controller"]
    for col in numeric_cols:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return frame


def compute_system_efficiency_score(df: pd.DataFrame) -> pd.Series:
    # Higher throughput and fairness are better; lower queue and wait are better.
    throughput = _min_max(df["average_throughput"])
    fairness = _min_max(df["average_fairness"])
    queue_penalty = 1.0 - _min_max(df["average_queue_length"])
    wait_penalty = 1.0 - _min_max(df["average_wait_time"])
    return 0.35 * throughput + 0.25 * fairness + 0.2 * queue_penalty + 0.2 * wait_penalty


def _min_max(values: pd.Series) -> pd.Series:
    low = values.min()
    high = values.max()
    if high - low < 1e-9:
        return pd.Series(np.ones(len(values)), index=values.index)
    return (values - low) / (high - low)

