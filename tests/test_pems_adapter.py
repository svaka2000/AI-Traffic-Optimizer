"""tests/test_pems_adapter.py

Tests for the Phase 9A PeMS CSV adapter:
  PeMSConnector.load_from_csv
  PeMSConnector.compute_hourly_demand_profile
  PeMSConnector.auto_detect_pems_files
  PeMSConnector.load_best_available
"""
from __future__ import annotations

import warnings
from io import StringIO
from pathlib import Path

import pandas as pd
import pytest

from traffic_ai.data_pipeline.pems_connector import PeMSConnector


# ---------------------------------------------------------------------------
# Helpers — build minimal synthetic PeMS CSV content
# ---------------------------------------------------------------------------

def _pems_csv_content(n_rows: int = 5, pct_observed: float = 1.0) -> str:
    """Return a minimal PeMS 5-Minute CSV string."""
    lines = [
        "Timestamp,Station,District,Freeway,Direction,Lane Type,"
        "Station Length,Samples,% Observed,Total Flow,Avg Occupancy,Avg Speed,"
        "Lane 1 Flow,Lane 1 Avg Occ,Lane 1 Avg Speed,Lane 1 Observed,"
        "Lane 2 Flow,Lane 2 Avg Occ,Lane 2 Avg Speed,Lane 2 Observed"
    ]
    for i in range(n_rows):
        hour = (7 + i) % 24
        ts = f"01/15/2024 {hour:02d}:00:00"
        lines.append(
            f"{ts},400456,11,5,N,ML,0.5,5,{pct_observed:.1f},120,0.15,55,"
            f"60,0.15,55,1,60,0.15,55,1"
        )
    return "\n".join(lines)


def _write_pems_csv(path: Path, content: str | None = None) -> Path:
    path.write_text(content or _pems_csv_content())
    return path


# ---------------------------------------------------------------------------
# Test 1: load_from_csv returns expected columns
# ---------------------------------------------------------------------------

def test_load_from_csv_returns_expected_columns(tmp_path: Path) -> None:
    """Parsed DataFrame has all required columns."""
    csv_path = _write_pems_csv(tmp_path / "pems_station_400456.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df = connector.load_from_csv(csv_path)

    expected = {
        "timestamp", "station_id", "hour_of_day", "day_of_week",
        "is_rush_hour", "total_flow_per_5min", "flow_per_lane_per_5min",
        "arrival_rate_per_sec", "occupancy", "avg_speed_mph",
        "pct_observed", "data_quality_ok",
    }
    assert expected.issubset(set(df.columns)), (
        f"Missing columns: {expected - set(df.columns)}"
    )
    assert len(df) == 5


# ---------------------------------------------------------------------------
# Test 2: compute_hourly_demand_profile returns 24-hour dict
# ---------------------------------------------------------------------------

def test_compute_hourly_profile_returns_24_hours(tmp_path: Path) -> None:
    """Profile dict has keys 0-23."""
    # Build a CSV with data across all hours of a weekday
    lines = [
        "Timestamp,Station,District,Freeway,Direction,Lane Type,"
        "Station Length,Samples,% Observed,Total Flow,Avg Occupancy,Avg Speed,"
        "Lane 1 Flow,Lane 1 Avg Occ,Lane 1 Avg Speed,Lane 1 Observed"
    ]
    # 3 weekdays × 24 hours = 72 rows so each hour has 3 days of data
    for day_offset in range(3):
        base_date = f"01/{15 + day_offset}/2024"
        for h in range(24):
            ts = f"{base_date} {h:02d}:00:00"
            lines.append(
                f"{ts},400456,11,5,N,ML,0.5,5,1.0,60,0.10,55,60,0.10,55,1"
            )
    csv_path = tmp_path / "pems_station_400456.csv"
    csv_path.write_text("\n".join(lines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df = connector.load_from_csv(csv_path)
    profile = connector.compute_hourly_demand_profile(df)

    assert set(profile.keys()) == set(range(24)), (
        f"Profile should have keys 0-23, got {sorted(profile.keys())}"
    )


# ---------------------------------------------------------------------------
# Test 3: data quality filter removes low-observed rows
# ---------------------------------------------------------------------------

def test_data_quality_filter_removes_low_observed(tmp_path: Path) -> None:
    """Rows with pct_observed < 0.5 are excluded from profile computation."""
    # 3 weekday-days of data: 1 good quality row per day at hour 8, 1 bad at hour 9
    lines = [
        "Timestamp,Station,District,Freeway,Direction,Lane Type,"
        "Station Length,Samples,% Observed,Total Flow,Avg Occupancy,Avg Speed,"
        "Lane 1 Flow,Lane 1 Avg Occ,Lane 1 Avg Speed,Lane 1 Observed"
    ]
    for day in range(3):
        date_str = f"01/{15 + day}/2024"
        # Hour 8 — good quality, high flow
        lines.append(
            f"{date_str} 08:00:00,400456,11,5,N,ML,0.5,5,1.0,300,0.20,55,300,0.20,55,1"
        )
        # Hour 9 — bad quality (pct_observed = 0.2), should be excluded
        lines.append(
            f"{date_str} 09:00:00,400456,11,5,N,ML,0.5,5,0.2,300,0.20,55,300,0.20,55,1"
        )
    csv_path = tmp_path / "pems_station_400456.csv"
    csv_path.write_text("\n".join(lines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df = connector.load_from_csv(csv_path)

    # Verify data_quality_ok is False for low-observed rows
    bad_rows = df[df["pct_observed"] < 0.5]
    assert (bad_rows["data_quality_ok"] == False).all(), (
        "Rows with pct_observed < 0.5 should have data_quality_ok=False"
    )

    # compute_hourly_demand_profile excludes those rows
    profile = connector.compute_hourly_demand_profile(df)
    # Hour 8 should have real data (3 good days); hour 9 only has bad-quality rows
    # → falls back to synthetic default 0.12
    assert profile[8] > 0.12, "Hour 8 with good data should exceed synthetic default"
    assert abs(profile[9] - 0.12) < 1e-6, "Hour 9 with bad data should fall back to 0.12"


# ---------------------------------------------------------------------------
# Test 4: arrival rate conversion is correct
# ---------------------------------------------------------------------------

def test_arrival_rate_conversion_is_correct(tmp_path: Path) -> None:
    """60 veh/5min total / 2 lanes / 300 sec = 0.10 veh/sec/lane."""
    lines = [
        "Timestamp,Station,District,Freeway,Direction,Lane Type,"
        "Station Length,Samples,% Observed,Total Flow,Avg Occupancy,Avg Speed,"
        "Lane 1 Flow,Lane 1 Avg Occ,Lane 1 Avg Speed,Lane 1 Observed,"
        "Lane 2 Flow,Lane 2 Avg Occ,Lane 2 Avg Speed,Lane 2 Observed",
        # 60 total flow, 2 lane columns → flow_per_lane = 30, rate = 30/300 = 0.10
        "01/15/2024 08:00:00,400456,11,5,N,ML,0.5,5,1.0,60,0.15,55,"
        "30,0.15,55,1,30,0.15,55,1",
    ]
    csv_path = tmp_path / "pems_station_400456.csv"
    csv_path.write_text("\n".join(lines))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df = connector.load_from_csv(csv_path)

    assert len(df) == 1
    # 60 veh / 2 lanes / 300 sec = 0.10 veh/sec/lane
    assert abs(df.iloc[0]["arrival_rate_per_sec"] - 0.10) < 1e-6, (
        f"Expected 0.10 veh/sec/lane, got {df.iloc[0]['arrival_rate_per_sec']}"
    )


# ---------------------------------------------------------------------------
# Test 5: auto_detect_pems_files finds matching CSVs
# ---------------------------------------------------------------------------

def test_auto_detect_finds_pems_files(tmp_path: Path) -> None:
    """auto_detect_pems_files finds pems_station_*.csv files."""
    (tmp_path / "pems_station_400456.csv").write_text("dummy")
    (tmp_path / "pems_station_999999.csv").write_text("dummy")
    (tmp_path / "other_data.csv").write_text("dummy")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    found = connector.auto_detect_pems_files(tmp_path)

    assert len(found) == 2
    names = {f.name for f in found}
    assert "pems_station_400456.csv" in names
    assert "pems_station_999999.csv" in names
    assert "other_data.csv" not in names


# ---------------------------------------------------------------------------
# Test 6: load_best_available returns synthetic when no files present
# ---------------------------------------------------------------------------

def test_load_best_available_synthetic_fallback(tmp_path: Path) -> None:
    """Returns synthetic source description when no PeMS files exist."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df, source = connector.load_best_available(tmp_path)

    assert df is None, "Should return None dataframe when no CSV found"
    assert "synthetic" in source.lower(), (
        f"Source should indicate synthetic fallback, got: {source!r}"
    )


# ---------------------------------------------------------------------------
# Test 7: load_best_available returns real_pems when CSV exists
# ---------------------------------------------------------------------------

def test_load_best_available_real_pems(tmp_path: Path) -> None:
    """Returns real_pems source description when valid CSV is present."""
    csv_path = _write_pems_csv(tmp_path / "pems_station_400456.csv")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        connector = PeMSConnector(station_id=400456)
    df, source = connector.load_best_available(tmp_path)

    assert df is not None, "Should return a DataFrame when CSV is found"
    assert "real_pems" in source.lower(), (
        f"Source should indicate real PeMS data, got: {source!r}"
    )
