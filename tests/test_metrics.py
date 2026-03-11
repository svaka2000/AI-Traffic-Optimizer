"""tests/test_metrics.py

Tests for traffic_ai.simulation.metrics.EpisodeMetricsTracker.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from traffic_ai.simulation.metrics import (
    CO2_KG_PER_MIN_PER_VEHICLE,
    EpisodeMetricsTracker,
    StepRecord,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tracker(tmp_path: Path) -> EpisodeMetricsTracker:
    return EpisodeMetricsTracker(
        output_dir=tmp_path / "metrics",
        controller_name="test_ctrl",
    )


def _make_info(
    total_queue: float = 10.0,
    avg_wait: float = 15.0,
    throughput: float = 5.0,
    green_switches: int = 2,
    spillback_events: int = 1,
    step_seconds: float = 1.0,
) -> dict:
    return {
        "total_queue": total_queue,
        "avg_wait": avg_wait,
        "throughput": throughput,
        "green_switches": green_switches,
        "spillback_events": spillback_events,
        "step_seconds": step_seconds,
    }


# ---------------------------------------------------------------------------
# CO2 estimate tests
# ---------------------------------------------------------------------------

def test_co2_estimate_nonnegative(tracker: EpisodeMetricsTracker, tmp_path: Path) -> None:
    """CO2 estimate should always be ≥ 0."""
    for step in range(10):
        info = _make_info(total_queue=float(step))
        record = tracker.record_step(step, info)
        assert record.estimated_co2_grams >= 0.0


def test_co2_estimate_scales_linearly_with_idle_time(tmp_path: Path) -> None:
    """CO2 estimate should scale linearly with total_queue × step_seconds."""
    t1 = EpisodeMetricsTracker(output_dir=tmp_path / "m1", controller_name="c1")
    t2 = EpisodeMetricsTracker(output_dir=tmp_path / "m2", controller_name="c2")

    # t2 has 2× the queue → should have 2× the CO2
    info1 = _make_info(total_queue=10.0, step_seconds=1.0)
    info2 = _make_info(total_queue=20.0, step_seconds=1.0)

    r1 = t1.record_step(0, info1)
    r2 = t2.record_step(0, info2)

    # CO2 = queue × (step_seconds/60) × CO2_KG_PER_MIN × 1000
    expected_ratio = 2.0
    assert abs(r2.estimated_co2_grams / max(r1.estimated_co2_grams, 1e-9) - expected_ratio) < 0.01


def test_co2_formula_matches_constant(tmp_path: Path) -> None:
    """Verify the exact CO2 formula: idle_time_min × CO2_KG_PER_MIN × 1000."""
    tracker = EpisodeMetricsTracker(output_dir=tmp_path / "m3", controller_name="c3")
    total_queue = 5.0
    step_seconds = 60.0  # 1 minute per step
    expected_co2 = total_queue * (step_seconds / 60.0) * CO2_KG_PER_MIN_PER_VEHICLE * 1_000.0

    info = _make_info(total_queue=total_queue, step_seconds=step_seconds)
    record = tracker.record_step(0, info)
    assert abs(record.estimated_co2_grams - expected_co2) < 0.001


# ---------------------------------------------------------------------------
# CSV export tests
# ---------------------------------------------------------------------------

def test_step_csv_written_after_record(tracker: EpisodeMetricsTracker) -> None:
    """Step CSV should exist after at least one step record."""
    info = _make_info()
    tracker.record_step(0, info)
    tracker._flush_step_csv()
    assert tracker.step_csv_path.exists()


def test_episode_csv_written_after_end_episode(tracker: EpisodeMetricsTracker) -> None:
    """Episode summary CSV should be written after end_episode()."""
    for step in range(5):
        tracker.record_step(step, _make_info(total_queue=float(step + 1)))
    summary = tracker.end_episode()
    assert tracker.episode_csv_path.exists()
    assert summary.n_steps == 5
    assert summary.controller == "test_ctrl"


def test_metrics_csv_written_after_each_episode(tmp_path: Path) -> None:
    """CSV should be appended after every episode."""
    t = EpisodeMetricsTracker(output_dir=tmp_path / "ep_test", controller_name="ctrl")
    for ep in range(3):
        for step in range(4):
            t.record_step(step, _make_info())
        t.end_episode()
    assert t.episode_csv_path.exists()
    import csv
    with open(t.episode_csv_path) as fh:
        rows = list(csv.DictReader(fh))
    assert len(rows) == 3, "Should have one row per episode"


def test_step_records_accumulate(tracker: EpisodeMetricsTracker) -> None:
    for step in range(8):
        tracker.record_step(step, _make_info())
    assert len(tracker.all_step_records) == 8


def test_episode_summary_mean_values(tmp_path: Path) -> None:
    """Mean avg_wait_time in summary should equal mean of individual step records."""
    t = EpisodeMetricsTracker(output_dir=tmp_path / "mean_test", controller_name="c")
    waits = [5.0, 10.0, 15.0, 20.0]
    for step, w in enumerate(waits):
        t.record_step(step, _make_info(avg_wait=w))
    summary = t.end_episode()
    expected_mean = sum(waits) / len(waits)
    assert abs(summary.mean_avg_wait_time - expected_mean) < 0.001


def test_multiple_episodes_increase_episode_count(tmp_path: Path) -> None:
    t = EpisodeMetricsTracker(output_dir=tmp_path / "multi_ep", controller_name="c")
    for _ in range(5):
        for step in range(3):
            t.record_step(step, _make_info())
        t.end_episode()
    assert len(t.all_episode_summaries) == 5
    assert t._episode == 5
