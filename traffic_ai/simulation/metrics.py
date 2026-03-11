"""traffic_ai/simulation/metrics.py

Per-step and per-episode metrics tracker for the MultiIntersectionNetwork.
Exports metrics to CSV after every episode.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


CO2_KG_PER_MIN_PER_VEHICLE: float = 0.0274  # idle CO2 proxy


@dataclass
class StepRecord:
    """Snapshot of simulation metrics at a single step."""

    episode: int
    step: int
    controller: str
    avg_wait_time: float
    max_queue_length: float
    throughput: float  # vehicles/min
    estimated_co2_grams: float
    green_light_switches: int
    spillback_events: int


@dataclass
class EpisodeSummary:
    """Aggregate metrics over a full episode."""

    episode: int
    controller: str
    mean_avg_wait_time: float
    mean_max_queue_length: float
    mean_throughput: float
    total_co2_grams: float
    total_green_switches: int
    total_spillback_events: int
    n_steps: int


_STEP_FIELDS: list[str] = [
    "episode", "step", "controller", "avg_wait_time",
    "max_queue_length", "throughput", "estimated_co2_grams",
    "green_light_switches", "spillback_events",
]

_EPISODE_FIELDS: list[str] = [
    "episode", "controller", "mean_avg_wait_time", "mean_max_queue_length",
    "mean_throughput", "total_co2_grams", "total_green_switches",
    "total_spillback_events", "n_steps",
]


class EpisodeMetricsTracker:
    """Track per-step and per-episode metrics and export to CSV.

    Parameters
    ----------
    output_dir:
        Directory where CSV files are written.
    controller_name:
        Label for the current controller (written to CSV).
    """

    def __init__(
        self,
        output_dir: str | Path = "artifacts/results",
        controller_name: str = "unknown",
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.controller_name = controller_name
        self._episode: int = 0
        self._step_records: list[StepRecord] = []
        self._episode_summaries: list[EpisodeSummary] = []
        self._current_episode_steps: list[StepRecord] = []

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record_step(
        self,
        step: int,
        info: dict[str, Any],
        prev_switches: int = 0,
    ) -> StepRecord:
        """Record metrics for a single simulation step.

        Parameters
        ----------
        step:
            Current step index within the episode.
        info:
            Info dict returned by ``MultiIntersectionNetwork.step()``.
        prev_switches:
            Green-light switch count at the previous step (used to compute delta).
        """
        total_queue = float(info.get("total_queue", 0.0))
        avg_wait = float(info.get("avg_wait", 0.0))
        throughput_veh_per_step = float(info.get("throughput", 0.0))
        # Convert throughput from veh/step to veh/min
        step_seconds: float = float(info.get("step_seconds", 1.0))
        throughput_veh_per_min = throughput_veh_per_step * (60.0 / max(step_seconds, 1e-9))

        # Idle time proxy: vehicles in queue × step duration in minutes
        idle_time_min = total_queue * (step_seconds / 60.0)
        co2_grams = idle_time_min * CO2_KG_PER_MIN_PER_VEHICLE * 1_000.0

        switches_total = int(info.get("green_switches", 0))
        switches_delta = max(0, switches_total - prev_switches)
        spillback = int(info.get("spillback_events", 0))

        record = StepRecord(
            episode=self._episode,
            step=step,
            controller=self.controller_name,
            avg_wait_time=avg_wait,
            max_queue_length=total_queue,
            throughput=throughput_veh_per_min,
            estimated_co2_grams=co2_grams,
            green_light_switches=switches_delta,
            spillback_events=spillback,
        )
        self._step_records.append(record)
        self._current_episode_steps.append(record)
        return record

    def end_episode(self) -> EpisodeSummary:
        """Finalise the current episode, compute summary, and write CSV."""
        steps = self._current_episode_steps
        n = max(len(steps), 1)

        summary = EpisodeSummary(
            episode=self._episode,
            controller=self.controller_name,
            mean_avg_wait_time=sum(s.avg_wait_time for s in steps) / n,
            mean_max_queue_length=sum(s.max_queue_length for s in steps) / n,
            mean_throughput=sum(s.throughput for s in steps) / n,
            total_co2_grams=sum(s.estimated_co2_grams for s in steps),
            total_green_switches=sum(s.green_light_switches for s in steps),
            total_spillback_events=sum(s.spillback_events for s in steps),
            n_steps=len(steps),
        )
        self._episode_summaries.append(summary)
        self._current_episode_steps = []
        self._episode += 1

        self._flush_step_csv(steps)
        self._flush_episode_csv(summary)
        return summary

    # ------------------------------------------------------------------
    # CSV export (append only the new records each call)
    # ------------------------------------------------------------------

    def _flush_step_csv(self, new_records: list[StepRecord] | None = None) -> Path:
        path = self.output_dir / "step_metrics.csv"
        write_header = not path.exists()
        records_to_write = new_records if new_records is not None else self._step_records
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_STEP_FIELDS)
            if write_header:
                writer.writeheader()
            for rec in records_to_write:
                writer.writerow(vars(rec))
        return path

    def _flush_episode_csv(self, new_summary: EpisodeSummary | None = None) -> Path:
        path = self.output_dir / "episode_summaries.csv"
        write_header = not path.exists()
        summaries_to_write = [new_summary] if new_summary is not None else self._episode_summaries
        with path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=_EPISODE_FIELDS)
            if write_header:
                writer.writeheader()
            for summary in summaries_to_write:
                writer.writerow(vars(summary))
        return path

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def step_csv_path(self) -> Path:
        return self.output_dir / "step_metrics.csv"

    @property
    def episode_csv_path(self) -> Path:
        return self.output_dir / "episode_summaries.csv"

    @property
    def all_step_records(self) -> list[StepRecord]:
        return list(self._step_records)

    @property
    def all_episode_summaries(self) -> list[EpisodeSummary]:
        return list(self._episode_summaries)
