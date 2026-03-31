"""traffic_ai/validation/rosecrans_validator.py

Rosecrans Corridor Validation.

Compares AITO simulation results against the verified real-world results
from the 2017 Rosecrans Street InSync deployment in San Diego:

  Source: San Diego Mayor Kevin Faulconer press release, March 2017.
          Verified by City of San Diego Transportation Department.
  Corridor: Rosecrans St, Hancock St to Nimitz Blvd (12 signals)
  Results:
    - 25% travel time reduction during rush hour
    - 53% stop reduction

This validator runs the Rosecrans scenario with GreedyAdaptive and
FixedTiming controllers, computes the improvement, and reports how close
the simulation result is to the real-world benchmark.

A result within ±10 percentage points of the real-world figure indicates
the simulation demand model is well-calibrated to San Diego conditions.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.greedy_adaptive import GreedyAdaptiveController
from traffic_ai.data_pipeline.pems_connector import PeMSConnector
from traffic_ai.scenarios.san_diego import SanDiegoScenario
from traffic_ai.simulation_engine.engine import TrafficNetworkSimulator

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Real-world benchmark constants
# Source: San Diego Mayor Kevin Faulconer, March 2017.
#         City of San Diego Transportation Department.
# Corridor: Rosecrans St, Hancock St to Nimitz Blvd (12 signals)
# System: In|Sync by Rhythm Engineering
# ---------------------------------------------------------------------------

ROSECRANS_BENCHMARK: dict[str, object] = {
    "travel_time_reduction_pct": 25.0,
    "stop_reduction_pct": 53.0,
    "source": (
        "San Diego Mayor Kevin Faulconer, March 2017. "
        "City of San Diego Transportation Department."
    ),
    "corridor": "Rosecrans St, Hancock St to Nimitz Blvd (12 signals)",
    "system": "In|Sync by Rhythm Engineering",
    "validation_tolerance_pct": 10.0,  # ±10 pp considered well-calibrated
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Results of the Rosecrans corridor validation run."""

    # Simulation results
    fixed_avg_wait_sec: float
    greedy_avg_wait_sec: float
    simulated_improvement_pct: float

    # Real-world benchmark
    benchmark_improvement_pct: float
    benchmark_source: str

    # Calibration assessment
    within_tolerance: bool
    gap_pct: float          # |simulated - benchmark|

    # Data source used
    demand_source: str      # "real_pems: ..." or "synthetic: ..."
    n_simulation_steps: int
    seed: int

    def summary_lines(self) -> list[str]:
        """Return human-readable summary lines for CLI output."""
        tolerance = float(ROSECRANS_BENCHMARK["validation_tolerance_pct"])
        lines = [
            "=" * 60,
            "AITO — Rosecrans Corridor Validation",
            "=" * 60,
            f"Demand source:    {self.demand_source}",
            f"Simulation steps: {self.n_simulation_steps}",
            "",
            "SIMULATION RESULTS:",
            f"  FixedTiming avg wait:    {self.fixed_avg_wait_sec:.2f} s",
            f"  GreedyAdaptive avg wait: {self.greedy_avg_wait_sec:.2f} s",
            f"  Simulated improvement:   {self.simulated_improvement_pct:.1f}%",
            "",
            "REAL-WORLD BENCHMARK (2017):",
            f"  Travel time reduction:   {self.benchmark_improvement_pct:.1f}%",
            f"  Source: {self.benchmark_source}",
            "",
            "CALIBRATION ASSESSMENT:",
            f"  Gap from benchmark:      {self.gap_pct:.1f} percentage points",
        ]
        if self.within_tolerance:
            lines += [
                f"  STATUS: PASS — within ±{tolerance:.0f} pp tolerance",
                "  Simulation is well-calibrated to San Diego conditions.",
            ]
        else:
            lines += [
                f"  STATUS: OUTSIDE TOLERANCE — gap exceeds ±{tolerance:.0f} pp",
                "  Consider downloading real PeMS data to improve calibration.",
            ]
        lines.append("=" * 60)
        return lines


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------

class RosecransValidator:
    """Run the Rosecrans corridor validation experiment.

    Compares GreedyAdaptiveController (InSync-style) against
    FixedTimingController on the 12-signal Rosecrans corridor scenario,
    calibrated to real PeMS demand data when available.

    Parameters
    ----------
    raw_data_dir:
        Directory to scan for PeMS CSV files (``data/raw/``).
    output_dir:
        Directory to write ``validation_report.json``.
    n_steps:
        Simulation steps per run (default 2000 ≈ 33 minutes real time).
    seed:
        Random seed for reproducibility.
    """

    def __init__(
        self,
        raw_data_dir: Path,
        output_dir: Path,
        n_steps: int = 2000,
        seed: int = 42,
    ) -> None:
        self.raw_data_dir = Path(raw_data_dir)
        self.output_dir = Path(output_dir)
        self.n_steps = n_steps
        self.seed = seed
        self.connector = PeMSConnector(station_id=400456)

    def run(self) -> ValidationResult:
        """Execute validation and return results.

        Steps
        -----
        1. Load demand calibration from real PeMS CSV or synthetic fallback.
        2. Build Rosecrans scenario config.
        3. Run FixedTimingController for baseline.
        4. Run GreedyAdaptiveController (InSync-style).
        5. Compute % improvement and compare against 25% benchmark.
        6. Save JSON report to output_dir/validation_report.json.
        """
        # 1. Demand calibration (real PeMS or synthetic)
        df, demand_source = self.connector.load_best_available(self.raw_data_dir)

        if df is not None:
            hourly_profile: dict[int, float] | None = (
                self.connector.compute_hourly_demand_profile(df)
            )
        else:
            hourly_profile = None

        # 2. Rosecrans scenario config
        config = SanDiegoScenario.rosecrans_corridor()
        config.steps = self.n_steps
        config.seed = self.seed

        # 3. FixedTiming baseline
        logger.info("Rosecrans validation: running FixedTimingController…")
        fixed_sim = TrafficNetworkSimulator(config)
        if hourly_profile:
            fixed_sim.demand.calibrate_from_pems_profile(hourly_profile)
        fixed_result = fixed_sim.run(FixedTimingController(), steps=self.n_steps)
        fixed_wait = float(fixed_result.aggregate.get("average_wait_time", 0.0))

        # 4. GreedyAdaptive (InSync-style)
        logger.info("Rosecrans validation: running GreedyAdaptiveController…")
        greedy_sim = TrafficNetworkSimulator(config)
        if hourly_profile:
            greedy_sim.demand.calibrate_from_pems_profile(hourly_profile)
        greedy_result = greedy_sim.run(GreedyAdaptiveController(), steps=self.n_steps)
        greedy_wait = float(greedy_result.aggregate.get("average_wait_time", 0.0))

        # 5. Compute improvement
        if fixed_wait > 0:
            improvement = (fixed_wait - greedy_wait) / fixed_wait * 100.0
        else:
            improvement = 0.0

        # 6. Compare against benchmark
        benchmark_pct = float(ROSECRANS_BENCHMARK["travel_time_reduction_pct"])
        gap = abs(improvement - benchmark_pct)
        tolerance = float(ROSECRANS_BENCHMARK["validation_tolerance_pct"])

        result = ValidationResult(
            fixed_avg_wait_sec=fixed_wait,
            greedy_avg_wait_sec=greedy_wait,
            simulated_improvement_pct=float(improvement),
            benchmark_improvement_pct=benchmark_pct,
            benchmark_source=str(ROSECRANS_BENCHMARK["source"]),
            within_tolerance=gap <= tolerance,
            gap_pct=float(gap),
            demand_source=demand_source,
            n_simulation_steps=self.n_steps,
            seed=self.seed,
        )

        self._save_report(result)
        return result

    def _save_report(self, result: ValidationResult) -> None:
        """Write validation_report.json to output_dir."""
        report = {
            "validation_type": "rosecrans_corridor",
            "corridor": str(ROSECRANS_BENCHMARK["corridor"]),
            "simulation": {
                "fixed_timing_avg_wait_sec": result.fixed_avg_wait_sec,
                "greedy_adaptive_avg_wait_sec": result.greedy_avg_wait_sec,
                "improvement_pct": result.simulated_improvement_pct,
                "demand_source": result.demand_source,
                "n_steps": result.n_simulation_steps,
                "seed": result.seed,
            },
            "real_world_benchmark": {
                "travel_time_reduction_pct": result.benchmark_improvement_pct,
                "stop_reduction_pct": float(ROSECRANS_BENCHMARK["stop_reduction_pct"]),
                "source": result.benchmark_source,
                "system": str(ROSECRANS_BENCHMARK["system"]),
            },
            "calibration_assessment": {
                "gap_percentage_points": result.gap_pct,
                "tolerance_percentage_points": float(
                    ROSECRANS_BENCHMARK["validation_tolerance_pct"]
                ),
                "within_tolerance": result.within_tolerance,
            },
        }

        path = self.output_dir / "validation_report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info("Validation report saved to %s", path)
