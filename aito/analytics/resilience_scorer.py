"""aito/analytics/resilience_scorer.py

GF11: Network Resilience Scoring.

Quantifies how well the AITO corridor maintains performance under
adverse conditions: sensor failures, high demand, incidents, and
probe data sparsity.

Resilience dimensions (FHWA Resilience Framework, 2022):
  1. Sensor Resilience    — performance at 0% loop detector availability
  2. Probe Resilience     — performance at low CV penetration (5–15%)
  3. Demand Resilience    — performance under demand surges (1.2x–2.0x)
  4. Incident Resilience  — performance with one approach blocked
  5. Recovery Speed       — time to return to baseline after disturbance

Score: 0–100, composite across all dimensions.
  90–100: Excellent (outperforms InSync under stress)
  70–89:  Good (maintains LOS C or better under stress)
  50–69:  Fair (degrades to LOS D under stress)
  < 50:   Poor (LOS E/F, needs hardware investment)

Reference:
  FHWA. (2022). Resilient Transportation Systems: A Framework for
  Planning and Operations. FHWA-HOP-22-008.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


class ResilienceGrade(str, Enum):
    EXCELLENT = "A"   # 90–100
    GOOD      = "B"   # 70–89
    FAIR      = "C"   # 50–69
    POOR      = "D"   # < 50


@dataclass
class ResilienceDimensionScore:
    """Score for one resilience dimension."""
    dimension: str
    score: float              # 0–100
    baseline_metric: float    # performance under normal conditions
    stressed_metric: float    # performance under stress
    degradation_pct: float    # how much worse under stress
    notes: str = ""

    @property
    def grade(self) -> ResilienceGrade:
        if self.score >= 90:
            return ResilienceGrade.EXCELLENT
        if self.score >= 70:
            return ResilienceGrade.GOOD
        if self.score >= 50:
            return ResilienceGrade.FAIR
        return ResilienceGrade.POOR


@dataclass
class ResilienceReport:
    """Complete resilience assessment for one corridor."""
    corridor_id: str
    corridor_name: str
    assessed_at: datetime

    sensor_score: ResilienceDimensionScore
    probe_score: ResilienceDimensionScore
    demand_score: ResilienceDimensionScore
    incident_score: ResilienceDimensionScore
    recovery_score: ResilienceDimensionScore

    @property
    def composite_score(self) -> float:
        weights = [0.25, 0.20, 0.25, 0.20, 0.10]
        scores = [
            self.sensor_score.score,
            self.probe_score.score,
            self.demand_score.score,
            self.incident_score.score,
            self.recovery_score.score,
        ]
        return sum(w * s for w, s in zip(weights, scores))

    @property
    def overall_grade(self) -> ResilienceGrade:
        score = self.composite_score
        if score >= 90:
            return ResilienceGrade.EXCELLENT
        if score >= 70:
            return ResilienceGrade.GOOD
        if score >= 50:
            return ResilienceGrade.FAIR
        return ResilienceGrade.POOR

    def summary(self) -> dict:
        return {
            "corridor": self.corridor_name,
            "composite_score": round(self.composite_score, 1),
            "overall_grade": self.overall_grade.value,
            "dimensions": {
                "sensor": round(self.sensor_score.score, 1),
                "probe_data": round(self.probe_score.score, 1),
                "demand_surge": round(self.demand_score.score, 1),
                "incident": round(self.incident_score.score, 1),
                "recovery": round(self.recovery_score.score, 1),
            },
            "weakest_dimension": min(
                [self.sensor_score, self.probe_score, self.demand_score,
                 self.incident_score, self.recovery_score],
                key=lambda d: d.score,
            ).dimension,
        }


# ---------------------------------------------------------------------------
# Individual resilience assessors
# ---------------------------------------------------------------------------

def score_sensor_resilience(
    n_detectors: int,
    n_operational: int,
    baseline_delay_s_veh: float,
    probe_penetration_rate: float = 0.28,
) -> ResilienceDimensionScore:
    """Score resilience to loop detector failures.

    AITO with probe data should maintain performance even at 0% detector availability.
    InSync degrades rapidly below 70% detector availability.
    """
    if n_detectors <= 0:
        # No detectors configured — fully probe-data-based
        availability = 0.0
        detector_dependency = 0.0
    else:
        availability = n_operational / n_detectors
        # Detector dependency: how much does performance rely on detectors?
        # With CV penetration >= 0.25, dependency is low
        detector_dependency = max(0.0, 1.0 - probe_penetration_rate * 3.0)

    # Stressed delay: if high detector dependency and low availability
    stress_factor = 1.0 + detector_dependency * (1.0 - availability) * 0.40
    stressed_delay = baseline_delay_s_veh * stress_factor
    degradation = (stressed_delay - baseline_delay_s_veh) / max(baseline_delay_s_veh, 1.0)

    # Score: 100 at zero degradation, linear decline
    score = max(0.0, 100.0 - degradation * 200.0)

    return ResilienceDimensionScore(
        dimension="sensor",
        score=round(score, 1),
        baseline_metric=baseline_delay_s_veh,
        stressed_metric=round(stressed_delay, 1),
        degradation_pct=round(degradation * 100, 1),
        notes=(
            f"{n_operational}/{n_detectors} detectors operational "
            f"({availability * 100:.0f}% availability), "
            f"probe penetration {probe_penetration_rate * 100:.0f}%"
        ),
    )


def score_probe_resilience(
    baseline_delay_s_veh: float,
    min_viable_penetration: float = 0.05,
    current_penetration: float = 0.28,
    delay_at_min_penetration: Optional[float] = None,
) -> ResilienceDimensionScore:
    """Score resilience when probe data becomes sparse.

    At 5% penetration (early EV adoption phase), how well does AITO still work?
    """
    # At minimum penetration, delay increases due to estimation uncertainty
    if delay_at_min_penetration is None:
        # Model: delay increases as 1/sqrt(penetration)
        stress_factor = math.sqrt(current_penetration / max(min_viable_penetration, 0.01))
        stressed_delay = baseline_delay_s_veh * min(3.0, stress_factor)
    else:
        stressed_delay = delay_at_min_penetration

    degradation = (stressed_delay - baseline_delay_s_veh) / max(baseline_delay_s_veh, 1.0)
    score = max(0.0, 100.0 - degradation * 150.0)

    return ResilienceDimensionScore(
        dimension="probe_data",
        score=round(score, 1),
        baseline_metric=baseline_delay_s_veh,
        stressed_metric=round(stressed_delay, 1),
        degradation_pct=round(degradation * 100, 1),
        notes=f"Probe penetration from {current_penetration * 100:.0f}% → {min_viable_penetration * 100:.0f}%",
    )


def score_demand_resilience(
    baseline_delay_s_veh: float,
    surge_factor: float = 1.5,
    n_intersections: int = 12,
) -> ResilienceDimensionScore:
    """Score resilience to demand surges (event, crash diversion).

    Uses HCM overflow delay model: delay grows sharply above v/c = 0.85.
    """
    # Simplified: at 1.5x demand, delay typically increases 2–4x for v/c > 0.85
    if surge_factor <= 1.0:
        stressed_delay = baseline_delay_s_veh
    elif surge_factor <= 1.2:
        stressed_delay = baseline_delay_s_veh * (1.0 + (surge_factor - 1.0) * 2.0)
    elif surge_factor <= 1.5:
        stressed_delay = baseline_delay_s_veh * (1.0 + (surge_factor - 1.0) * 4.0)
    else:
        stressed_delay = baseline_delay_s_veh * (1.0 + (surge_factor - 1.0) * 8.0)

    degradation = (stressed_delay - baseline_delay_s_veh) / max(baseline_delay_s_veh, 1.0)
    score = max(0.0, 100.0 - degradation * 100.0)

    return ResilienceDimensionScore(
        dimension="demand_surge",
        score=round(score, 1),
        baseline_metric=baseline_delay_s_veh,
        stressed_metric=round(stressed_delay, 1),
        degradation_pct=round(degradation * 100, 1),
        notes=f"Demand surge {surge_factor:.1f}x baseline across {n_intersections} intersections",
    )


def score_incident_resilience(
    baseline_delay_s_veh: float,
    n_intersections: int = 12,
    incident_capacity_reduction: float = 0.50,
) -> ResilienceDimensionScore:
    """Score resilience when one approach is blocked (incident/construction)."""
    # One intersection loses 50% capacity → queue builds → upstream spillback
    # Spillback propagates: simplified linear model
    n_affected = min(3, max(1, n_intersections // 4))
    avg_delay_increase = incident_capacity_reduction * 2.0 * (n_affected / n_intersections)
    stressed_delay = baseline_delay_s_veh * (1.0 + avg_delay_increase)

    degradation = (stressed_delay - baseline_delay_s_veh) / max(baseline_delay_s_veh, 1.0)
    score = max(0.0, 100.0 - degradation * 150.0)

    return ResilienceDimensionScore(
        dimension="incident",
        score=round(score, 1),
        baseline_metric=baseline_delay_s_veh,
        stressed_metric=round(stressed_delay, 1),
        degradation_pct=round(degradation * 100, 1),
        notes=f"One approach at {(1-incident_capacity_reduction)*100:.0f}% capacity, {n_affected} downstream intersections affected",
    )


def score_recovery_resilience(
    incident_duration_min: float = 30.0,
    retiming_interval_min: float = 5.0,
    auto_retiming_enabled: bool = True,
) -> ResilienceDimensionScore:
    """Score how quickly the system recovers to baseline after a disturbance.

    With GF7 continuous retiming, AITO re-optimizes within 5 minutes of
    detecting performance degradation.  Manual systems take hours.
    """
    if auto_retiming_enabled:
        # AITO retiming: detect in ~5 min, reoptimize in ~2 min, deploy in ~1 min
        recovery_time_min = retiming_interval_min + 3.0
    else:
        # Manual: engineer must be called, review, upload plan
        recovery_time_min = 120.0  # typical manual response time

    # Score based on recovery time vs. incident duration
    recovery_ratio = recovery_time_min / max(incident_duration_min, 1.0)
    score = max(0.0, 100.0 - recovery_ratio * 50.0)

    return ResilienceDimensionScore(
        dimension="recovery",
        score=round(score, 1),
        baseline_metric=incident_duration_min,
        stressed_metric=recovery_time_min,
        degradation_pct=0.0,  # not applicable for recovery metric
        notes=f"Recovery time: {recovery_time_min:.0f} min (auto-retiming: {auto_retiming_enabled})",
    )


# ---------------------------------------------------------------------------
# ResilienceScorer — main class
# ---------------------------------------------------------------------------

class ResilienceScorer:
    """Assess AITO corridor resilience across all dimensions.

    Usage:
        scorer = ResilienceScorer(corridor)
        report = scorer.score(
            baseline_delay_s_veh=28.5,
            probe_penetration_rate=0.28,
        )
        print(f"Overall grade: {report.overall_grade.value} ({report.composite_score:.1f}/100)")
    """

    def __init__(self, corridor) -> None:
        self.corridor = corridor

    def score(
        self,
        baseline_delay_s_veh: float = 30.0,
        probe_penetration_rate: float = 0.28,
        n_detectors: int = 0,
        n_operational_detectors: int = 0,
        demand_surge_factor: float = 1.5,
        incident_capacity_reduction: float = 0.50,
        auto_retiming_enabled: bool = True,
        retiming_interval_min: float = 5.0,
    ) -> ResilienceReport:
        """Compute full resilience report."""
        n = len(self.corridor.intersections)

        sensor = score_sensor_resilience(
            n_detectors=n_detectors,
            n_operational=n_operational_detectors,
            baseline_delay_s_veh=baseline_delay_s_veh,
            probe_penetration_rate=probe_penetration_rate,
        )
        probe = score_probe_resilience(
            baseline_delay_s_veh=baseline_delay_s_veh,
            current_penetration=probe_penetration_rate,
        )
        demand = score_demand_resilience(
            baseline_delay_s_veh=baseline_delay_s_veh,
            surge_factor=demand_surge_factor,
            n_intersections=n,
        )
        incident = score_incident_resilience(
            baseline_delay_s_veh=baseline_delay_s_veh,
            n_intersections=n,
            incident_capacity_reduction=incident_capacity_reduction,
        )
        recovery = score_recovery_resilience(
            auto_retiming_enabled=auto_retiming_enabled,
            retiming_interval_min=retiming_interval_min,
        )

        return ResilienceReport(
            corridor_id=self.corridor.id,
            corridor_name=self.corridor.name,
            assessed_at=datetime.utcnow(),
            sensor_score=sensor,
            probe_score=probe,
            demand_score=demand,
            incident_score=incident,
            recovery_score=recovery,
        )

    def compare_vs_insync(
        self,
        baseline_delay_s_veh: float = 30.0,
        probe_penetration_rate: float = 0.28,
    ) -> dict:
        """Compare AITO resilience against InSync degradation characteristics.

        InSync degradation model based on published SDOT after-study data:
        - 0% detectors → 40% delay increase
        - 50% detectors → 15% delay increase
        - 1.5x demand → 60% delay increase
        """
        aito_report = self.score(
            baseline_delay_s_veh=baseline_delay_s_veh,
            probe_penetration_rate=probe_penetration_rate,
            auto_retiming_enabled=True,
        )

        insync_sensor_degraded = baseline_delay_s_veh * 1.40  # 0% detectors
        insync_demand_degraded = baseline_delay_s_veh * 1.60  # 1.5x demand
        insync_recovery_min = 120.0  # manual retiming

        return {
            "aito_composite_score": round(aito_report.composite_score, 1),
            "aito_grade": aito_report.overall_grade.value,
            "sensor_failure": {
                "aito_delay_s_veh": round(aito_report.sensor_score.stressed_metric, 1),
                "insync_delay_s_veh": round(insync_sensor_degraded, 1),
                "aito_better_by_pct": round(
                    (insync_sensor_degraded - aito_report.sensor_score.stressed_metric)
                    / insync_sensor_degraded * 100, 1
                ),
            },
            "demand_surge_1.5x": {
                "aito_delay_s_veh": round(aito_report.demand_score.stressed_metric, 1),
                "insync_delay_s_veh": round(insync_demand_degraded, 1),
            },
            "recovery_time_min": {
                "aito": aito_report.recovery_score.stressed_metric,
                "insync": insync_recovery_min,
                "aito_faster_by_min": round(
                    insync_recovery_min - aito_report.recovery_score.stressed_metric, 0
                ),
            },
        }
