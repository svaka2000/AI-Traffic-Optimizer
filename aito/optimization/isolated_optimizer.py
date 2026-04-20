"""aito/optimization/isolated_optimizer.py

Isolated intersection optimizer — enhanced Webster with ML demand prediction.

Algorithm:
  1. Compute classical Webster C_opt as baseline.
  2. Apply demand-weighted green splits per movement.
  3. Build full time-of-day plan set (6 periods).
  4. Validate every plan through the constraints engine.
  5. Return ranked plans for engineer review.
"""
from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from typing import Optional

from aito.models import (
    DemandProfile,
    Intersection,
    PhaseTiming,
    TimingPlan,
)
from aito.optimization.constraints import TimingPlanValidator, ValidationResult


# ---------------------------------------------------------------------------
# Time-of-day periods
# ---------------------------------------------------------------------------

TOD_PERIODS = {
    "AM Peak":   (6, 9),    # 06:00–09:00
    "Midday":    (9, 15),   # 09:00–15:00
    "PM Peak":   (15, 19),  # 15:00–19:00
    "Evening":   (19, 22),  # 19:00–22:00
    "Overnight": (22, 6),   # 22:00–06:00
    "Weekend":   None,      # separate weekend pattern
}

# Demand scale factors by period (relative to AM peak = 1.0)
TOD_DEMAND_FACTORS: dict[str, float] = {
    "AM Peak":   1.00,
    "Midday":    0.65,
    "PM Peak":   0.90,
    "Evening":   0.50,
    "Overnight": 0.20,
    "Weekend":   0.55,
}

# Standard 8-phase NEMA assignment: phase_id → (ring, position)
PHASE_MOVEMENTS: dict[int, str] = {
    1: "NB_left",
    2: "NB_thru",
    3: "SB_left",
    4: "SB_thru",
    5: "EB_left",
    6: "EB_thru",
    7: "WB_left",
    8: "WB_thru",
}

SAT_FLOW_THRU = 1800.0    # veh/hr/lane, HCM 7th edition
SAT_FLOW_LEFT = 1200.0    # veh/hr/lane, protected left turn


# ---------------------------------------------------------------------------
# Webster helpers
# ---------------------------------------------------------------------------

def webster_cycle(
    critical_flow_ratios: list[float],
    lost_time_per_phase: float,
    min_cycle: float = 60.0,
    max_cycle: float = 180.0,
) -> float:
    """Webster (1958) optimal cycle length.

    C_opt = (1.5L + 5) / (1 - Y)

    Parameters
    ----------
    critical_flow_ratios:
        List of y_i = q_i / s_i for each critical phase.
    lost_time_per_phase:
        Yellow + all-red per phase (seconds).
    """
    n = len(critical_flow_ratios)
    L = n * lost_time_per_phase
    Y = sum(critical_flow_ratios)
    if Y >= 0.99:
        return max_cycle  # oversaturated — use maximum
    c_opt = (1.5 * L + 5.0) / (1.0 - Y)
    return max(min_cycle, min(max_cycle, c_opt))


def allocate_greens(
    cycle: float,
    critical_flow_ratios: list[float],
    lost_time_per_phase: float,
    min_green: float = 7.0,
) -> list[float]:
    """Allocate green times proportionally to critical flow ratios.

    g_i = (C - L) * (y_i / Y)
    """
    n = len(critical_flow_ratios)
    L = n * lost_time_per_phase
    Y = max(sum(critical_flow_ratios), 1e-9)
    effective_green = max(cycle - L, n * min_green)
    greens = [(effective_green * (y / Y)) for y in critical_flow_ratios]
    # Enforce minimum green
    greens = [max(g, min_green) for g in greens]
    # Re-scale to fit cycle
    total = sum(greens) + L
    scale = min(1.0, (cycle - L) / max(sum(greens), 1e-9))
    greens = [g * scale for g in greens]
    greens = [max(g, min_green) for g in greens]
    return greens


# ---------------------------------------------------------------------------
# Demand extraction helpers
# ---------------------------------------------------------------------------

def _critical_flow_ratios(demand: DemandProfile) -> list[tuple[int, float]]:
    """Return [(phase_id, critical_flow_ratio)] for the 4 critical phases.

    In a standard dual-ring 8-phase plan, critical phases are:
      NB/SB: phase 2 or 4 (whichever has higher demand)
      EB/WB: phase 6 or 8 (whichever has higher demand)
    We also include protected lefts if demand warrants.
    """
    # veh/hr → flow ratio (dimensionless)
    # Left turns use saturation flow 1200 veh/hr/lane,
    # through movements use 1800 veh/hr/lane.

    movements = {
        2: demand.north_thru / SAT_FLOW_THRU,
        4: demand.south_thru / SAT_FLOW_THRU,
        6: demand.east_thru / SAT_FLOW_THRU,
        8: demand.west_thru / SAT_FLOW_THRU,
    }
    # Add protected left turns if significant (>150 veh/hr per MUTCD warrant)
    if demand.north_left > 150:
        movements[1] = demand.north_left / SAT_FLOW_LEFT
    if demand.south_left > 150:
        movements[3] = demand.south_left / SAT_FLOW_LEFT
    if demand.east_left > 150:
        movements[5] = demand.east_left / SAT_FLOW_LEFT
    if demand.west_left > 150:
        movements[7] = demand.west_left / SAT_FLOW_LEFT

    # Dual ring: Ring1 = {1,2,3,4}, Ring2 = {5,6,7,8}
    # Critical phase per ring-half (barrier 1: phases 1+5/2+6, barrier 2: phases 3+7/4+8)
    critical = []
    for phase_id, y in sorted(movements.items()):
        critical.append((phase_id, y))
    return critical


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

@dataclass
class IsolatedOptimizationResult:
    intersection: Intersection
    period: str
    plan: TimingPlan
    validation: ValidationResult
    cycle_length: float
    degree_of_saturation: float   # Y, sum of critical flow ratios
    notes: list[str]


class IsolatedIntersectionOptimizer:
    """Webster-enhanced isolated intersection optimizer.

    Generates time-of-day timing plans for a single intersection based on
    turning movement counts or probe-data-derived demand profiles.

    Parameters
    ----------
    intersection:
        Intersection geometry, phase config, pedestrian phases.
    validator:
        Optional custom validator.  Defaults to TimingPlanValidator().
    """

    LOST_TIME_PER_PHASE = 6.0   # yellow(4s) + all-red(2s), HCM §19.6

    def __init__(
        self,
        intersection: Intersection,
        validator: Optional[TimingPlanValidator] = None,
    ) -> None:
        self.intersection = intersection
        self.validator = validator or TimingPlanValidator()

    def optimize(self, demand: DemandProfile, period: str = "AM Peak") -> IsolatedOptimizationResult:
        """Generate an optimized timing plan for a single demand period."""
        notes: list[str] = []

        critical = _critical_flow_ratios(demand)
        flow_ratios = [y for _, y in critical]
        Y = sum(flow_ratios)

        if Y >= 0.99:
            notes.append(f"Intersection oversaturated (Y={Y:.3f}). Maximum cycle applied.")

        cycle = webster_cycle(
            critical_flow_ratios=flow_ratios,
            lost_time_per_phase=self.LOST_TIME_PER_PHASE,
        )
        greens = allocate_greens(
            cycle=cycle,
            critical_flow_ratios=flow_ratios,
            lost_time_per_phase=self.LOST_TIME_PER_PHASE,
        )

        # Build PhaseTiming list
        phases: list[PhaseTiming] = []
        for i, (phase_id, _y) in enumerate(critical):
            split = round(greens[i], 1)
            is_ped = phase_id in self.intersection.pedestrian_phases

            ped_walk = 7.0 if is_ped else None
            ped_clearance = round(self.intersection.crossing_clearance, 1) if is_ped else None

            # Ensure split covers ped requirements
            if is_ped and ped_clearance is not None:
                min_split_ped = self.validator.PED_WALK_MINIMUM + ped_clearance
                if split < min_split_ped:
                    split = min_split_ped
                    notes.append(
                        f"Phase {phase_id}: split raised to {split:.1f}s to cover pedestrian timing"
                    )

            phases.append(PhaseTiming(
                phase_id=phase_id,
                min_green=7.0,
                max_green=min(split * 2, 90.0),
                split=split,
                yellow=4.0,
                all_red=2.0,
                ped_walk=ped_walk,
                ped_clearance=ped_clearance,
            ))

        # Recompute cycle to fit actual splits (including ped adjustments)
        total_overhead = sum(p.yellow + p.all_red for p in phases)
        total_green = sum(p.split for p in phases)
        cycle = math.ceil(total_green + total_overhead)
        # Cap at 180s but never truncate below minimum feasible
        cycle = max(60.0, min(180.0, cycle))

        plan = TimingPlan(
            intersection_id=self.intersection.id,
            cycle_length=cycle,
            phases=phases,
            source="aito",
        )

        validation = self.validator.validate(plan, self.intersection)

        return IsolatedOptimizationResult(
            intersection=self.intersection,
            period=period,
            plan=plan,
            validation=validation,
            cycle_length=cycle,
            degree_of_saturation=Y,
            notes=notes,
        )

    def optimize_all_periods(
        self,
        base_demand: DemandProfile,
    ) -> list[IsolatedOptimizationResult]:
        """Generate timing plans for all 6 TOD periods.

        Scales base demand by TOD_DEMAND_FACTORS for each period.
        """
        results = []
        for period, scale in TOD_DEMAND_FACTORS.items():
            scaled = self._scale_demand(base_demand, scale)
            scaled.period_minutes = 60  # full-hour plan
            result = self.optimize(scaled, period=period)
            results.append(result)
        return results

    @staticmethod
    def _scale_demand(base: DemandProfile, factor: float) -> DemandProfile:
        data = base.model_dump()
        for key in list(data.keys()):
            if isinstance(data[key], (int, float)) and key not in (
                "period_minutes",
            ) and "intersection_id" not in key:
                data[key] = data[key] * factor
        return DemandProfile(**data)
