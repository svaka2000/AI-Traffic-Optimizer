"""aito/deployment/timing_plan.py

Timing plan data model helpers and NEMA ring-barrier utilities.
"""
from __future__ import annotations

import math
from aito.models import TimingPlan, PhaseTiming, Intersection
from aito.optimization.constraints import TimingPlanValidator, compute_yellow, compute_all_red


def build_default_plan(intersection: Intersection, cycle: float = 120.0) -> TimingPlan:
    """Build a minimal valid timing plan for an intersection.

    Uses 4 critical phases (2, 4, 6, 8) — the standard through/thru movements.
    Automatically scales cycle to accommodate pedestrian timing requirements.
    """
    # Use standard 4-phase critical movements only
    critical_phases = [2, 4, 6, 8]
    req_yellow = math.ceil(compute_yellow(intersection.approach_speed_mph) * 2) / 2
    req_yellow = max(req_yellow, 3.0)
    req_ar = math.ceil(compute_all_red(intersection.crossing_distance_ft, approach_speed_mph=intersection.approach_speed_mph) * 2) / 2
    req_ar = max(req_ar, 1.0)
    overhead = req_yellow + req_ar

    # Compute minimum split per phase including pedestrian requirements
    phase_min_splits: list[float] = []
    phase_ped: list[tuple[bool, float | None, float | None]] = []
    for pid in critical_phases:
        is_ped = pid in intersection.pedestrian_phases
        ped_walk = 7.0 if is_ped else None
        ped_cl = round(intersection.crossing_clearance, 1) if is_ped else None
        ms = 7.0
        if is_ped and ped_cl is not None:
            ms = max(ms, 7.0 + ped_cl)
        phase_min_splits.append(ms)
        phase_ped.append((is_ped, ped_walk, ped_cl))

    # Minimum feasible cycle
    min_feasible = len(critical_phases) * overhead + sum(phase_min_splits)
    cycle = max(cycle, min_feasible + 2.0)
    cycle = min(180.0, cycle)

    effective_green = max(cycle - len(critical_phases) * overhead, sum(phase_min_splits))
    extra = effective_green - sum(phase_min_splits)
    extra_each = max(0.0, extra / len(critical_phases))

    phases_out: list[PhaseTiming] = []
    for pid, ms, (is_ped, ped_walk, ped_cl) in zip(critical_phases, phase_min_splits, phase_ped):
        split = round(ms + extra_each, 1)
        phases_out.append(PhaseTiming(
            phase_id=pid,
            min_green=7.0,
            max_green=min(split * 2, 90.0),
            split=split,
            yellow=req_yellow,
            all_red=req_ar,
            ped_walk=ped_walk,
            ped_clearance=ped_cl,
        ))

    return TimingPlan(
        intersection_id=intersection.id,
        cycle_length=math.ceil(cycle),
        phases=phases_out,
        source="aito_default",
    )
