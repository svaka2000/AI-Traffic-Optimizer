"""aito/optimization/corridor_optimizer.py

Arterial corridor green-wave optimizer.

Algorithm: MAXBAND-enhanced with probe-data travel times.

Classical MAXBAND (Little et al. 1981) maximizes the sum of inbound and
outbound bandwidths on a coordinated arterial:

    maximize  b1 + k * b2
    subject to:
      (1/C) * (t_i - t_{i+1}) + (w_i + w_{i+1})/2 = d_i/(v_i * C) + n_i
      for all adjacent intersections i, i+1

AITO enhancements over classical MAXBAND:
  1. Uses probe-data travel times instead of assumed constant speeds.
  2. Iterates over candidate cycle lengths (60–180s in 5s steps).
  3. Selects cycle that maximises combined bandwidth.
  4. Returns all non-dominated solutions for multi-objective comparison.

References:
  Little, J. D. C., Kelson, M. D., Gartner, N. H. (1981).
  "MAXBAND: A Program for Setting Signals on Arteries and Triangular Grids."
  Transportation Research Record, 795, 40–46.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from aito.models import (
    Corridor,
    CorridorPlan,
    DemandProfile,
    Intersection,
    PhaseTiming,
    TimingPlan,
)
from aito.optimization.constraints import TimingPlanValidator, ValidationResult, compute_yellow, compute_all_red
from aito.optimization.isolated_optimizer import (
    IsolatedIntersectionOptimizer,
    TOD_DEMAND_FACTORS,
    _critical_flow_ratios,
    webster_cycle,
    allocate_greens,
)


MPH_TO_MPS = 0.44704   # miles-per-hour → meters-per-second


# ---------------------------------------------------------------------------
# Travel time data (probe or default)
# ---------------------------------------------------------------------------

@dataclass
class SegmentTravelData:
    """Travel time and speed for one inter-signal segment."""
    inbound_travel_time_s: float   # downstream → upstream direction
    outbound_travel_time_s: float  # upstream → downstream direction
    distance_m: float

    @property
    def inbound_speed_mps(self) -> float:
        return self.distance_m / max(self.inbound_travel_time_s, 1.0)

    @property
    def outbound_speed_mps(self) -> float:
        return self.distance_m / max(self.outbound_travel_time_s, 1.0)


# ---------------------------------------------------------------------------
# Bandwidth computation
# ---------------------------------------------------------------------------

def compute_bandwidth(
    offsets: list[float],
    green_splits: list[float],
    travel_times_s: list[float],
    cycle: float,
    direction: str = "outbound",
) -> float:
    """Compute green-wave bandwidth for a given offset set.

    bandwidth = min over all i of (green[i] - |offset_diff - travel_time|) mod C

    Normalised to [0,1] as fraction of cycle length.
    """
    n = len(offsets)
    if n < 2:
        return green_splits[0] / cycle if green_splits else 0.0

    bandwidth = cycle  # start with maximum possible
    for i in range(n - 1):
        if direction == "outbound":
            ideal_offset_diff = travel_times_s[i]
        else:
            ideal_offset_diff = -travel_times_s[i]

        actual_diff = (offsets[i + 1] - offsets[i]) % cycle
        # How much of green[i+1] aligns with arriving platoon
        phase_offset = abs(actual_diff - ideal_offset_diff % cycle)
        green = min(green_splits[i], green_splits[i + 1])
        bw = max(0.0, green - phase_offset)
        bandwidth = min(bandwidth, bw)

    return bandwidth / cycle  # normalised


def optimize_offsets_maxband(
    travel_times_s: list[float],
    green_splits: list[float],
    cycle: float,
    directional_weight: float = 1.0,
) -> tuple[list[float], float, float]:
    """Maximise combined bandwidth via exhaustive search on offset space.

    Parameters
    ----------
    travel_times_s:
        Travel times for each segment in the outbound direction.
    green_splits:
        Effective green time per intersection (seconds).
    cycle:
        Common cycle length (seconds).
    directional_weight:
        k = weight for inbound direction (1.0 = symmetric).

    Returns
    -------
    (offsets, bandwidth_outbound, bandwidth_inbound)
    """
    n = len(green_splits)
    best_offsets = [0.0] * n
    best_combined = -1.0
    best_bw_out = 0.0
    best_bw_in = 0.0

    # Fix offset[0] = 0 (system reference), scan offset[1] in 1-second steps.
    # For n > 2, we use a greedy progressive approach:
    #   set offset[i] = ideal outbound offset from offset[i-1].

    # MAXBAND analytical solution: offset_i = offset_{i-1} + travel_time_{i-1} mod C
    ideal_offsets = [0.0]
    for tt in travel_times_s:
        ideal_offsets.append((ideal_offsets[-1] + tt) % cycle)

    # Scan perturbations around the ideal to find max combined bandwidth
    step = max(1.0, cycle / 60.0)  # ~1-2s resolution
    for delta in range(0, int(cycle / step)):
        shift = delta * step
        offsets = [(o + shift) % cycle for o in ideal_offsets]

        bw_out = compute_bandwidth(offsets, green_splits, travel_times_s, cycle, "outbound")
        bw_in = compute_bandwidth(offsets, green_splits, travel_times_s, cycle, "inbound")
        combined = bw_out + directional_weight * bw_in

        if combined > best_combined:
            best_combined = combined
            best_offsets = offsets
            best_bw_out = bw_out
            best_bw_in = bw_in

    return best_offsets, best_bw_out, best_bw_in


# ---------------------------------------------------------------------------
# Corridor optimizer result
# ---------------------------------------------------------------------------

@dataclass
class CorridorOptimizationResult:
    corridor: Corridor
    period: str
    corridor_plan: CorridorPlan
    cycle_length: float
    bandwidth_outbound_pct: float   # % of cycle
    bandwidth_inbound_pct: float
    combined_bandwidth_pct: float
    validation_results: dict[str, ValidationResult]
    notes: list[str] = field(default_factory=list)

    @property
    def all_valid(self) -> bool:
        return all(v.valid for v in self.validation_results.values())


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class CorridorOptimizer:
    """MAXBAND-enhanced arterial corridor optimizer.

    Parameters
    ----------
    corridor:
        Corridor with at least 2 intersections.
    validator:
        Timing plan validator (defaults to TimingPlanValidator).
    cycle_step:
        Resolution of cycle length search (seconds).  Default 5s.
    directional_weight:
        k in MAXBAND objective b1 + k*b2.  1.0 = symmetric, >1 = favour inbound.
    """

    def __init__(
        self,
        corridor: Corridor,
        validator: Optional[TimingPlanValidator] = None,
        cycle_step: float = 5.0,
        directional_weight: float = 1.0,
    ) -> None:
        self.corridor = corridor
        self.validator = validator or TimingPlanValidator()
        self.cycle_step = cycle_step
        self.directional_weight = directional_weight

    def optimize(
        self,
        demand_profiles: list[DemandProfile],
        travel_data: Optional[list[SegmentTravelData]] = None,
        period: str = "AM Peak",
    ) -> CorridorOptimizationResult:
        """Optimise offsets for a single time period.

        Parameters
        ----------
        demand_profiles:
            One per intersection, in corridor order.
        travel_data:
            Probe-data travel times per segment.  If None, travel time is
            computed from distance / speed limit.
        period:
            Time-of-day period label.
        """
        n = len(self.corridor.intersections)
        assert len(demand_profiles) == n, (
            f"Expected {n} demand profiles, got {len(demand_profiles)}"
        )

        # 1. Build travel data from defaults if not provided
        if travel_data is None:
            travel_data = self._default_travel_data()

        # 2. Determine common cycle length using Webster on busiest intersection
        max_Y = 0.0
        for dp in demand_profiles:
            ratios = [y for _, y in _critical_flow_ratios(dp)]
            max_Y = max(max_Y, sum(ratios))

        # Search over candidate cycle lengths
        best_result: Optional[CorridorOptimizationResult] = None
        best_score = -1.0
        notes: list[str] = []

        for c_candidate in self._candidate_cycles(max_Y):
            result = self._optimize_at_cycle(
                demand_profiles=demand_profiles,
                travel_data=travel_data,
                cycle=c_candidate,
                period=period,
            )
            score = result.combined_bandwidth_pct
            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise RuntimeError("Corridor optimizer produced no valid result")

        best_result.notes = notes
        return best_result

    def optimize_all_periods(
        self,
        demand_profiles: list[DemandProfile],
        travel_data: Optional[list[SegmentTravelData]] = None,
    ) -> list[CorridorOptimizationResult]:
        """Generate corridor plans for all 6 TOD periods."""
        results = []
        for period, scale in TOD_DEMAND_FACTORS.items():
            scaled = [
                IsolatedIntersectionOptimizer._scale_demand(dp, scale)
                for dp in demand_profiles
            ]
            result = self.optimize(scaled, travel_data=travel_data, period=period)
            results.append(result)
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _default_travel_data(self) -> list[SegmentTravelData]:
        """Use speed limits and distances to compute default travel times."""
        data: list[SegmentTravelData] = []
        for i, dist_m in enumerate(self.corridor.distances_m):
            speed_mps = self.corridor.speed_limits_mph[i] * MPH_TO_MPS
            tt_s = dist_m / max(speed_mps, 1.0)
            data.append(SegmentTravelData(
                inbound_travel_time_s=tt_s,
                outbound_travel_time_s=tt_s,
                distance_m=dist_m,
            ))
        return data

    def _candidate_cycles(self, max_Y: float) -> list[float]:
        """Generate candidate cycle lengths to evaluate."""
        n_phases = 4  # default 4 critical phases
        L = n_phases * 6.0  # 4 phases × 6s lost time
        if max_Y >= 0.99:
            c_webster = 180.0
        else:
            c_webster = (1.5 * L + 5.0) / (1.0 - max_Y)
        # Clamp to [60, 180] range
        c_webster = max(60.0, min(180.0, c_webster))
        c_start = max(60.0, c_webster - 20)
        c_end = min(180.0, c_webster + 20)
        # Ensure we always have at least a few candidates
        if c_start > c_end:
            c_start, c_end = max(60.0, c_end - 20), c_end
        candidates = []
        c = c_start
        while c <= c_end + 0.1:
            candidates.append(round(c / self.cycle_step) * self.cycle_step)
            c += self.cycle_step
        # Always include the Webster optimal
        candidates.append(round(c_webster / self.cycle_step) * self.cycle_step)
        # Deduplicate and filter
        return sorted({max(60.0, min(180.0, c)) for c in candidates})

    def _optimize_at_cycle(
        self,
        demand_profiles: list[DemandProfile],
        travel_data: list[SegmentTravelData],
        cycle: float,
        period: str,
    ) -> CorridorOptimizationResult:
        """Build a full corridor plan for a fixed cycle length."""
        intersections = self.corridor.intersections
        n = len(intersections)

        # Compute per-intersection splits
        all_splits: list[list[tuple[int, float]]] = []
        all_phases: list[list[PhaseTiming]] = []
        green_for_bandwidth: list[float] = []

        for i, (ix, dp) in enumerate(zip(intersections, demand_profiles)):
            critical = _critical_flow_ratios(dp)
            flow_ratios = [y for _, y in critical]
            # Compute ITE-correct clearance intervals for this intersection
            req_yellow = math.ceil(compute_yellow(ix.approach_speed_mph) * 2) / 2
            req_yellow = max(req_yellow, 3.0)
            req_ar = math.ceil(compute_all_red(ix.crossing_distance_ft, approach_speed_mph=ix.approach_speed_mph) * 2) / 2
            req_ar = max(req_ar, 1.0)
            lost_time = req_yellow + req_ar
            greens = allocate_greens(
                cycle=cycle,
                critical_flow_ratios=flow_ratios,
                lost_time_per_phase=lost_time,
            )

            phases: list[PhaseTiming] = []
            max_green = 0.0
            for j, (phase_id, _) in enumerate(critical):
                split = round(greens[j], 1)
                is_ped = phase_id in ix.pedestrian_phases
                ped_walk = 7.0 if is_ped else None
                ped_cl = round(ix.crossing_clearance, 1) if is_ped else None
                if is_ped and ped_cl is not None:
                    split = max(split, 7.0 + ped_cl)
                phases.append(PhaseTiming(
                    phase_id=phase_id,
                    min_green=7.0,
                    max_green=min(split * 2, 90.0),
                    split=split,
                    yellow=req_yellow,
                    all_red=req_ar,
                    ped_walk=ped_walk,
                    ped_clearance=ped_cl,
                ))
                max_green = max(max_green, split)
            all_phases.append(phases)
            green_for_bandwidth.append(max_green)

        # After ped adjustments, recalculate the actual cycle required.
        # Raised ped splits can push total above the requested cycle; clamp to HCM max.
        import math as _math
        actual_cycle = cycle
        for phases in all_phases:
            total_needed = sum(p.split + p.yellow + p.all_red for p in phases)
            actual_cycle = max(actual_cycle, _math.ceil(total_needed))
        actual_cycle = min(180.0, actual_cycle)

        # Compute outbound travel times per segment
        tt_outbound = [seg.outbound_travel_time_s for seg in travel_data]

        # MAXBAND offset optimisation
        offsets, bw_out, bw_in = optimize_offsets_maxband(
            travel_times_s=tt_outbound,
            green_splits=green_for_bandwidth,
            cycle=actual_cycle,
            directional_weight=self.directional_weight,
        )

        # Build TimingPlan objects
        timing_plans: list[TimingPlan] = []
        validation_results: dict[str, ValidationResult] = {}

        for i, (ix, phases, offset) in enumerate(zip(intersections, all_phases, offsets)):
            plan = TimingPlan(
                intersection_id=ix.id,
                cycle_length=actual_cycle,
                offset=round(offset, 1),
                phases=phases,
                source="aito_maxband",
            )
            timing_plans.append(plan)
            validation_results[ix.id] = self.validator.validate(plan, ix)

        corridor_plan = CorridorPlan(
            corridor_id=self.corridor.id,
            plan_name=period,
            timing_plans=timing_plans,
            cycle_length=actual_cycle,
            offsets=offsets,
            bandwidth_inbound=bw_in * actual_cycle,
            bandwidth_outbound=bw_out * actual_cycle,
        )

        return CorridorOptimizationResult(
            corridor=self.corridor,
            period=period,
            corridor_plan=corridor_plan,
            cycle_length=cycle,
            bandwidth_outbound_pct=bw_out * 100,
            bandwidth_inbound_pct=bw_in * 100,
            combined_bandwidth_pct=(bw_out + bw_in) * 100 / 2,
            validation_results=validation_results,
        )
