"""aito/optimization/multi_objective.py

Multi-objective Pareto optimizer for AITO.

Generates and ranks corridor plans across 5 objectives simultaneously:
  1. Delay       — average intersection delay (s/veh), HCM methodology
  2. Emissions   — CO2 kg/hr, EPA MOVES2014b idle factors
  3. Stops       — stops per vehicle
  4. Safety      — conflict index (surrogate measure)
  5. Equity      — std deviation of delay across approaches

This is AITO's core differentiator: no competitor optimizes all five
simultaneously and presents the tradeoffs to engineers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

from aito.models import (
    Corridor,
    CorridorPlan,
    DemandProfile,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    ParetoSolution,
    Intersection,
    TimingPlan,
    PhaseTiming,
)
from aito.optimization.corridor_optimizer import (
    CorridorOptimizer,
    CorridorOptimizationResult,
    SegmentTravelData,
)
from aito.optimization.isolated_optimizer import (
    _critical_flow_ratios,
    SAT_FLOW_THRU,
    SAT_FLOW_LEFT,
)


# ---------------------------------------------------------------------------
# HCM delay model
# ---------------------------------------------------------------------------

def hcm_delay(
    cycle: float,
    green: float,
    flow_ratio: float,
    capacity_veh_hr: float = 900.0,
    volume_veh_hr: float = 500.0,
    incremental_delay_factor: float = 0.9,
) -> float:
    """HCM 7th Edition intersection delay (s/veh).

    d = d1 + d2 + d3

    d1 = uniform delay (Webster approximation):
         0.5 * C * (1 - g/C)^2 / (1 - min(1, X) * g/C)

    d2 = incremental delay:
         900 * T * [(X-1) + sqrt((X-1)^2 + 8*k*I*X/(c*T))]

    where X = v/c (degree of saturation).
    """
    g = green
    C = max(cycle, g + 1.0)
    g_ratio = min(g / C, 0.99)
    volume_veh_s = volume_veh_hr / 3600.0
    c_veh_s = max(capacity_veh_hr / 3600.0, volume_veh_s + 0.001)
    X = min(volume_veh_s / c_veh_s, 2.0)

    # d1 — uniform delay
    d1 = (0.5 * C * (1 - g_ratio) ** 2) / max((1 - min(1.0, X) * g_ratio), 0.01)

    # d2 — incremental delay (simplified, T=0.25 hr)
    T = 0.25
    k = 0.5  # pre-timed signal
    I = incremental_delay_factor
    d2 = 900 * T * (
        (X - 1) + math.sqrt(max((X - 1) ** 2 + 8 * k * I * X / (c_veh_s * 3600 * T), 0))
    )

    return round(d1 + d2, 2)


# ---------------------------------------------------------------------------
# Emissions model
# ---------------------------------------------------------------------------

# EPA MOVES3 idle emission factors
IDLE_FUEL_GAL_HR = 0.16          # gallons/hr/vehicle
CO2_PER_GALLON_KG = 8.887        # kg CO2/gallon gasoline
STOP_PENALTY_GAL = 0.0022        # extra gallons per stop event

def compute_emissions_kg_hr(
    avg_queue_veh: float,
    stops_per_hr: float,
    idle_fuel_rate: float = IDLE_FUEL_GAL_HR,
    co2_per_gallon: float = CO2_PER_GALLON_KG,
    stop_penalty: float = STOP_PENALTY_GAL,
) -> float:
    """CO2 kg per hour for a single intersection approach."""
    idle_fuel = avg_queue_veh * idle_fuel_rate
    stop_fuel = stops_per_hr * stop_penalty
    return (idle_fuel + stop_fuel) * co2_per_gallon


# ---------------------------------------------------------------------------
# Conflict index (safety surrogate)
# ---------------------------------------------------------------------------

def conflict_index(
    num_phases: int,
    max_green_ratio: float,
    volume_veh_hr: float,
) -> float:
    """Simplified conflict point index.

    Higher throughput at short greens → more conflict risk.
    Normalized 0-100 (higher = worse safety).
    """
    # Proportional to volume × (1 - max_green_ratio)
    return min(100.0, volume_veh_hr * (1.0 - max_green_ratio) / 100.0)


# ---------------------------------------------------------------------------
# Equity score
# ---------------------------------------------------------------------------

def equity_score(delays_by_approach: list[float]) -> float:
    """Standard deviation of per-approach delays.  Lower = more equitable."""
    if len(delays_by_approach) < 2:
        return 0.0
    mean = sum(delays_by_approach) / len(delays_by_approach)
    variance = sum((d - mean) ** 2 for d in delays_by_approach) / len(delays_by_approach)
    return math.sqrt(variance)


# ---------------------------------------------------------------------------
# Score a corridor plan
# ---------------------------------------------------------------------------

@dataclass
class PlanScores:
    delay_s_veh: float
    emissions_kg_hr: float
    stops_per_veh: float
    safety_index: float
    equity_std: float


def score_corridor_plan(
    corridor: Corridor,
    corridor_plan: CorridorPlan,
    demand_profiles: list[DemandProfile],
) -> PlanScores:
    """Evaluate a CorridorPlan across all 5 objectives."""
    all_delays: list[float] = []
    all_emissions: list[float] = []
    all_stops: list[float] = []
    all_safety: list[float] = []

    for ix, plan, dp in zip(
        corridor.intersections, corridor_plan.timing_plans, demand_profiles
    ):
        critical = _critical_flow_ratios(dp)
        total_volume = (
            dp.north_thru + dp.south_thru + dp.east_thru + dp.west_thru +
            dp.north_left + dp.south_left + dp.east_left + dp.west_left
        )

        approach_delays: list[float] = []
        for phase_id, y in critical:
            pt = next((p for p in plan.phases if p.phase_id == phase_id), None)
            if pt is None:
                continue

            volume = total_volume / max(len(critical), 1)
            capacity = SAT_FLOW_THRU * (pt.split / plan.cycle_length)
            d = hcm_delay(
                cycle=plan.cycle_length,
                green=pt.split,
                flow_ratio=y,
                capacity_veh_hr=capacity,
                volume_veh_hr=volume,
            )
            approach_delays.append(d)

        if not approach_delays:
            continue

        avg_delay = sum(approach_delays) / len(approach_delays)
        all_delays.append(avg_delay)

        # Stops: approximate from Webster
        g_ratio = sum(p.split for p in plan.phases) / (plan.cycle_length * len(plan.phases))
        stops = max(0.0, (1.0 - g_ratio) * 0.9)  # fraction of vehicles stopped
        all_stops.append(stops)

        # Emissions
        avg_queue = total_volume / max(3600 / plan.cycle_length, 1)
        stops_hr = stops * total_volume
        all_emissions.append(compute_emissions_kg_hr(avg_queue, stops_hr))

        # Safety
        max_gr = max((p.split / plan.cycle_length for p in plan.phases), default=0.5)
        all_safety.append(conflict_index(len(plan.phases), max_gr, total_volume))

    avg_delay = sum(all_delays) / max(len(all_delays), 1)
    avg_emissions = sum(all_emissions)
    avg_stops = sum(all_stops) / max(len(all_stops), 1)
    avg_safety = sum(all_safety) / max(len(all_safety), 1)
    eq = equity_score(all_delays)

    return PlanScores(
        delay_s_veh=round(avg_delay, 2),
        emissions_kg_hr=round(avg_emissions, 2),
        stops_per_veh=round(avg_stops, 3),
        safety_index=round(avg_safety, 2),
        equity_std=round(eq, 2),
    )


# ---------------------------------------------------------------------------
# Pareto dominance
# ---------------------------------------------------------------------------

def dominates(a: PlanScores, b: PlanScores) -> bool:
    """Return True if plan A dominates plan B (all objectives ≤, at least one <)."""
    a_vals = (a.delay_s_veh, a.emissions_kg_hr, a.stops_per_veh, a.safety_index, a.equity_std)
    b_vals = (b.delay_s_veh, b.emissions_kg_hr, b.stops_per_veh, b.safety_index, b.equity_std)
    return all(av <= bv for av, bv in zip(a_vals, b_vals)) and any(
        av < bv for av, bv in zip(a_vals, b_vals)
    )


def pareto_front(solutions: list[tuple[CorridorPlan, PlanScores]]) -> list[int]:
    """Return indices of non-dominated solutions."""
    n = len(solutions)
    dominated = [False] * n
    for i in range(n):
        for j in range(n):
            if i != j and not dominated[j] and dominates(solutions[j][1], solutions[i][1]):
                dominated[i] = True
                break
    return [i for i in range(n) if not dominated[i]]


# ---------------------------------------------------------------------------
# Main optimizer
# ---------------------------------------------------------------------------

class MultiObjectiveOptimizer:
    """Multi-objective Pareto corridor optimizer.

    Generates a family of corridor plans at different cycle lengths and
    directional weights, then returns the Pareto frontier.

    Parameters
    ----------
    corridor:
        The corridor to optimize.
    """

    CYCLE_CANDIDATES = list(range(60, 185, 5))
    DIRECTIONAL_WEIGHTS = [0.8, 1.0, 1.2]  # favour outbound, symmetric, favour inbound

    def __init__(self, corridor: Corridor) -> None:
        self.corridor = corridor

    def optimize(
        self,
        request: OptimizationRequest,
        travel_data: Optional[list[SegmentTravelData]] = None,
    ) -> OptimizationResult:
        """Run multi-objective optimization and return Pareto solutions."""
        import time
        t0 = time.time()

        demand_profiles = request.demand_profiles
        candidate_plans: list[tuple[CorridorPlan, PlanScores]] = []

        # Generate candidate plans across cycle lengths and directional weights
        for weight in self.DIRECTIONAL_WEIGHTS:
            opt = CorridorOptimizer(
                self.corridor,
                cycle_step=5.0,
                directional_weight=weight,
            )
            for cycle in self._filter_cycles(request):
                try:
                    result = opt._optimize_at_cycle(
                        demand_profiles=demand_profiles,
                        travel_data=travel_data or opt._default_travel_data(),
                        cycle=float(cycle),
                        period="AM Peak",
                    )
                    # Skip plans that fail validation (e.g., cycle too short for ped)
                    if not result.all_valid:
                        continue
                    scores = score_corridor_plan(
                        self.corridor, result.corridor_plan, demand_profiles
                    )
                    candidate_plans.append((result.corridor_plan, scores))
                except Exception:
                    continue

        if not candidate_plans:
            raise RuntimeError("MultiObjectiveOptimizer produced no candidate plans")

        # Find Pareto front
        pareto_indices = pareto_front(candidate_plans)
        pareto_plans = [candidate_plans[i] for i in pareto_indices]

        # Convert to ParetoSolution objects
        solutions: list[ParetoSolution] = []
        for plan, scores in pareto_plans:
            desc = self._describe_tradeoff(scores, [s for _, s in pareto_plans])
            solutions.append(ParetoSolution(
                plan=plan,
                delay_score=scores.delay_s_veh,
                emissions_score=scores.emissions_kg_hr,
                stops_score=scores.stops_per_veh,
                safety_score=scores.safety_index,
                equity_score=scores.equity_std,
                description=desc,
            ))

        # Recommended = lowest weighted sum
        weights = {
            OptimizationObjective.DELAY:     0.4,
            OptimizationObjective.EMISSIONS: 0.2,
            OptimizationObjective.STOPS:     0.2,
            OptimizationObjective.SAFETY:    0.1,
            OptimizationObjective.EQUITY:    0.1,
        }
        best = min(
            solutions,
            key=lambda s: (
                weights.get(OptimizationObjective.DELAY, 0) * s.delay_score +
                weights.get(OptimizationObjective.EMISSIONS, 0) * s.emissions_score +
                weights.get(OptimizationObjective.STOPS, 0) * s.stops_score * 100 +
                weights.get(OptimizationObjective.SAFETY, 0) * s.safety_score +
                weights.get(OptimizationObjective.EQUITY, 0) * s.equity_score
            ),
        )

        elapsed = time.time() - t0
        return OptimizationResult(
            corridor_id=self.corridor.id,
            pareto_solutions=solutions,
            recommended_solution=best,
            computation_seconds=round(elapsed, 2),
        )

    def _filter_cycles(self, request: OptimizationRequest) -> list[int]:
        return [
            c for c in self.CYCLE_CANDIDATES
            if request.min_cycle <= c <= request.max_cycle
        ]

    @staticmethod
    def _describe_tradeoff(scores: PlanScores, all_scores: list[PlanScores]) -> str:
        """Human-readable tradeoff description for engineers."""
        parts: list[str] = []
        delays = [s.delay_s_veh for s in all_scores]
        min_d, max_d = min(delays), max(delays)
        if scores.delay_s_veh == min_d:
            parts.append("Minimum delay")
        elif scores.delay_s_veh == max_d:
            parts.append("Higher delay")
        else:
            parts.append(f"Moderate delay ({scores.delay_s_veh:.0f}s/veh)")

        emissions = [s.emissions_kg_hr for s in all_scores]
        if scores.emissions_kg_hr == min(emissions):
            parts.append("lowest emissions")

        if scores.equity_std <= 5.0:
            parts.append("high equity")

        return ", ".join(parts) if parts else f"Delay={scores.delay_s_veh:.0f}s/veh"
