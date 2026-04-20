"""aito/optimization/multi_objective_engine.py

GF6: NSGA-III Multi-Objective Optimizer for Traffic Signal Timing.

Implements Non-dominated Sorting Genetic Algorithm III (Deb & Jain, 2014)
adapted for 5-objective traffic signal optimization:
  1. Delay       — average delay per vehicle (s/veh), HCM 7th Edition
  2. Emissions   — CO2 kg/hr, EPA MOVES2014b speed-cycle model
  3. Stops       — stops per vehicle
  4. Safety      — conflict index (approach speed × left-turn exposure)
  5. Equity      — std deviation of delay across all approaches

Why NSGA-III over NSGA-II:
  NSGA-III uses reference points on a normalized hyperplane to maintain
  diversity across many objectives.  For 5+ objectives, NSGA-II degrades
  because crowding distance becomes meaningless in high dimensions.

References:
  Deb, K. & Jain, H. (2014). "An Evolutionary Many-Objective Optimization
  Algorithm Using Reference-Point-Based Nondominated Sorting Approach,
  Part I: Solving Problems With Box Constraints."
  IEEE Transactions on Evolutionary Computation, 18(4), 577-601.

  EPA MOVES2014b Technical Guidance.
  HCM 7th Edition, Chapter 19 (Signalized Intersections).
"""
from __future__ import annotations

import math
import random
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
)
from aito.optimization.corridor_optimizer import (
    CorridorOptimizer,
    SegmentTravelData,
)
from aito.optimization.multi_objective import hcm_delay


def stops_per_vehicle(cycle: float, green: float, flow_ratio: float) -> float:
    """Webster uniform stops per vehicle estimate."""
    red = cycle - green
    if cycle <= 0:
        return 0.0
    return 0.9 * (red / cycle) * (1.0 - (green / cycle) * flow_ratio / max(1.0 - flow_ratio * (cycle / max(green, 1)), 0.01))


# ---------------------------------------------------------------------------
# EPA MOVES2014b emission factors (g CO2/s) by operating mode
# ---------------------------------------------------------------------------

MOVES_CO2_G_S: dict[str, float] = {
    "idle":          1.38,
    "decel":         2.15,
    "accel_low":     4.22,
    "accel_high":    8.41,
    "cruise_low":    2.50,
    "cruise_high":   3.80,
}

# Speed-bin emission factors (g CO2 / vehicle-mile)
MOVES_CO2_G_VMT: dict[str, float] = {
    "0_5_mph":   1200.0,
    "5_15_mph":   620.0,
    "15_25_mph":  380.0,
    "25_35_mph":  310.0,
    "35_45_mph":  295.0,
    "45_55_mph":  305.0,
    "55_65_mph":  330.0,
}


def moves_co2_rate(speed_mph: float) -> float:
    """EPA MOVES2014b CO2 emission rate (g/vehicle-mile) by speed bin."""
    if speed_mph < 5:
        return MOVES_CO2_G_VMT["0_5_mph"]
    if speed_mph < 15:
        return MOVES_CO2_G_VMT["5_15_mph"]
    if speed_mph < 25:
        return MOVES_CO2_G_VMT["15_25_mph"]
    if speed_mph < 35:
        return MOVES_CO2_G_VMT["25_35_mph"]
    if speed_mph < 45:
        return MOVES_CO2_G_VMT["35_45_mph"]
    if speed_mph < 55:
        return MOVES_CO2_G_VMT["45_55_mph"]
    return MOVES_CO2_G_VMT["55_65_mph"]


def intersection_co2_kg_hr(
    cycle: float,
    avg_green_ratio: float,
    volume_veh_hr: float,
    approach_speed_mph: float,
) -> float:
    """Estimate CO2 kg/hr at one intersection approach using MOVES2014b.

    Vehicles spend (1 - avg_green_ratio) of cycle time idling/decelerating.
    Running phase: modeled at 60% of approach speed (congested).
    """
    idle_fraction = max(0.0, 1.0 - avg_green_ratio)
    # Idle: MOVES idle rate
    idle_co2_g_s = MOVES_CO2_G_S["idle"]
    # Running: use speed-bin model at approach speed
    run_speed = approach_speed_mph * 0.6  # congested approach
    run_rate = moves_co2_rate(run_speed)  # g/mile
    run_speed_mps = run_speed * 0.44704
    # Running emission in g/s: (g/mile) * (miles/s) = (g/mile) * (mps / 1609.34)
    run_co2_g_s = run_rate * (run_speed_mps / 1609.34)

    avg_co2_g_s = idle_fraction * idle_co2_g_s + (1 - idle_fraction) * run_co2_g_s
    total_g_hr = avg_co2_g_s * 3600 * volume_veh_hr / max(volume_veh_hr, 1)
    return total_g_hr / 1000.0  # g → kg


# ---------------------------------------------------------------------------
# Chromosome: one candidate timing solution
# ---------------------------------------------------------------------------

@dataclass
class TimingChromosome:
    """NSGA-III candidate solution encoding."""
    cycle: float                    # seconds
    green_ratios: list[float]       # effective green / cycle, per intersection
    offsets: list[float]            # coordination offsets, per intersection

    # Objective values (lower = better for all)
    delay: float = 0.0
    emissions: float = 0.0
    stops: float = 0.0
    safety: float = 0.0
    equity: float = 0.0

    # NSGA-III metadata
    rank: int = 0
    reference_point_idx: int = -1
    niche_count: int = 0

    @property
    def objectives(self) -> list[float]:
        return [self.delay, self.emissions, self.stops, self.safety, self.equity]


# ---------------------------------------------------------------------------
# NSGA-III reference points (Das & Dennis simplex lattice)
# ---------------------------------------------------------------------------

def generate_reference_points(n_obj: int, n_divisions: int) -> list[list[float]]:
    """Generate simplex lattice reference points on normalized hyperplane.

    Total points = C(n_obj + n_divisions - 1, n_divisions).
    For 5 objectives, n_divisions=4 gives C(8,4) = 70 reference points.
    """
    def _recursive(n_remaining, n_pts, prefix):
        if n_remaining == 1:
            yield prefix + [n_pts]
            return
        for i in range(n_pts + 1):
            yield from _recursive(n_remaining - 1, n_pts - i, prefix + [i])

    raw = list(_recursive(n_obj, n_divisions, []))
    return [[v / n_divisions for v in pt] for pt in raw]


def normalize_objectives(
    population: list[TimingChromosome],
    ideal: list[float],
    nadir: list[float],
) -> list[list[float]]:
    """Translate and normalize objective vectors to [0, 1] hyperplane."""
    result = []
    for ind in population:
        norm = []
        for j, obj in enumerate(ind.objectives):
            denom = max(nadir[j] - ideal[j], 1e-9)
            norm.append((obj - ideal[j]) / denom)
        result.append(norm)
    return result


def associate_reference_points(
    norm_objs: list[list[float]],
    ref_points: list[list[float]],
) -> list[int]:
    """Associate each individual to its nearest reference point."""
    associations = []
    for norm_obj in norm_objs:
        best_rp = 0
        best_dist = float("inf")
        for i, rp in enumerate(ref_points):
            # Perpendicular distance from norm_obj to reference line through rp
            rp_norm = math.sqrt(sum(v ** 2 for v in rp)) or 1.0
            dot = sum(norm_obj[j] * rp[j] for j in range(len(rp))) / rp_norm
            dist_sq = sum((norm_obj[j] - dot * rp[j] / rp_norm) ** 2 for j in range(len(rp)))
            if dist_sq < best_dist:
                best_dist = dist_sq
                best_rp = i
        associations.append(best_rp)
    return associations


# ---------------------------------------------------------------------------
# Non-dominated sorting (fast non-dominated sort, Deb 2002)
# ---------------------------------------------------------------------------

def dominates(a: list[float], b: list[float]) -> bool:
    """Return True if objective vector a dominates b (all ≤, at least one <)."""
    all_leq = all(ai <= bi for ai, bi in zip(a, b))
    any_lt  = any(ai <  bi for ai, bi in zip(a, b))
    return all_leq and any_lt


def fast_non_dominated_sort(population: list[TimingChromosome]) -> list[list[int]]:
    """Partition population into Pareto fronts F1, F2, ...

    Returns list of fronts, each containing indices into population.
    """
    n = len(population)
    S: list[list[int]] = [[] for _ in range(n)]   # dominated-by sets
    n_dom: list[int] = [0] * n                    # domination counter

    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if dominates(population[p].objectives, population[q].objectives):
                S[p].append(q)
            elif dominates(population[q].objectives, population[p].objectives):
                n_dom[p] += 1
        if n_dom[p] == 0:
            population[p].rank = 1
            fronts[0].append(p)

    i = 0
    while fronts[i]:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    population[q].rank = i + 2
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return [f for f in fronts if f]


# ---------------------------------------------------------------------------
# Objective evaluation
# ---------------------------------------------------------------------------

def evaluate_chromosome(
    chrom: TimingChromosome,
    demand_profiles: list[DemandProfile],
    corridor: Corridor,
) -> None:
    """Compute all 5 objective values in-place."""
    from aito.optimization.isolated_optimizer import _critical_flow_ratios, SAT_FLOW_THRU

    delays: list[float] = []
    co2_total: float = 0.0
    stops_total: list[float] = []
    conflict_total: float = 0.0

    n = len(corridor.intersections)
    green_ratio = chrom.green_ratios[0] if chrom.green_ratios else 0.4

    for i, (ix, dp) in enumerate(zip(corridor.intersections, demand_profiles)):
        gr = chrom.green_ratios[i] if i < len(chrom.green_ratios) else green_ratio
        critical = _critical_flow_ratios(dp)
        vol_total = sum(v for v, _ in critical) * 3600  # veh/hr
        cap = gr * SAT_FLOW_THRU

        d = hcm_delay(chrom.cycle, gr * chrom.cycle, sum(y for _, y in critical), cap, vol_total)
        delays.append(d)

        co2_total += intersection_co2_kg_hr(
            chrom.cycle, gr, vol_total, ix.approach_speed_mph
        )
        st = stops_per_vehicle(chrom.cycle, gr * chrom.cycle, sum(y for _, y in critical))
        stops_total.append(st)

        # Safety: left-turn exposure × approach speed proxy
        lt_flow = (dp.north_left + dp.south_left + dp.east_left + dp.west_left)
        conflict_total += lt_flow * ix.approach_speed_mph / 1000.0

    chrom.delay = sum(delays) / max(n, 1)
    chrom.emissions = co2_total
    chrom.stops = sum(stops_total) / max(n, 1)
    chrom.safety = conflict_total / max(n, 1)
    # Equity: std of per-intersection delays
    mean_d = chrom.delay
    chrom.equity = math.sqrt(sum((d - mean_d) ** 2 for d in delays) / max(n, 1))


# ---------------------------------------------------------------------------
# Genetic operators
# ---------------------------------------------------------------------------

def _random_chromosome(
    n_intersections: int,
    min_cycle: float,
    max_cycle: float,
    rng: random.Random,
) -> TimingChromosome:
    cycle = rng.uniform(min_cycle, max_cycle)
    green_ratios = [rng.uniform(0.30, 0.65) for _ in range(n_intersections)]
    offsets = [rng.uniform(0, cycle) for _ in range(n_intersections)]
    offsets[0] = 0.0  # reference intersection
    return TimingChromosome(cycle=cycle, green_ratios=green_ratios, offsets=offsets)


def crossover(
    parent_a: TimingChromosome,
    parent_b: TimingChromosome,
    rng: random.Random,
    eta_c: float = 15.0,
) -> tuple[TimingChromosome, TimingChromosome]:
    """Simulated Binary Crossover (SBX) for real-valued chromosomes."""
    n = len(parent_a.green_ratios)

    def _sbx_scalar(x1: float, x2: float, lo: float, hi: float) -> tuple[float, float]:
        if abs(x1 - x2) < 1e-9:
            return x1, x2
        u = rng.random()
        beta = (2 * u) ** (1.0 / (eta_c + 1)) if u <= 0.5 else (1.0 / (2 * (1 - u))) ** (1.0 / (eta_c + 1))
        c1 = 0.5 * ((1 + beta) * x1 + (1 - beta) * x2)
        c2 = 0.5 * ((1 - beta) * x1 + (1 + beta) * x2)
        return max(lo, min(hi, c1)), max(lo, min(hi, c2))

    c_cycle1, c_cycle2 = _sbx_scalar(parent_a.cycle, parent_b.cycle, 60.0, 180.0)

    gr1, gr2 = [], []
    off1, off2 = [0.0], [0.0]  # reference intersection always 0
    for i in range(n):
        g1, g2 = _sbx_scalar(parent_a.green_ratios[i], parent_b.green_ratios[i], 0.20, 0.75)
        gr1.append(g1); gr2.append(g2)

    for i in range(1, n):
        o1, o2 = _sbx_scalar(parent_a.offsets[i], parent_b.offsets[i], 0.0, c_cycle1)
        off1.append(o1); off2.append(o2)

    return (
        TimingChromosome(cycle=c_cycle1, green_ratios=gr1, offsets=off1),
        TimingChromosome(cycle=c_cycle2, green_ratios=gr2, offsets=off2),
    )


def mutate(
    chrom: TimingChromosome,
    rng: random.Random,
    mutation_rate: float = 0.1,
    eta_m: float = 20.0,
) -> TimingChromosome:
    """Polynomial mutation."""
    def _poly_mutate(x: float, lo: float, hi: float) -> float:
        if rng.random() > mutation_rate:
            return x
        u = rng.random()
        delta = ((2 * u) ** (1.0 / (eta_m + 1)) - 1) if u < 0.5 else (1 - (2 * (1 - u)) ** (1.0 / (eta_m + 1)))
        return max(lo, min(hi, x + delta * (hi - lo)))

    new_cycle = _poly_mutate(chrom.cycle, 60.0, 180.0)
    new_gr = [_poly_mutate(g, 0.20, 0.75) for g in chrom.green_ratios]
    new_off = [0.0] + [_poly_mutate(o, 0.0, new_cycle) for o in chrom.offsets[1:]]
    return TimingChromosome(cycle=new_cycle, green_ratios=new_gr, offsets=new_off)


# ---------------------------------------------------------------------------
# NSGA-III optimizer
# ---------------------------------------------------------------------------

@dataclass
class NSGAIIIResult:
    """Output from the NSGA-III optimizer."""
    pareto_front: list[TimingChromosome]
    all_generations: list[list[TimingChromosome]]
    n_generations: int
    n_evaluations: int
    objective_names: list[str] = field(default_factory=lambda: [
        "delay_s_veh", "emissions_kg_hr", "stops_per_veh",
        "safety_index", "equity_s"
    ])

    @property
    def best_delay(self) -> TimingChromosome:
        return min(self.pareto_front, key=lambda c: c.delay)

    @property
    def best_emissions(self) -> TimingChromosome:
        return min(self.pareto_front, key=lambda c: c.emissions)

    @property
    def balanced(self) -> TimingChromosome:
        """Individual closest to the normalized ideal point."""
        objs = [c.objectives for c in self.pareto_front]
        ideal = [min(o[j] for o in objs) for j in range(5)]
        nadir = [max(o[j] for o in objs) for j in range(5)]
        def _dist(c):
            return sum(
                ((c.objectives[j] - ideal[j]) / max(nadir[j] - ideal[j], 1e-9)) ** 2
                for j in range(5)
            )
        return min(self.pareto_front, key=_dist)


class NSGAIIIOptimizer:
    """NSGA-III for 5-objective traffic signal timing.

    Parameters
    ----------
    corridor : Corridor
    population_size : int
        Must be divisible by 4 (crossover pairs). Default 80.
    n_generations : int
        Number of evolutionary generations. Default 50.
    n_divisions : int
        Reference point lattice divisions. Default 4 (gives 70 ref points for 5 obj).
    seed : int
        Random seed for reproducibility.
    """

    N_OBJECTIVES = 5

    def __init__(
        self,
        corridor: Corridor,
        population_size: int = 80,
        n_generations: int = 50,
        n_divisions: int = 4,
        seed: int = 42,
    ) -> None:
        self.corridor = corridor
        self.pop_size = max(4, (population_size // 4) * 4)
        self.n_gen = n_generations
        self.ref_points = generate_reference_points(self.N_OBJECTIVES, n_divisions)
        self.rng = random.Random(seed)

    def optimize(
        self,
        demand_profiles: list[DemandProfile],
        min_cycle: float = 60.0,
        max_cycle: float = 180.0,
        travel_data: Optional[list[SegmentTravelData]] = None,
    ) -> NSGAIIIResult:
        """Run NSGA-III and return the Pareto front."""
        n = len(self.corridor.intersections)
        assert len(demand_profiles) == n

        # Initialise population
        pop = [
            _random_chromosome(n, min_cycle, max_cycle, self.rng)
            for _ in range(self.pop_size)
        ]
        for ind in pop:
            evaluate_chromosome(ind, demand_profiles, self.corridor)

        all_gens: list[list[TimingChromosome]] = []
        n_evals = self.pop_size

        for gen in range(self.n_gen):
            # Generate offspring via tournament → crossover → mutate
            offspring = self._generate_offspring(pop, n, min_cycle, max_cycle)
            for ind in offspring:
                evaluate_chromosome(ind, demand_profiles, self.corridor)
            n_evals += len(offspring)

            # Combine + sort
            combined = pop + offspring
            fronts = fast_non_dominated_sort(combined)

            # Build next population using reference-point-based niching
            next_pop: list[TimingChromosome] = []
            for front in fronts:
                if len(next_pop) + len(front) <= self.pop_size:
                    next_pop.extend(combined[i] for i in front)
                else:
                    # Fill remaining slots using niching on this front
                    remaining = self.pop_size - len(next_pop)
                    selected = self._niching_select(
                        [combined[i] for i in front],
                        next_pop,
                        remaining,
                    )
                    next_pop.extend(selected)
                    break

            pop = next_pop
            all_gens.append(list(pop))

        # Extract Pareto front
        pareto_indices = fast_non_dominated_sort(pop)[0]
        pareto = [pop[i] for i in pareto_indices]

        return NSGAIIIResult(
            pareto_front=pareto,
            all_generations=all_gens,
            n_generations=self.n_gen,
            n_evaluations=n_evals,
        )

    def _tournament_select(self, pop: list[TimingChromosome]) -> TimingChromosome:
        a, b = self.rng.sample(pop, 2)
        return a if a.rank <= b.rank else b

    def _generate_offspring(
        self,
        pop: list[TimingChromosome],
        n_intersections: int,
        min_cycle: float,
        max_cycle: float,
    ) -> list[TimingChromosome]:
        offspring = []
        while len(offspring) < self.pop_size:
            p1 = self._tournament_select(pop)
            p2 = self._tournament_select(pop)
            c1, c2 = crossover(p1, p2, self.rng)
            offspring.append(mutate(c1, self.rng))
            offspring.append(mutate(c2, self.rng))
        return offspring[:self.pop_size]

    def _niching_select(
        self,
        front: list[TimingChromosome],
        already_selected: list[TimingChromosome],
        n_needed: int,
    ) -> list[TimingChromosome]:
        """Select n_needed individuals from front using reference-point niching."""
        if len(front) <= n_needed:
            return front

        all_pop = already_selected + front
        objs = [ind.objectives for ind in all_pop]
        ideal = [min(o[j] for o in objs) for j in range(self.N_OBJECTIVES)]
        nadir = [max(o[j] for o in objs) for j in range(self.N_OBJECTIVES)]
        norm_objs = normalize_objectives(all_pop, ideal, nadir)

        # Count niche occupancy in already-selected
        rp_count: dict[int, int] = {i: 0 for i in range(len(self.ref_points))}
        front_norm_start = len(already_selected)
        for idx in range(front_norm_start):
            assoc = associate_reference_points([norm_objs[idx]], self.ref_points)[0]
            rp_count[assoc] = rp_count.get(assoc, 0) + 1

        selected = []
        front_associations = associate_reference_points(
            norm_objs[front_norm_start:], self.ref_points
        )

        # Iteratively pick from least-crowded reference points
        available = list(range(len(front)))
        self.rng.shuffle(available)

        while len(selected) < n_needed and available:
            # Find ref point with minimum niche count
            min_count = min(rp_count[front_associations[i]] for i in available)
            candidates = [i for i in available if rp_count[front_associations[i]] == min_count]
            chosen = self.rng.choice(candidates)
            selected.append(front[chosen])
            rp_count[front_associations[chosen]] += 1
            available.remove(chosen)

        return selected

    def to_optimization_result(
        self,
        nsga_result: NSGAIIIResult,
        request: "OptimizationRequest",
        corridor: Corridor,
        computation_seconds: float = 0.0,
    ) -> OptimizationResult:
        """Convert NSGA-III result to AITO OptimizationResult for API compatibility."""
        from aito.models import OptimizationResult, ParetoSolution, CorridorPlan
        from aito.optimization.corridor_optimizer import CorridorOptimizer

        # Build corridor plans for Pareto front members
        pareto_solutions: list[ParetoSolution] = []
        corridor_optimizer = CorridorOptimizer(corridor)

        for chrom in nsga_result.pareto_front:
            # Use CorridorOptimizer to build a proper plan at this cycle
            try:
                opt_result = corridor_optimizer._optimize_at_cycle(
                    demand_profiles=request.demand_profiles,
                    travel_data=corridor_optimizer._default_travel_data(),
                    cycle=chrom.cycle,
                    period="NSGA-III",
                )
                plan = opt_result.corridor_plan
            except Exception:
                continue

            desc = (
                f"Cycle={chrom.cycle:.0f}s, "
                f"Delay={chrom.delay:.1f}s/veh, "
                f"CO2={chrom.emissions:.1f}kg/hr"
            )
            pareto_solutions.append(ParetoSolution(
                plan=plan,
                delay_score=chrom.delay,
                emissions_score=chrom.emissions,
                stops_score=chrom.stops,
                safety_score=chrom.safety,
                equity_score=chrom.equity,
                description=desc,
            ))

        if not pareto_solutions:
            # Fallback to single default result
            opt_result = corridor_optimizer.optimize(request.demand_profiles)
            plan = opt_result.corridor_plan
            pareto_solutions = [ParetoSolution(
                plan=plan,
                delay_score=30.0,
                emissions_score=50.0,
                stops_score=0.6,
                safety_score=0.5,
                equity_score=5.0,
                description="Default fallback",
            )]

        # Pick balanced solution as recommended
        balanced = nsga_result.balanced
        rec_idx = 0
        if pareto_solutions:
            best_dist = float("inf")
            for idx, sol in enumerate(pareto_solutions):
                d = (sol.delay_score - balanced.delay) ** 2
                if d < best_dist:
                    best_dist = d
                    rec_idx = idx

        return OptimizationResult(
            corridor_id=corridor.id,
            pareto_solutions=pareto_solutions,
            recommended_solution=pareto_solutions[rec_idx],
            computation_seconds=computation_seconds,
        )
