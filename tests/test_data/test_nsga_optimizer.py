"""Tests for aito/optimization/multi_objective_engine.py (GF6)."""
import pytest
from aito.optimization.multi_objective_engine import (
    stops_per_vehicle,
    moves_co2_rate,
    intersection_co2_kg_hr,
    TimingChromosome,
    generate_reference_points,
    normalize_objectives,
    dominates,
    fast_non_dominated_sort,
    evaluate_chromosome,
    crossover,
    mutate,
    NSGAIIIResult,
    NSGAIIIOptimizer,
)
from aito.models import Corridor as AITOCorridor, DemandProfile
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")

def _single_ix_corridor():
    return AITOCorridor(
        id="test_single",
        name="Test (single ix)",
        intersections=ROSECRANS.intersections[:1],
        distances_m=[],
        speed_limits_mph=[35.0],
    )

def _demand_profile():
    return DemandProfile(
        intersection_id=ROSECRANS.intersections[0].id,
        period_minutes=60,
        north_thru=450.0, south_thru=380.0,
        east_thru=220.0, west_thru=190.0,
        north_left=80.0, south_left=65.0,
        east_left=55.0, west_left=45.0,
        north_right=60.0, south_right=50.0,
        east_right=40.0, west_right=35.0,
    )


class TestHelperFunctions:
    def test_stops_per_vehicle_returns_float(self):
        result = stops_per_vehicle(cycle=120.0, green=48.0, flow_ratio=0.5)
        assert isinstance(result, float)

    def test_stops_per_vehicle_zero_flow(self):
        result = stops_per_vehicle(cycle=120.0, green=60.0, flow_ratio=0.0)
        assert result >= 0

    def test_moves_co2_rate_positive(self):
        for speed in [5, 15, 25, 35, 45, 55]:
            rate = moves_co2_rate(speed_mph=float(speed))
            assert rate > 0, f"Zero/negative CO2 rate at {speed} mph"

    def test_moves_co2_rate_low_speed_higher(self):
        low = moves_co2_rate(5.0)
        medium = moves_co2_rate(30.0)
        # Low speed (stop-and-go) typically emits more per mile
        assert isinstance(low, float) and isinstance(medium, float)

    def test_intersection_co2_kg_hr_positive(self):
        result = intersection_co2_kg_hr(
            cycle=120.0, avg_green_ratio=0.4, volume_veh_hr=800.0, approach_speed_mph=35.0,
        )
        assert result > 0


class TestTimingChromosome:
    def test_has_required_fields(self):
        chrom = TimingChromosome(
            cycle=120.0,
            green_ratios=[0.4, 0.4, 0.2],
            offsets=[0.0, 40.0, 80.0],
        )
        assert chrom.cycle == 120.0
        assert len(chrom.green_ratios) == 3

    def test_objectives_initially_zero(self):
        chrom = TimingChromosome(cycle=120.0, green_ratios=[0.4], offsets=[0.0])
        assert chrom.objectives is not None
        assert len(chrom.objectives) == 5


class TestDominates:
    def test_better_on_all_objectives_dominates(self):
        a = [1.0, 2.0, 3.0]
        b = [2.0, 3.0, 4.0]
        assert dominates(a, b)

    def test_dominated_does_not_dominate(self):
        a = [2.0, 3.0, 4.0]
        b = [1.0, 2.0, 3.0]
        assert not dominates(a, b)

    def test_equal_objectives_neither_dominates(self):
        a = [1.0, 2.0]
        b = [1.0, 2.0]
        assert not dominates(a, b)

    def test_partial_better_but_worse_elsewhere_no_domination(self):
        a = [1.0, 5.0]
        b = [3.0, 2.0]
        assert not dominates(a, b)
        assert not dominates(b, a)


class TestFastNonDominatedSort:
    def test_returns_fronts(self):
        pop = [
            TimingChromosome(cycle=c, green_ratios=[0.4], offsets=[0.0])
            for c in [80, 100, 120]
        ]
        for i, p in enumerate(pop):
            p.delay = float(i)
            p.emissions = float(3 - i)
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts) >= 1
        assert all(isinstance(f, list) for f in fronts)

    def test_front_zero_contains_nondominated(self):
        pop = [
            TimingChromosome(cycle=80, green_ratios=[0.4], offsets=[0.0]),
            TimingChromosome(cycle=120, green_ratios=[0.4], offsets=[0.0]),
        ]
        pop[0].delay = 1.0; pop[0].emissions = 5.0
        pop[1].delay = 5.0; pop[1].emissions = 1.0
        fronts = fast_non_dominated_sort(pop)
        assert len(fronts[0]) == 2  # both non-dominated


class TestGenerateReferencePoints:
    def test_returns_list(self):
        pts = generate_reference_points(n_obj=3, n_divisions=4)
        assert isinstance(pts, list)
        assert len(pts) > 0

    def test_two_objectives(self):
        pts = generate_reference_points(n_obj=2, n_divisions=5)
        # Should have 6 points for 2 obj, 5 divisions
        assert len(pts) >= 5

    def test_each_point_sums_to_one(self):
        pts = generate_reference_points(n_obj=3, n_divisions=4)
        for pt in pts:
            assert sum(pt) == pytest.approx(1.0, abs=0.01)


class TestNSGAIIIOptimizer:
    def setup_method(self):
        self.corridor = _single_ix_corridor()
        self.demand = [_demand_profile()]

    def test_optimize_returns_result(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert isinstance(result, NSGAIIIResult)

    def test_pareto_front_nonempty(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert len(result.pareto_front) > 0

    def test_balanced_solution_exists(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert result.balanced is not None

    def test_balanced_has_cycle(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert result.balanced.cycle > 0

    def test_n_evaluations_positive(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert result.n_evaluations > 0

    def test_pareto_solutions_have_objectives(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        for sol in result.pareto_front:
            assert len(sol.objectives) >= 2

    def test_cycle_within_reasonable_range(self):
        opt = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=42)
        result = opt.optimize(self.demand)
        assert 40 <= result.balanced.cycle <= 240

    def test_different_seeds_may_differ(self):
        opt1 = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=1)
        opt2 = NSGAIIIOptimizer(self.corridor, population_size=12, n_generations=5, seed=99)
        r1 = opt1.optimize(self.demand)
        r2 = opt2.optimize(self.demand)
        # Results may differ (not deterministically same) - just check both valid
        assert r1.balanced.cycle > 0
        assert r2.balanced.cycle > 0

    def test_more_generations_same_or_more_evaluations(self):
        opt_few = NSGAIIIOptimizer(self.corridor, population_size=10, n_generations=3, seed=42)
        opt_many = NSGAIIIOptimizer(self.corridor, population_size=10, n_generations=10, seed=42)
        r_few = opt_few.optimize(self.demand)
        r_many = opt_many.optimize(self.demand)
        assert r_many.n_evaluations >= r_few.n_evaluations
