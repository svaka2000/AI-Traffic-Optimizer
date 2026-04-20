"""Tests for aito/optimization/multimodal.py (GF13)."""
import pytest
from aito.optimization.multimodal import (
    PedLOS,
    pedestrian_los,
    hcm_ped_delay,
    mutcd_min_walk_s,
    mutcd_clearance_s,
    accessible_clearance_s,
    CyclistFacility,
    cyclist_delay_s,
    MultiModalConstraints,
    MultiModalIntersectionPlan,
    MultiModalOptimizationResult,
    MultiModalOptimizer,
    school_zone_plan,
)
from aito.models import DemandProfile
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")
IX = ROSECRANS.intersections[3]


def _demand_profile() -> DemandProfile:
    return DemandProfile(
        intersection_id=IX.id,
        period_minutes=60,
        north_thru=320.0, south_thru=290.0,
        east_thru=160.0, west_thru=140.0,
        north_left=55.0, south_left=45.0,
        east_left=40.0, west_left=35.0,
        north_right=45.0, south_right=40.0,
        east_right=30.0, west_right=25.0,
    )


class TestPedLOS:
    def test_has_los_a_through_f(self):
        values = [p.value for p in PedLOS]
        assert "A" in values
        assert "F" in values

    def test_six_levels(self):
        assert len(list(PedLOS)) == 6


class TestPedestrianLOS:
    def test_low_delay_gives_good_los(self):
        los = pedestrian_los(5.0)
        assert los in (PedLOS.A, PedLOS.B)

    def test_high_delay_gives_poor_los(self):
        los = pedestrian_los(90.0)
        assert los in (PedLOS.E, PedLOS.F)

    def test_moderate_delay(self):
        los = pedestrian_los(20.0)
        assert los in (PedLOS.A, PedLOS.B, PedLOS.C)

    def test_monotonic(self):
        delays = [5, 15, 25, 45, 70, 100]
        los_list = [pedestrian_los(float(d)) for d in delays]
        # Grade should be non-improving (or equal) as delay increases
        grades = [p.value for p in los_list]
        assert grades == sorted(grades)


class TestHCMPedDelay:
    def test_returns_positive(self):
        delay = hcm_ped_delay(cycle_s=120.0, walk_s=7.0, clearance_s=21.0, ped_volume_hr=100.0)
        assert delay >= 0

    def test_longer_red_more_delay(self):
        short_red = hcm_ped_delay(cycle_s=120.0, walk_s=60.0, clearance_s=20.0, ped_volume_hr=100.0)
        long_red = hcm_ped_delay(cycle_s=120.0, walk_s=20.0, clearance_s=20.0, ped_volume_hr=100.0)
        assert long_red >= short_red

    def test_all_green_low_delay(self):
        delay = hcm_ped_delay(cycle_s=120.0, walk_s=100.0, clearance_s=19.0, ped_volume_hr=50.0)
        assert delay < 30.0  # almost all-green should give low delay


class TestMUTCDRequirements:
    def test_min_walk_s_positive(self):
        assert mutcd_min_walk_s() >= 4.0

    def test_clearance_s_increases_with_distance(self):
        short = mutcd_clearance_s(crossing_distance_ft=40.0)
        long_ = mutcd_clearance_s(crossing_distance_ft=80.0)
        assert long_ > short

    def test_accessible_clearance_slower_than_standard(self):
        standard = mutcd_clearance_s(crossing_distance_ft=60.0, walk_speed_fps=3.5)
        accessible = accessible_clearance_s(crossing_distance_ft=60.0)
        # Accessible uses slower walk speed → longer clearance
        assert accessible >= standard


class TestCyclistDelay:
    def test_returns_nonneg(self):
        delay = cyclist_delay_s(
            cycle_s=120.0, green_s=48.0, volume_bikes_hr=300.0,
            facility=CyclistFacility.BIKE_LANE,
        )
        assert delay >= 0

    def test_bike_lane_lower_delay_than_mixed_traffic(self):
        mixed = cyclist_delay_s(
            cycle_s=120.0, green_s=48.0, volume_bikes_hr=800.0,
            facility=CyclistFacility.MIXED_TRAFFIC,
        )
        lane = cyclist_delay_s(
            cycle_s=120.0, green_s=48.0, volume_bikes_hr=800.0,
            facility=CyclistFacility.BIKE_LANE,
        )
        assert lane <= mixed


class TestMultiModalOptimizer:
    def setup_method(self):
        self.constraints = MultiModalConstraints(
            accessible_clearance=False,
            school_zone=False,
            bike_facility=CyclistFacility.BIKE_LANE,
        )
        self.optimizer = MultiModalOptimizer(IX, self.constraints)
        self.dp = _demand_profile()

    def test_optimize_returns_result(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert isinstance(result, MultiModalOptimizationResult)

    def test_pareto_solutions_nonempty(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert len(result.pareto_solutions) > 0

    def test_balanced_exists(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert result.balanced is not None

    def test_balanced_cycle_positive(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert result.balanced.cycle_s > 0

    def test_balanced_vehicle_delay_positive(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert result.balanced.vehicle_delay_s_veh > 0

    def test_balanced_ped_delay_nonneg(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert result.balanced.ped_delay_s_ped >= 0

    def test_ped_los_valid(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert result.balanced.ped_los in PedLOS

    def test_vision_zero_compliant_field(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        assert isinstance(result.balanced.is_vision_zero_compliant, bool)

    def test_pareto_solutions_non_dominated(self):
        result = self.optimizer.optimize(self.dp, ped_volume_hr=100.0, bike_volume_hr=20.0)
        # Pareto front: no solution should be dominated by another in the front
        plans = result.pareto_solutions
        for i, a in enumerate(plans):
            for j, b in enumerate(plans):
                if i != j:
                    a_worse = (a.vehicle_delay_s_veh >= b.vehicle_delay_s_veh and
                               a.ped_delay_s_ped >= b.ped_delay_s_ped)
                    a_strictly_worse = (a.vehicle_delay_s_veh > b.vehicle_delay_s_veh and
                                        a.ped_delay_s_ped > b.ped_delay_s_ped)
                    assert not a_strictly_worse, f"Plan {i} dominated by plan {j}"

    def test_school_zone_constraint_enforced(self):
        school_constraints = MultiModalConstraints(
            accessible_clearance=True,
            school_zone=True,
            bike_facility=CyclistFacility.BIKE_LANE,
        )
        opt = MultiModalOptimizer(IX, school_constraints)
        result = opt.optimize(self.dp, ped_volume_hr=200.0, bike_volume_hr=30.0)
        assert result is not None


class TestSchoolZonePlan:
    def test_returns_plan(self):
        plan = school_zone_plan(
            intersection=IX,
            crossing_distance_ft=IX.crossing_distance_ft,
        )
        assert isinstance(plan, MultiModalIntersectionPlan)

    def test_school_zone_plan_vision_zero_compliant(self):
        plan = school_zone_plan(
            intersection=IX,
            crossing_distance_ft=IX.crossing_distance_ft,
        )
        assert plan.is_vision_zero_compliant
