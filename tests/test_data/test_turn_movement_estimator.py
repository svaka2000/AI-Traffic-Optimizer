"""Tests for aito/data/turn_movement_estimator.py (GF5)."""
import pytest
from datetime import datetime, timedelta

from aito.data.turn_movement_estimator import (
    bearing,
    heading_change,
    classify_approach,
    classify_movement,
    Approach,
    Movement,
    IntersectionBBox,
    TrajectoryPoint,
    VehicleTrajectory,
    TurnMovementRecord,
    TurnMovementCounts,
    TrajectoryTurnMovementEstimator,
    generate_synthetic_trajectories,
    estimate_corridor_demands,
)
from aito.data.san_diego_inventory import DOWNTOWN_INTERSECTIONS, DOWNTOWN_CORRIDOR

NOW = datetime(2025, 6, 1, 7, 0)
IX_LAT = 32.7502
IX_LON = -117.1286


# ---------------------------------------------------------------------------
# Unit tests: geometry
# ---------------------------------------------------------------------------

class TestBearing:
    def test_due_north(self):
        brg = bearing(0.0, 0.0, 1.0, 0.0)
        assert abs(brg - 0.0) < 1.0

    def test_due_east(self):
        brg = bearing(0.0, 0.0, 0.0, 1.0)
        assert abs(brg - 90.0) < 1.0

    def test_due_south(self):
        brg = bearing(1.0, 0.0, 0.0, 0.0)
        assert abs(brg - 180.0) < 1.0

    def test_due_west(self):
        brg = bearing(0.0, 1.0, 0.0, 0.0)
        assert abs(brg - 270.0) < 1.0


class TestHeadingChange:
    def test_right_turn(self):
        delta = heading_change(90.0, 180.0)
        assert abs(delta - 90.0) < 1.0

    def test_left_turn(self):
        delta = heading_change(90.0, 0.0)
        assert abs(delta + 90.0) < 1.0

    def test_through(self):
        delta = heading_change(180.0, 180.0)
        assert abs(delta) < 1.0

    def test_wrap_around(self):
        delta = heading_change(10.0, 350.0)
        assert abs(delta + 20.0) < 1.0


class TestClassifyApproach:
    def test_northbound(self):
        # Vehicle heading north (bearing ~0)
        assert classify_approach(5.0) == Approach.NORTH

    def test_eastbound(self):
        assert classify_approach(90.0) == Approach.EAST

    def test_southbound(self):
        assert classify_approach(180.0) == Approach.SOUTH

    def test_westbound(self):
        assert classify_approach(270.0) == Approach.WEST

    def test_northwest(self):
        # 315 = NW boundary, should be NORTH
        assert classify_approach(315.0) == Approach.NORTH


class TestClassifyMovement:
    def test_through(self):
        assert classify_movement(5.0) == Movement.THROUGH
        assert classify_movement(-10.0) == Movement.THROUGH

    def test_right_turn(self):
        assert classify_movement(90.0) == Movement.RIGHT_TURN

    def test_left_turn(self):
        assert classify_movement(-90.0) == Movement.LEFT_TURN

    def test_u_turn(self):
        assert classify_movement(180.0) == Movement.U_TURN
        assert classify_movement(-175.0) == Movement.U_TURN

    def test_slight_right(self):
        assert classify_movement(50.0) == Movement.RIGHT_TURN


# ---------------------------------------------------------------------------
# IntersectionBBox
# ---------------------------------------------------------------------------

class TestIntersectionBBox:
    def test_contains_center(self):
        bbox = IntersectionBBox(IX_LAT, IX_LON, radius_m=75.0)
        assert bbox.contains(IX_LAT, IX_LON)

    def test_excludes_distant_point(self):
        bbox = IntersectionBBox(IX_LAT, IX_LON, radius_m=75.0)
        # ~200m north
        assert not bbox.contains(IX_LAT + 0.002, IX_LON)

    def test_includes_nearby(self):
        bbox = IntersectionBBox(IX_LAT, IX_LON, radius_m=75.0)
        # ~50m north (0.00045 deg ≈ 50m)
        assert bbox.contains(IX_LAT + 0.00045, IX_LON)


# ---------------------------------------------------------------------------
# VehicleTrajectory
# ---------------------------------------------------------------------------

class TestVehicleTrajectory:
    def _make_traj(self, entry_heading=180.0, exit_heading=180.0):
        """Make a trajectory heading south (entry_heading=180 → south approach)."""
        pts = []
        # Approach from north
        for i in range(3):
            pts.append(TrajectoryPoint(
                vehicle_id="v001",
                timestamp=NOW + timedelta(seconds=i * 5),
                latitude=IX_LAT + 0.001 * (2 - i),
                longitude=IX_LON,
                speed_mph=25.0,
            ))
        # Depart going south
        for i in range(3):
            pts.append(TrajectoryPoint(
                vehicle_id="v001",
                timestamp=NOW + timedelta(seconds=15 + i * 5),
                latitude=IX_LAT - 0.001 * (i + 1),
                longitude=IX_LON,
                speed_mph=25.0,
            ))
        traj = VehicleTrajectory(vehicle_id="v001", points=pts)
        return traj

    def test_duration(self):
        traj = self._make_traj()
        assert traj.duration_s > 0

    def test_entry_bearing_south(self):
        traj = self._make_traj()
        brg = traj.entry_bearing()
        # Moving from north lat toward intersection → bearing south ~180°
        assert brg is not None
        assert 150 < brg < 210

    def test_exit_bearing_south(self):
        traj = self._make_traj()
        brg = traj.exit_bearing()
        assert brg is not None
        assert 150 < brg < 210


# ---------------------------------------------------------------------------
# TrajectoryTurnMovementEstimator
# ---------------------------------------------------------------------------

class TestTrajectoryEstimator:
    def _make_estimator(self):
        bbox = IntersectionBBox(IX_LAT, IX_LON, radius_m=120.0)
        return TrajectoryTurnMovementEstimator(
            intersection_id="test_ix",
            bbox=bbox,
            penetration_rate=0.28,
        )

    def test_process_synthetic_batch(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=200, period_start=NOW
        )
        records = estimator.process_batch(trajs)
        assert len(records) > 0
        assert all(isinstance(r, TurnMovementRecord) for r in records)

    def test_movement_types_present(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=300, period_start=NOW
        )
        estimator.process_batch(trajs)
        movements = {r.movement for r in estimator._records}
        assert Movement.THROUGH in movements
        assert Movement.LEFT_TURN in movements or Movement.RIGHT_TURN in movements

    def test_aggregate_returns_counts(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=200, period_start=NOW
        )
        estimator.process_batch(trajs)
        period_end = NOW + timedelta(minutes=15)
        counts = estimator.aggregate(NOW, period_end)
        assert counts.total_observed > 0
        assert counts.intersection_id == "test_ix"

    def test_demand_profile_positive_volumes(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=200, period_start=NOW
        )
        estimator.process_batch(trajs)
        profile = estimator.estimate_demand_profile(NOW)
        total = (profile.north_thru + profile.south_thru +
                 profile.east_thru + profile.west_thru)
        assert total > 0, "At least one through movement should have demand"

    def test_penetration_expansion(self):
        counts = TurnMovementCounts(
            intersection_id="test",
            period_start=NOW,
            period_end=NOW + timedelta(minutes=15),
            N_TH=28, S_TH=22, E_TH=12, W_TH=6,
            penetration_rate=0.28,
        )
        expanded = counts.expand()
        # 28 probes / 0.28 = 100 true vehicles
        assert abs(expanded.N_TH - 100) < 3

    def test_demand_profile_veh_per_hour(self):
        # 28 probes in 15 min period × (1/0.28) = 100 veh / 15 min = 400 veh/hr
        counts = TurnMovementCounts(
            intersection_id="test",
            period_start=NOW,
            period_end=NOW + timedelta(minutes=15),
            N_TH=28,
            penetration_rate=0.28,
        )
        profile = counts.to_demand_profile(period_minutes=15)
        assert abs(profile.north_thru - 400.0) < 20.0

    def test_clear_resets(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=100, period_start=NOW
        )
        estimator.process_batch(trajs)
        assert estimator.record_count > 0
        estimator.clear()
        assert estimator.record_count == 0

    def test_confidence_filter(self):
        estimator = self._make_estimator()
        trajs = generate_synthetic_trajectories(
            IX_LAT, IX_LON, n_vehicles=200, period_start=NOW
        )
        estimator.process_batch(trajs)
        period_end = NOW + timedelta(minutes=15)
        counts_strict = estimator.aggregate(NOW, period_end, min_confidence=0.95)
        counts_lax = estimator.aggregate(NOW, period_end, min_confidence=0.0)
        # Strict threshold should filter some records
        assert counts_strict.total_observed <= counts_lax.total_observed


# ---------------------------------------------------------------------------
# Multi-intersection corridor pipeline
# ---------------------------------------------------------------------------

class TestCorridorDemandEstimation:
    def test_estimate_corridor_demands(self):
        intersections = DOWNTOWN_INTERSECTIONS[:2]
        traj_sets = [
            generate_synthetic_trajectories(ix.latitude, ix.longitude,
                                            n_vehicles=150, period_start=NOW)
            for ix in intersections
        ]
        profiles = estimate_corridor_demands(
            intersections=intersections,
            trajectories_per_intersection=traj_sets,
            period_start=NOW,
        )
        assert len(profiles) == 2
        for profile in profiles:
            total = (profile.north_thru + profile.south_thru +
                     profile.east_thru + profile.west_thru)
            assert total > 0


# ---------------------------------------------------------------------------
# Synthetic trajectory generator
# ---------------------------------------------------------------------------

class TestSyntheticGenerator:
    def test_generates_expected_count(self):
        # n_vehicles=200, penetration_rate=0.28 → ~56 CVs
        trajs = generate_synthetic_trajectories(IX_LAT, IX_LON, n_vehicles=200,
                                                period_start=NOW, penetration_rate=0.28)
        assert 40 <= len(trajs) <= 70

    def test_trajectories_near_intersection(self):
        bbox = IntersectionBBox(IX_LAT, IX_LON, radius_m=120.0)
        trajs = generate_synthetic_trajectories(IX_LAT, IX_LON, n_vehicles=200,
                                                period_start=NOW)
        # Most trajectories should pass through the bbox
        passing = sum(
            1 for t in trajs
            if any(bbox.contains(p.latitude, p.longitude) for p in t.points)
        )
        assert passing > len(trajs) * 0.5

    def test_all_vehicle_ids_unique(self):
        trajs = generate_synthetic_trajectories(IX_LAT, IX_LON, n_vehicles=100,
                                                period_start=NOW)
        ids = [t.vehicle_id for t in trajs]
        assert len(ids) == len(set(ids))

    def test_timestamps_in_period(self):
        trajs = generate_synthetic_trajectories(IX_LAT, IX_LON, n_vehicles=50,
                                                period_start=NOW, period_minutes=15)
        period_end = NOW + timedelta(minutes=15)
        for traj in trajs:
            assert traj.points[0].timestamp >= NOW
            assert traj.points[0].timestamp < period_end
