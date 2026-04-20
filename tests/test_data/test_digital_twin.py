"""Tests for aito/simulation/digital_twin.py (GF4)."""
import pytest
from aito.simulation.digital_twin import (
    FundamentalDiagram,
    CTMCell,
    CalibrationObservation,
    calibrate_fd,
    SimulationResult,
    DigitalTwin,
)
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")
DISTANCES = ROSECRANS.distances_m
N_SEGS = len(DISTANCES)
FREE_FLOW = [35.0] * N_SEGS


class TestFundamentalDiagram:
    def test_instantiation_from_speed_and_capacity(self):
        fd = FundamentalDiagram(
            free_flow_speed_mps=15.6,
            capacity_veh_s=0.5,
            jam_density_veh_m=0.15,
        )
        assert fd is not None

    def test_free_flow_speed_positive(self):
        fd = FundamentalDiagram(free_flow_speed_mps=15.6, capacity_veh_s=0.5, jam_density_veh_m=0.15)
        assert fd.free_flow_speed_mps > 0

    def test_capacity_positive(self):
        fd = FundamentalDiagram(free_flow_speed_mps=15.6, capacity_veh_s=0.5, jam_density_veh_m=0.15)
        assert fd.capacity_veh_s > 0

    def test_backward_wave_speed_positive(self):
        fd = FundamentalDiagram(free_flow_speed_mps=15.6, capacity_veh_s=0.5, jam_density_veh_m=0.15)
        w = fd.backward_wave_speed_mps
        assert w > 0


def _make_obs(speed_mps: float, travel_time_s: float = 60.0, distance_m: float = 500.0):
    from datetime import datetime
    return CalibrationObservation(
        segment_id="rosecrans_seg0",
        timestamp=datetime(2026, 4, 17, 8, 0, 0),
        observed_speed_mps=speed_mps,
        observed_travel_time_s=travel_time_s,
        distance_m=distance_m,
    )


class TestCalibratefd:
    def test_returns_fundamental_diagram(self):
        obs = [
            _make_obs(15.6),
            _make_obs(11.2),
            _make_obs(4.5),
        ]
        fd, stats = calibrate_fd(obs)
        assert isinstance(fd, FundamentalDiagram)

    def test_calibrated_fd_has_positive_capacity(self):
        obs = [_make_obs(15.6)]
        fd, stats = calibrate_fd(obs)
        assert fd.capacity_veh_s > 0

    def test_calibrated_fd_free_flow_from_observations(self):
        obs = [_make_obs(15.6), _make_obs(14.0)]
        fd, stats = calibrate_fd(obs)
        assert fd.free_flow_speed_mps > 0
        assert "n_observations" in stats


class TestDigitalTwin:
    def setup_method(self):
        self.twin = DigitalTwin(
            corridor_id=ROSECRANS.id,
            segment_distances_m=DISTANCES,
            free_flow_speeds_mph=FREE_FLOW,
        )

    def test_instantiation(self):
        assert self.twin is not None

    def test_simulate_returns_result(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert isinstance(result, SimulationResult)

    def test_simulation_has_density_history(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert len(result.density_history) > 0

    def test_simulation_avg_travel_time_positive(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert result.avg_travel_time_s > 0

    def test_simulation_avg_speed_positive(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert result.avg_speed_mph > 0

    def test_simulation_co2_nonneg(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert result.co2_kg >= 0

    def test_all_red_slower_than_all_green(self):
        all_green = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.20,
            duration_s=600.0,
        )
        all_red = self.twin.simulate(
            signal_green_per_segment=[False] * N_SEGS,
            demand_veh_s=0.20,
            duration_s=600.0,
        )
        assert all_red.avg_travel_time_s >= all_green.avg_travel_time_s

    def test_higher_demand_more_co2(self):
        low_demand = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.10,
            duration_s=600.0,
        )
        high_demand = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.50,
            duration_s=600.0,
        )
        assert high_demand.co2_kg >= low_demand.co2_kg

    def test_density_history_cells_count(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        # All steps should have the same cell count
        first_len = len(result.density_history[0])
        for step in result.density_history:
            assert len(step) == first_len

    def test_simulation_labels_tracked(self):
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
            scenario_label="test_scenario",
        )
        assert result is not None

    def test_alternating_signal_pattern(self):
        pattern = [i % 2 == 0 for i in range(N_SEGS)]
        result = self.twin.simulate(
            signal_green_per_segment=pattern,
            demand_veh_s=0.28,
            duration_s=900.0,
        )
        assert result.avg_travel_time_s > 0

    def test_calibrate_from_observations(self):
        obs = [
            _make_obs(15.6),
            _make_obs(11.2),
        ]
        self.twin.calibrate(obs)
        # After calibration twin should still simulate
        result = self.twin.simulate(
            signal_green_per_segment=[True] * N_SEGS,
            demand_veh_s=0.28,
            duration_s=300.0,
        )
        assert result is not None
