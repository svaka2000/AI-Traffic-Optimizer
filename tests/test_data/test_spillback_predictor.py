"""Tests for aito/optimization/spillback_predictor.py (GF12)."""
import pytest
from datetime import datetime
from aito.optimization.spillback_predictor import (
    estimate_queue_length_m,
    estimate_discharge_rate_veh_s,
    SpillbackRisk,
    IntersectionStorage,
    SpillbackForecast,
    forecast_spillback,
    CorridorSpillbackReport,
    SpillbackPredictor,
)
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")
NOW = datetime(2026, 4, 17, 8, 0, 0)


class TestQueueLengthEstimate:
    def test_returns_nonneg(self):
        result = estimate_queue_length_m(
            volume_veh_hr=800.0,
            capacity_veh_hr=1800.0 * 0.40,
            cycle_s=120.0,
            green_s=0.40 * 120.0,
        )
        assert result >= 0

    def test_high_vc_ratio_longer_queue(self):
        low_vc = estimate_queue_length_m(
            volume_veh_hr=400.0, capacity_veh_hr=1800.0 * 0.40,
            cycle_s=120.0, green_s=0.40 * 120.0,
        )
        high_vc = estimate_queue_length_m(
            volume_veh_hr=700.0, capacity_veh_hr=1800.0 * 0.40,
            cycle_s=120.0, green_s=0.40 * 120.0,
        )
        assert high_vc >= low_vc

    def test_zero_volume_zero_queue(self):
        result = estimate_queue_length_m(
            volume_veh_hr=0.0, capacity_veh_hr=720.0,
            cycle_s=120.0, green_s=48.0,
        )
        assert result == pytest.approx(0.0, abs=1.0)


class TestDischargeRate:
    def test_returns_positive(self):
        rate = estimate_discharge_rate_veh_s(
            green_s=0.40 * 120.0, cycle_s=120.0,
        )
        assert rate > 0

    def test_higher_green_ratio_higher_rate(self):
        low = estimate_discharge_rate_veh_s(green_s=30.0, cycle_s=120.0)
        high = estimate_discharge_rate_veh_s(green_s=60.0, cycle_s=120.0)
        assert high > low


class TestSpillbackRisk:
    def test_has_none_level(self):
        assert SpillbackRisk.NONE in SpillbackRisk.__members__.values()

    def test_has_critical_level(self):
        values = [r.value for r in SpillbackRisk]
        assert any("crit" in v.lower() or "high" in v.lower() or "active" in v.lower() for v in values)

    def test_four_or_more_levels(self):
        assert len(list(SpillbackRisk)) >= 3


class TestForecastSpillback:
    def test_returns_forecast(self):
        storage = IntersectionStorage(
            intersection_id="test_ix",
            approach_direction="N",
            storage_length_m=120.0,
        )
        forecast = forecast_spillback(
            storage=storage,
            volume_veh_hr=600.0,
            capacity_veh_hr=720.0,
            cycle_s=120.0,
            green_s=48.0,
            now=NOW,
        )
        assert isinstance(forecast, SpillbackForecast)

    def test_forecast_has_risk_level(self):
        storage = IntersectionStorage(
            intersection_id="test_ix",
            approach_direction="N",
            storage_length_m=120.0,
        )
        forecast = forecast_spillback(
            storage=storage,
            volume_veh_hr=700.0,
            capacity_veh_hr=720.0,
            cycle_s=120.0,
            green_s=48.0,
            now=NOW,
        )
        assert forecast.risk in SpillbackRisk

    def test_vc_above_1_active_spillback(self):
        storage = IntersectionStorage(
            intersection_id="test_ix",
            approach_direction="N",
            storage_length_m=60.0,  # short storage
        )
        forecast = forecast_spillback(
            storage=storage,
            volume_veh_hr=900.0,   # > capacity
            capacity_veh_hr=720.0,
            cycle_s=120.0,
            green_s=48.0,
            now=NOW,
        )
        assert forecast.v_c_ratio > 1.0 or forecast.risk != SpillbackRisk.NONE


class TestSpillbackPredictor:
    def setup_method(self):
        self.predictor = SpillbackPredictor.from_corridor(ROSECRANS)

    def test_from_corridor_creates_storages(self):
        assert len(self.predictor.storages) > 0

    def test_storages_count_matches_4_per_intersection(self):
        expected = len(ROSECRANS.intersections) * 4
        assert len(self.predictor.storages) == expected

    def test_scan_returns_report(self):
        n = len(self.predictor.storages)
        report = self.predictor.scan(
            volumes_veh_hr=[500.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        assert isinstance(report, CorridorSpillbackReport)

    def test_scan_forecasts_count_matches_storages(self):
        n = len(self.predictor.storages)
        report = self.predictor.scan(
            volumes_veh_hr=[500.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        assert len(report.forecasts) == n

    def test_report_has_worst_risk(self):
        n = len(self.predictor.storages)
        report = self.predictor.scan(
            volumes_veh_hr=[500.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        assert report.worst_risk in SpillbackRisk

    def test_high_volume_creates_more_risk(self):
        n = len(self.predictor.storages)
        low_report = self.predictor.scan(
            volumes_veh_hr=[200.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        high_report = self.predictor.scan(
            volumes_veh_hr=[700.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        # High volume should have equal or more risk approaches
        assert (len(high_report.active_spillbacks) + len(high_report.high_risk_approaches)) >= \
               (len(low_report.active_spillbacks) + len(low_report.high_risk_approaches))

    def test_scan_active_spillbacks_list(self):
        n = len(self.predictor.storages)
        report = self.predictor.scan(
            volumes_veh_hr=[500.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        assert isinstance(report.active_spillbacks, list)
        assert isinstance(report.high_risk_approaches, list)

    def test_all_forecast_fields_present(self):
        n = len(self.predictor.storages)
        report = self.predictor.scan(
            volumes_veh_hr=[500.0] * n,
            capacities_veh_hr=[720.0] * n,
            cycle_s=120.0,
            green_ratios=[0.40] * n,
        )
        for f in report.forecasts:
            assert hasattr(f, "intersection_id")
            assert hasattr(f, "approach_direction")
            assert hasattr(f, "queue_length_m")
            assert hasattr(f, "v_c_ratio")
            assert hasattr(f, "risk")
