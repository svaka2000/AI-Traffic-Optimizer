"""Tests for before/after analysis and ROI calculator."""
import pytest
from datetime import datetime, timedelta
from aito.data.san_diego_inventory import get_corridor
from aito.models import PerformanceMetrics
from aito.analytics.before_after import BeforeAfterAnalyzer, improvement_pct, paired_t_statistic
from aito.analytics.roi_calculator import ROICalculator


@pytest.fixture
def corridor():
    return get_corridor("rosecrans")


@pytest.fixture
def before_metrics():
    now = datetime(2025, 6, 1, 7, 0)
    return PerformanceMetrics(
        period_start=now,
        period_end=now + timedelta(hours=2),
        avg_delay_sec=52.0,
        avg_travel_time_sec=480.0,
        arrival_on_green_pct=42.0,
        split_failure_pct=18.0,
        stops_per_veh=2.8,
        co2_kg_hr=68.0,
        throughput_veh_hr=1650.0,
    )


@pytest.fixture
def after_metrics():
    now = datetime(2025, 6, 1, 7, 0)
    return PerformanceMetrics(
        period_start=now,
        period_end=now + timedelta(hours=2),
        avg_delay_sec=38.0,
        avg_travel_time_sec=370.0,
        arrival_on_green_pct=65.0,
        split_failure_pct=7.0,
        stops_per_veh=1.4,
        co2_kg_hr=52.0,
        throughput_veh_hr=1750.0,
    )


def test_before_after_improvement_positive(corridor, before_metrics, after_metrics):
    analyzer = BeforeAfterAnalyzer(corridor, daily_vehicles=28000)
    result = analyzer.analyze(before_metrics, after_metrics)
    assert result.travel_time_improvement_pct > 0
    assert result.delay_improvement_pct > 0
    assert result.co2_reduction_pct > 0
    assert result.stops_reduction_pct > 0


def test_annual_savings_positive(corridor, before_metrics, after_metrics):
    analyzer = BeforeAfterAnalyzer(corridor, daily_vehicles=28000)
    result = analyzer.analyze(before_metrics, after_metrics)
    assert result.annual_veh_hours_saved > 0
    assert result.annual_fuel_saved_gallons > 0
    assert result.annual_co2_reduction_tonnes > 0
    assert result.annual_cost_savings_usd > 0


def test_improvement_pct_formula():
    assert improvement_pct(100.0, 75.0) == 25.0
    assert improvement_pct(0.0, 50.0) == 0.0  # avoid division by zero


def test_paired_t_statistic():
    before = [50, 52, 48, 55, 51]
    after = [38, 40, 36, 42, 39]
    t, df = paired_t_statistic(before, after)
    assert t > 1.96  # should be significant
    assert df == 4


def test_roi_benefit_cost_positive(corridor):
    calc = ROICalculator(corridor)
    report = calc.calculate(
        delay_reduction_s_veh=14.0,
        stops_reduction_pct=50.0,
        co2_reduction_pct=23.0,
        daily_vehicles=28000,
        aito_annual_cost_usd=72000,
        before_co2_kg_hr=68.0,
    )
    assert report.benefit_cost_ratio > 1.0
    assert report.annual_total_benefit_usd > report.aito_annual_cost_usd
    assert report.simple_payback_months < 12.0  # < 1 year
    assert report.npv_usd > 0


def test_roi_summary_contains_key_metrics(corridor):
    calc = ROICalculator(corridor)
    report = calc.calculate(
        delay_reduction_s_veh=14.0,
        stops_reduction_pct=50.0,
        co2_reduction_pct=23.0,
        daily_vehicles=28000,
        aito_annual_cost_usd=72000,
        before_co2_kg_hr=68.0,
    )
    summary = report.summary()
    assert "Benefit-Cost" in summary
    assert "NPV" in summary
    assert "Fuel" in summary


def test_roi_per_intersection_metrics(corridor):
    calc = ROICalculator(corridor)
    report = calc.calculate(
        delay_reduction_s_veh=14.0,
        stops_reduction_pct=50.0,
        co2_reduction_pct=23.0,
        daily_vehicles=28000,
        aito_annual_cost_usd=72000,
    )
    assert report.num_intersections == len(corridor.intersections)
    assert report.benefit_per_intersection_usd > 0


def test_atspm_calculator():
    from datetime import datetime
    from aito.data.atspm import ATSPMCalculator, generate_synthetic_events

    calc = ATSPMCalculator("test_ix", cycle_length_s=120.0)
    now = datetime(2025, 1, 1, 7, 0)
    events = generate_synthetic_events("test_ix", now, cycle_length_s=120.0, aog_pct=65.0, num_cycles=10)
    assert len(events) > 0

    metrics = calc.compute_metrics(events, now, now.replace(hour=9))
    assert 0 <= metrics.arrival_on_green_pct <= 100
    assert 0 <= metrics.split_failure_pct <= 100
    assert metrics.platoon_ratio > 0
