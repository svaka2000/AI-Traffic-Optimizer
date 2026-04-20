"""Tests for aito/analytics/carbon_accountant.py (GF2)."""
import pytest
from datetime import datetime
from aito.analytics.carbon_accountant import (
    OperatingMode,
    fleet_co2_g_s,
    CycleOperatingModeProfile,
    IntersectionEmissions,
    CorridorEmissionsReport,
    EmissionsDelta,
    CarbonAccountant,
    MOVES_CO2_G_S,
    CA_FLEET_MIX,
)


class TestOperatingMode:
    def test_all_modes_have_entries_in_rate_table(self):
        for mode in OperatingMode:
            assert mode in MOVES_CO2_G_S
            assert MOVES_CO2_G_S[mode] > 0

    def test_idle_rate_is_approx_1_38(self):
        assert MOVES_CO2_G_S[OperatingMode.IDLE] == pytest.approx(1.38, abs=0.01)

    def test_high_accel_highest_rate(self):
        assert MOVES_CO2_G_S[OperatingMode.ACCEL_HIGH] > MOVES_CO2_G_S[OperatingMode.IDLE]
        assert MOVES_CO2_G_S[OperatingMode.ACCEL_HIGH] > MOVES_CO2_G_S[OperatingMode.CRUISE_MED]


class TestCAFleetMix:
    def test_fractions_sum_to_one(self):
        assert sum(CA_FLEET_MIX.values()) == pytest.approx(1.0, abs=0.01)

    def test_all_fractions_nonneg(self):
        for k, v in CA_FLEET_MIX.items():
            assert v >= 0, f"{k} is negative"


class TestFleetCo2GS:
    def test_returns_positive_for_all_modes(self):
        for mode in OperatingMode:
            assert fleet_co2_g_s(mode) > 0

    def test_default_uses_ca_fleet_mix(self):
        result = fleet_co2_g_s(OperatingMode.IDLE)
        assert isinstance(result, float)
        assert result > 0

    def test_ev_only_mix_returns_zero_for_idle(self):
        ev_mix = {"light_duty_ev": 1.0}
        result = fleet_co2_g_s(OperatingMode.IDLE, fleet_mix=ev_mix)
        assert result == pytest.approx(0.0, abs=1e-9)

    def test_high_accel_higher_than_idle(self):
        idle_rate = fleet_co2_g_s(OperatingMode.IDLE)
        accel_rate = fleet_co2_g_s(OperatingMode.ACCEL_HIGH)
        assert accel_rate > idle_rate


class TestCycleOperatingModeProfile:
    def test_from_signal_timing_returns_profile(self):
        profile = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=48.0,
            volume_veh_hr=800.0, saturation_flow=1800.0, approach_speed_mph=35.0,
        )
        assert profile is not None

    def test_fractions_all_nonneg(self):
        profile = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=48.0, volume_veh_hr=800.0,
        )
        assert profile.idle_frac >= 0
        assert profile.decel_frac >= 0
        assert profile.cruise_frac >= 0
        assert profile.accel_frac >= 0

    def test_fractions_sum_approx_one(self):
        profile = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=48.0, volume_veh_hr=800.0,
        )
        total = profile.idle_frac + profile.decel_frac + profile.cruise_frac + profile.accel_frac
        assert total == pytest.approx(1.0, abs=0.05)

    def test_longer_red_more_idle(self):
        short_red = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=90.0, volume_veh_hr=800.0,
        )
        long_red = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=30.0, volume_veh_hr=800.0,
        )
        assert long_red.idle_frac > short_red.idle_frac

    def test_low_volume_less_idle(self):
        low_vol = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=48.0, volume_veh_hr=100.0,
        )
        high_vol = CycleOperatingModeProfile.from_signal_timing(
            cycle_s=120.0, green_s=48.0, volume_veh_hr=1600.0,
        )
        assert high_vol.idle_frac >= low_vol.idle_frac


class TestCarbonAccountantEstimateQuick:
    def test_returns_positive(self):
        result = CarbonAccountant.estimate_quick(
            n_intersections=5, avg_aadt=20000, avg_delay_reduction_s_veh=15.0,
        )
        assert result > 0

    def test_zero_delay_reduction_zero_result(self):
        result = CarbonAccountant.estimate_quick(
            n_intersections=5, avg_aadt=20000, avg_delay_reduction_s_veh=0.0,
        )
        assert result == pytest.approx(0.0, abs=1e-6)

    def test_more_intersections_more_reduction(self):
        r3 = CarbonAccountant.estimate_quick(n_intersections=3, avg_aadt=20000, avg_delay_reduction_s_veh=10.0)
        r12 = CarbonAccountant.estimate_quick(n_intersections=12, avg_aadt=20000, avg_delay_reduction_s_veh=10.0)
        assert r12 > r3
        assert r12 / r3 == pytest.approx(4.0, rel=0.05)

    def test_higher_aadt_proportional(self):
        r_low = CarbonAccountant.estimate_quick(n_intersections=5, avg_aadt=10000, avg_delay_reduction_s_veh=10.0)
        r_high = CarbonAccountant.estimate_quick(n_intersections=5, avg_aadt=20000, avg_delay_reduction_s_veh=10.0)
        assert r_high / r_low == pytest.approx(2.0, rel=0.05)

    def test_larger_delay_reduction_proportional(self):
        r5 = CarbonAccountant.estimate_quick(n_intersections=5, avg_aadt=20000, avg_delay_reduction_s_veh=5.0)
        r10 = CarbonAccountant.estimate_quick(n_intersections=5, avg_aadt=20000, avg_delay_reduction_s_veh=10.0)
        assert r10 / r5 == pytest.approx(2.0, rel=0.05)

    def test_rosecrans_result_in_plausible_range(self):
        result = CarbonAccountant.estimate_quick(
            n_intersections=12, avg_aadt=28000, avg_delay_reduction_s_veh=17.0,
        )
        assert 500 < result < 50000

    def test_monotonic_in_intersections(self):
        results = [
            CarbonAccountant.estimate_quick(n_intersections=n, avg_aadt=20000, avg_delay_reduction_s_veh=10.0)
            for n in [1, 3, 6, 10, 20]
        ]
        assert results == sorted(results)


class TestEmissionsDelta:
    def _make_report(self, co2_kg_hr: float) -> CorridorEmissionsReport:
        from datetime import datetime
        ix = IntersectionEmissions(
            intersection_id="ix1",
            intersection_name="Test Intersection",
            period_start=datetime(2025, 1, 1, 7),
            period_end=datetime(2025, 1, 1, 8),
            approach_emissions_kg_hr={"N": co2_kg_hr / 4, "S": co2_kg_hr / 4,
                                      "E": co2_kg_hr / 4, "W": co2_kg_hr / 4},
            total_co2_kg_hr=co2_kg_hr,
            total_co2_kg_day=co2_kg_hr * 24,
            mode_emissions_kg_hr={},
            volume_veh_hr=1000.0,
            avg_green_ratio=0.4,
        )
        return CorridorEmissionsReport(
            corridor_id="test",
            corridor_name="Test",
            period_label="AM Peak",
            computation_datetime=datetime.now(),
            intersections=[ix],
        )

    def test_reduction_positive_when_optimized_lower(self):
        baseline = self._make_report(co2_kg_hr=100.0)
        optimized = self._make_report(co2_kg_hr=75.0)
        delta = optimized.delta_vs(baseline)
        assert delta.reduction_kg_hr == pytest.approx(25.0, rel=0.01)

    def test_reduction_tonnes_year_positive(self):
        baseline = self._make_report(co2_kg_hr=100.0)
        optimized = self._make_report(co2_kg_hr=80.0)
        delta = optimized.delta_vs(baseline)
        assert delta.reduction_tonnes_year > 0

    def test_reduction_pct_correct(self):
        baseline = self._make_report(co2_kg_hr=100.0)
        optimized = self._make_report(co2_kg_hr=80.0)
        delta = optimized.delta_vs(baseline)
        assert delta.reduction_pct == pytest.approx(20.0, rel=0.01)

    def test_summary_returns_dict(self):
        baseline = self._make_report(co2_kg_hr=100.0)
        optimized = self._make_report(co2_kg_hr=80.0)
        delta = optimized.delta_vs(baseline)
        s = delta.summary()
        assert "reduction_tonnes_year" in s
        assert "reduction_pct" in s

    def test_corridor_report_totals(self):
        report = self._make_report(co2_kg_hr=100.0)
        assert report.total_co2_kg_hr == pytest.approx(100.0, rel=0.01)
        assert report.total_co2_kg_day == pytest.approx(2400.0, rel=0.01)
        assert report.total_co2_tonnes_year == pytest.approx(2400.0 * 365 / 1000.0, rel=0.01)
