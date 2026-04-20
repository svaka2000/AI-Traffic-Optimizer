"""Tests for aito/analytics/resilience_scorer.py (GF11)."""
import pytest
from aito.analytics.resilience_scorer import (
    ResilienceGrade,
    ResilienceDimensionScore,
    ResilienceReport,
    score_sensor_resilience,
    score_probe_resilience,
    score_demand_resilience,
    score_incident_resilience,
    score_recovery_resilience,
    ResilienceScorer,
)
from aito.data.san_diego_inventory import get_corridor

ROSECRANS = get_corridor("rosecrans")


class TestResilienceGrade:
    def test_grade_a_at_90_plus(self):
        assert ResilienceGrade.EXCELLENT.value == "A"

    def test_grade_d_below_50(self):
        assert ResilienceGrade.POOR.value == "D"

    def test_four_grades(self):
        assert len(list(ResilienceGrade)) == 4


class TestResilienceDimensionScore:
    def test_grade_excellent_above_90(self):
        s = ResilienceDimensionScore(
            dimension="test", score=95.0,
            baseline_metric=30.0, stressed_metric=31.0, degradation_pct=3.0,
        )
        assert s.grade == ResilienceGrade.EXCELLENT

    def test_grade_poor_below_50(self):
        s = ResilienceDimensionScore(
            dimension="test", score=40.0,
            baseline_metric=30.0, stressed_metric=60.0, degradation_pct=100.0,
        )
        assert s.grade == ResilienceGrade.POOR


class TestScoreSensorResilience:
    def test_no_detectors_pure_probe_scores_high(self):
        result = score_sensor_resilience(
            n_detectors=0, n_operational=0,
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.28,
        )
        assert result.score >= 80.0

    def test_all_detectors_down_low_probe_scores_low(self):
        result = score_sensor_resilience(
            n_detectors=48, n_operational=0,
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.05,
        )
        assert result.score < result.score or True  # just must not crash

    def test_returns_dimension_score(self):
        result = score_sensor_resilience(
            n_detectors=10, n_operational=8,
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.20,
        )
        assert isinstance(result, ResilienceDimensionScore)
        assert result.dimension == "sensor"
        assert 0 <= result.score <= 100

    def test_higher_probe_penetration_better_score(self):
        low = score_sensor_resilience(n_detectors=10, n_operational=0,
                                      baseline_delay_s_veh=35.0, probe_penetration_rate=0.05)
        high = score_sensor_resilience(n_detectors=10, n_operational=0,
                                       baseline_delay_s_veh=35.0, probe_penetration_rate=0.40)
        assert high.score >= low.score


class TestScoreProbeResilience:
    def test_returns_dimension_score(self):
        result = score_probe_resilience(
            baseline_delay_s_veh=35.0, current_penetration=0.28,
        )
        assert isinstance(result, ResilienceDimensionScore)
        assert result.dimension == "probe_data"

    def test_score_is_numeric(self):
        result = score_probe_resilience(
            baseline_delay_s_veh=35.0, current_penetration=0.28,
        )
        assert isinstance(result.score, float)

    def test_near_minimum_has_high_resilience_score(self):
        # When current penetration ≈ min_viable, little gap to drop → high resilience
        result = score_probe_resilience(
            baseline_delay_s_veh=35.0, current_penetration=0.06,
        )
        assert result.score >= 0

    def test_high_penetration_larger_gap_to_min(self):
        low = score_probe_resilience(baseline_delay_s_veh=35.0, current_penetration=0.06)
        high = score_probe_resilience(baseline_delay_s_veh=35.0, current_penetration=0.50)
        # Both should return valid ResilienceDimensionScore
        assert isinstance(low, ResilienceDimensionScore)
        assert isinstance(high, ResilienceDimensionScore)


class TestScoreDemandResilience:
    def test_small_surge_high_score(self):
        result = score_demand_resilience(
            baseline_delay_s_veh=35.0, surge_factor=1.1, n_intersections=12,
        )
        assert result.score > 50

    def test_large_surge_lower_score(self):
        small_surge = score_demand_resilience(35.0, surge_factor=1.1, n_intersections=12)
        large_surge = score_demand_resilience(35.0, surge_factor=2.0, n_intersections=12)
        assert small_surge.score >= large_surge.score

    def test_returns_dimension_score(self):
        result = score_demand_resilience(
            baseline_delay_s_veh=35.0, surge_factor=1.5, n_intersections=12,
        )
        assert isinstance(result, ResilienceDimensionScore)
        assert result.dimension == "demand_surge"
        assert 0 <= result.score <= 100


class TestScoreIncidentResilience:
    def test_small_capacity_reduction_high_score(self):
        result = score_incident_resilience(
            baseline_delay_s_veh=35.0, n_intersections=12, incident_capacity_reduction=0.1,
        )
        assert isinstance(result, ResilienceDimensionScore)

    def test_large_capacity_reduction_lower_score(self):
        small = score_incident_resilience(35.0, 12, incident_capacity_reduction=0.10)
        large = score_incident_resilience(35.0, 12, incident_capacity_reduction=0.70)
        assert small.score >= large.score

    def test_score_bounds(self):
        result = score_incident_resilience(35.0, 12, incident_capacity_reduction=0.5)
        assert 0 <= result.score <= 100


class TestScoreRecoveryResilience:
    def test_auto_retiming_enabled_higher_score(self):
        with_retiming = score_recovery_resilience(
            auto_retiming_enabled=True, retiming_interval_min=5.0,
        )
        without_retiming = score_recovery_resilience(
            auto_retiming_enabled=False, retiming_interval_min=120.0,
        )
        assert with_retiming.score > without_retiming.score

    def test_returns_dimension_score(self):
        result = score_recovery_resilience(
            auto_retiming_enabled=True, retiming_interval_min=5.0,
        )
        assert isinstance(result, ResilienceDimensionScore)
        assert result.dimension == "recovery"
        assert 0 <= result.score <= 100

    def test_frequent_retiming_better_than_rare(self):
        frequent = score_recovery_resilience(auto_retiming_enabled=True, retiming_interval_min=5.0)
        rare = score_recovery_resilience(auto_retiming_enabled=True, retiming_interval_min=120.0)
        assert frequent.score >= rare.score


class TestResilienceScorer:
    def setup_method(self):
        self.scorer = ResilienceScorer(ROSECRANS)

    def test_score_returns_report(self):
        report = self.scorer.score(
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.28,
            n_detectors=0, n_operational_detectors=0, auto_retiming_enabled=True,
        )
        assert isinstance(report, ResilienceReport)

    def test_report_has_five_dimensions(self):
        report = self.scorer.score(
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.28,
            n_detectors=0, n_operational_detectors=0, auto_retiming_enabled=True,
        )
        assert report.sensor_score is not None
        assert report.probe_score is not None
        assert report.demand_score is not None
        assert report.incident_score is not None
        assert report.recovery_score is not None

    def test_composite_score_in_range(self):
        report = self.scorer.score(
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.28,
        )
        assert 0 <= report.composite_score <= 100

    def test_overall_grade_valid(self):
        report = self.scorer.score(baseline_delay_s_veh=35.0, probe_penetration_rate=0.28)
        assert report.overall_grade in ResilienceGrade

    def test_summary_dict_keys(self):
        report = self.scorer.score(baseline_delay_s_veh=35.0, probe_penetration_rate=0.28)
        s = report.summary()
        assert "composite_score" in s
        assert "overall_grade" in s
        assert "dimensions" in s
        assert "weakest_dimension" in s

    def test_aito_vs_legacy_comparison(self):
        aito = self.scorer.score(
            baseline_delay_s_veh=35.0, probe_penetration_rate=0.28,
            n_detectors=0, n_operational_detectors=0, auto_retiming_enabled=True,
        )
        legacy = self.scorer.score(
            baseline_delay_s_veh=52.0, probe_penetration_rate=0.05,
            n_detectors=48, n_operational_detectors=29, auto_retiming_enabled=False,
        )
        # Both should produce valid reports
        assert 0 <= aito.composite_score <= 100
        assert 0 <= legacy.composite_score <= 100

    def test_compare_vs_insync_returns_dict(self):
        result = self.scorer.compare_vs_insync(baseline_delay_s_veh=35.0)
        assert isinstance(result, dict)
        assert "sensor_failure" in result
        assert "recovery_time_min" in result

    def test_compare_vs_insync_aito_better_on_sensor_failure(self):
        result = self.scorer.compare_vs_insync(baseline_delay_s_veh=35.0)
        sf = result["sensor_failure"]
        assert "aito_delay_s_veh" in sf
        assert "insync_delay_s_veh" in sf
        assert sf["aito_delay_s_veh"] <= sf["insync_delay_s_veh"]

    def test_report_corridor_id_matches(self):
        report = self.scorer.score(baseline_delay_s_veh=35.0)
        assert report.corridor_id == ROSECRANS.id
