"""Tests for aito/data/probe_fusion.py (GF1)."""
import pytest
from datetime import datetime, timedelta

from aito.data.probe_fusion import (
    ProbeFusionEngine,
    SourceObservation,
    CongestionState,
    FusedSegmentEstimate,
    CorridorFusedEstimate,
    estimate_demand_from_probes,
    estimate_penetration_rate,
    fuse_all_tod_periods,
    _congestion_from_ratio,
    _fuse_observations,
    SOURCE_WEIGHTS,
)


NOW = datetime(2025, 6, 1, 7, 30)
SEGMENT_DISTANCES = [420.0, 250.0, 190.0]
FREE_FLOW_SPEEDS = [35.0, 35.0, 35.0]
CORRIDOR_ID = "test_corridor"


# ---------------------------------------------------------------------------
# Unit tests: helpers
# ---------------------------------------------------------------------------

class TestCongestionClassification:
    def test_free_flow(self):
        assert _congestion_from_ratio(0.95) == CongestionState.FREE

    def test_moderate(self):
        assert _congestion_from_ratio(0.80) == CongestionState.MODERATE

    def test_heavy(self):
        assert _congestion_from_ratio(0.60) == CongestionState.HEAVY

    def test_breakdown(self):
        assert _congestion_from_ratio(0.40) == CongestionState.BREAKDOWN

    def test_boundary_free(self):
        assert _congestion_from_ratio(0.90) == CongestionState.FREE

    def test_boundary_breakdown(self):
        assert _congestion_from_ratio(0.50) == CongestionState.HEAVY


class TestPenetrationRateEstimation:
    def test_28pct_penetration(self):
        rate = estimate_penetration_rate(
            probe_flow_veh_hr=280.0,
            reference_aadt=10000,
            peak_hour_factor=0.10,
        )
        assert abs(rate - 0.28) < 0.01

    def test_clamped_at_one(self):
        rate = estimate_penetration_rate(5000.0, 1000, 0.10)
        assert rate == 1.0

    def test_zero_aadt(self):
        rate = estimate_penetration_rate(100.0, 0, 0.10)
        assert rate == 0.0


class TestDemandFromProbes:
    def test_basic_expansion(self):
        # At 28% penetration, 280 probes → ~1000 true vehicles
        demand = estimate_demand_from_probes(280.0, 0.28)
        assert 950 < demand < 1050

    def test_zero_penetration(self):
        assert estimate_demand_from_probes(100.0, 0.0) == 0.0


# ---------------------------------------------------------------------------
# Unit tests: _fuse_observations
# ---------------------------------------------------------------------------

class TestFuseObservations:
    def _make_obs(self, source, tt, speed, samples=10, conf=0.9):
        return SourceObservation(
            source=source,
            segment_id="seg0",
            timestamp=NOW,
            travel_time_s=tt,
            speed_mph=speed,
            sample_count=samples,
            confidence=conf,
        )

    def test_single_source(self):
        obs = [self._make_obs("inrix", 45.0, 35.0)]
        result = _fuse_observations(obs, 35.0, NOW, NOW + timedelta(minutes=15), 420.0)
        assert abs(result.travel_time_s - 45.0) < 5.0
        assert result.source_count == 1
        assert result.fused_confidence > 0.5

    def test_multi_source_converging(self):
        # Three sources agree → high confidence, tight CI
        obs = [
            self._make_obs("connected_vehicle", 48.0, 31.0),
            self._make_obs("inrix", 50.0, 30.0),
            self._make_obs("here", 47.0, 32.0),
        ]
        result = _fuse_observations(obs, 35.0, NOW, NOW + timedelta(minutes=15), 420.0)
        assert 44 < result.travel_time_s < 54
        assert result.source_count == 3
        assert result.fused_confidence > 0.8
        assert result.travel_time_lower_s < result.travel_time_s < result.travel_time_upper_s

    def test_high_quality_source_outweights_low(self):
        # CV (0.95) vs waze (0.72) with different readings
        obs = [
            self._make_obs("connected_vehicle", 40.0, 37.0, samples=20),
            self._make_obs("waze", 70.0, 21.0, samples=3),
        ]
        result = _fuse_observations(obs, 35.0, NOW, NOW + timedelta(minutes=15), 420.0)
        # Should be closer to CV reading (40s) than waze (70s)
        assert result.travel_time_s < 55.0

    def test_no_observations_fallback(self):
        result = _fuse_observations([], 35.0, NOW, NOW + timedelta(minutes=15), 420.0)
        free_flow_tt = 420.0 / (35.0 * 0.44704)
        assert abs(result.travel_time_s - free_flow_tt) < 2.0
        assert result.fused_confidence == 0.0
        assert result.source_count == 0

    def test_source_weights_documented(self):
        assert SOURCE_WEIGHTS["connected_vehicle"] > SOURCE_WEIGHTS["waze"]
        assert SOURCE_WEIGHTS["inrix"] > SOURCE_WEIGHTS["smartphone_gps"]


# ---------------------------------------------------------------------------
# Integration tests: ProbeFusionEngine
# ---------------------------------------------------------------------------

class TestProbeFusionEngine:
    def _make_engine(self):
        return ProbeFusionEngine(
            corridor_id=CORRIDOR_ID,
            segment_distances_m=SEGMENT_DISTANCES,
            free_flow_speeds_mph=FREE_FLOW_SPEEDS,
        )

    def test_init(self):
        engine = self._make_engine()
        assert engine.corridor_id == CORRIDOR_ID
        assert len(engine.segment_distances_m) == 3

    def test_fuse_empty_returns_freeflow(self):
        engine = self._make_engine()
        result = engine.fuse(NOW)
        assert isinstance(result, CorridorFusedEstimate)
        assert len(result.segments) == 3
        for seg in result.segments:
            assert seg.fused_confidence == 0.0
            assert seg.travel_time_s > 0

    def test_ingest_and_fuse(self):
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW, congestion_factor=0.75)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        assert len(result.segments) == 3
        for seg in result.segments:
            assert seg.travel_time_s > 0
            assert seg.fused_confidence > 0.3

    def test_congestion_detected(self):
        engine = self._make_engine()
        # Heavy congestion (50% speed reduction)
        obs = engine.generate_synthetic_observations(NOW, congestion_factor=0.45)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        # At 45% of free flow, expect breakdown
        states = [seg.congestion_state for seg in result.segments]
        assert any(s in (CongestionState.HEAVY, CongestionState.BREAKDOWN) for s in states)

    def test_freeflow_detected(self):
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW, congestion_factor=0.95)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        states = [seg.congestion_state for seg in result.segments]
        assert any(s == CongestionState.FREE for s in states)

    def test_corridor_travel_time_sum(self):
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        total = result.corridor_travel_time_s
        assert total == sum(s.travel_time_s for s in result.segments)

    def test_worst_congestion(self):
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW, congestion_factor=0.40)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        assert result.worst_congestion_state in (CongestionState.HEAVY, CongestionState.BREAKDOWN)

    def test_clear_resets_buffer(self):
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW)
        engine.ingest(obs)
        engine.clear()
        result = engine.fuse(NOW)
        for seg in result.segments:
            assert seg.fused_confidence == 0.0

    def test_fuse_to_travel_data(self):
        from aito.optimization.corridor_optimizer import SegmentTravelData
        engine = self._make_engine()
        obs = engine.generate_synthetic_observations(NOW)
        engine.ingest(obs)
        travel_data = engine.fuse_to_travel_data(NOW)
        assert len(travel_data) == 3
        for td in travel_data:
            assert isinstance(td, SegmentTravelData)
            assert td.outbound_travel_time_s > 0
            assert td.inbound_travel_time_s >= td.outbound_travel_time_s

    def test_window_filtering(self):
        engine = self._make_engine()
        # Observations in a different window shouldn't appear in fused result
        other_time = NOW + timedelta(hours=2)
        obs = engine.generate_synthetic_observations(other_time)
        engine.ingest(obs)
        result = engine.fuse(NOW)  # requesting NOW window
        for seg in result.segments:
            assert seg.fused_confidence == 0.0  # no data in this window

    def test_source_count_multiple_sources(self):
        engine = self._make_engine()
        # Generate with 3 sources
        obs = engine.generate_synthetic_observations(NOW, n_sources=3)
        engine.ingest(obs)
        result = engine.fuse(NOW)
        for seg in result.segments:
            assert seg.source_count >= 1


# ---------------------------------------------------------------------------
# TOD period fusion
# ---------------------------------------------------------------------------

class TestTODFusion:
    def test_all_periods_generated(self):
        engine = ProbeFusionEngine(CORRIDOR_ID, SEGMENT_DISTANCES, FREE_FLOW_SPEEDS)
        results = fuse_all_tod_periods(engine, NOW)
        assert len(results) == 6
        expected_periods = {"AM Peak", "Midday", "PM Peak", "Evening", "Overnight", "Weekend"}
        assert set(results.keys()) == expected_periods

    def test_am_peak_slower_than_overnight(self):
        engine = ProbeFusionEngine(CORRIDOR_ID, SEGMENT_DISTANCES, FREE_FLOW_SPEEDS)
        results = fuse_all_tod_periods(engine, NOW)
        am_tt = results["AM Peak"].corridor_travel_time_s
        overnight_tt = results["Overnight"].corridor_travel_time_s
        assert am_tt > overnight_tt
