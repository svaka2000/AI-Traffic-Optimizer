"""tests/test_detection.py

Tests for traffic_ai.simulation_engine.detection:
  DetectorConfig, DetectionSystem, DetectorType.
"""
from __future__ import annotations

import numpy as np
import pytest

from traffic_ai.simulation_engine.detection import (
    DetectionSystem,
    DetectorConfig,
    DetectorType,
)


def _make_rng(seed: int = 0) -> np.random.Generator:
    return np.random.default_rng(seed)


# ------------------------------------------------------------------
# Disabled (default) system is a passthrough
# ------------------------------------------------------------------

def test_disabled_detector_never_fails() -> None:
    """When enabled=False, is_failed is always False regardless of updates."""
    cfg = DetectorConfig(enabled=False)
    det = DetectionSystem(cfg, _make_rng())
    for step in range(1000):
        det.update(step, 1.0)
    assert not det.is_failed


def test_disabled_detector_no_obs_degradation() -> None:
    """When enabled=False, degrade_observation returns obs unchanged."""
    cfg = DetectorConfig(enabled=False, detector_type=DetectorType.VIDEO)
    det = DetectionSystem(cfg, _make_rng())
    obs = {"queue_ns": 10.0, "queue_ew": 5.0}
    result = det.degrade_observation(obs, hour_of_day=8)  # glare hour
    assert result is obs  # exact same dict object returned


# ------------------------------------------------------------------
# Loop detector: binary failure model
# ------------------------------------------------------------------

def test_loop_detector_fails_and_triggers_fixed_timing() -> None:
    """A loop detector with 100% failure rate fails immediately."""
    cfg = DetectorConfig(
        enabled=True,
        detector_type=DetectorType.LOOP,
        failure_rate_per_hour=9999.0,   # guaranteed failure
        repair_time_hours=1.0,
    )
    det = DetectionSystem(cfg, _make_rng(42))
    det.update(step=0, step_seconds=1.0)
    assert det.is_failed
    assert det.status == "failed"


def test_loop_detector_repairs_after_repair_time() -> None:
    """Loop detector recovers after repair_time_hours steps have elapsed."""
    steps_per_hour = 3600
    cfg = DetectorConfig(
        enabled=True,
        detector_type=DetectorType.LOOP,
        failure_rate_per_hour=9999.0,
        repair_time_hours=1.0,           # 1 hr = 3600 steps at 1 s/step
    )
    det = DetectionSystem(cfg, _make_rng(42))
    det.update(step=0, step_seconds=1.0)
    assert det.is_failed

    # Advance exactly to the repair step (repair_time_hours=1 → repair at step 3600).
    # Stop at step 3600 (inclusive) so the repair is applied but the post-repair
    # failure roll (guaranteed at failure_rate=9999) does not run another iteration.
    for step in range(1, steps_per_hour + 1):
        det.update(step=step, step_seconds=1.0)
    assert not det.is_failed, "Detector should have repaired after repair_time_hours"


def test_loop_detector_no_obs_degradation_when_operational() -> None:
    """Operational loop detector does not degrade observations."""
    cfg = DetectorConfig(enabled=True, detector_type=DetectorType.LOOP)
    det = DetectionSystem(cfg, _make_rng())
    obs = {"queue_ns": 20.0, "queue_ew": 10.0}
    result = det.degrade_observation(obs, hour_of_day=12)
    assert result == obs


# ------------------------------------------------------------------
# Video detector: graded degradation
# ------------------------------------------------------------------

def test_video_glare_adds_noise() -> None:
    """Video detector adds noise to queue observations during solar glare hours."""
    rng = np.random.default_rng(0)
    cfg = DetectorConfig(
        enabled=True,
        detector_type=DetectorType.VIDEO,
        video_glare_hours=(7, 8, 17, 18),
    )
    det = DetectionSystem(cfg, rng)
    obs = {"queue_ns": 50.0, "queue_ew": 50.0, "sim_step": 100.0}

    # Run many degradations and check noise was applied
    values = [det.degrade_observation(obs, hour_of_day=8)["queue_ns"] for _ in range(200)]
    assert not all(v == 50.0 for v in values), "Glare should introduce non-zero noise"
    assert all(v >= 0.0 for v in values), "Degraded values must remain non-negative"


def test_video_no_noise_outside_glare_window() -> None:
    """Video detector outside glare/night window returns unchanged observations."""
    cfg = DetectorConfig(
        enabled=True,
        detector_type=DetectorType.VIDEO,
        video_glare_hours=(7, 8, 17, 18),
        video_night_degradation=0.30,
    )
    det = DetectionSystem(cfg, _make_rng())
    obs = {"queue_ns": 30.0, "queue_ew": 15.0}
    result = det.degrade_observation(obs, hour_of_day=12)  # midday — no glare or night
    assert result == obs


# ------------------------------------------------------------------
# No-detection fallback
# ------------------------------------------------------------------

def test_none_detector_always_failed() -> None:
    """DetectorType.NONE means always failed — controller always runs fixed timing."""
    cfg = DetectorConfig(enabled=True, detector_type=DetectorType.NONE)
    det = DetectionSystem(cfg, _make_rng())
    det.update(step=0, step_seconds=1.0)
    assert det.is_failed


def test_fallback_action_alternates() -> None:
    """Fixed-timing fallback alternates NS/EW every FALLBACK_CYCLE_STEPS."""
    cfg = DetectorConfig(enabled=True, detector_type=DetectorType.NONE)
    det = DetectionSystem(cfg, _make_rng())
    cycle = DetectionSystem.FALLBACK_CYCLE_STEPS
    assert det.fallback_action(0) == "NS_THROUGH"
    assert det.fallback_action(cycle - 1) == "NS_THROUGH"
    assert det.fallback_action(cycle) == "EW_THROUGH"
    assert det.fallback_action(2 * cycle) == "NS_THROUGH"
