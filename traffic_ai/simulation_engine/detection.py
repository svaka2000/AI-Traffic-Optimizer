"""traffic_ai/simulation_engine/detection.py

Detector reliability model for traffic signal controllers.

Based on Steve Celniker's description (Senior Traffic Engineer, City of San Diego):
  "Inductive loop detectors are cheap, work well, easily broken by street
   construction.  Without working detection, actuated control defaults to
   fixed timing.  Video detectors are expensive and have trouble with glare
   at sunrise/sunset (7–9 am, 4–6 pm) and reduced accuracy at night."

Detector Types
--------------
LOOP   — Inductive loop: high reliability when intact, binary failure mode.
         Repair requires street crew → 24–48 hour MTTR.
         When failed: intersection falls back to fixed-cycle operation.

VIDEO  — Video detection: continuous degradation from glare (sunrise/sunset)
         and night lighting.  Does NOT trigger hard fallback — controller
         keeps running but receives noisy observations.

RADAR  — Reliable but expensive.  Modelled as loop with lower failure rate.

NONE   — No detection.  Controller always receives zero-count observations
         and runs in fixed-timing fallback unconditionally.

References
----------
Federal Highway Administration (FHWA), "Traffic Detector Handbook", 3rd ed.,
FHWA-HRT-06-108, 2006.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class DetectorType(Enum):
    LOOP  = "inductive_loop"
    VIDEO = "video"
    RADAR = "radar"
    NONE  = "none"


@dataclass(slots=True)
class DetectorConfig:
    """Configuration parameters for the detection reliability model.

    Attributes
    ----------
    detector_type : DetectorType
        Hardware type.  Determines failure mode behaviour.
    failure_rate_per_hour : float
        Probability of detector failure per operating hour.
        Inductive loop baseline ≈ 0.001 /hr ≈ 1 % per month (FHWA, 2006).
    repair_time_hours : float
        Mean time to repair (MTTR) after failure.
        Loop detectors require street crew: 24–48 hr typical.
    video_glare_hours : tuple[int, ...]
        Hours of day (0–23) when solar glare significantly degrades video.
        San Diego sunrise window ≈ 7–9, sunset window ≈ 17–18.
    video_night_degradation : float
        Fractional accuracy loss at night (hours 20–6). 0.30 = 30 % loss.
    noise_std_degraded : float
        Gaussian noise std as a fraction of the true observation value when
        the video detector is degraded (glare or night conditions).
    enabled : bool
        Master switch.  When False, DetectionSystem is a no-op passthrough.
    """
    detector_type: DetectorType         = DetectorType.LOOP
    failure_rate_per_hour: float        = 0.001
    repair_time_hours: float            = 24.0
    video_glare_hours: tuple            = (7, 8, 17, 18)
    video_night_degradation: float      = 0.30
    noise_std_degraded: float           = 0.15
    enabled: bool                       = False


class DetectionSystem:
    """Per-intersection detector reliability model.

    When a loop detector fails, the intersection reverts to fixed timing
    (the controller's action is overridden with an alternating NS/EW plan).
    When a video detector is degraded, observations are returned with added
    Gaussian noise but the controller is not overridden.

    Parameters
    ----------
    config : DetectorConfig
        Hardware configuration and failure parameters.
    rng : np.random.Generator
        Shared RNG for reproducible stochastic failure events.
    intersection_id : int
        Used for logging and metrics attribution.
    """

    # Fixed-timing fallback: alternate NS/EW every 30 steps (30 s default)
    FALLBACK_CYCLE_STEPS: int = 30

    def __init__(
        self,
        config: DetectorConfig,
        rng: np.random.Generator,
        intersection_id: int = 0,
    ) -> None:
        self.config = config
        self.rng = rng
        self.intersection_id = intersection_id

        self._failed: bool = False
        self._failed_at_step: Optional[int] = None
        self._repair_step: Optional[int] = None

    # ------------------------------------------------------------------
    # State update
    # ------------------------------------------------------------------

    def update(self, step: int, step_seconds: float) -> None:
        """Roll for failure or repair each simulation step."""
        if not self.config.enabled:
            return
        if self.config.detector_type == DetectorType.NONE:
            self._failed = True  # No detector → always in fallback
            return

        steps_per_hour = 3600.0 / step_seconds

        if self._failed:
            if self._repair_step is not None and step >= self._repair_step:
                self._failed = False
                self._failed_at_step = None
                self._repair_step = None
        else:
            fail_prob = self.config.failure_rate_per_hour / steps_per_hour
            if self.rng.random() < fail_prob:
                self._failed = True
                self._failed_at_step = step
                repair_steps = int(self.config.repair_time_hours * steps_per_hour)
                self._repair_step = step + repair_steps

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    @property
    def is_failed(self) -> bool:
        """True when detection is lost and controller must fall back to fixed timing."""
        if not self.config.enabled:
            return False
        if self.config.detector_type == DetectorType.NONE:
            return True
        return self._failed

    @property
    def status(self) -> str:
        """Human-readable detector status string."""
        if not self.config.enabled:
            return "disabled"
        if self.config.detector_type == DetectorType.NONE:
            return "no_detector"
        if self._failed:
            return "failed"
        return "operational"

    def fallback_action(self, step: int) -> str:
        """Fixed-timing fallback phase (alternates NS/EW every FALLBACK_CYCLE_STEPS)."""
        return "NS_THROUGH" if (step // self.FALLBACK_CYCLE_STEPS) % 2 == 0 else "EW_THROUGH"

    # ------------------------------------------------------------------
    # Observation degradation (video only)
    # ------------------------------------------------------------------

    def degrade_observation(
        self,
        obs: dict[str, float],
        hour_of_day: int,
    ) -> dict[str, float]:
        """Return noisy copy of obs when video detector is degraded.

        For loop/radar detectors: returns obs unchanged (no degradation
        when operational — binary failure model).

        For video detectors: adds Gaussian noise during glare windows and
        at night.  Queue and flow keys only; non-queue metadata unchanged.
        """
        if not self.config.enabled:
            return obs
        if self.config.detector_type != DetectorType.VIDEO:
            return obs  # Loop/radar: no graded degradation

        noise_std = 0.0
        if hour_of_day in self.config.video_glare_hours:
            noise_std = 0.25  # Heavy solar glare: ±25 % noise
        elif hour_of_day >= 20 or hour_of_day < 6:
            noise_std = self.config.video_night_degradation

        if noise_std == 0.0:
            return obs

        # Apply multiplicative Gaussian noise to queue-related keys only
        _QUEUE_KEYS = {
            "queue_ns", "queue_ew", "queue_ns_through", "queue_ew_through",
            "queue_ns_left", "queue_ew_left", "total_queue",
            "upstream_queue",
        }
        return {
            k: max(0.0, v * (1.0 + self.rng.normal(0.0, noise_std)))
            if k in _QUEUE_KEYS else v
            for k, v in obs.items()
        }
