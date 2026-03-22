"""Predictive congestion module: forecasts traffic 5-15 minutes ahead.

Uses historical pattern learning and exponential smoothing to predict
future queue states, enabling proactive signal timing adjustments.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Deque

import numpy as np


@dataclass(slots=True)
class PredictionResult:
    """Result of a congestion prediction."""
    horizon_minutes: float
    predicted_queue: float
    predicted_wait_sec: float
    confidence: float  # 0-1
    trend: str  # "increasing", "decreasing", "stable"
    recommended_action: str  # "extend_green", "switch_phase", "maintain"


class CongestionForecaster:
    """Predicts congestion 5-15 minutes ahead using historical patterns.

    Combines:
    1. Exponential smoothing for trend detection
    2. Time-of-day seasonal patterns (24h cycle)
    3. Linear extrapolation for short-term prediction
    """

    def __init__(
        self,
        history_window: int = 300,  # 5 minutes of history at 1 step/sec
        seasonal_bins: int = 96,     # 15-minute bins across 24 hours
        smoothing_alpha: float = 0.15,
        seed: int = 42,
    ) -> None:
        self.history_window = history_window
        self.seasonal_bins = seasonal_bins
        self.alpha = smoothing_alpha

        # Per-intersection history
        self._queue_history: dict[int, Deque[float]] = {}
        self._wait_history: dict[int, Deque[float]] = {}

        # Seasonal patterns (learned over time)
        self._seasonal_queue: dict[int, np.ndarray] = {}
        self._seasonal_counts: dict[int, np.ndarray] = {}

        # Exponential moving average
        self._ema: dict[int, float] = {}

    def update(
        self,
        intersection_id: int,
        queue: float,
        wait_sec: float,
        sim_step: int,
        step_seconds: float = 1.0,
    ) -> None:
        """Feed a new observation to the forecaster."""
        if intersection_id not in self._queue_history:
            self._queue_history[intersection_id] = deque(maxlen=self.history_window)
            self._wait_history[intersection_id] = deque(maxlen=self.history_window)
            self._seasonal_queue[intersection_id] = np.zeros(self.seasonal_bins)
            self._seasonal_counts[intersection_id] = np.zeros(self.seasonal_bins)
            self._ema[intersection_id] = queue

        self._queue_history[intersection_id].append(queue)
        self._wait_history[intersection_id].append(wait_sec)

        # Update EMA
        self._ema[intersection_id] = (
            self.alpha * queue + (1 - self.alpha) * self._ema[intersection_id]
        )

        # Update seasonal pattern
        hour = (sim_step * step_seconds / 3600.0) % 24.0
        bin_idx = int(hour / 24.0 * self.seasonal_bins) % self.seasonal_bins
        self._seasonal_queue[intersection_id][bin_idx] += queue
        self._seasonal_counts[intersection_id][bin_idx] += 1

    def predict(
        self,
        intersection_id: int,
        horizon_minutes: float,
        current_step: int,
        step_seconds: float = 1.0,
    ) -> PredictionResult:
        """Predict congestion at given horizon (5-15 minutes ahead)."""
        if intersection_id not in self._queue_history:
            return PredictionResult(
                horizon_minutes=horizon_minutes,
                predicted_queue=0.0,
                predicted_wait_sec=0.0,
                confidence=0.0,
                trend="stable",
                recommended_action="maintain",
            )

        history = list(self._queue_history[intersection_id])
        wait_history = list(self._wait_history[intersection_id])
        n = len(history)

        if n < 10:
            return PredictionResult(
                horizon_minutes=horizon_minutes,
                predicted_queue=history[-1] if history else 0.0,
                predicted_wait_sec=wait_history[-1] if wait_history else 0.0,
                confidence=0.1,
                trend="stable",
                recommended_action="maintain",
            )

        # 1. Trend from linear regression on recent window
        recent_window = min(60, n)  # Last 60 steps
        recent = np.array(history[-recent_window:])
        x = np.arange(recent_window, dtype=np.float64)
        slope, intercept = np.polyfit(x, recent, 1)

        # 2. Seasonal component
        future_hour = ((current_step + horizon_minutes * 60 / step_seconds) * step_seconds / 3600.0) % 24.0
        future_bin = int(future_hour / 24.0 * self.seasonal_bins) % self.seasonal_bins
        seasonal_avg = (
            self._seasonal_queue[intersection_id][future_bin]
            / max(self._seasonal_counts[intersection_id][future_bin], 1)
        )

        # 3. Combine: trend extrapolation + seasonal adjustment
        horizon_steps = horizon_minutes * 60 / step_seconds
        trend_prediction = intercept + slope * (recent_window + horizon_steps)
        ema = self._ema[intersection_id]

        # Weighted combination
        if self._seasonal_counts[intersection_id][future_bin] > 5:
            predicted_queue = 0.4 * trend_prediction + 0.3 * seasonal_avg + 0.3 * ema
        else:
            predicted_queue = 0.5 * trend_prediction + 0.5 * ema

        predicted_queue = max(0.0, predicted_queue)

        # Predict wait time proportionally
        current_wait = wait_history[-1] if wait_history else 0.0
        current_queue = history[-1] if history else 1.0
        wait_ratio = current_wait / max(current_queue, 0.1)
        predicted_wait = predicted_queue * wait_ratio

        # Confidence based on data quantity and prediction stability
        data_confidence = min(1.0, n / self.history_window)
        stability = 1.0 / (1.0 + abs(slope) * 10)
        confidence = 0.6 * data_confidence + 0.4 * stability

        # Trend classification
        if slope > 0.05:
            trend = "increasing"
        elif slope < -0.05:
            trend = "decreasing"
        else:
            trend = "stable"

        # Recommended action
        if trend == "increasing" and predicted_queue > history[-1] * 1.3:
            recommended = "extend_green"
        elif trend == "decreasing" and predicted_queue < history[-1] * 0.7:
            recommended = "switch_phase"
        else:
            recommended = "maintain"

        return PredictionResult(
            horizon_minutes=horizon_minutes,
            predicted_queue=predicted_queue,
            predicted_wait_sec=predicted_wait,
            confidence=confidence,
            trend=trend,
            recommended_action=recommended,
        )

    def predict_corridor(
        self,
        n_intersections: int,
        horizon_minutes: float,
        current_step: int,
        step_seconds: float = 1.0,
    ) -> list[PredictionResult]:
        """Predict congestion across all corridor intersections."""
        return [
            self.predict(i, horizon_minutes, current_step, step_seconds)
            for i in range(n_intersections)
        ]
