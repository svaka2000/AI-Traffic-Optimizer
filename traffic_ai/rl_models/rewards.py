"""traffic_ai/rl_models/rewards.py

Composite reward shaping for traffic signal RL controllers.

The RewardShaper assembles a multi-objective reward signal that balances:
- Queue minimization (primary objective)
- Throughput maximization (secondary)
- Wait-time reduction
- Phase-change stability (penalise rapid flickering)
- Emergency vehicle priority (hard requirement)
- Fairness (penalise large NS/EW imbalance)
- Improvement over baseline (incremental reward)

Unlike a single-objective reward that only penalises queue length, this shaper
provides richer gradient signal and discourages degenerate policies such as
always granting green to the same direction.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class RewardWeights:
    """Configurable weights for each reward component."""
    queue_penalty: float = -1.0
    throughput_bonus: float = 0.5
    wait_penalty: float = -0.3
    phase_change_penalty: float = -0.2
    emergency_cleared_bonus: float = 2.0
    fairness_penalty: float = -0.1
    delay_reduction_bonus: float = 0.4


class RewardShaper:
    """Compute a composite, normalised reward signal for a single step.

    All inputs are normalised by configurable maximum values so that each
    component contributes on a comparable scale (roughly -1 to +1 per term).

    Parameters
    ----------
    weights:
        Per-component reward weights. Pass ``None`` to use defaults.
    max_queue:
        Normalisation constant for queue length.
    max_throughput:
        Normalisation constant for throughput.
    max_wait:
        Normalisation constant for wait time.
    fairness_threshold:
        Queue imbalance (vehicles) below which no fairness penalty is applied.
    """

    def __init__(
        self,
        weights: Optional[RewardWeights] = None,
        max_queue: float = 240.0,
        max_throughput: float = 4.0,
        max_wait: float = 300.0,
        fairness_threshold: float = 10.0,
    ) -> None:
        self.w = weights if weights is not None else RewardWeights()
        self.max_queue = max(max_queue, 1.0)
        self.max_throughput = max(max_throughput, 1.0)
        self.max_wait = max(max_wait, 1.0)
        self.fairness_threshold = fairness_threshold

    def compute(
        self,
        total_queue: float,
        throughput: float,
        total_wait_seconds: float,
        phase_changed: bool,
        emergency_vehicle_cleared: bool = False,
        queue_ns: float = 0.0,
        queue_ew: float = 0.0,
        delay_reduction_vs_baseline: float = 0.0,
    ) -> float:
        """Compute the composite reward for a single simulation step.

        Parameters
        ----------
        total_queue:
            Sum of all queued vehicles across all lanes/intersections.
        throughput:
            Vehicles departed this step.
        total_wait_seconds:
            Cumulative wait seconds accumulated this step.
        phase_changed:
            True if the signal phase was switched this step.
        emergency_vehicle_cleared:
            True if an emergency vehicle was cleared (bonus reward).
        queue_ns:
            North-South queue total (for fairness penalty).
        queue_ew:
            East-West queue total (for fairness penalty).
        delay_reduction_vs_baseline:
            Fractional improvement in delay over the fixed-timing baseline (0–1).

        Returns
        -------
        float
            Scalar composite reward.
        """
        queue_term = self.w.queue_penalty * (total_queue / self.max_queue)
        throughput_term = self.w.throughput_bonus * (throughput / self.max_throughput)
        wait_term = self.w.wait_penalty * (total_wait_seconds / self.max_wait)
        change_term = self.w.phase_change_penalty * float(phase_changed)
        emergency_term = self.w.emergency_cleared_bonus * float(emergency_vehicle_cleared)

        imbalance = abs(queue_ns - queue_ew)
        fairness_term = (
            self.w.fairness_penalty * max(0.0, imbalance - self.fairness_threshold)
            / self.max_queue
        )

        delay_term = self.w.delay_reduction_bonus * max(0.0, min(1.0, delay_reduction_vs_baseline))

        return (
            queue_term
            + throughput_term
            + wait_term
            + change_term
            + emergency_term
            + fairness_term
            + delay_term
        )
