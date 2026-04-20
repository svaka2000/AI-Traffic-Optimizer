"""aito/optimization/continuous_retiming.py

GF7: Continuous Auto-Retiming Engine.

Monitors real-time probe data quality metrics and triggers automatic
re-optimization when performance degrades beyond defined thresholds.

Core value proposition:
  AITO's #1 competitive advantage over InSync: InSync requires on-site
  engineers for re-timing.  AITO retimes itself continuously, targeting
  the "24-hours-a-day optimization" that Steve Celniker highlighted.

Architecture:
  RetimingMonitor runs as a background service, polling probe data
  and ATSPM quality metrics every `check_interval_s` seconds.  When
  degradation is detected (delay drift, bandwidth collapse, cycle
  infeasibility), it schedules a re-optimization job.

Triggers:
  1. Delay drift:    avg intersection delay exceeds baseline + threshold_pct
  2. Bandwidth drop: green-wave bandwidth falls below min_bandwidth_pct
  3. Cycle overload: any intersection shows split failure > threshold_pct
  4. Demand shift:   probe-inferred volume changes > demand_shift_pct
  5. Time-of-day:    new TOD period begins (scheduled trigger)

Thresholds calibrated to San Diego InSync deployment benchmarks:
  delay drift > 15% triggers re-evaluation
  bandwidth < 20% triggers offset recalculation
  split failure > 30% triggers full re-timing
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional, Callable


# ---------------------------------------------------------------------------
# Performance snapshot — baseline & current
# ---------------------------------------------------------------------------

@dataclass
class PerformanceSnapshot:
    """Point-in-time corridor performance metrics."""
    timestamp: datetime
    corridor_id: str

    # Delay metrics
    avg_delay_s_veh: float          # average intersection delay
    max_delay_s_veh: float          # worst intersection

    # Bandwidth metrics
    bandwidth_outbound_pct: float   # green-wave bandwidth % of cycle
    bandwidth_inbound_pct: float

    # Quality metrics
    split_failure_pct: float        # fraction of cycles with split failure
    arrival_on_green_pct: float     # Purdue coordination quality
    probe_confidence: float         # mean probe data confidence [0, 1]

    # Demand
    volume_veh_hr: float            # total approach volume

    # Plan metadata
    cycle_s: float
    active_plan_id: str = ""


@dataclass
class RetimingTrigger:
    """Record of a detected degradation event."""
    triggered_at: datetime
    trigger_type: str           # delay_drift | bandwidth_drop | split_failure | demand_shift | tod_transition
    severity: float             # 0–1
    description: str
    recommended_action: str     # full_retiming | offset_only | cycle_adjust | no_action
    baseline_value: float = 0.0
    current_value: float = 0.0
    threshold: float = 0.0


class RetimingAction(str, Enum):
    NO_ACTION       = "no_action"
    OFFSET_ONLY     = "offset_only"     # just recompute offsets, keep cycle
    CYCLE_ADJUST    = "cycle_adjust"    # adjust cycle ±5s
    FULL_RETIMING   = "full_retiming"   # full Webster + MAXBAND reopt
    NSGA_RETIMING   = "nsga_retiming"   # full NSGA-III (expensive, rare)


# ---------------------------------------------------------------------------
# Degradation detectors
# ---------------------------------------------------------------------------

def detect_delay_drift(
    baseline: PerformanceSnapshot,
    current: PerformanceSnapshot,
    threshold_pct: float = 0.15,
) -> Optional[RetimingTrigger]:
    """Detect when average delay has drifted > threshold above baseline."""
    if baseline.avg_delay_s_veh <= 0:
        return None
    drift = (current.avg_delay_s_veh - baseline.avg_delay_s_veh) / baseline.avg_delay_s_veh
    if drift > threshold_pct:
        severity = min(1.0, drift / (threshold_pct * 3))
        action = (RetimingAction.FULL_RETIMING.value if drift > 0.30
                  else RetimingAction.CYCLE_ADJUST.value if drift > 0.20
                  else RetimingAction.OFFSET_ONLY.value)
        return RetimingTrigger(
            triggered_at=current.timestamp,
            trigger_type="delay_drift",
            severity=severity,
            description=f"Delay drifted +{drift * 100:.1f}% above baseline ({baseline.avg_delay_s_veh:.1f}→{current.avg_delay_s_veh:.1f} s/veh)",
            recommended_action=action,
            baseline_value=baseline.avg_delay_s_veh,
            current_value=current.avg_delay_s_veh,
            threshold=threshold_pct,
        )
    return None


def detect_bandwidth_collapse(
    baseline: PerformanceSnapshot,
    current: PerformanceSnapshot,
    min_bandwidth_pct: float = 20.0,
    drop_threshold_pct: float = 0.25,
) -> Optional[RetimingTrigger]:
    """Detect green-wave bandwidth collapse."""
    combined_bw = (current.bandwidth_outbound_pct + current.bandwidth_inbound_pct) / 2
    baseline_bw = (baseline.bandwidth_outbound_pct + baseline.bandwidth_inbound_pct) / 2
    drop = (baseline_bw - combined_bw) / max(baseline_bw, 1.0)

    if combined_bw < min_bandwidth_pct or drop > drop_threshold_pct:
        severity = min(1.0, max(
            (min_bandwidth_pct - combined_bw) / max(min_bandwidth_pct, 1.0),
            drop / (drop_threshold_pct * 2),
        ))
        return RetimingTrigger(
            triggered_at=current.timestamp,
            trigger_type="bandwidth_drop",
            severity=severity,
            description=f"Green-wave bandwidth fell to {combined_bw:.1f}% (was {baseline_bw:.1f}%)",
            recommended_action=RetimingAction.OFFSET_ONLY.value,
            baseline_value=baseline_bw,
            current_value=combined_bw,
            threshold=min_bandwidth_pct,
        )
    return None


def detect_split_failure(
    current: PerformanceSnapshot,
    threshold_pct: float = 0.30,
) -> Optional[RetimingTrigger]:
    """Detect excessive split failures (demand exceeds phase capacity)."""
    if current.split_failure_pct > threshold_pct:
        severity = min(1.0, current.split_failure_pct / (threshold_pct * 2))
        return RetimingTrigger(
            triggered_at=current.timestamp,
            trigger_type="split_failure",
            severity=severity,
            description=f"Split failure rate {current.split_failure_pct * 100:.0f}% exceeds {threshold_pct * 100:.0f}% threshold",
            recommended_action=RetimingAction.FULL_RETIMING.value,
            baseline_value=0.0,
            current_value=current.split_failure_pct,
            threshold=threshold_pct,
        )
    return None


def detect_demand_shift(
    baseline: PerformanceSnapshot,
    current: PerformanceSnapshot,
    threshold_pct: float = 0.20,
) -> Optional[RetimingTrigger]:
    """Detect significant volume change requiring re-optimization."""
    if baseline.volume_veh_hr <= 0:
        return None
    shift = abs(current.volume_veh_hr - baseline.volume_veh_hr) / baseline.volume_veh_hr
    if shift > threshold_pct:
        direction = "increase" if current.volume_veh_hr > baseline.volume_veh_hr else "decrease"
        severity = min(1.0, shift / (threshold_pct * 2))
        action = (RetimingAction.FULL_RETIMING.value if shift > 0.40
                  else RetimingAction.CYCLE_ADJUST.value)
        return RetimingTrigger(
            triggered_at=current.timestamp,
            trigger_type="demand_shift",
            severity=severity,
            description=f"Demand {direction} {shift * 100:.0f}%: {baseline.volume_veh_hr:.0f}→{current.volume_veh_hr:.0f} veh/hr",
            recommended_action=action,
            baseline_value=baseline.volume_veh_hr,
            current_value=current.volume_veh_hr,
            threshold=threshold_pct,
        )
    return None


def detect_tod_transition(
    now: datetime,
    last_transition: datetime,
    tod_periods: Optional[dict] = None,
) -> Optional[RetimingTrigger]:
    """Detect scheduled time-of-day period boundary."""
    from aito.optimization.isolated_optimizer import TOD_PERIODS

    periods = tod_periods or TOD_PERIODS
    for name, hours in periods.items():
        if hours is None:
            continue
        start_h, end_h = hours
        # Check if we've just crossed a period boundary (within last 5 min)
        current_h = now.hour + now.minute / 60.0
        prev_h = (now - timedelta(minutes=5)).hour + (now - timedelta(minutes=5)).minute / 60.0
        in_new = start_h <= current_h < end_h
        was_in_prev = start_h <= prev_h < end_h
        if in_new and not was_in_prev:
            return RetimingTrigger(
                triggered_at=now,
                trigger_type="tod_transition",
                severity=0.5,
                description=f"Time-of-day transition → {name}",
                recommended_action=RetimingAction.FULL_RETIMING.value,
            )
    return None


# ---------------------------------------------------------------------------
# Retiming schedule
# ---------------------------------------------------------------------------

@dataclass
class RetimingJob:
    """Scheduled re-optimization work item."""
    job_id: str
    corridor_id: str
    trigger: RetimingTrigger
    scheduled_at: datetime
    action: RetimingAction
    priority: int = 5           # 1 = highest (split failure), 10 = lowest
    completed: bool = False
    result_plan_id: Optional[str] = None

    @classmethod
    def from_trigger(cls, trigger: RetimingTrigger, corridor_id: str) -> "RetimingJob":
        import uuid
        action_map = {
            RetimingAction.OFFSET_ONLY.value:    (RetimingAction.OFFSET_ONLY, 7),
            RetimingAction.CYCLE_ADJUST.value:   (RetimingAction.CYCLE_ADJUST, 5),
            RetimingAction.FULL_RETIMING.value:  (RetimingAction.FULL_RETIMING, 3),
            RetimingAction.NSGA_RETIMING.value:  (RetimingAction.NSGA_RETIMING, 8),
        }
        action, priority = action_map.get(trigger.recommended_action, (RetimingAction.NO_ACTION, 10))
        # High-severity split failures are highest priority
        if trigger.trigger_type == "split_failure":
            priority = 1
        return cls(
            job_id=str(uuid.uuid4())[:8],
            corridor_id=corridor_id,
            trigger=trigger,
            scheduled_at=trigger.triggered_at,
            action=action,
            priority=priority,
        )


# ---------------------------------------------------------------------------
# RetimingMonitor — core service
# ---------------------------------------------------------------------------

@dataclass
class MonitorConfig:
    """Configuration for the retiming monitor."""
    check_interval_s: float = 300.0          # check every 5 minutes
    delay_drift_threshold: float = 0.15      # 15% delay increase
    bandwidth_min_pct: float = 20.0          # green-wave minimum
    split_failure_threshold: float = 0.30    # 30% cycles failing
    demand_shift_threshold: float = 0.20     # 20% volume change
    min_retiming_interval_s: float = 1800.0  # don't retiming more often than 30 min
    max_jobs_queued: int = 10
    enable_nsga: bool = False                # NSGA-III (expensive) only for major events


class RetimingMonitor:
    """Continuous auto-retiming service for AITO corridors.

    Usage:
        monitor = RetimingMonitor("rosecrans", config)
        monitor.set_baseline(initial_snapshot)
        # ... poll probe data ...
        triggers = monitor.evaluate(current_snapshot)
        jobs = monitor.pending_jobs
    """

    def __init__(
        self,
        corridor_id: str,
        config: Optional[MonitorConfig] = None,
    ) -> None:
        self.corridor_id = corridor_id
        self.config = config or MonitorConfig()
        self._baseline: Optional[PerformanceSnapshot] = None
        self._last_retiming: Optional[datetime] = None
        self._last_tod_transition: Optional[datetime] = None
        self._jobs: list[RetimingJob] = []
        self._history: list[RetimingTrigger] = []

    def set_baseline(self, snapshot: PerformanceSnapshot) -> None:
        """Establish performance baseline for drift detection."""
        self._baseline = snapshot

    def evaluate(self, current: PerformanceSnapshot) -> list[RetimingTrigger]:
        """Evaluate current performance and return any triggers fired.

        Side effect: queues RetimingJob for each actionable trigger.
        """
        triggers: list[RetimingTrigger] = []
        cfg = self.config

        if self._baseline is None:
            self.set_baseline(current)
            return []

        # Check minimum retiming interval
        if (self._last_retiming is not None and
                (current.timestamp - self._last_retiming).total_seconds()
                < cfg.min_retiming_interval_s):
            return []

        # Run detectors
        t = detect_delay_drift(self._baseline, current, cfg.delay_drift_threshold)
        if t: triggers.append(t)

        t = detect_bandwidth_collapse(self._baseline, current, cfg.bandwidth_min_pct)
        if t: triggers.append(t)

        t = detect_split_failure(current, cfg.split_failure_threshold)
        if t: triggers.append(t)

        t = detect_demand_shift(self._baseline, current, cfg.demand_shift_threshold)
        if t: triggers.append(t)

        if self._last_tod_transition:
            t = detect_tod_transition(current.timestamp, self._last_tod_transition)
            if t:
                triggers.append(t)
                self._last_tod_transition = current.timestamp

        # Queue jobs for actionable triggers
        for trigger in triggers:
            if trigger.recommended_action != RetimingAction.NO_ACTION.value:
                # Upgrade to NSGA if configured and severity high
                if (cfg.enable_nsga and trigger.severity > 0.8):
                    trigger.recommended_action = RetimingAction.NSGA_RETIMING.value
                job = RetimingJob.from_trigger(trigger, self.corridor_id)
                self._jobs.append(job)
                self._jobs.sort(key=lambda j: j.priority)
                if len(self._jobs) > cfg.max_jobs_queued:
                    self._jobs = self._jobs[:cfg.max_jobs_queued]

        self._history.extend(triggers)
        if triggers:
            self._last_retiming = current.timestamp

        return triggers

    @property
    def pending_jobs(self) -> list[RetimingJob]:
        return [j for j in self._jobs if not j.completed]

    @property
    def trigger_history(self) -> list[RetimingTrigger]:
        return list(self._history)

    def mark_completed(self, job_id: str, result_plan_id: Optional[str] = None) -> None:
        for job in self._jobs:
            if job.job_id == job_id:
                job.completed = True
                job.result_plan_id = result_plan_id
                # Update baseline after successful retiming
                break

    def update_baseline(self, snapshot: PerformanceSnapshot) -> None:
        """Call after successful retiming to reset drift detection."""
        self._baseline = snapshot

    def retiming_frequency_hr(self) -> float:
        """Average retimings per hour from history."""
        if len(self._history) < 2:
            return 0.0
        span_hr = (
            (self._history[-1].triggered_at - self._history[0].triggered_at).total_seconds()
            / 3600.0
        )
        return len(self._history) / max(span_hr, 0.01)

    def summary(self) -> dict:
        return {
            "corridor_id": self.corridor_id,
            "triggers_total": len(self._history),
            "pending_jobs": len(self.pending_jobs),
            "last_retiming": self._last_retiming.isoformat() if self._last_retiming else None,
            "retiming_frequency_per_hr": round(self.retiming_frequency_hr(), 2),
            "trigger_breakdown": {
                t: sum(1 for h in self._history if h.trigger_type == t)
                for t in {"delay_drift", "bandwidth_drop", "split_failure",
                          "demand_shift", "tod_transition"}
            },
        }
