"""aito/data/atspm.py

Automated Traffic Signal Performance Measures (ATSPM) integration.

Computes industry-standard metrics from high-resolution controller event logs:
  - Arrival on Green (AoG) percentage
  - Split failure percentage
  - Purdue Coordination Diagram
  - Platoon ratio

References:
  FHWA-HOP-14-031 "Signal Timing Manual" 2nd Edition
  Utah ATSPM open source reference implementation
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from aito.data.schemas import ATSPMEvent, ATSPMMetrics


# NTCIP controller event codes (common across manufacturers)
EC_PHASE_GREEN = 1        # Phase begins green
EC_PHASE_YELLOW = 8       # Phase begins yellow
EC_PHASE_RED = 11         # Phase begins red
EC_DETECTOR_ON = 81       # Vehicle detected (detector actuated)
EC_DETECTOR_OFF = 82      # Detector de-actuated
EC_COORD_FORCE_OFF = 22   # Coordinated phase force-off


class ATSPMCalculator:
    """Compute ATSPM performance metrics from controller event logs.

    Parameters
    ----------
    intersection_id:
        Identifier for the intersection being analyzed.
    cycle_length_s:
        Nominal cycle length in seconds (for period normalization).
    """

    def __init__(self, intersection_id: str, cycle_length_s: float = 120.0) -> None:
        self.intersection_id = intersection_id
        self.cycle_length_s = cycle_length_s

    def compute_metrics(
        self,
        events: list[ATSPMEvent],
        period_start: datetime,
        period_end: datetime,
    ) -> ATSPMMetrics:
        """Compute all ATSPM metrics for a time period."""
        aog = self._arrival_on_green(events)
        sf = self._split_failure_pct(events)
        pr = self._platoon_ratio(events)
        pcs = self._purdue_coordination_score(aog, pr)

        return ATSPMMetrics(
            intersection_id=self.intersection_id,
            period_start=period_start,
            period_end=period_end,
            arrival_on_green_pct=round(aog, 1),
            split_failure_pct=round(sf, 1),
            platoon_ratio=round(pr, 2),
            purdue_coordination_score=round(pcs, 1),
        )

    def _arrival_on_green(self, events: list[ATSPMEvent]) -> float:
        """Estimate % arrivals during green.

        Approximated from detector actuations relative to phase green periods.
        """
        green_intervals: list[tuple[datetime, datetime]] = []
        phase_green_start: Optional[datetime] = None

        # Use phase 2 (major through) as representative
        for e in sorted(events, key=lambda x: x.timestamp):
            if e.event_code == EC_PHASE_GREEN and e.parameter == 2:
                phase_green_start = e.timestamp
            elif e.event_code == EC_PHASE_YELLOW and e.parameter == 2:
                if phase_green_start:
                    green_intervals.append((phase_green_start, e.timestamp))
                    phase_green_start = None

        detector_arrivals = [
            e.timestamp for e in events if e.event_code == EC_DETECTOR_ON
        ]

        if not detector_arrivals:
            return 50.0  # default assumption if no data

        on_green = sum(
            1 for arr in detector_arrivals
            if any(g_start <= arr <= g_end for g_start, g_end in green_intervals)
        )
        return 100.0 * on_green / max(len(detector_arrivals), 1)

    def _split_failure_pct(self, events: list[ATSPMEvent]) -> float:
        """Estimate % of phases that end in split failure.

        A split failure occurs when a phase reaches max green while a
        vehicle is still detected (queue not served).
        """
        # Simplified: if force-off events occur near detector actuation → split failure
        force_offs = [e.timestamp for e in events if e.event_code == EC_COORD_FORCE_OFF]
        detector_at_force_off = 0
        window = timedelta(seconds=3)

        for fo in force_offs:
            nearby = [
                e for e in events
                if e.event_code == EC_DETECTOR_ON and abs((e.timestamp - fo).total_seconds()) < window.total_seconds()
            ]
            if nearby:
                detector_at_force_off += 1

        if not force_offs:
            return 0.0
        return 100.0 * detector_at_force_off / len(force_offs)

    def _platoon_ratio(self, events: list[ATSPMEvent]) -> float:
        """Platoon ratio = (arrivals during green / cycle capacity) / (total arrivals / capacity).

        PR > 1.0 = good coordination (green wave), PR < 1.0 = poor coordination.
        """
        total = sum(1 for e in events if e.event_code == EC_DETECTOR_ON)
        if total == 0:
            return 1.0
        aog_ratio = self._arrival_on_green(events) / 100.0
        return aog_ratio / 0.5  # normalized against random arrivals (50% AoG)

    @staticmethod
    def _purdue_coordination_score(aog_pct: float, platoon_ratio: float) -> float:
        """Composite score 0–100.  >70 = good coordination."""
        return min(100.0, aog_pct * 0.6 + min(platoon_ratio, 2.0) * 20.0)


def generate_synthetic_events(
    intersection_id: str,
    period_start: datetime,
    cycle_length_s: float = 120.0,
    aog_pct: float = 65.0,
    num_cycles: int = 20,
) -> list[ATSPMEvent]:
    """Generate realistic synthetic ATSPM events for testing and demo."""
    import random
    rng = random.Random(42)
    events: list[ATSPMEvent] = []
    t = period_start

    for _ in range(num_cycles):
        # Phase 2 green
        green_start = t
        green_duration = cycle_length_s * 0.45
        events.append(ATSPMEvent(
            device_id=intersection_id,
            timestamp=green_start,
            event_code=EC_PHASE_GREEN,
            parameter=2,
        ))
        # Detector arrivals — simulate AoG
        for _ in range(rng.randint(5, 20)):
            if rng.random() < aog_pct / 100.0:
                arr = green_start + timedelta(seconds=rng.uniform(0, green_duration))
            else:
                arr = t + timedelta(seconds=rng.uniform(green_duration, cycle_length_s))
            events.append(ATSPMEvent(
                device_id=intersection_id,
                timestamp=arr,
                event_code=EC_DETECTOR_ON,
                parameter=1,
            ))
        # Yellow
        events.append(ATSPMEvent(
            device_id=intersection_id,
            timestamp=t + timedelta(seconds=green_duration),
            event_code=EC_PHASE_YELLOW,
            parameter=2,
        ))
        t += timedelta(seconds=cycle_length_s)

    return sorted(events, key=lambda e: e.timestamp)
