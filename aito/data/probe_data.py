"""aito/data/probe_data.py

Probe data adapters — INRIX, HERE, Waze, synthetic.

AITO uses probe data instead of broken loop detectors.  This module
provides a common interface across providers.

In production, replace SyntheticProbeAdapter with INRIXAdapter or
HEREAdapter by supplying valid API keys via environment variables.
"""
from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from typing import Protocol, runtime_checkable

from aito.data.schemas import TravelTimeSegment, SpeedSegment


@runtime_checkable
class ProbeDataProvider(Protocol):
    """Interface for any probe data source."""

    async def get_travel_times(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: timedelta = timedelta(minutes=15),
    ) -> list[TravelTimeSegment]: ...

    async def get_speeds(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[SpeedSegment]: ...


class SyntheticProbeAdapter:
    """Generates realistic synthetic probe data for demo and testing.

    Simulates AM/PM peak congestion patterns on a San Diego-style arterial.
    """

    def __init__(self, free_flow_speed_mph: float = 35.0, seed: int = 42) -> None:
        self.free_flow_speed_mph = free_flow_speed_mph
        self.rng = random.Random(seed)

    async def get_travel_times(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: timedelta = timedelta(minutes=15),
    ) -> list[TravelTimeSegment]:
        """Generate 15-min travel time segments between start and end."""
        results: list[TravelTimeSegment] = []
        current = start_time
        seg_idx = 0
        while current < end_time:
            speed = self._speed_at_time(current)
            tt = self._distance_m_for_segment(seg_idx) / (speed * 0.44704)
            results.append(TravelTimeSegment(
                segment_id=f"{corridor_id}_seg{seg_idx}",
                start_time=current,
                end_time=current + resolution,
                travel_time_s=round(tt, 1),
                speed_mph=round(speed, 1),
                confidence=0.85 + self.rng.uniform(-0.1, 0.1),
            ))
            current += resolution
            seg_idx += 1
        return results

    async def get_speeds(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
    ) -> list[SpeedSegment]:
        results: list[SpeedSegment] = []
        current = start_time
        while current < end_time:
            speed = self._speed_at_time(current)
            cong = self._congestion_label(speed)
            results.append(SpeedSegment(
                segment_id=corridor_id,
                timestamp=current,
                speed_mph=round(speed, 1),
                free_flow_speed_mph=self.free_flow_speed_mph,
                congestion_level=cong,
            ))
            current += timedelta(minutes=15)
        return results

    def get_travel_times_sync(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: timedelta = timedelta(minutes=15),
    ) -> list[TravelTimeSegment]:
        """Synchronous version for use in demo scripts."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
        return loop.run_until_complete(
            self.get_travel_times(corridor_id, start_time, end_time, resolution)
        )

    def _speed_at_time(self, dt: datetime) -> float:
        """Speed as function of time of day — models AM/PM congestion."""
        h = dt.hour + dt.minute / 60.0
        # AM peak: 7–9 AM (speed drops to ~60% of free flow)
        if 7 <= h < 9:
            factor = 0.60 + 0.20 * abs(h - 8.0)
        # PM peak: 4–6 PM
        elif 16 <= h < 18:
            factor = 0.60 + 0.20 * abs(h - 17.0)
        # Midday moderate
        elif 9 <= h < 16:
            factor = 0.85
        # Night free-flow
        else:
            factor = 0.95
        noise = self.rng.uniform(-0.05, 0.05)
        return max(5.0, self.free_flow_speed_mph * (factor + noise))

    @staticmethod
    def _distance_m_for_segment(seg_idx: int) -> float:
        """Default distance: 400m per segment."""
        return 400.0

    @staticmethod
    def _congestion_label(speed_mph: float) -> str:
        if speed_mph > 35:
            return "free"
        if speed_mph > 20:
            return "moderate"
        if speed_mph > 10:
            return "heavy"
        return "stop-and-go"


class INRIXAdapter:
    """INRIX XD segment travel times (production adapter).

    Requires INRIX_API_TOKEN environment variable.
    XD segment IDs must be configured per corridor.
    """

    def __init__(self, api_token: str, timeout_s: float = 10.0) -> None:
        self.api_token = api_token
        self.timeout_s = timeout_s
        self._base_url = "https://api.iq.inrix.com/v1"

    async def get_travel_times(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: timedelta = timedelta(minutes=15),
    ) -> list[TravelTimeSegment]:
        # In production: call INRIX Traffic API
        # GET /segments?ids=<xd_ids>&startTime=...&endTime=...&resolution=15
        raise NotImplementedError(
            "INRIXAdapter requires a valid INRIX_API_TOKEN and XD segment mapping. "
            "Use SyntheticProbeAdapter for demo/testing."
        )

    async def get_speeds(self, corridor_id: str, start_time: datetime, end_time: datetime) -> list[SpeedSegment]:
        raise NotImplementedError("INRIXAdapter.get_speeds not implemented")


class HEREAdapter:
    """HERE Traffic Flow API adapter (production)."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key

    async def get_travel_times(
        self,
        corridor_id: str,
        start_time: datetime,
        end_time: datetime,
        resolution: timedelta = timedelta(minutes=15),
    ) -> list[TravelTimeSegment]:
        raise NotImplementedError(
            "HEREAdapter requires a valid HERE_API_KEY. "
            "Use SyntheticProbeAdapter for demo/testing."
        )

    async def get_speeds(self, corridor_id: str, start_time: datetime, end_time: datetime) -> list[SpeedSegment]:
        raise NotImplementedError("HEREAdapter.get_speeds not implemented")
