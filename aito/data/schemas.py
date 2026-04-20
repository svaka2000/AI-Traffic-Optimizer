"""aito/data/schemas.py — Pydantic schemas for inbound data."""
from __future__ import annotations

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class TravelTimeSegment(BaseModel):
    segment_id: str
    start_time: datetime
    end_time: datetime
    travel_time_s: float
    speed_mph: float
    confidence: float = 1.0   # 0–1


class SpeedSegment(BaseModel):
    segment_id: str
    timestamp: datetime
    speed_mph: float
    free_flow_speed_mph: float
    congestion_level: str = "free"   # free, moderate, heavy, stop-and-go


class ATSPMEvent(BaseModel):
    """High-resolution controller event log entry."""
    device_id: str
    timestamp: datetime
    event_code: int
    parameter: int


class ATSPMMetrics(BaseModel):
    """Derived ATSPM performance metrics."""
    intersection_id: str
    period_start: datetime
    period_end: datetime
    arrival_on_green_pct: float
    split_failure_pct: float
    platoon_ratio: float
    purdue_coordination_score: float    # 0–100


class DetectorReading(BaseModel):
    detector_id: str
    timestamp: datetime
    volume_veh_hr: float
    occupancy_pct: float
    speed_mph: Optional[float] = None
    operational: bool = True
