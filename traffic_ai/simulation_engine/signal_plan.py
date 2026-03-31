"""traffic_ai/simulation_engine/signal_plan.py

Signal timing data structures per MUTCD and HCM 7th edition.

MovementType maps AITO's 4-phase simplified model onto the NEMA TS-2
dual-ring 8-phase structure used by all U.S. signal controllers.

NEMA Phase Mapping (simplified to 4 actions for AITO)
------------------------------------------------------
Action 0 → NS_THROUGH  (NEMA phases 2+6, the "coordinated phases")
Action 1 → EW_THROUGH  (NEMA phases 6+2, cross-street through)
Action 2 → NS_LEFT     (NEMA phases 1+5, protected NS left turns)
Action 3 → EW_LEFT     (NEMA phases 5+1, protected EW left turns)

PhaseConstraints encapsulates HCM/MUTCD timing bounds so controllers and
the engine share a single source of truth for clearance intervals, minimum
greens, and pedestrian requirements.

DetailedSignalState extends IntersectionState with per-intersection
real-time signal tracking: pedestrian calls, LPI enforcement, preemption.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class MovementType(Enum):
    """AITO 4-phase movement types (simplified NEMA dual-ring mapping)."""
    NS_THROUGH = "ns_through"   # NEMA phases 2+6 — major arterial through
    NS_LEFT    = "ns_left"      # NEMA phases 1+5 — protected NS left turns
    EW_THROUGH = "ew_through"   # NEMA phases 6+2 — cross-street through
    EW_LEFT    = "ew_left"      # NEMA phases 5+1 — protected EW left turns
    NS_PED     = "ns_ped"       # Pedestrian crossing, NS direction
    EW_PED     = "ew_ped"       # Pedestrian crossing, EW direction


@dataclass(slots=True)
class PhaseConstraints:
    """Signal timing constraints per MUTCD Table 4D-2 and HCM 7th edition §19.6.

    Attributes
    ----------
    min_green_sec : float
        Minimum green interval in seconds before phase change is allowed.
        MUTCD Table 4D-2 minimum = 7.0 s.
    max_green_sec : float
        Maximum green interval in seconds (typical urban arterial = 60 s).
    yellow_sec : float
        Yellow change interval per ITE clearance standard = 4.0 s.
        (HCM 7th ed. uses 3–5 s depending on approach speed.)
    all_red_sec : float
        All-red clearance interval after yellow = 2.0 s (HCM 7th ed. §19.6).
    ped_walk_sec : float
        Walk signal minimum per MUTCD §4E.06 = 7.0 s.
    ped_clearance_sec : float
        Flashing don't-walk clearance interval per MUTCD §4E.08 = 15.0 s.
    min_cycle_sec : float
        Webster minimum cycle length = 60 s.
    max_cycle_sec : float
        Webster maximum cycle length = 180 s.
    """
    min_green_sec: float    = 7.0
    max_green_sec: float    = 60.0
    yellow_sec: float       = 4.0
    all_red_sec: float      = 2.0
    ped_walk_sec: float     = 7.0
    ped_clearance_sec: float = 15.0
    min_cycle_sec: float    = 60.0
    max_cycle_sec: float    = 180.0

    @property
    def clearance_sec(self) -> float:
        """Total clearance interval = yellow + all-red (seconds)."""
        return self.yellow_sec + self.all_red_sec

    @property
    def lost_time_per_phase(self) -> float:
        """Lost time per phase change (yellow + all-red, seconds)."""
        return self.clearance_sec


@dataclass(slots=True)
class DetailedSignalState:
    """Real-time signal state at one intersection.

    Tracks pedestrian calls, LPI (Leading Pedestrian Interval), and
    preemption state separately from IntersectionState so the engine
    can enforce ADA pedestrian requirements and emergency preemption
    without tangling those concerns with queue dynamics.
    """
    current_movement: MovementType      = MovementType.NS_THROUGH
    steps_in_phase: int                 = 0
    clearance_steps_remaining: int      = 0
    is_in_clearance: bool               = False

    # Pedestrian demand tracking (stochastic calls each step)
    ped_call_ns: bool                   = False   # button pressed, NS crossing
    ped_call_ew: bool                   = False   # button pressed, EW crossing
    ped_wait_steps_ns: int              = 0       # steps since last NS ped service
    ped_wait_steps_ew: int              = 0       # steps since last EW ped service

    # Emergency / bus preemption
    preempt_active: bool                = False
    preempt_steps_remaining: int        = 0
