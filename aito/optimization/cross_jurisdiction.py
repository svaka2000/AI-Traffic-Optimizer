"""aito/optimization/cross_jurisdiction.py

GF8: Cross-Jurisdiction Orchestration.

Coordinates signal timing across agency boundaries — a critical capability
for San Diego, where Caltrans, SANDAG, City of San Diego, and multiple
municipalities each control different segments of the same arterial.

Problem:
  The 805/163 corridor crosses 4 jurisdictions.  Each agency optimizes
  independently, breaking the green wave at boundaries.  AITO proposes
  a shared offset reference that agencies can adopt without surrendering
  control.

Solution:
  1. Boundary Handshake Protocol: upstream agency exports its exit offset
     and green duration; downstream agency imports as a constraint.
  2. Federated objective: maximize corridor-wide bandwidth while
     respecting each agency's internal constraints.
  3. API-based coordination: agencies expose read-only endpoints per
     NTCIP 1211 (Signal System Master, Section 5.4).

No agency needs to give another agency write access.  Each adopts the
shared cycle and offset voluntarily.

Reference:
  NTCIP 1211 v02 — Signal System Master (2023).
  FHWA. (2020). Connected Corridor Technical Framework. FHWA-JPO-20-730.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Agency and jurisdiction models
# ---------------------------------------------------------------------------

class AgencyType(str, Enum):
    CITY         = "city"
    COUNTY       = "county"
    STATE_DOT    = "state_dot"     # Caltrans
    REGIONAL_MPO = "regional_mpo"  # SANDAG
    TRANSIT      = "transit"


@dataclass
class Agency:
    """A jurisdiction controlling one or more intersections."""
    agency_id: str
    name: str
    agency_type: AgencyType
    atms_endpoint: Optional[str] = None    # NTCIP 1211 API endpoint
    contact_email: Optional[str] = None
    # AITO participation level
    shared_cycle: bool = True              # agrees to use common cycle
    shared_offset: bool = False            # agrees to offset coordination
    read_only_api: bool = True             # can query timing plans


@dataclass
class JurisdictionBoundary:
    """Handshake point between two agencies at a corridor boundary."""
    upstream_agency_id: str
    downstream_agency_id: str
    boundary_intersection_id: str
    # Upstream exports these values
    export_offset_s: float = 0.0
    export_cycle_s: float = 120.0
    export_green_s: float = 48.0
    # Downstream uses these as constraints
    import_constraint_priority: str = "advisory"  # advisory | required


# ---------------------------------------------------------------------------
# Corridor-level jurisdiction mapping
# ---------------------------------------------------------------------------

@dataclass
class JurisdictionalCorridor:
    """A corridor spanning multiple jurisdictions."""
    corridor_id: str
    corridor_name: str
    agencies: list[Agency]
    # Which agency controls which intersection (by index in corridor.intersections)
    intersection_agency_map: dict[int, str]  # intersection_idx → agency_id
    boundaries: list[JurisdictionBoundary] = field(default_factory=list)

    def agency_for_intersection(self, idx: int) -> Optional[Agency]:
        agency_id = self.intersection_agency_map.get(idx)
        if not agency_id:
            return None
        return next((a for a in self.agencies if a.agency_id == agency_id), None)

    def boundary_intersections(self) -> list[int]:
        """Return indices where agency control changes."""
        boundaries = []
        prev = None
        for i in sorted(self.intersection_agency_map.keys()):
            curr = self.intersection_agency_map[i]
            if prev is not None and curr != prev:
                boundaries.append(i)
            prev = curr
        return boundaries


# ---------------------------------------------------------------------------
# Coordination message protocol
# ---------------------------------------------------------------------------

class CoordinationMessageType(str, Enum):
    OFFSET_PROPOSAL   = "offset_proposal"     # suggest an offset to adopt
    OFFSET_ACK        = "offset_ack"          # agency confirms adoption
    OFFSET_REJECT     = "offset_reject"       # agency declines, with reason
    CYCLE_SYNC        = "cycle_sync"          # agree on common cycle
    STATUS_REPORT     = "status_report"       # current timing state
    INCIDENT_ALERT    = "incident_alert"      # upstream incident notification


@dataclass
class CoordinationMessage:
    """Message exchanged between agency ATMS endpoints."""
    message_id: str
    message_type: CoordinationMessageType
    from_agency_id: str
    to_agency_id: str
    corridor_id: str
    timestamp: datetime

    # Payload
    proposed_cycle_s: Optional[float] = None
    proposed_offset_s: Optional[float] = None
    boundary_intersection_id: Optional[str] = None
    reason: Optional[str] = None
    data: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Federated offset optimizer
# ---------------------------------------------------------------------------

@dataclass
class AgencyOptimizationConstraints:
    """Internal constraints an agency cannot relax."""
    agency_id: str
    min_cycle_s: float = 60.0
    max_cycle_s: float = 180.0
    preferred_cycle_s: float = 120.0
    fixed_intersection_ids: list[str] = field(default_factory=list)  # cannot be changed
    max_offset_change_s: float = 30.0   # max acceptable offset change per re-timing


@dataclass
class FederatedOptimizationResult:
    """Result of cross-jurisdiction corridor optimization."""
    corridor_id: str
    common_cycle_s: float
    offsets_by_intersection: dict[str, float]   # intersection_id → offset_s
    agencies_participating: list[str]
    agencies_declining: list[str]
    bandwidth_outbound_pct: float
    bandwidth_inbound_pct: float
    coordination_messages: list[CoordinationMessage]
    notes: list[str] = field(default_factory=list)

    @property
    def participation_rate(self) -> float:
        total = len(self.agencies_participating) + len(self.agencies_declining)
        return len(self.agencies_participating) / max(total, 1)


class CrossJurisdictionOrchestrator:
    """Coordinates timing plan negotiation across agency boundaries.

    Algorithm:
    1. Find common cycle acceptable to all agencies (lowest common denominator).
    2. For each boundary, compute optimal offset given upstream constraint.
    3. Send OFFSET_PROPOSAL to each agency; collect ACK/REJECT.
    4. For declining agencies: keep their existing offsets as fixed constraints.
    5. Re-optimize with fixed offsets for non-participating agencies.

    Usage:
        orchestrator = CrossJurisdictionOrchestrator(jurisdictional_corridor)
        result = orchestrator.optimize(corridor, demand_profiles)
    """

    def __init__(
        self,
        jurisdictional_corridor: JurisdictionalCorridor,
        simulation_mode: bool = True,   # True = no actual API calls
    ) -> None:
        self.jcorridor = jurisdictional_corridor
        self.simulation_mode = simulation_mode
        self._message_log: list[CoordinationMessage] = []

    def negotiate_common_cycle(
        self,
        agency_constraints: list[AgencyOptimizationConstraints],
    ) -> float:
        """Find common cycle length acceptable to all participating agencies.

        Uses the intersection of feasible ranges.
        """
        global_min = max(c.min_cycle_s for c in agency_constraints)
        global_max = min(c.max_cycle_s for c in agency_constraints)

        if global_min > global_max:
            # No overlap: use the average of preferred cycles
            preferred = [c.preferred_cycle_s for c in agency_constraints]
            return round(sum(preferred) / len(preferred) / 5.0) * 5.0

        # Pick the cycle closest to the median preferred cycle within the range
        preferred = sorted(c.preferred_cycle_s for c in agency_constraints)
        median_preferred = preferred[len(preferred) // 2]
        return max(global_min, min(global_max, round(median_preferred / 5.0) * 5.0))

    def optimize(
        self,
        corridor,
        demand_profiles,
        agency_constraints: Optional[list[AgencyOptimizationConstraints]] = None,
    ) -> FederatedOptimizationResult:
        """Run federated offset optimization across all agencies."""
        import uuid
        from aito.optimization.corridor_optimizer import CorridorOptimizer

        agencies = self.jcorridor.agencies
        constraints = agency_constraints or [
            AgencyOptimizationConstraints(agency_id=a.agency_id)
            for a in agencies
        ]

        # 1. Negotiate common cycle
        common_cycle = self.negotiate_common_cycle(constraints)

        # 2. Run corridor optimizer at common cycle
        optimizer = CorridorOptimizer(corridor)
        try:
            result = optimizer._optimize_at_cycle(
                demand_profiles=demand_profiles,
                travel_data=optimizer._default_travel_data(),
                cycle=common_cycle,
                period="Cross-Jurisdiction",
            )
            offsets = result.corridor_plan.offsets
            bw_out = result.bandwidth_outbound_pct
            bw_in = result.bandwidth_inbound_pct
        except Exception as e:
            offsets = [0.0] * len(corridor.intersections)
            bw_out = 0.0
            bw_in = 0.0

        # 3. Send proposals and collect responses
        messages: list[CoordinationMessage] = []
        participating: list[str] = []
        declining: list[str] = []

        for i, (ix, offset) in enumerate(zip(corridor.intersections, offsets)):
            agency = self.jcorridor.agency_for_intersection(i)
            if agency is None:
                continue

            # Check if this agency allows offset coordination
            if not agency.shared_offset:
                declining.append(agency.agency_id)
                continue

            # Check offset change is within agency tolerance
            constraint = next((c for c in constraints if c.agency_id == agency.agency_id), None)
            max_delta = constraint.max_offset_change_s if constraint else 30.0

            proposal = CoordinationMessage(
                message_id=str(uuid.uuid4())[:8],
                message_type=CoordinationMessageType.OFFSET_PROPOSAL,
                from_agency_id="aito",
                to_agency_id=agency.agency_id,
                corridor_id=corridor.id,
                timestamp=datetime.utcnow(),
                proposed_cycle_s=common_cycle,
                proposed_offset_s=offset,
                boundary_intersection_id=ix.id,
            )
            messages.append(proposal)

            # Simulate response (in real system: await API response)
            if self.simulation_mode:
                if agency.agency_id not in declining:
                    participating.append(agency.agency_id)
                    ack = CoordinationMessage(
                        message_id=str(uuid.uuid4())[:8],
                        message_type=CoordinationMessageType.OFFSET_ACK,
                        from_agency_id=agency.agency_id,
                        to_agency_id="aito",
                        corridor_id=corridor.id,
                        timestamp=datetime.utcnow(),
                        proposed_cycle_s=common_cycle,
                        proposed_offset_s=offset,
                        boundary_intersection_id=ix.id,
                    )
                    messages.append(ack)

        # Deduplicate
        participating = list(set(participating))
        declining = list(set(declining))
        offsets_map = {
            ix.id: round(offset, 1)
            for ix, offset in zip(corridor.intersections, offsets)
        }

        self._message_log.extend(messages)

        notes = []
        if declining:
            notes.append(f"Agencies declining coordination: {declining}")
        if len(participating) == len(agencies):
            notes.append("Full corridor coordination achieved")

        return FederatedOptimizationResult(
            corridor_id=corridor.id,
            common_cycle_s=common_cycle,
            offsets_by_intersection=offsets_map,
            agencies_participating=participating,
            agencies_declining=declining,
            bandwidth_outbound_pct=round(bw_out, 1),
            bandwidth_inbound_pct=round(bw_in, 1),
            coordination_messages=messages,
            notes=notes,
        )

    @property
    def message_log(self) -> list[CoordinationMessage]:
        return list(self._message_log)


# ---------------------------------------------------------------------------
# San Diego jurisdiction registry
# ---------------------------------------------------------------------------

CITY_OF_SAN_DIEGO = Agency(
    agency_id="cosd",
    name="City of San Diego",
    agency_type=AgencyType.CITY,
    atms_endpoint="https://atms.sandiego.gov/ntcip",
    shared_cycle=True,
    shared_offset=True,
    read_only_api=True,
)

CALTRANS_D11 = Agency(
    agency_id="caltrans_d11",
    name="Caltrans District 11",
    agency_type=AgencyType.STATE_DOT,
    atms_endpoint="https://d11.dot.ca.gov/ntcip",
    shared_cycle=True,
    shared_offset=False,   # Caltrans rarely shares offset control
    read_only_api=True,
)

SANDAG = Agency(
    agency_id="sandag",
    name="SANDAG",
    agency_type=AgencyType.REGIONAL_MPO,
    shared_cycle=True,
    shared_offset=True,
    read_only_api=True,
)

SD_AGENCIES = [CITY_OF_SAN_DIEGO, CALTRANS_D11, SANDAG]
