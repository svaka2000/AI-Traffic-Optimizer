"""Tests for aito/optimization/cross_jurisdiction.py (GF8)."""
import pytest
from aito.optimization.cross_jurisdiction import (
    AgencyType,
    Agency,
    JurisdictionBoundary,
    JurisdictionalCorridor,
    CoordinationMessageType,
    CoordinationMessage,
    AgencyOptimizationConstraints,
    FederatedOptimizationResult,
    CrossJurisdictionOrchestrator,
    SD_AGENCIES,
    CITY_OF_SAN_DIEGO,
    CALTRANS_D11,
    SANDAG,
)
from aito.data.san_diego_inventory import get_corridor
from aito.models import DemandProfile

ROSECRANS = get_corridor("rosecrans")
N_IX = len(ROSECRANS.intersections)


def _make_jcorridor() -> JurisdictionalCorridor:
    return JurisdictionalCorridor(
        corridor_id=ROSECRANS.id,
        corridor_name=ROSECRANS.name,
        agencies=SD_AGENCIES,
        intersection_agency_map={
            i: ("cosd" if i < 8 else "caltrans_d11")
            for i in range(N_IX)
        },
    )


def _demand_profiles():
    return [
        DemandProfile(
            intersection_id=ix.id,
            period_minutes=60,
            north_thru=400.0, south_thru=350.0,
            east_thru=180.0, west_thru=160.0,
            north_left=70.0, south_left=60.0,
            east_left=50.0, west_left=40.0,
            north_right=50.0, south_right=45.0,
            east_right=35.0, west_right=30.0,
        )
        for ix in ROSECRANS.intersections
    ]


class TestAgency:
    def test_sd_agencies_list(self):
        assert len(SD_AGENCIES) >= 2

    def test_cosd_exists(self):
        assert CITY_OF_SAN_DIEGO is not None
        assert CITY_OF_SAN_DIEGO.agency_id == "cosd"

    def test_caltrans_d11_exists(self):
        assert CALTRANS_D11 is not None
        assert "caltrans" in CALTRANS_D11.agency_id.lower()

    def test_sandag_exists(self):
        assert SANDAG is not None

    def test_agency_has_required_fields(self):
        agency = CITY_OF_SAN_DIEGO
        assert hasattr(agency, "agency_id")
        assert hasattr(agency, "name")
        assert hasattr(agency, "agency_type")


class TestAgencyType:
    def test_has_multiple_types(self):
        assert len(list(AgencyType)) >= 2


class TestJurisdictionalCorridor:
    def test_instantiation(self):
        jc = _make_jcorridor()
        assert jc is not None

    def test_agency_count(self):
        jc = _make_jcorridor()
        assert len(jc.agencies) >= 2

    def test_intersection_agency_map(self):
        jc = _make_jcorridor()
        assert len(jc.intersection_agency_map) == N_IX

    def test_corridor_id_matches(self):
        jc = _make_jcorridor()
        assert jc.corridor_id == ROSECRANS.id


class TestAgencyOptimizationConstraints:
    def test_instantiation_with_agency_id(self):
        c = AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0)
        assert c.agency_id == "cosd"
        assert c.preferred_cycle_s == 110.0

    def test_default_cycle_bounds(self):
        c = AgencyOptimizationConstraints("cosd")
        assert c.min_cycle_s < c.max_cycle_s

    def test_preferred_within_bounds(self):
        c = AgencyOptimizationConstraints("cosd", preferred_cycle_s=120.0)
        assert c.min_cycle_s <= c.preferred_cycle_s <= c.max_cycle_s


class TestCrossJurisdictionOrchestrator:
    def setup_method(self):
        self.jcorridor = _make_jcorridor()
        self.orch = CrossJurisdictionOrchestrator(self.jcorridor, simulation_mode=True)
        self.demand = _demand_profiles()

    def test_instantiation(self):
        assert self.orch is not None

    def test_negotiate_common_cycle_returns_float(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        ]
        cycle = self.orch.negotiate_common_cycle(constraints)
        assert isinstance(cycle, float)
        assert cycle > 0

    def test_negotiated_cycle_between_min_max(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0,
                                           min_cycle_s=90.0, max_cycle_s=150.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0,
                                           min_cycle_s=100.0, max_cycle_s=160.0),
        ]
        cycle = self.orch.negotiate_common_cycle(constraints)
        assert 90 <= cycle <= 160

    def test_consensus_of_same_preference(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=120.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=120.0),
        ]
        cycle = self.orch.negotiate_common_cycle(constraints)
        assert cycle == pytest.approx(120.0, rel=0.01)

    def test_optimize_returns_result(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        ]
        result = self.orch.optimize(ROSECRANS, self.demand, agency_constraints=constraints)
        assert isinstance(result, FederatedOptimizationResult)

    def test_result_has_offsets(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        ]
        result = self.orch.optimize(ROSECRANS, self.demand, agency_constraints=constraints)
        assert isinstance(result.offsets_by_intersection, dict)

    def test_result_has_coordination_messages(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        ]
        result = self.orch.optimize(ROSECRANS, self.demand, agency_constraints=constraints)
        assert isinstance(result.coordination_messages, list)

    def test_result_participation_rate(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=110.0),
            AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        ]
        result = self.orch.optimize(ROSECRANS, self.demand, agency_constraints=constraints)
        assert 0 <= result.participation_rate <= 1.0

    def test_bandwidth_fields_in_result(self):
        constraints = [
            AgencyOptimizationConstraints("cosd", preferred_cycle_s=120.0),
        ]
        result = self.orch.optimize(ROSECRANS, self.demand, agency_constraints=constraints)
        assert hasattr(result, "bandwidth_outbound_pct")
        assert hasattr(result, "bandwidth_inbound_pct")
