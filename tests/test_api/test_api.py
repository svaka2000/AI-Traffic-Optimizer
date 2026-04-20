"""Tests for AITO FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from datetime import datetime, timedelta
from aito.api.app import app, _corridors, _optimization_jobs, _timing_plans
from aito.data.san_diego_inventory import get_corridor


@pytest.fixture(autouse=True)
def reset_state():
    """Clear in-memory stores between tests."""
    _corridors.clear()
    _optimization_jobs.clear()
    _timing_plans.clear()
    yield
    _corridors.clear()
    _optimization_jobs.clear()
    _timing_plans.clear()


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def corridor():
    return get_corridor("downtown")


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_create_and_get_corridor(client, corridor):
    resp = client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                       headers={"Content-Type": "application/json"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == corridor.name

    get_resp = client.get(f"/api/v1/corridors/{corridor.id}")
    assert get_resp.status_code == 200
    assert get_resp.json()["id"] == corridor.id


def test_list_corridors(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    resp = client.list("/api/v1/corridors") if hasattr(client, "list") else client.get("/api/v1/corridors")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_get_nonexistent_corridor(client):
    resp = client.get("/api/v1/corridors/nonexistent-id")
    assert resp.status_code == 404


def test_optimize_corridor(client, corridor):
    # Register corridor
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})

    # Build demand profiles
    from aito.models import DemandProfile
    demands = [
        DemandProfile(
            intersection_id=ix.id,
            north_thru=500, south_thru=400, east_thru=300, west_thru=250,
        ).model_dump()
        for ix in corridor.intersections
    ]

    payload = {
        "demand_profiles": demands,
        "min_cycle": 70.0,
        "max_cycle": 150.0,  # downtown has 50ft crossings requiring ~126s minimum cycle
    }
    resp = client.post(f"/api/v1/corridors/{corridor.id}/optimize", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "pareto_solutions" in data
    assert "recommended_solution" in data
    assert len(data["pareto_solutions"]) >= 1


def test_optimize_wrong_demand_count(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    from aito.models import DemandProfile
    demands = [DemandProfile(intersection_id="x").model_dump()]  # wrong count
    resp = client.post(f"/api/v1/corridors/{corridor.id}/optimize", json={"demand_profiles": demands})
    assert resp.status_code == 422


def test_before_after_endpoint(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    now = datetime.utcnow()
    def make_metrics(delay):
        return {
            "period_start": now.isoformat(),
            "period_end": (now + timedelta(hours=2)).isoformat(),
            "avg_delay_sec": delay,
            "avg_travel_time_sec": delay * 10,
            "arrival_on_green_pct": 50.0,
            "split_failure_pct": 10.0,
            "stops_per_veh": 2.0,
            "co2_kg_hr": 50.0,
            "throughput_veh_hr": 1500.0,
        }
    payload = {"before": make_metrics(52.0), "after": make_metrics(38.0), "daily_vehicles": 20000}
    resp = client.post(f"/api/v1/corridors/{corridor.id}/analytics/before-after", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["delay_improvement_pct"] > 0


def test_roi_endpoint(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    payload = {
        "delay_reduction_s_veh": 14.0,
        "stops_reduction_pct": 50.0,
        "co2_reduction_pct": 23.0,
        "daily_vehicles": 20000,
        "aito_annual_cost_usd": 24000.0,
    }
    resp = client.post(f"/api/v1/corridors/{corridor.id}/analytics/roi", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["benefit_cost_ratio"] > 1.0


def test_synchro_export_no_plans(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    resp = client.get(f"/api/v1/corridors/{corridor.id}/export/synchro")
    assert resp.status_code == 404  # No plans yet


def test_delete_corridor(client, corridor):
    client.post("/api/v1/corridors", content=corridor.model_dump_json(),
                headers={"Content-Type": "application/json"})
    resp = client.delete(f"/api/v1/corridors/{corridor.id}")
    assert resp.status_code == 200
    get_resp = client.get(f"/api/v1/corridors/{corridor.id}")
    assert get_resp.status_code == 404
