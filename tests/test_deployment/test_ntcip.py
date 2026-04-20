"""Tests for NTCIP deployment client."""
import asyncio
import pytest
from aito.data.san_diego_inventory import get_corridor
from aito.deployment.ntcip_client import MockNTCIPClient, SynchroCSVExporter
from aito.deployment.timing_plan import build_default_plan


@pytest.fixture
def corridor():
    return get_corridor("downtown")


@pytest.fixture
def plans_and_intersections(corridor):
    # build_default_plan auto-calculates minimum cycle for ped requirements
    plans = [build_default_plan(ix) for ix in corridor.intersections]
    for p, ix in zip(plans, corridor.intersections):
        p.intersection_id = ix.id
    return plans, corridor.intersections


def test_mock_client_write_valid_plan(plans_and_intersections):
    plans, intersections = plans_and_intersections
    client = MockNTCIPClient()

    async def run():
        return await client.write_timing_plan(
            host="192.168.1.101",
            community="write",
            plan=plans[0],
            intersection=intersections[0],
            plan_number=2,
        )

    result = asyncio.run(run())
    assert result.success


def test_mock_client_activate_plan(plans_and_intersections):
    plans, intersections = plans_and_intersections
    client = MockNTCIPClient()

    async def run():
        await client.write_timing_plan(
            host="192.168.1.101",
            community="write",
            plan=plans[0],
            intersection=intersections[0],
            plan_number=2,
        )
        return await client.activate_plan("192.168.1.101", "write", 2)

    result = asyncio.run(run())
    assert result.success


def test_mock_client_activate_missing_plan():
    client = MockNTCIPClient()

    async def run():
        return await client.activate_plan("192.168.1.101", "write", 99)

    result = asyncio.run(run())
    assert not result.success


def test_invalid_plan_rejected(corridor):
    from aito.models import PhaseTiming, TimingPlan
    ix = corridor.intersections[0]
    bad_plan = TimingPlan(
        intersection_id=ix.id,
        cycle_length=120,
        phases=[
            PhaseTiming(phase_id=2, min_green=7, max_green=40, split=3.0,  # too short
                        yellow=4, all_red=2, ped_walk=7, ped_clearance=18),
        ],
    )
    client = MockNTCIPClient()

    async def run():
        return await client.write_timing_plan("192.168.1.101", "write", bad_plan, ix)

    result = asyncio.run(run())
    assert not result.success
    assert "Validation" in result.message


def test_synchro_csv_export(plans_and_intersections):
    plans, intersections = plans_and_intersections
    exporter = SynchroCSVExporter()
    csv = exporter.export(plans, intersections)
    assert "[TIMING]" in csv
    assert "INTID" in csv
    assert len(csv.strip().split("\n")) > 2  # Header + data rows


def test_synchro_csv_contains_all_intersections(plans_and_intersections):
    plans, intersections = plans_and_intersections
    exporter = SynchroCSVExporter()
    csv = exporter.export(plans, intersections)
    for ix in intersections:
        assert ix.name in csv


def test_build_default_plan_valid(corridor):
    from aito.optimization.constraints import TimingPlanValidator
    validator = TimingPlanValidator()
    for ix in corridor.intersections:
        plan = build_default_plan(ix, cycle=120.0)
        plan.intersection_id = ix.id
        result = validator.validate(plan, ix)
        assert result.valid, f"Default plan invalid for {ix.name}: {result.errors}"
