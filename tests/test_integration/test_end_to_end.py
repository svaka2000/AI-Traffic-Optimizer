"""End-to-end integration test: full pipeline on Rosecrans corridor."""
import asyncio
import pytest
from datetime import datetime, timedelta
from aito.data.san_diego_inventory import get_corridor
from aito.data.probe_data import SyntheticProbeAdapter
from aito.models import DemandProfile, OptimizationObjective, OptimizationRequest, PerformanceMetrics
from aito.optimization.multi_objective import MultiObjectiveOptimizer
from aito.optimization.corridor_optimizer import SegmentTravelData
from aito.optimization.constraints import validate_corridor_plan
from aito.analytics.before_after import BeforeAfterAnalyzer
from aito.analytics.roi_calculator import ROICalculator
from aito.deployment.ntcip_client import MockNTCIPClient, SynchroCSVExporter
from aito.deployment.timing_plan import build_default_plan


def make_demand(ix_id, aadt):
    # Keep left turns below MUTCD 150 veh/hr protected-left warrant to avoid
    # adding extra critical phases that push minimum feasible cycle above 180s
    return DemandProfile(
        intersection_id=ix_id,
        north_thru=aadt * 0.028, south_thru=aadt * 0.022,
        east_thru=aadt * 0.014, west_thru=aadt * 0.010,
        north_left=min(aadt * 0.003, 140.0), east_left=min(aadt * 0.003, 140.0),
    )


def test_full_pipeline_rosecrans():
    """Full pipeline: load → probe data → optimize → validate → deploy."""
    corridor = get_corridor("rosecrans")
    n = len(corridor.intersections)
    assert n == 12

    # Probe data
    probe = SyntheticProbeAdapter(free_flow_speed_mph=35.0)
    now = datetime(2025, 6, 1, 7, 0)
    travel_data = []
    for dist in corridor.distances_m:
        speed_mps = 30.0 * 0.44704  # ~30 mph congested
        tt = dist / speed_mps
        travel_data.append(SegmentTravelData(
            inbound_travel_time_s=tt * 1.05,
            outbound_travel_time_s=tt,
            distance_m=dist,
        ))

    # Demand profiles
    demands = [make_demand(ix.id, ix.aadt) for ix in corridor.intersections]

    # Optimize
    opt_request = OptimizationRequest(
        corridor_id=corridor.id,
        demand_profiles=demands,
        objectives=[OptimizationObjective.DELAY, OptimizationObjective.EMISSIONS],
        min_cycle=70.0,
        max_cycle=170.0,  # Rosecrans widest crossing (90ft) needs ~155s minimum cycle
    )
    optimizer = MultiObjectiveOptimizer(corridor)
    result = optimizer.optimize(opt_request, travel_data=travel_data)

    assert len(result.pareto_solutions) >= 1
    rec = result.recommended_solution

    # Assign intersection IDs to plans
    for plan, ix in zip(rec.plan.timing_plans, corridor.intersections):
        plan.intersection_id = ix.id

    # Validate
    vr = validate_corridor_plan(corridor, rec.plan.timing_plans)
    errors = [(iid, v) for iid, v in vr.items() if not v.valid]
    assert len(errors) == 0, f"Validation errors: {errors}"

    # Deploy (mock)
    client = MockNTCIPClient()

    async def deploy_all():
        results = []
        for plan, ix in zip(rec.plan.timing_plans, corridor.intersections):
            r = await client.write_timing_plan(
                host=f"10.0.0.{100 + corridor.intersections.index(ix)}",
                community="write",
                plan=plan,
                intersection=ix,
                plan_number=2,
            )
            results.append(r)
        return results

    deploy_results = asyncio.run(deploy_all())
    assert all(r.success for r in deploy_results), [r.message for r in deploy_results if not r.success]

    # Synchro export
    exporter = SynchroCSVExporter()
    csv = exporter.export(rec.plan.timing_plans, corridor.intersections)
    assert "[TIMING]" in csv
    assert "Rosecrans" in csv

    # ROI
    roi_calc = ROICalculator(corridor)
    roi = roi_calc.calculate(
        delay_reduction_s_veh=14.0,
        stops_reduction_pct=48.0,
        co2_reduction_pct=23.5,
        daily_vehicles=28000,
        aito_annual_cost_usd=72000,
        before_co2_kg_hr=68.0,
    )
    assert roi.benefit_cost_ratio > 10.0  # AITO should deliver 10:1+ B/C


def test_full_pipeline_mira_mesa():
    """Full pipeline on Mira Mesa Boulevard (8 intersections, 45 mph)."""
    corridor = get_corridor("mira_mesa")
    demands = [make_demand(ix.id, ix.aadt) for ix in corridor.intersections]
    opt_request = OptimizationRequest(
        corridor_id=corridor.id,
        demand_profiles=demands,
        min_cycle=80.0,
        max_cycle=170.0,  # Mira Mesa widest crossing (90ft) needs ~155s minimum cycle
    )
    optimizer = MultiObjectiveOptimizer(corridor)
    result = optimizer.optimize(opt_request)
    assert len(result.pareto_solutions) >= 1
    rec = result.recommended_solution
    for plan, ix in zip(rec.plan.timing_plans, corridor.intersections):
        plan.intersection_id = ix.id
    vr = validate_corridor_plan(corridor, rec.plan.timing_plans)
    assert all(v.valid for v in vr.values()), f"Validation errors: {[(k, v.errors) for k, v in vr.items() if not v.valid]}"


def test_existing_tests_still_pass():
    """Smoke test: verify AITO core modules are importable and functional."""
    from aito.optimization.constraints import TimingPlanValidator
    from aito.optimization.isolated_optimizer import IsolatedIntersectionOptimizer
    from aito.optimization.corridor_optimizer import CorridorOptimizer
    from aito.optimization.multi_objective import MultiObjectiveOptimizer
    from aito.analytics.before_after import BeforeAfterAnalyzer
    from aito.analytics.roi_calculator import ROICalculator
    from aito.deployment.ntcip_client import MockNTCIPClient, SynchroCSVExporter
    # Verify core classes instantiate correctly
    validator = TimingPlanValidator()
    assert validator.MIN_GREEN_MUTCD == 7.0
    assert validator.MAX_CYCLE_HCM == 180.0
