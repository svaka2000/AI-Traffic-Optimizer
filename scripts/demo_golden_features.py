#!/usr/bin/env python3
"""scripts/demo_golden_features.py

AITO — All 15 Golden Features showcase demo.

Runs a representative demonstration of every Golden Feature in sequence,
using the Rosecrans Street and Mira Mesa corridors in San Diego.

Target runtime: < 90 seconds.
"""
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def hdr(n: int, title: str):
    print(f"\n{'─'*60}")
    print(f"  GF{n:02d}: {title}")
    print('─'*60)


def ok(msg: str):
    print(f"  ✓  {msg}")


def info(msg: str):
    print(f"       {msg}")


def main():
    t0 = time.time()
    print("\n" + "="*60)
    print("  AITO — 15 Golden Features Demo")
    print("  San Diego, California")
    print("="*60)

    # Pre-load corridors once
    from aito.data.san_diego_inventory import get_corridor
    rosecrans = get_corridor("rosecrans")
    mira_mesa = get_corridor("mira_mesa")
    now = datetime.now()

    # =========================================================================
    # GF1: Probe Data Fusion
    # =========================================================================
    hdr(1, "Detection-Free Probe Data Fusion")
    from aito.data.probe_fusion import (
        ProbeFusionEngine, SourceObservation, fuse_all_tod_periods, SOURCE_WEIGHTS,
    )

    n_segs = len(rosecrans.distances_m)
    engine = ProbeFusionEngine(
        corridor_id=rosecrans.id,
        segment_distances_m=rosecrans.distances_m,
        free_flow_speeds_mph=[35.0] * n_segs,
    )
    # Inject synthetic observations from 4 sources for segment 1
    seg_id = f"{rosecrans.id}_seg1"
    sources = [
        ("connected_vehicle", 22.0, 65.5, 0.95),
        ("inrix",             19.5, 73.8, 0.88),
        ("here",              21.0, 68.6, 0.85),
        ("waze",              18.0, 80.0, 0.72),
    ]
    obs_list = [
        SourceObservation(source=src, segment_id=seg_id, timestamp=now,
                          travel_time_s=tt, speed_mph=spd, confidence=conf)
        for src, spd, tt, conf in sources
    ]
    engine.ingest(obs_list)
    corridor_est = engine.fuse(now)
    first_seg = next((s for s in corridor_est.segments if s.segment_id == seg_id), corridor_est.segments[0])
    ok(f"Fused {first_seg.source_count} probe sources → {first_seg.speed_mph:.1f} mph ({first_seg.congestion_state.value})")
    ok(f"Confidence: {first_seg.fused_confidence:.3f}  Sources: {', '.join(first_seg.sources_used)}")

    tod_states = fuse_all_tod_periods(engine, now)
    ok(f"TOD profiles generated: {len(tod_states)} periods")
    for period, est in list(tod_states.items())[:3]:
        avg_spd = est.mean_speed_mph
        cong = est.worst_congestion_state.value
        info(f"{period:<12} {avg_spd:.1f} mph  [{cong}]")

    # =========================================================================
    # GF5: Turn Movement Estimation
    # =========================================================================
    hdr(5, "Turn Movement Estimation from Trajectories")
    from aito.data.turn_movement_estimator import (
        TrajectoryTurnMovementEstimator, IntersectionBBox, generate_synthetic_trajectories,
    )
    ix = rosecrans.intersections[5]
    bbox = IntersectionBBox(center_lat=ix.latitude, center_lon=ix.longitude, radius_m=80.0)
    estimator = TrajectoryTurnMovementEstimator(
        intersection_id=ix.id,
        bbox=bbox,
        penetration_rate=0.10,
    )
    trajs = generate_synthetic_trajectories(ix.latitude, ix.longitude, n_vehicles=200, seed=42)
    classified = 0
    for t in trajs:
        rec = estimator.process_trajectory(t)
        if rec:
            classified += 1

    period_start = now.replace(hour=7, minute=0, second=0, microsecond=0)
    period_end = period_start + timedelta(hours=1)
    tmc = estimator.aggregate(period_start, period_end)
    expanded = tmc.expand()
    ok(f"Classified {classified}/{len(trajs)} trajectories at {ix.name}")
    ok(f"N thru: {expanded.N_TH:.0f}  N left: {expanded.N_LT:.0f}  N right: {expanded.N_RT:.0f}  (expanded @ 10% penetration)")

    # =========================================================================
    # GF6: Multi-Objective Optimization (NSGA-III)
    # =========================================================================
    hdr(6, "Emissions-Weighted Multi-Objective Optimization (NSGA-III)")
    from aito.optimization.multi_objective_engine import NSGAIIIOptimizer
    from aito.models import DemandProfile

    # Use a single-intersection corridor slice for speed in demo
    from aito.models import Corridor as AITOCorridor
    single_ix_corridor = AITOCorridor(
        id=rosecrans.id + "_demo",
        name=rosecrans.name + " (demo)",
        intersections=rosecrans.intersections[:1],
        distances_m=[],
        speed_limits_mph=[35.0],
    )
    demand_list = [DemandProfile(
        intersection_id=rosecrans.intersections[0].id,
        period_minutes=60,
        north_thru=450.0, south_thru=380.0,
        east_thru=220.0, west_thru=190.0,
        north_left=80.0, south_left=65.0,
        east_left=55.0, west_left=45.0,
        north_right=60.0, south_right=50.0,
        east_right=40.0, west_right=35.0,
    )]
    nsga = NSGAIIIOptimizer(single_ix_corridor, population_size=24, n_generations=20, seed=42)
    nsga_result = nsga.optimize(demand_list)
    balanced = nsga_result.balanced
    ok(f"NSGA-III: {len(nsga_result.pareto_front)} Pareto solutions ({nsga_result.n_evaluations} evaluations)")
    ok(f"Balanced: cycle={balanced.cycle:.0f}s  delay={balanced.objectives[0]:.1f}s/veh  CO2={balanced.objectives[1]:.1f}kg/hr")

    # =========================================================================
    # GF7: Continuous Auto-Retiming
    # =========================================================================
    hdr(7, "Continuous Auto-Retiming (Kill the 3–5 Year Cycle)")
    from aito.optimization.continuous_retiming import RetimingMonitor, MonitorConfig, PerformanceSnapshot

    monitor = RetimingMonitor(rosecrans.id, MonitorConfig(min_retiming_interval_s=0))

    def make_snap(day: int, base_delay: float) -> PerformanceSnapshot:
        return PerformanceSnapshot(
            timestamp=now - timedelta(days=13 - day),
            corridor_id=rosecrans.id,
            avg_delay_s_veh=base_delay * (1.0 + day * 0.015),
            max_delay_s_veh=base_delay * (1.0 + day * 0.020) * 1.4,
            bandwidth_outbound_pct=45.0 - day * 0.8,
            bandwidth_inbound_pct=40.0 - day * 0.6,
            split_failure_pct=0.08 + day * 0.018,
            arrival_on_green_pct=62.0 - day * 1.1,
            probe_confidence=0.85,
            volume_veh_hr=1600.0 + day * 15,
            cycle_s=120.0,
        )

    # Set baseline on day 0
    baseline_snap = make_snap(0, 35.0)
    monitor.set_baseline(baseline_snap)

    all_triggers = []
    for day in range(1, 14):
        snap = make_snap(day, 35.0)
        triggers = monitor.evaluate(snap)
        all_triggers.extend(triggers)

    ok(f"Drift detection over 13 days: {len(all_triggers)} retiming triggers")
    for t in all_triggers[:3]:
        info(f"[{t.trigger_type}] {t.description}  severity={t.severity:.2f}")
    ok(f"Pending retiming jobs: {len(monitor.pending_jobs)}")

    # =========================================================================
    # GF12: Queue Spillback Predictor
    # =========================================================================
    hdr(12, "Queue Spillback Prediction & Prevention")
    from aito.optimization.spillback_predictor import SpillbackPredictor

    predictor = SpillbackPredictor.from_corridor(rosecrans)
    n_storages = len(predictor.storages)
    import random as _rng; _r = _rng.Random(42)
    vols = [_r.randint(350, 700) for _ in range(n_storages)]
    caps = [1800 * 0.40] * n_storages
    grs  = [0.40] * n_storages
    report = predictor.scan(
        volumes_veh_hr=vols,
        capacities_veh_hr=caps,
        cycle_s=120.0,
        green_ratios=grs,
    )
    at_risk = report.active_spillbacks + report.high_risk_approaches
    ok(f"Spillback analysis: {len(at_risk)}/{len(report.forecasts)} approaches HIGH/ACTIVE risk")
    ok(f"Worst corridor risk: {report.worst_risk.value.upper()}")
    for f in at_risk[:2]:
        info(f"{f.intersection_id} [{f.approach_direction}]: queue={f.queue_length_m:.0f}m  vc={f.v_c_ratio:.2f}  {f.risk.value.upper()}")

    # =========================================================================
    # GF2: Carbon Accounting
    # =========================================================================
    hdr(2, "Real-Time Intersection Carbon Accounting (EPA MOVES2014b)")
    from aito.analytics.carbon_accountant import CarbonAccountant

    reduction_t_yr = CarbonAccountant.estimate_quick(
        n_intersections=len(rosecrans.intersections),
        avg_aadt=rosecrans.aadt,
        avg_delay_reduction_s_veh=17.0,
    )
    ok(f"Annual CO2 reduction:  {reduction_t_yr:,.0f} tonnes/year")
    info(f"Equivalent to taking {reduction_t_yr / 4.6:.0f} passenger cars off the road permanently")
    info(f"Methodology: EPA MOVES2014b idle=1.38g/s × delay reduction × CARB EMFAC2021 fleet mix")

    # =========================================================================
    # GF9: Carbon Credit Pipeline
    # =========================================================================
    hdr(9, "Carbon Credit Pipeline (Verra VCS / Gold Standard / CARB LCFS)")
    from aito.analytics.carbon_credits import CarbonCreditPipeline

    baseline_t = reduction_t_yr / 0.235
    pipeline = CarbonCreditPipeline(corridor_name=rosecrans.name, investment_cost_usd=72000.0)
    package = pipeline.build_package(
        corridor_id=rosecrans.id,
        baseline_co2_tonnes_year=baseline_t,
        optimized_co2_tonnes_year=baseline_t - reduction_t_yr,
        benchmark_adoption_pct=0.05,
    )
    eligible = {k: v for k, v in package.projections.items() if v.eligible}
    best = max(eligible, key=lambda k: eligible[k].net_usd_year) if eligible else None
    ok(f"Creditable reduction: {package.creditable_tonnes_year:,.0f} tonnes/year (after 3% leakage)")
    ok(f"Additionality: {package.additionality.value}")
    for mkt, proj in eligible.items():
        ok(f"{mkt}: ${proj.net_usd_year:,.0f}/year")
    if best:
        info(f"Best market: {best} → ${eligible[best].net_usd_year:,.0f}/year revenue")

    # =========================================================================
    # GF4: Digital Twin
    # =========================================================================
    hdr(4, "Auto-Calibrating Digital Twin (CTM)")
    from aito.simulation.digital_twin import DigitalTwin

    n_segs_twin = len(rosecrans.distances_m)
    twin = DigitalTwin(
        corridor_id=rosecrans.id,
        segment_distances_m=rosecrans.distances_m,
        free_flow_speeds_mph=[35.0] * n_segs_twin,
    )
    # Alternate green/red for segments (simplified wave)
    green_pattern = [i % 2 == 0 for i in range(n_segs_twin)]
    sim_result = twin.simulate(
        signal_green_per_segment=green_pattern,
        demand_veh_s=0.28,
        duration_s=900.0,
    )
    n_steps_sim = len(sim_result.density_history)
    n_cells_sim = len(sim_result.density_history[0]) if sim_result.density_history else 0
    ok(f"CTM simulation: {n_cells_sim} cells × {n_steps_sim} steps")
    ok(f"Avg travel time: {sim_result.avg_travel_time_s:.1f}s  avg speed: {sim_result.avg_speed_mph:.1f} mph  CO2: {sim_result.co2_kg:.2f} kg")

    # =========================================================================
    # GF15: What-If Scenarios
    # =========================================================================
    hdr(15, "What-If Scenario Engine")
    from aito.simulation.what_if import WhatIfEngine, DemandScenario, IncidentScenario

    engine_wi = WhatIfEngine(rosecrans, digital_twin=twin)
    baseline_wi = engine_wi.run_baseline(demand_veh_s=0.28, duration_s=900.0)

    sc1 = DemandScenario(
        name="Padres Game Surge",
        description="Padres game surge: +50% demand",
        scale_factor=1.5,
        demand_veh_s=0.28,
    )
    res1 = engine_wi.run(sc1)
    comp1 = engine_wi.compare(baseline_wi, res1, sc1)
    ok(f"Demand +50%: travel time Δ={comp1.travel_time_delta_pct:+.1f}%  verdict={comp1.verdict}")

    sc2 = IncidentScenario(
        name="Midway Crash",
        description="Crash on Midway segment — 40% capacity",
        segment_idx=3,
        capacity_reduction_pct=0.60,
        demand_veh_s=0.28,
    )
    res2 = engine_wi.run(sc2)
    comp2 = engine_wi.compare(baseline_wi, res2, sc2)
    ok(f"Incident (40% capacity): travel time Δ={comp2.travel_time_delta_pct:+.1f}%  verdict={comp2.verdict}")

    # =========================================================================
    # GF3: Event-Aware Optimization
    # =========================================================================
    hdr(3, "Predictive Event-Aware Optimization")
    from aito.optimization.event_aware import EventAwareOptimizer, CalendarEvent, EventType

    game_start = now.replace(hour=19, minute=10, second=0) + timedelta(days=1)
    padres_game = CalendarEvent(
        event_id="padres_2026_042",
        name="Padres vs. Dodgers",
        venue_lat=32.7076,
        venue_lon=-117.1570,
        event_type=EventType.SPORTS_MAJOR,
        start_time=game_start,
        end_time=game_start + timedelta(hours=3, minutes=15),
        expected_attendance=41500,
        venue_capacity=45000,
    )
    event_opt = EventAwareOptimizer(rosecrans)
    event_opt.register_event(padres_game)
    result = event_opt._optimized_plans[padres_game.event_id]
    n_plans = sum(1 for p in [result.pre_event_plan, result.egress_plan] if p is not None)
    ok(f"Event: {padres_game.name} @ Petco Park")
    ok(f"Inbound peak:  {padres_game.inbound_peak_veh_hr:.0f} veh/hr  (pre-event plan activates {padres_game.pre_deploy_time.strftime('%H:%M')})")
    ok(f"Egress peak:   {padres_game.outbound_peak_veh_hr:.0f} veh/hr  (ends {padres_game.egress_end_time.strftime('%H:%M')})")
    ok(f"Generated {n_plans} timing plan phases")

    # =========================================================================
    # GF13: Multi-Modal Optimization
    # =========================================================================
    hdr(13, "Pedestrian & Cyclist Priority Optimization")
    from aito.optimization.multimodal import MultiModalOptimizer, MultiModalConstraints, CyclistFacility

    ix13 = rosecrans.intersections[3]
    constraints = MultiModalConstraints(
        accessible_clearance=False,
        school_zone=False,
        bike_facility=CyclistFacility.BIKE_LANE,
    )
    from aito.models import DemandProfile as DP
    dp13 = DP(
        intersection_id=ix13.id, period_minutes=60,
        north_thru=320.0, south_thru=290.0, east_thru=160.0, west_thru=140.0,
        north_left=55.0, south_left=45.0, east_left=40.0, west_left=35.0,
        north_right=45.0, south_right=40.0, east_right=30.0, west_right=25.0,
    )
    mm_opt = MultiModalOptimizer(ix13, constraints)
    mm_result = mm_opt.optimize(dp13, ped_volume_hr=120.0, bike_volume_hr=25.0)
    ok(f"Multi-modal Pareto: {len(mm_result.pareto_solutions)} solutions")
    bal = mm_result.balanced
    ok(f"Balanced plan: cycle={bal.cycle_s:.0f}s  veh={bal.vehicle_delay_s_veh:.1f}s/veh  ped={bal.ped_delay_s_ped:.1f}s/ped  LOS-{bal.ped_los.value}")
    ok(f"MUTCD compliant: {bal.is_vision_zero_compliant}")

    # =========================================================================
    # GF8: Cross-Jurisdiction Coordination
    # =========================================================================
    hdr(8, "Cross-Jurisdiction Corridor Orchestration")
    from aito.optimization.cross_jurisdiction import (
        CrossJurisdictionOrchestrator, JurisdictionalCorridor,
        AgencyOptimizationConstraints, SD_AGENCIES,
    )

    jcorridor = JurisdictionalCorridor(
        corridor_id=rosecrans.id,
        corridor_name=rosecrans.name,
        agencies=SD_AGENCIES,
        intersection_agency_map={
            i: ("cosd" if i < 8 else "caltrans_d11")
            for i in range(len(rosecrans.intersections))
        },
    )
    orch = CrossJurisdictionOrchestrator(jcorridor, simulation_mode=True)
    constraints_xj = [
        AgencyOptimizationConstraints("cosd",         preferred_cycle_s=110.0),
        AgencyOptimizationConstraints("caltrans_d11", preferred_cycle_s=130.0),
        AgencyOptimizationConstraints("sandag",       preferred_cycle_s=120.0),
    ]
    common_cycle = orch.negotiate_common_cycle(constraints_xj)
    ok(f"Common cycle negotiated: {common_cycle:.0f}s  (COSD pref=110s, Caltrans pref=130s)")
    ok(f"Agencies: COSD + Caltrans D11 + SANDAG  ({len(SD_AGENCIES)} jurisdictions)")

    # =========================================================================
    # GF10: Natural Language Interface
    # =========================================================================
    hdr(10, "Natural Language Signal Engineering Interface")
    from aito.interface.nl_engineer import NLEngineerSession, classify_query

    session = NLEngineerSession(corridor=rosecrans)
    queries = [
        "Why is the cycle length so long at Midway Drive?",
        "What happens if I reduce the cycle to 90 seconds?",
        "Compare AITO to InSync",
        "What is the carbon impact of Rosecrans?",
    ]
    for q in queries:
        qtype = classify_query(q)
        ok(f"Query classified → {qtype:<18}  \"{q[:55]}\"")

    resp = session.ask(queries[2])
    ok(f"Response generated ({len(resp.answer)} chars, API_used={resp.used_claude_api})")
    info(resp.answer[:120] + "...")

    # =========================================================================
    # GF11: Resilience Scoring
    # =========================================================================
    hdr(11, "Network Resilience Scoring")
    from aito.analytics.resilience_scorer import ResilienceScorer

    scorer = ResilienceScorer(rosecrans)
    report = scorer.score(
        baseline_delay_s_veh=35.0,
        probe_penetration_rate=0.28,
        n_detectors=0,
        n_operational_detectors=0,
        auto_retiming_enabled=True,
    )
    s = report.summary()
    ok(f"Composite score: {s['composite_score']:.1f}/100  Grade: {s['overall_grade']}")
    for dim, score in s["dimensions"].items():
        ok(f"  {dim:<20} {score:.1f}")
    ok(f"Weakest dimension: {s['weakest_dimension']}")

    # =========================================================================
    # GF14: Federated Learning
    # =========================================================================
    hdr(14, "Federated Learning — Cross-City Knowledge Transfer")
    from aito.ml.federated_learning import (
        FederatedLearningServer, CityClient, generate_synthetic_city_update,
    )

    server = FederatedLearningServer(min_cities_for_aggregation=3)
    cities = [
        CityClient("san_diego",    "San Diego",    n_corridors=200, avg_aadt=28000, state="CA"),
        CityClient("los_angeles",  "Los Angeles",  n_corridors=800, avg_aadt=42000, state="CA"),
        CityClient("san_francisco","San Francisco", n_corridors=150, avg_aadt=22000, state="CA"),
    ]
    for city in cities:
        server.enroll_city(city)
    ok(f"Enrolled {server.n_enrolled_cities} cities in federated network")

    for i, city in enumerate(cities):
        update = generate_synthetic_city_update(
            city_id=city.city_id,
            n_samples=5000 + i * 1500,
            tod_variation=0.08 + i * 0.02,
            seed=42 + i,
        )
        server.submit_update(update)
    ok(f"Submitted {server.pending_update_count} differentially-private model updates (ε=1.0)")

    global_model = server.aggregate()
    ok(f"FedAvg round {global_model.round_number}: {global_model.n_cities} cities  {global_model.total_samples:,} samples")
    ok(f"Convergence delta: {global_model.convergence_delta}")
    am_factor = global_model.demand_model.tod_factors.get("AM Peak", 1.0)
    ok(f"Global AM Peak demand factor: {am_factor:.3f}")

    # =========================================================================
    # Summary
    # =========================================================================
    elapsed = time.time() - t0
    print("\n" + "="*60)
    print("  ALL 15 GOLDEN FEATURES DEMONSTRATED")
    print("="*60)
    print(f"  Runtime:              {elapsed:.1f}s")
    print(f"  Corridors used:       {rosecrans.name}, {mira_mesa.name}")
    print(f"  CO2 reduction/yr:     {reduction_t_yr:,.0f} tonnes  (Rosecrans alone)")
    print(f"  Carbon revenue/yr:    ${eligible[best].net_usd_year:,.0f}  ({best})")
    print(f"  Resilience grade:     {s['overall_grade']}  ({s['composite_score']:.0f}/100)")
    print(f"  Federated cities:     {global_model.n_cities}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
