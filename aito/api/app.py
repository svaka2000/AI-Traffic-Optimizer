"""aito/api/app.py

AITO REST API — FastAPI application.

Endpoints:
  GET  /health
  POST /corridors
  GET  /corridors/{id}
  POST /corridors/{id}/optimize
  GET  /corridors/{id}/plans
  POST /corridors/{id}/deploy
  GET  /corridors/{id}/analytics
  POST /corridors/{id}/report

All routes are versioned under /api/v1/.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from aito.models import (
    Corridor,
    DemandProfile,
    OptimizationObjective,
    OptimizationRequest,
    OptimizationResult,
    PerformanceMetrics,
    TimingPlan,
)
from aito.optimization.multi_objective import MultiObjectiveOptimizer
from aito.analytics.before_after import BeforeAfterAnalyzer, BeforeAfterAnalysis
from aito.analytics.roi_calculator import ROICalculator

# GF module imports (lazy-imported in routes to keep startup fast)
# GF1: probe_fusion, GF2: carbon_accountant, GF3: event_aware,
# GF9: carbon_credits, GF11: resilience_scorer, GF13: multimodal,
# GF14: federated_learning, GF15: what_if


# ---------------------------------------------------------------------------
# In-memory store (replace with PostgreSQL in production)
# ---------------------------------------------------------------------------

_corridors: dict[str, Corridor] = {}
_optimization_jobs: dict[str, dict] = {}
_timing_plans: dict[str, list[TimingPlan]] = {}  # corridor_id → plans


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AITO — AI Traffic Optimization",
    description=(
        "Cloud-based signal timing optimization platform for government "
        "transportation agencies. Generates optimized timing plans using "
        "probe data, multi-objective optimization, and NTCIP 1202 deployment."
    ),
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class OptimizeRequest(BaseModel):
    demand_profiles: list[DemandProfile]
    objectives: list[OptimizationObjective] = [
        OptimizationObjective.DELAY,
        OptimizationObjective.EMISSIONS,
    ]
    min_cycle: float = 60.0
    max_cycle: float = 180.0


class BeforeAfterRequest(BaseModel):
    before: PerformanceMetrics
    after: PerformanceMetrics
    daily_vehicles: int = 20000


class ROIRequest(BaseModel):
    delay_reduction_s_veh: float
    stops_reduction_pct: float
    co2_reduction_pct: float
    daily_vehicles: int
    aito_annual_cost_usd: float
    before_co2_kg_hr: float = 50.0


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["system"])
def health_check():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat(), "version": "1.0.0"}


# ---------------------------------------------------------------------------
# Corridors
# ---------------------------------------------------------------------------

@app.post("/api/v1/corridors", tags=["corridors"], response_model=Corridor)
def create_corridor(corridor: Corridor):
    """Register a new corridor in the AITO system."""
    _corridors[corridor.id] = corridor
    return corridor


@app.get("/api/v1/corridors", tags=["corridors"])
def list_corridors():
    """List all registered corridors."""
    return list(_corridors.values())


@app.get("/api/v1/corridors/{corridor_id}", tags=["corridors"])
def get_corridor(corridor_id: str):
    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    return corridor


@app.delete("/api/v1/corridors/{corridor_id}", tags=["corridors"])
def delete_corridor(corridor_id: str):
    if corridor_id not in _corridors:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    del _corridors[corridor_id]
    return {"deleted": corridor_id}


# ---------------------------------------------------------------------------
# Optimization
# ---------------------------------------------------------------------------

@app.post("/api/v1/corridors/{corridor_id}/optimize", tags=["optimization"])
def optimize_corridor(corridor_id: str, request: OptimizeRequest) -> dict:
    """Run multi-objective optimization for a corridor.

    Returns a set of Pareto-optimal timing plans with a recommended solution.
    """
    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    if len(request.demand_profiles) != len(corridor.intersections):
        raise HTTPException(
            status_code=422,
            detail=(
                f"demand_profiles length {len(request.demand_profiles)} must equal "
                f"corridor intersection count {len(corridor.intersections)}"
            ),
        )

    opt_request = OptimizationRequest(
        corridor_id=corridor_id,
        demand_profiles=request.demand_profiles,
        objectives=request.objectives,
        min_cycle=request.min_cycle,
        max_cycle=request.max_cycle,
    )

    optimizer = MultiObjectiveOptimizer(corridor)
    result = optimizer.optimize(opt_request)

    # Store plans for deployment
    _timing_plans[corridor_id] = result.recommended_solution.plan.timing_plans
    _optimization_jobs[result.request_id] = result.model_dump()

    return result.model_dump()


@app.get("/api/v1/corridors/{corridor_id}/plans", tags=["optimization"])
def get_timing_plans(corridor_id: str):
    """Get the current timing plans for a corridor."""
    if corridor_id not in _corridors:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")
    plans = _timing_plans.get(corridor_id, [])
    return {"corridor_id": corridor_id, "plans": [p.model_dump() for p in plans]}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.post("/api/v1/corridors/{corridor_id}/analytics/before-after", tags=["analytics"])
def before_after_analysis(corridor_id: str, request: BeforeAfterRequest) -> dict:
    """Compute before/after performance comparison."""
    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    analyzer = BeforeAfterAnalyzer(
        corridor=corridor,
        daily_vehicles=request.daily_vehicles,
    )
    result = analyzer.analyze(before=request.before, after=request.after)
    return {
        "travel_time_improvement_pct": result.travel_time_improvement_pct,
        "delay_improvement_pct": result.delay_improvement_pct,
        "co2_reduction_pct": result.co2_reduction_pct,
        "stops_reduction_pct": result.stops_reduction_pct,
        "annual_veh_hours_saved": result.annual_veh_hours_saved,
        "annual_fuel_saved_gallons": result.annual_fuel_saved_gallons,
        "annual_co2_reduction_tonnes": result.annual_co2_reduction_tonnes,
        "annual_cost_savings_usd": result.annual_cost_savings_usd,
        "statistical_test": {
            "t_statistic": result.statistical_test.t_statistic,
            "significant": result.statistical_test.statistically_significant,
        },
        "notes": result.notes,
    }


@app.post("/api/v1/corridors/{corridor_id}/analytics/roi", tags=["analytics"])
def roi_analysis(corridor_id: str, request: ROIRequest) -> dict:
    """Compute ROI analysis for AITO deployment."""
    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    calc = ROICalculator(corridor=corridor)
    report = calc.calculate(
        delay_reduction_s_veh=request.delay_reduction_s_veh,
        stops_reduction_pct=request.stops_reduction_pct,
        co2_reduction_pct=request.co2_reduction_pct,
        daily_vehicles=request.daily_vehicles,
        aito_annual_cost_usd=request.aito_annual_cost_usd,
        before_co2_kg_hr=request.before_co2_kg_hr,
    )
    return {
        "benefit_cost_ratio": report.benefit_cost_ratio,
        "simple_payback_months": report.simple_payback_months,
        "npv_usd": report.npv_usd,
        "annual_total_benefit_usd": report.annual_total_benefit_usd,
        "annual_veh_hours_saved": report.annual_veh_hours_saved,
        "annual_fuel_saved_gallons": report.annual_fuel_saved_gallons,
        "annual_co2_reduction_tonnes": report.annual_co2_reduction_tonnes,
        "summary": report.summary(),
    }


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

@app.get("/api/v1/corridors/{corridor_id}/export/synchro", tags=["export"])
def export_synchro(corridor_id: str) -> dict:
    """Export timing plans in Synchro UTDF CSV format."""
    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    plans = _timing_plans.get(corridor_id, [])
    if not plans:
        raise HTTPException(status_code=404, detail="No timing plans found. Run optimization first.")

    from aito.deployment.ntcip_client import SynchroCSVExporter
    exporter = SynchroCSVExporter()
    csv_content = exporter.export(plans, corridor.intersections)
    return {"corridor_id": corridor_id, "format": "synchro_utdf", "csv": csv_content}


# ---------------------------------------------------------------------------
# GF1: Probe Data Fusion
# ---------------------------------------------------------------------------

class ProbeObservationRequest(BaseModel):
    segment_id: str
    source: str = "connected_vehicle"   # connected_vehicle | inrix | here | waze
    speed_mph: float
    travel_time_s: float
    confidence: float = 0.95
    timestamp: Optional[datetime] = None


class ProbeFuseRequest(BaseModel):
    corridor_id: str
    segment_distance_m: float = 400.0
    observations: list[ProbeObservationRequest]


@app.post("/api/v1/probe/fuse", tags=["probe-data"])
def fuse_probe_data(request: ProbeFuseRequest) -> dict:
    """Fuse multi-source probe observations into corridor traffic state estimates.

    Sources: connected_vehicle, inrix, here, waze, smartphone_gps.
    Returns per-segment fused speed, travel time, congestion state, and confidence.
    """
    from aito.data.probe_fusion import ProbeFusionEngine, SourceObservation

    # Build segment lists from unique segment IDs in observations
    seg_ids = list(dict.fromkeys(o.segment_id for o in request.observations))
    n_segs = len(seg_ids)
    if n_segs == 0:
        raise HTTPException(status_code=422, detail="No observations provided.")

    engine = ProbeFusionEngine(
        corridor_id=request.corridor_id,
        segment_distances_m=[request.segment_distance_m] * n_segs,
        free_flow_speeds_mph=[35.0] * n_segs,
    )

    now_ts = datetime.utcnow()
    obs_list = [
        SourceObservation(
            source=obs.source,
            segment_id=obs.segment_id,
            timestamp=obs.timestamp or now_ts,
            travel_time_s=obs.travel_time_s,
            speed_mph=obs.speed_mph,
            confidence=obs.confidence,
        )
        for obs in request.observations
    ]
    engine.ingest(obs_list)
    corridor_est = engine.fuse(now_ts)

    return {
        "corridor_id": request.corridor_id,
        "n_segments": len(corridor_est.segments),
        "corridor_travel_time_s": round(corridor_est.corridor_travel_time_s, 1),
        "mean_speed_mph": round(corridor_est.mean_speed_mph, 1),
        "worst_congestion": corridor_est.worst_congestion_state.value,
        "segments": [
            {
                "segment_id": s.segment_id,
                "fused_speed_mph": s.speed_mph,
                "fused_travel_time_s": s.travel_time_s,
                "congestion_state": s.congestion_state.value,
                "confidence": s.fused_confidence,
                "source_count": s.source_count,
                "sources_used": s.sources_used,
            }
            for s in corridor_est.segments
        ],
    }


# ---------------------------------------------------------------------------
# GF2 + GF9: Carbon Accounting & Credits
# ---------------------------------------------------------------------------

class CarbonEstimateRequest(BaseModel):
    n_intersections: int = 12
    avg_aadt: int = 28000
    baseline_delay_s_veh: float = 52.0
    optimized_delay_s_veh: float = 35.0


class CarbonCreditsRequest(BaseModel):
    baseline_co2_tonnes_year: float
    optimized_co2_tonnes_year: float
    corridor_name: str = "Unknown"
    investment_cost_usd: float = 72000.0
    benchmark_adoption_pct: float = 0.05


@app.post("/api/v1/corridors/{corridor_id}/carbon", tags=["carbon"])
def estimate_carbon(corridor_id: str, request: CarbonEstimateRequest) -> dict:
    """Estimate CO2 emissions reduction from signal optimization.

    Uses EPA MOVES2014b idle emission factor (1.38 g/s per vehicle).
    Returns annual reduction in tonnes CO2.
    """
    from aito.analytics.carbon_accountant import CarbonAccountant

    delay_reduction = request.baseline_delay_s_veh - request.optimized_delay_s_veh
    reduction_tonnes_year = CarbonAccountant.estimate_quick(
        n_intersections=request.n_intersections,
        avg_aadt=request.avg_aadt,
        avg_delay_reduction_s_veh=max(0.0, delay_reduction),
    )
    # Back-calculate baseline using 23.5% reduction ratio as reference
    reduction_pct = delay_reduction / max(request.baseline_delay_s_veh, 1.0) * 100.0
    baseline_approx = reduction_tonnes_year / max(reduction_pct / 100.0, 0.01)
    optimized_approx = baseline_approx - reduction_tonnes_year

    return {
        "corridor_id": corridor_id,
        "delay_reduction_s_veh": round(delay_reduction, 1),
        "reduction_tonnes_year": round(reduction_tonnes_year, 1),
        "baseline_co2_tonnes_year_approx": round(baseline_approx, 0),
        "optimized_co2_tonnes_year_approx": round(optimized_approx, 0),
        "reduction_pct_delay": round(reduction_pct, 1),
        "methodology": "EPA MOVES2014b idle rate (1.38 g/s) × delay reduction × fleet mix",
    }


@app.post("/api/v1/corridors/{corridor_id}/carbon/credits", tags=["carbon"])
def estimate_carbon_credits(corridor_id: str, request: CarbonCreditsRequest) -> dict:
    """Estimate carbon credit revenue potential from signal optimization.

    Evaluates Verra VCS, Gold Standard, CARB LCFS, and CARB Cap-and-Trade markets.
    Returns eligible markets, revenue estimates, and additionality assessment.
    """
    from aito.analytics.carbon_credits import CarbonCreditPipeline, CreditMarket

    pipeline = CarbonCreditPipeline(
        corridor_name=request.corridor_name,
        investment_cost_usd=request.investment_cost_usd,
    )
    package = pipeline.build_package(
        corridor_id=corridor_id,
        baseline_co2_tonnes_year=request.baseline_co2_tonnes_year,
        optimized_co2_tonnes_year=request.optimized_co2_tonnes_year,
        benchmark_adoption_pct=request.benchmark_adoption_pct,
    )

    eligible_markets = {
        name: round(proj.net_usd_year, 0)
        for name, proj in package.projections.items()
        if proj.eligible
    }
    best_market = max(eligible_markets, key=eligible_markets.get) if eligible_markets else None

    return {
        "corridor_id": corridor_id,
        "creditable_tonnes_year": round(package.creditable_tonnes_year, 1),
        "additionality": package.additionality.value,
        "additionality_notes": package.additionality_notes,
        "eligible_markets": eligible_markets,
        "best_market": best_market,
        "best_market_revenue_usd_year": eligible_markets.get(best_market) if best_market else None,
    }


# ---------------------------------------------------------------------------
# GF11: Network Resilience Scoring
# ---------------------------------------------------------------------------

class ResilienceRequest(BaseModel):
    baseline_delay_s_veh: float = 30.0
    probe_penetration_rate: float = 0.28
    n_detectors: int = 0
    n_operational_detectors: int = 0
    demand_surge_factor: float = 1.5
    incident_capacity_reduction: float = 0.50
    auto_retiming_enabled: bool = True
    retiming_interval_min: float = 5.0


@app.post("/api/v1/corridors/{corridor_id}/resilience", tags=["resilience"])
def score_resilience(corridor_id: str, request: ResilienceRequest) -> dict:
    """Score corridor resilience across 5 dimensions (0–100 each).

    Dimensions: sensor failure, probe sparsity, demand surge, incident, recovery speed.
    Returns composite grade (A–D) and AITO vs InSync comparison.
    """
    from aito.analytics.resilience_scorer import ResilienceScorer

    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    scorer = ResilienceScorer(corridor)
    report = scorer.score(
        baseline_delay_s_veh=request.baseline_delay_s_veh,
        probe_penetration_rate=request.probe_penetration_rate,
        n_detectors=request.n_detectors,
        n_operational_detectors=request.n_operational_detectors,
        demand_surge_factor=request.demand_surge_factor,
        incident_capacity_reduction=request.incident_capacity_reduction,
        auto_retiming_enabled=request.auto_retiming_enabled,
        retiming_interval_min=request.retiming_interval_min,
    )
    comparison = scorer.compare_vs_insync(
        baseline_delay_s_veh=request.baseline_delay_s_veh,
        probe_penetration_rate=request.probe_penetration_rate,
    )
    return {
        "corridor_id": corridor_id,
        "summary": report.summary(),
        "vs_insync": comparison,
    }


# ---------------------------------------------------------------------------
# GF15: What-If Scenario Engine
# ---------------------------------------------------------------------------

class WhatIfRequest(BaseModel):
    scenario_type: str = "demand"   # demand | closure | timing | event | incident
    description: str = "What-if scenario"
    # Demand scenario
    demand_multiplier: float = 1.3
    # Closure scenario
    closed_segment_idx: int = 0
    # Timing scenario
    cycle_override_s: Optional[float] = None
    green_ratio_override: Optional[float] = None
    # Incident scenario
    incident_segment_idx: int = 0
    incident_capacity_fraction: float = 0.5
    incident_duration_min: float = 30.0


@app.post("/api/v1/corridors/{corridor_id}/whatif", tags=["scenarios"])
def run_whatif(corridor_id: str, request: WhatIfRequest) -> dict:
    """Run a what-if scenario against the corridor digital twin.

    Supported scenarios: demand surge, lane closure, timing plan change, incident.
    Returns side-by-side comparison of baseline vs scenario performance.
    """
    from aito.simulation.what_if import (
        WhatIfEngine, DemandScenario, ClosureScenario,
        TimingPlanScenario, IncidentScenario, ScenarioType,
    )
    from aito.simulation.digital_twin import DigitalTwin

    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    twin = DigitalTwin(corridor)
    engine = WhatIfEngine(twin)

    stype = request.scenario_type.lower()
    if stype == "demand":
        scenario = DemandScenario(
            description=request.description,
            demand_multiplier=request.demand_multiplier,
        )
    elif stype == "closure":
        scenario = ClosureScenario(
            description=request.description,
            closed_segment_idx=request.closed_segment_idx,
        )
    elif stype == "timing":
        scenario = TimingPlanScenario(
            description=request.description,
            cycle_override_s=request.cycle_override_s,
            green_ratio_override=request.green_ratio_override or 0.45,
        )
    elif stype == "incident":
        scenario = IncidentScenario(
            description=request.description,
            incident_segment_idx=request.incident_segment_idx,
            capacity_fraction=request.incident_capacity_fraction,
            duration_min=request.incident_duration_min,
        )
    else:
        raise HTTPException(status_code=422, detail=f"Unknown scenario_type '{stype}'")

    comparison = engine.run_scenario(scenario)
    return {
        "corridor_id": corridor_id,
        "scenario": request.scenario_type,
        "description": request.description,
        "baseline": {
            "avg_delay_s_veh": round(comparison.baseline_delay, 1),
            "avg_speed_mph": round(comparison.baseline_speed_mph, 1),
            "throughput_veh_hr": round(comparison.baseline_throughput, 0),
        },
        "scenario_result": {
            "avg_delay_s_veh": round(comparison.scenario_delay, 1),
            "avg_speed_mph": round(comparison.scenario_speed_mph, 1),
            "throughput_veh_hr": round(comparison.scenario_throughput, 0),
        },
        "delta_delay_pct": round(comparison.delay_delta_pct, 1),
        "verdict": comparison.verdict,
    }


# ---------------------------------------------------------------------------
# GF3: Event-Aware Optimization
# ---------------------------------------------------------------------------

class EventRegistrationRequest(BaseModel):
    event_id: str
    name: str
    venue_name: str
    venue_lat: float
    venue_lon: float
    event_type: str = "sports_major"   # sports_major | sports_minor | concert_large | convention
    start_time: datetime
    end_time: datetime
    expected_attendance: int = 40000
    max_capacity: int = 45000


@app.post("/api/v1/corridors/{corridor_id}/events", tags=["events"])
def register_event(corridor_id: str, request: EventRegistrationRequest) -> dict:
    """Register an upcoming event and generate pre-emptive timing plans.

    AITO generates inbound (pre-event) and egress (post-event) timing plans
    that activate before demand arrives, not after.
    """
    from aito.optimization.event_aware import EventAwareOptimizer, CalendarEvent, EventType

    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    try:
        etype = EventType(request.event_type)
    except ValueError:
        etype = EventType.SPORTS_MAJOR

    event = CalendarEvent(
        event_id=request.event_id,
        name=request.name,
        venue_name=request.venue_name,
        venue_lat=request.venue_lat,
        venue_lon=request.venue_lon,
        event_type=etype,
        start_time=request.start_time,
        end_time=request.end_time,
        expected_attendance=request.expected_attendance,
        max_capacity=request.max_capacity,
    )

    optimizer = EventAwareOptimizer(corridor)
    optimizer.register_event(event)
    result = optimizer._optimize_for_event(event)

    return {
        "corridor_id": corridor_id,
        "event_id": request.event_id,
        "event_name": request.name,
        "inbound_peak_veh_hr": round(event.inbound_peak_veh_hr, 0),
        "outbound_peak_veh_hr": round(event.outbound_peak_veh_hr, 0),
        "pre_deploy_time": event.pre_deploy_time.isoformat(),
        "egress_end_time": event.egress_end_time.isoformat(),
        "n_timing_plans": len(result.timing_plans),
        "plans_summary": [
            {
                "phase": p.phase,
                "active_from": p.active_from.isoformat(),
                "active_until": p.active_until.isoformat(),
                "cycle_s": p.cycle_s,
                "description": p.description,
            }
            for p in result.timing_plans
        ],
    }


@app.get("/api/v1/corridors/{corridor_id}/events", tags=["events"])
def list_events(corridor_id: str) -> dict:
    """List upcoming events registered for this corridor."""
    return {"corridor_id": corridor_id, "note": "Use POST to register events. Events are not persisted across restarts."}


# ---------------------------------------------------------------------------
# GF13: Multi-Modal Optimization
# ---------------------------------------------------------------------------

class MultiModalRequest(BaseModel):
    intersection_idx: int = 0
    ped_volume_hr: float = 100.0
    bike_volume_hr: float = 20.0
    min_cycle_s: float = 60.0
    max_cycle_s: float = 180.0
    accessible_clearance: bool = False
    school_zone: bool = False
    bike_facility: str = "bike_lane"   # mixed | bike_lane | protected | shared_path


@app.post("/api/v1/corridors/{corridor_id}/multimodal", tags=["multimodal"])
def multimodal_optimize(corridor_id: str, request: MultiModalRequest) -> dict:
    """Optimize one intersection for all modes: vehicles, pedestrians, cyclists.

    Returns Pareto-optimal plans with vehicle-optimal, ped-optimal, and balanced solutions.
    All plans are validated against MUTCD 2023 pedestrian timing requirements.
    """
    from aito.optimization.multimodal import (
        MultiModalOptimizer, MultiModalConstraints, CyclistFacility,
    )

    corridor = _corridors.get(corridor_id)
    if not corridor:
        raise HTTPException(status_code=404, detail=f"Corridor {corridor_id} not found")

    n = len(corridor.intersections)
    if request.intersection_idx >= n:
        raise HTTPException(
            status_code=422,
            detail=f"intersection_idx {request.intersection_idx} out of range (corridor has {n} intersections)"
        )

    intersection = corridor.intersections[request.intersection_idx]

    try:
        facility = CyclistFacility(request.bike_facility)
    except ValueError:
        facility = CyclistFacility.BIKE_LANE

    constraints = MultiModalConstraints(
        accessible_clearance=request.accessible_clearance,
        school_zone=request.school_zone,
        bike_facility=facility,
        bike_volume_hr=request.bike_volume_hr,
    )

    # Build a minimal demand profile for the optimizer
    from aito.models import DemandProfile
    demand = DemandProfile(
        intersection_id=intersection.id,
        period_minutes=60,
        north_thru=300.0, south_thru=280.0,
        east_thru=180.0, west_thru=160.0,
        north_left=60.0, south_left=50.0,
        east_left=40.0, west_left=35.0,
        north_right=40.0, south_right=35.0,
        east_right=30.0, west_right=25.0,
    )

    optimizer = MultiModalOptimizer(intersection, constraints)
    result = optimizer.optimize(
        demand_profile=demand,
        ped_volume_hr=request.ped_volume_hr,
        bike_volume_hr=request.bike_volume_hr,
        min_cycle_s=request.min_cycle_s,
        max_cycle_s=request.max_cycle_s,
    )

    def plan_dict(p) -> dict:
        return {
            "cycle_s": p.cycle_s,
            "ped_walk_s": p.ped_walk_s,
            "ped_clearance_s": p.ped_clearance_s,
            "vehicle_delay_s_veh": p.vehicle_delay_s_veh,
            "ped_delay_s_ped": p.ped_delay_s_ped,
            "bike_delay_s_bike": p.bike_delay_s_bike,
            "ped_los": p.ped_los.value,
            "vision_zero_compliant": p.is_vision_zero_compliant,
            "compliance_notes": p.compliance_notes,
        }

    return {
        "corridor_id": corridor_id,
        "intersection_id": intersection.id,
        "intersection_idx": request.intersection_idx,
        "pareto_solutions": len(result.pareto_solutions),
        "vehicle_optimal": plan_dict(result.vehicle_optimal),
        "ped_optimal": plan_dict(result.ped_optimal),
        "balanced": plan_dict(result.balanced),
    }


# ---------------------------------------------------------------------------
# GF14: Federated Learning
# ---------------------------------------------------------------------------

# Module-level server instance (single server per process)
_fed_server = _fed_server = None


def _get_fed_server() -> Any:
    global _fed_server
    if _fed_server is None:
        from aito.ml.federated_learning import FederatedLearningServer
        _fed_server = FederatedLearningServer(min_cities_for_aggregation=2)
    return _fed_server


class FederatedCityRequest(BaseModel):
    city_id: str
    city_name: str
    n_corridors: int = 50
    avg_aadt: int = 25000
    state: str = "CA"


class FederatedUpdateRequest(BaseModel):
    city_id: str
    round_number: int = 1
    n_samples: int = 5000
    tod_variation: float = 0.10
    seed: int = 42
    dp_epsilon: float = 1.0


@app.post("/api/v1/federated/enroll", tags=["federated-learning"])
def federated_enroll(request: FederatedCityRequest) -> dict:
    """Enroll a city in the AITO federated learning network.

    Cities share only differentially-private model weight updates — never raw data.
    """
    from aito.ml.federated_learning import CityClient
    server = _get_fed_server()
    city = CityClient(
        city_id=request.city_id,
        city_name=request.city_name,
        n_corridors=request.n_corridors,
        avg_aadt=request.avg_aadt,
        state=request.state,
    )
    server.enroll_city(city)
    return {
        "enrolled": True,
        "city_id": request.city_id,
        "total_enrolled": server.n_enrolled_cities,
    }


@app.post("/api/v1/federated/submit", tags=["federated-learning"])
def federated_submit(request: FederatedUpdateRequest) -> dict:
    """Submit a differentially-private model update from a city.

    DP noise (Laplace mechanism, ε=1.0) is applied server-side before aggregation.
    Raw demand data never leaves the city.
    """
    from aito.ml.federated_learning import generate_synthetic_city_update
    server = _get_fed_server()

    try:
        update = generate_synthetic_city_update(
            city_id=request.city_id,
            n_samples=request.n_samples,
            tod_variation=request.tod_variation,
            seed=request.seed,
        )
        update.round_number = request.round_number
        update.dp_epsilon = request.dp_epsilon
        server.submit_update(update)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    return {
        "accepted": True,
        "city_id": request.city_id,
        "pending_updates": server.pending_update_count,
        "ready_to_aggregate": server.pending_update_count >= server.min_cities,
    }


@app.post("/api/v1/federated/aggregate", tags=["federated-learning"])
def federated_aggregate() -> dict:
    """Run FedAvg aggregation across all pending city updates.

    Requires at least min_cities_for_aggregation updates. Returns the
    new global model's TOD demand factors and emission corrections.
    """
    server = _get_fed_server()
    global_model = server.aggregate()
    if global_model is None:
        return {
            "aggregated": False,
            "reason": f"Need at least {server.min_cities} cities. Currently {server.pending_update_count} pending.",
        }

    return {
        "aggregated": True,
        "round_number": global_model.round_number,
        "n_cities": global_model.n_cities,
        "total_samples": global_model.total_samples,
        "convergence_delta": global_model.convergence_delta,
        "global_tod_factors": global_model.demand_model.tod_factors,
        "global_emission_correction": global_model.emission_model.fleet_correction,
    }


@app.get("/api/v1/federated/status", tags=["federated-learning"])
def federated_status() -> dict:
    """Get federated learning server status."""
    server = _get_fed_server()
    return server.status()
