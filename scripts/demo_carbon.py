#!/usr/bin/env python3
"""scripts/demo_carbon.py

AITO Carbon Intelligence Demo — Rosecrans Street corridor, San Diego.

Demonstrates Golden Features 2, 9, and 11:
  GF2:  Real-time intersection carbon accounting (EPA MOVES2014b)
  GF9:  Carbon credit pipeline (Verra VCS / Gold Standard / CARB LCFS)
  GF11: Network resilience scoring (5-dimension, A–D grade)

Target runtime: < 30 seconds.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from aito.data.san_diego_inventory import get_corridor
from aito.analytics.carbon_accountant import CarbonAccountant
from aito.analytics.carbon_credits import CarbonCreditPipeline, CreditMarket, assess_additionality
from aito.analytics.resilience_scorer import ResilienceScorer


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print('='*60)


def main():
    print("\n" + "="*60)
    print("  AITO — Carbon Intelligence Demo")
    print("  Rosecrans Street · City of San Diego")
    print("="*60)
    t0 = time.time()

    corridor = get_corridor("rosecrans")
    n = len(corridor.intersections)
    print(f"\nCorridor: {corridor.name}")
    print(f"Intersections: {n}  |  AADT: {corridor.aadt:,}  |  Length: {sum(corridor.distances_m)/1000:.1f} km")

    # -------------------------------------------------------------------------
    # GF2: Carbon Accounting
    # -------------------------------------------------------------------------
    section("GF2 · Real-Time Carbon Accounting  (EPA MOVES2014b)")

    baseline_delay = 52.0    # HCM LOS D — typical San Diego fixed-time
    optimized_delay = 35.0   # AITO target — LOS C
    delay_reduction = baseline_delay - optimized_delay

    reduction_tonnes_year = CarbonAccountant.estimate_quick(
        n_intersections=n,
        avg_aadt=corridor.aadt,
        avg_delay_reduction_s_veh=delay_reduction,
    )
    # Estimate baseline from ratio: baseline ≈ reduction / (reduction_pct/100)
    # Use empirical 23.5% reduction from UMich Birmingham pilot
    baseline_tonnes_year = reduction_tonnes_year / 0.235
    optimized_tonnes_year = baseline_tonnes_year - reduction_tonnes_year
    reduction_pct = delay_reduction / baseline_delay * 100.0

    print(f"\n  Baseline (fixed-time 120s cycle, {baseline_delay:.0f}s delay):")
    print(f"    Annual CO2:          {baseline_tonnes_year:>8,.0f}  tonnes/year")
    print(f"\n  AITO-optimized ({optimized_delay:.0f}s delay, LOS C):")
    print(f"    Annual CO2:          {optimized_tonnes_year:>8,.0f}  tonnes/year")
    print(f"\n  Reduction:")
    print(f"    Tonnes/year:         {reduction_tonnes_year:>8,.0f}  t CO2")
    print(f"    Delay reduction:     {delay_reduction:>6.0f}  s/veh  ({reduction_pct:.1f}%)")
    print(f"\n  Methodology: EPA MOVES2014b operating modes + CARB EMFAC2021 fleet mix")
    print(f"  (Idle: 1.38 g/s · High accel: 8.41 g/s · Cruise 25mph: 2.14 g/s)")

    # -------------------------------------------------------------------------
    # GF9: Carbon Credit Pipeline
    # -------------------------------------------------------------------------
    section("GF9 · Carbon Credit Pipeline  (Verra / Gold Standard / CARB LCFS)")

    pipeline = CarbonCreditPipeline(
        corridor_name=corridor.name,
        investment_cost_usd=72000.0,
    )
    package = pipeline.build_package(
        corridor_id=corridor.id,
        baseline_co2_tonnes_year=baseline_tonnes_year,
        optimized_co2_tonnes_year=optimized_tonnes_year,
        benchmark_adoption_pct=0.05,
    )

    print(f"\n  Gross CO2 reduction:   {package.reduction_tonnes_year:>6,.0f}  t/year")
    print(f"  After 3% leakage:      {package.creditable_tonnes_year:>6,.0f}  t/year (creditable)")
    print(f"  Additionality status:  {package.additionality.value}")

    print(f"\n  Market prices (USD/tonne):")
    best_rev = 0.0
    best_market = None
    for market_name, proj in package.projections.items():
        if proj.eligible:
            eligible_str = "✓"
            print(f"    {market_name:<22} ${proj.net_usd_year:>8,.0f}/year   {eligible_str}")
            if proj.net_usd_year > best_rev:
                best_rev = proj.net_usd_year
                best_market = market_name
        else:
            reason = proj.ineligibility_reason or "Not eligible"
            print(f"    {market_name:<22}  ✗  {reason}")

    if best_market:
        print(f"\n  → Recommended market:  {best_market}")
        print(f"  → Estimated revenue:   ${best_rev:,.0f}/year")
        print(f"\n  Note: {package.additionality_notes}")

    # -------------------------------------------------------------------------
    # GF11: Resilience Scoring
    # -------------------------------------------------------------------------
    section("GF11 · Network Resilience Scoring  (FHWA Framework)")

    scorer = ResilienceScorer(corridor)

    # Scenario A: AITO with probe data, no loop detectors
    report_aito = scorer.score(
        baseline_delay_s_veh=35.0,
        probe_penetration_rate=0.28,
        n_detectors=0,
        n_operational_detectors=0,
        demand_surge_factor=1.5,
        auto_retiming_enabled=True,
        retiming_interval_min=5.0,
    )

    # Scenario B: Legacy (InSync-equivalent) with 60% detector availability
    report_legacy = scorer.score(
        baseline_delay_s_veh=52.0,
        probe_penetration_rate=0.05,
        n_detectors=48,
        n_operational_detectors=29,   # 60% availability — typical San Diego
        demand_surge_factor=1.5,
        auto_retiming_enabled=False,
        retiming_interval_min=120.0,
    )

    print(f"\n  {'Dimension':<22} {'AITO':>8} {'Legacy':>8}")
    print(f"  {'-'*40}")
    dims = [
        ("Sensor failure",    report_aito.sensor_score.score,   report_legacy.sensor_score.score),
        ("Probe sparsity",    report_aito.probe_score.score,    report_legacy.probe_score.score),
        ("Demand surge 1.5x", report_aito.demand_score.score,   report_legacy.demand_score.score),
        ("Incident response", report_aito.incident_score.score, report_legacy.incident_score.score),
        ("Recovery speed",    report_aito.recovery_score.score, report_legacy.recovery_score.score),
    ]
    for name, aito_s, leg_s in dims:
        delta = "▲" if aito_s > leg_s else ("▼" if aito_s < leg_s else "=")
        print(f"  {name:<22} {aito_s:>7.1f}  {leg_s:>7.1f}  {delta}")

    print(f"  {'-'*40}")
    print(f"  {'COMPOSITE':<22} {report_aito.composite_score:>7.1f}  {report_legacy.composite_score:>7.1f}")
    print(f"  {'GRADE':<22} {'Grade ' + report_aito.overall_grade.value:>8}  {'Grade ' + report_legacy.overall_grade.value:>8}")

    cmp = scorer.compare_vs_insync(baseline_delay_s_veh=35.0)
    print(f"\n  Sensor failure stress:")
    print(f"    AITO delay:          {cmp['sensor_failure']['aito_delay_s_veh']:.1f}  s/veh")
    print(f"    InSync delay:        {cmp['sensor_failure']['insync_delay_s_veh']:.1f}  s/veh")
    print(f"    AITO advantage:      {cmp['sensor_failure']['aito_better_by_pct']:.1f}%")
    print(f"  Recovery after incident:")
    print(f"    AITO:                {cmp['recovery_time_min']['aito']:.0f}  minutes")
    print(f"    InSync (manual):     {cmp['recovery_time_min']['insync']:.0f}  minutes")
    print(f"    AITO faster by:      {cmp['recovery_time_min']['aito_faster_by_min']:.0f}  minutes")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    section("Summary")
    print(f"\n  CO2 reduced/year:      {reduction_tonnes_year:,.0f}  tonnes")
    print(f"  Carbon credit revenue: ${best_rev:,.0f}/year  ({best_market})")
    print(f"  Resilience grade:       {report_aito.overall_grade.value}  ({report_aito.composite_score:.0f}/100)")
    print(f"\n  Runtime: {time.time() - t0:.1f}s")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
