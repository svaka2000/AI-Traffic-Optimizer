"""aito/analytics/carbon_credits.py

GF9: Carbon Credit Pipeline — Verra VCS, Gold Standard, CARB Offset Protocol.

Converts verified AITO CO2 reductions into monetizable carbon credits
under three major voluntary and compliance market frameworks.

This module does NOT issue credits — it prepares the documentation
package and revenue projections needed to engage offset registries.

Frameworks supported:
  1. Verra VCS (VM0036 / VM0038 — Transport)
     Market price: ~$15-30/tonne (voluntary, 2024)
  2. Gold Standard Transport GS-T
     Market price: ~$20-50/tonne (premium, 2024)
  3. CARB Low Carbon Fuel Standard (LCFS)
     Market price: ~$50-80/tonne (California compliance, 2024)
  4. CARB Cap-and-Trade Offset Protocol (OPT)
     Requires ARB approval; ~$20-35/tonne

Reference:
  Verra. (2022). VM0036 Methodology for Rewarding Reductions of GHG
  Emissions from Avoided Traffic Congestion. v1.0.
  California Air Resources Board. (2022). Low Carbon Fuel Standard
  Regulation, 17 CCR §95480 et seq.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, date
from enum import Enum
from typing import Optional


# ---------------------------------------------------------------------------
# Market price assumptions (USD/tonne CO2e, 2024 midpoints)
# ---------------------------------------------------------------------------

class CreditMarket(str, Enum):
    VERRA_VCS    = "verra_vcs"
    GOLD_STANDARD = "gold_standard"
    CARB_LCFS    = "carb_lcfs"
    CARB_CAP_TRADE = "carb_cap_trade"


MARKET_PRICE_USD_TONNE: dict[CreditMarket, float] = {
    CreditMarket.VERRA_VCS:       22.0,
    CreditMarket.GOLD_STANDARD:   35.0,
    CreditMarket.CARB_LCFS:       65.0,
    CreditMarket.CARB_CAP_TRADE:  28.0,
}

# Conservative discount factors (verification uncertainty, buffer pool, fees)
MARKET_DISCOUNT: dict[CreditMarket, float] = {
    CreditMarket.VERRA_VCS:       0.80,   # 20% to buffer pool
    CreditMarket.GOLD_STANDARD:   0.82,
    CreditMarket.CARB_LCFS:       0.92,   # smallest discount (compliance market)
    CreditMarket.CARB_CAP_TRADE:  0.78,
}

# Eligibility requirements (simplified)
ELIGIBILITY_MIN_TONNES_YEAR: dict[CreditMarket, float] = {
    CreditMarket.VERRA_VCS:       100.0,   # minimum project scale
    CreditMarket.GOLD_STANDARD:   50.0,
    CreditMarket.CARB_LCFS:       0.0,     # no minimum
    CreditMarket.CARB_CAP_TRADE:  25000.0, # large offset projects only
}


# ---------------------------------------------------------------------------
# Additionality assessment
# ---------------------------------------------------------------------------

class AdditionalityLevel(str, Enum):
    HIGH    = "high"      # clearly additional: no regulatory mandate, significant cost
    MEDIUM  = "medium"    # likely additional: some common practice concerns
    LOW     = "low"       # questionable: may be baseline practice
    FAILED  = "failed"    # not additional: required by regulation


def assess_additionality(
    is_regulatory_requirement: bool = False,
    has_existing_adaptive_control: bool = False,
    investment_cost_usd: float = 0.0,
    benchmark_adoption_pct: float = 0.05,  # fraction of similar agencies using AITO
) -> AdditionalityLevel:
    """Simple additionality assessment (Verra regulatory surplus test).

    Returns assessment of whether AITO's CO2 reductions are 'additional'
    (i.e., wouldn't happen without carbon finance).
    """
    if is_regulatory_requirement:
        return AdditionalityLevel.FAILED
    if has_existing_adaptive_control:
        return AdditionalityLevel.LOW
    if investment_cost_usd > 50000 and benchmark_adoption_pct < 0.15:
        return AdditionalityLevel.HIGH
    if benchmark_adoption_pct < 0.30:
        return AdditionalityLevel.MEDIUM
    return AdditionalityLevel.LOW


# ---------------------------------------------------------------------------
# Baseline emissions methodology
# ---------------------------------------------------------------------------

@dataclass
class BaselineEmissions:
    """Counterfactual emissions absent AITO (for additionality proof)."""
    corridor_id: str
    methodology: str              # "fixed_time" or "pre_existing_adaptive"
    co2_kg_hr: float              # business-as-usual emissions
    co2_tonnes_year: float        # annualized
    data_sources: list[str] = field(default_factory=list)
    confidence: float = 0.85      # [0,1] — affects credit issuance discount


# ---------------------------------------------------------------------------
# Credit issuance projection
# ---------------------------------------------------------------------------

@dataclass
class CreditIssuanceProjection:
    """Carbon credit revenue projection for one market."""
    market: CreditMarket
    tonnes_year: float            # verified reduction eligible for crediting
    gross_usd_year: float         # pre-discount revenue
    net_usd_year: float           # after buffer pool and fees
    price_per_tonne: float
    discount_applied: float
    eligible: bool
    ineligibility_reason: Optional[str] = None
    crediting_period_years: int = 7     # Verra default crediting period


@dataclass
class CarbonCreditPackage:
    """Complete carbon credit documentation package for one corridor."""
    corridor_id: str
    corridor_name: str
    generated_at: datetime

    # Emission accounting
    baseline_co2_tonnes_year: float
    optimized_co2_tonnes_year: float
    reduction_tonnes_year: float
    reduction_pct: float

    # Additionality
    additionality: AdditionalityLevel
    additionality_notes: str

    # Market projections
    projections: dict[str, CreditIssuanceProjection]   # market.value → projection

    # Leakage deduction (emissions displaced to other areas)
    leakage_deduction_pct: float = 0.03   # 3% standard for urban transport

    @property
    def creditable_tonnes_year(self) -> float:
        """Tonnes eligible after leakage deduction."""
        return self.reduction_tonnes_year * (1.0 - self.leakage_deduction_pct)

    @property
    def best_market(self) -> Optional[CreditIssuanceProjection]:
        eligible = [p for p in self.projections.values() if p.eligible]
        if not eligible:
            return None
        return max(eligible, key=lambda p: p.net_usd_year)

    @property
    def max_annual_revenue_usd(self) -> float:
        best = self.best_market
        return best.net_usd_year if best else 0.0

    @property
    def total_crediting_period_revenue_usd(self) -> float:
        best = self.best_market
        if not best:
            return 0.0
        return best.net_usd_year * best.crediting_period_years

    def summary(self) -> dict:
        best = self.best_market
        return {
            "corridor": self.corridor_name,
            "reduction_tonnes_year": round(self.reduction_tonnes_year, 1),
            "creditable_tonnes_year": round(self.creditable_tonnes_year, 1),
            "reduction_pct": round(self.reduction_pct, 1),
            "additionality": self.additionality.value,
            "best_market": best.market.value if best else None,
            "max_annual_revenue_usd": round(self.max_annual_revenue_usd),
            "7yr_revenue_usd": round(self.total_crediting_period_revenue_usd),
            "markets_eligible": [
                p.market.value for p in self.projections.values() if p.eligible
            ],
        }


# ---------------------------------------------------------------------------
# CarbonCreditPipeline — main class
# ---------------------------------------------------------------------------

class CarbonCreditPipeline:
    """Prepares carbon credit documentation for AITO corridors.

    Usage:
        pipeline = CarbonCreditPipeline(corridor_name="Rosecrans St")
        package = pipeline.build_package(
            corridor_id=corridor.id,
            baseline_co2_tonnes_year=285.0,
            optimized_co2_tonnes_year=218.0,
        )
        print(f"Max annual revenue: ${package.max_annual_revenue_usd:,.0f}")
    """

    def __init__(
        self,
        corridor_name: str,
        state: str = "CA",
        investment_cost_usd: float = 72000.0,
    ) -> None:
        self.corridor_name = corridor_name
        self.state = state
        self.investment_cost_usd = investment_cost_usd

    def build_package(
        self,
        corridor_id: str,
        baseline_co2_tonnes_year: float,
        optimized_co2_tonnes_year: float,
        is_regulatory_requirement: bool = False,
        has_existing_adaptive: bool = False,
        benchmark_adoption_pct: float = 0.05,
        leakage_pct: float = 0.03,
    ) -> CarbonCreditPackage:
        """Build complete credit package from before/after emission totals."""
        reduction = baseline_co2_tonnes_year - optimized_co2_tonnes_year
        reduction_pct = (reduction / max(baseline_co2_tonnes_year, 1.0)) * 100.0

        additionality = assess_additionality(
            is_regulatory_requirement=is_regulatory_requirement,
            has_existing_adaptive_control=has_existing_adaptive,
            investment_cost_usd=self.investment_cost_usd,
            benchmark_adoption_pct=benchmark_adoption_pct,
        )

        creditable = reduction * (1.0 - leakage_pct)

        projections: dict[str, CreditIssuanceProjection] = {}
        for market in CreditMarket:
            min_scale = ELIGIBILITY_MIN_TONNES_YEAR[market]
            eligible = (
                creditable >= min_scale
                and additionality != AdditionalityLevel.FAILED
            )
            reason = None
            if not eligible:
                if creditable < min_scale:
                    reason = f"Below minimum {min_scale} tonnes/year for {market.value}"
                elif additionality == AdditionalityLevel.FAILED:
                    reason = "Additionality failed: regulatory requirement"
            elif self.state != "CA" and market == CreditMarket.CARB_LCFS:
                eligible = False
                reason = "CARB LCFS only available in California"

            price = MARKET_PRICE_USD_TONNE[market]
            discount = MARKET_DISCOUNT[market]
            gross = creditable * price if eligible else 0.0
            net = gross * discount if eligible else 0.0

            projections[market.value] = CreditIssuanceProjection(
                market=market,
                tonnes_year=round(creditable, 2),
                gross_usd_year=round(gross),
                net_usd_year=round(net),
                price_per_tonne=price,
                discount_applied=discount,
                eligible=eligible,
                ineligibility_reason=reason,
            )

        # Build additionality notes
        notes_map = {
            AdditionalityLevel.HIGH:   "Project clearly additional: high cost, low adoption rate in peer agencies",
            AdditionalityLevel.MEDIUM: "Likely additional: modest market penetration, not yet common practice",
            AdditionalityLevel.LOW:    "Additionality questionable: review common practice test carefully",
            AdditionalityLevel.FAILED: "Not additional: emissions reduction is regulatory requirement",
        }

        return CarbonCreditPackage(
            corridor_id=corridor_id,
            corridor_name=self.corridor_name,
            generated_at=datetime.utcnow(),
            baseline_co2_tonnes_year=round(baseline_co2_tonnes_year, 2),
            optimized_co2_tonnes_year=round(optimized_co2_tonnes_year, 2),
            reduction_tonnes_year=round(reduction, 2),
            reduction_pct=round(reduction_pct, 1),
            additionality=additionality,
            additionality_notes=notes_map[additionality],
            projections=projections,
            leakage_deduction_pct=leakage_pct,
        )

    @staticmethod
    def estimate_network_revenue(
        n_corridors: int,
        avg_reduction_tonnes_year: float,
        market: CreditMarket = CreditMarket.CARB_LCFS,
        crediting_years: int = 7,
    ) -> dict:
        """Quick network-scale revenue estimate for investor presentations."""
        price = MARKET_PRICE_USD_TONNE[market]
        discount = MARKET_DISCOUNT[market]
        annual_net = n_corridors * avg_reduction_tonnes_year * price * discount
        total_7yr = annual_net * crediting_years

        return {
            "n_corridors": n_corridors,
            "market": market.value,
            "annual_reduction_tonnes": round(n_corridors * avg_reduction_tonnes_year, 0),
            "annual_revenue_usd": round(annual_net),
            "7yr_revenue_usd": round(total_7yr),
            "price_per_tonne_usd": price,
        }


# ---------------------------------------------------------------------------
# MRV documentation helpers (Monitoring, Reporting, Verification)
# ---------------------------------------------------------------------------

@dataclass
class MRVRecord:
    """Monitoring, Reporting, Verification data record for one credit period."""
    corridor_id: str
    period_start: date
    period_end: date
    monitoring_method: str = "aito_probe_fusion"    # data source for MRV
    baseline_co2_tonnes: float = 0.0
    project_co2_tonnes: float = 0.0
    net_reduction_tonnes: float = 0.0
    data_quality_flag: str = "A"    # A=high, B=medium, C=low
    verified: bool = False
    verifier_name: Optional[str] = None
    verification_date: Optional[date] = None

    @property
    def reduction_pct(self) -> float:
        if self.baseline_co2_tonnes <= 0:
            return 0.0
        return (self.net_reduction_tonnes / self.baseline_co2_tonnes) * 100.0


def generate_mrv_report(
    package: CarbonCreditPackage,
    period_start: date,
    period_end: date,
) -> MRVRecord:
    """Generate an MRV record from a credit package."""
    period_days = (period_end - period_start).days
    period_years = period_days / 365.25

    return MRVRecord(
        corridor_id=package.corridor_id,
        period_start=period_start,
        period_end=period_end,
        baseline_co2_tonnes=round(package.baseline_co2_tonnes_year * period_years, 2),
        project_co2_tonnes=round(package.optimized_co2_tonnes_year * period_years, 2),
        net_reduction_tonnes=round(package.creditable_tonnes_year * period_years, 2),
        monitoring_method="aito_probe_fusion",
    )
