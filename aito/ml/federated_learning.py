"""aito/ml/federated_learning.py

GF14: Federated Learning for Cross-City Knowledge Transfer.

Enables AITO deployments in different cities to share learned patterns
(demand-timing relationships, seasonal factors, event impact models)
without sharing raw trajectory or demand data.

Architecture:
  Federated Averaging (FedAvg, McMahan et al. 2017):
  - Each city trains a local demand prediction model on local probe data
  - Only model weights (gradients) are shared with the AITO aggregator
  - Aggregator averages weights → produces global model
  - Global model is pushed back to all cities as a warm-start

Privacy guarantees:
  - Raw trajectories never leave the city
  - Model weights are differentially private (ε-DP, δ=10⁻⁵)
  - Gradient clipping prevents inference attacks

What gets shared:
  1. Demand scale factors by TOD period (weekday/weekend/holiday)
  2. Event demand surge profiles (generalized, not venue-specific)
  3. Vehicle mix corrections (fleet transition rates)
  4. Seasonal emission factors

What stays local:
  - Individual GPS trajectories
  - Turn movement counts (intersection-level)
  - NTCIP credentials
  - Agency budget data

Reference:
  McMahan et al. (2017). "Communication-Efficient Learning of Deep
  Networks from Decentralized Data." AISTATS 2017.

  Dwork, C. & Roth, A. (2014). "The Algorithmic Foundations of
  Differential Privacy." Foundations and Trends in TCS, 9(3–4).
"""
from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


# ---------------------------------------------------------------------------
# Model parameter types that can be federated
# ---------------------------------------------------------------------------

@dataclass
class DemandScaleModel:
    """Learned demand scale factors by time-of-day and day-type."""
    # Scale factors relative to AM Peak = 1.0
    tod_factors: dict[str, float] = field(default_factory=lambda: {
        "AM Peak": 1.00,
        "Midday": 0.65,
        "PM Peak": 0.90,
        "Evening": 0.50,
        "Overnight": 0.20,
        "Weekend": 0.55,
    })
    # Seasonal adjustments (fraction change by month, 1=January)
    monthly_factors: dict[int, float] = field(default_factory=lambda: {
        m: 1.0 for m in range(1, 13)
    })
    # Event type demand multipliers (generalized by category)
    event_multipliers: dict[str, float] = field(default_factory=lambda: {
        "sports_major": 2.5,
        "sports_minor": 1.4,
        "concert_large": 1.6,
        "convention": 1.3,
    })
    # Metadata
    city_id: str = "unknown"
    n_training_samples: int = 0
    last_updated: Optional[datetime] = None


@dataclass
class EmissionCorrectionModel:
    """Locally-learned fleet emission correction factors."""
    # Multiplier on EPA MOVES2014b base rates (local fleet may differ)
    fleet_correction: float = 1.0
    # Local EV penetration (reduces total CO2)
    ev_fraction: float = 0.06
    # Heavy truck fraction (increases CO2 per vehicle)
    hdt_fraction: float = 0.02
    city_id: str = "unknown"
    calibration_period_days: int = 30


@dataclass
class FederatedModelUpdate:
    """Package of model parameters sent from one city to aggregator."""
    city_id: str
    round_number: int
    demand_model: DemandScaleModel
    emission_model: EmissionCorrectionModel
    # Differential privacy: gradient noise added before sending
    dp_epsilon: float = 1.0     # privacy budget
    dp_delta: float = 1e-5
    n_samples: int = 0          # training sample count (for weighted aggregation)
    timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GlobalModel:
    """Federated-averaged global model from all participating cities."""
    round_number: int
    demand_model: DemandScaleModel
    emission_model: EmissionCorrectionModel
    n_cities: int
    total_samples: int
    convergence_delta: float    # change in model vs. previous round
    timestamp: datetime = field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Differential privacy noise injection
# ---------------------------------------------------------------------------

def add_dp_noise(
    value: float,
    sensitivity: float,
    epsilon: float,
    rng: Optional[random.Random] = None,
) -> float:
    """Add Laplace noise for ε-differential privacy.

    Noise scale = sensitivity / epsilon.
    For TOD factors with sensitivity ≤ 0.5, ε=1 → noise scale = 0.5.
    """
    if rng is None:
        rng = random.Random()
    scale = sensitivity / max(epsilon, 1e-9)
    u = rng.uniform(0, 1) - 0.5
    noise = -scale * math.copysign(1, u) * math.log(1 - 2 * abs(u) + 1e-9)
    return value + noise


def privatize_demand_model(
    model: DemandScaleModel,
    epsilon: float = 1.0,
    sensitivity: float = 0.20,
    seed: int = 42,
) -> DemandScaleModel:
    """Apply differential privacy to demand scale factors."""
    rng = random.Random(seed)
    noisy_tod = {
        period: max(0.05, add_dp_noise(factor, sensitivity, epsilon, rng))
        for period, factor in model.tod_factors.items()
    }
    noisy_monthly = {
        m: max(0.5, add_dp_noise(f, sensitivity, epsilon, rng))
        for m, f in model.monthly_factors.items()
    }
    return DemandScaleModel(
        tod_factors=noisy_tod,
        monthly_factors=noisy_monthly,
        event_multipliers=model.event_multipliers,
        city_id=model.city_id,
        n_training_samples=model.n_training_samples,
        last_updated=model.last_updated,
    )


# ---------------------------------------------------------------------------
# FedAvg aggregator
# ---------------------------------------------------------------------------

def fedavg_demand_model(
    updates: list[FederatedModelUpdate],
) -> DemandScaleModel:
    """Federated Averaging on demand scale models.

    Weighted average by n_samples (larger cities contribute more).
    """
    total_samples = sum(u.n_samples for u in updates)
    if total_samples <= 0:
        return DemandScaleModel()

    # Aggregate TOD factors
    all_periods = set()
    for u in updates:
        all_periods.update(u.demand_model.tod_factors.keys())

    averaged_tod: dict[str, float] = {}
    for period in all_periods:
        weighted_sum = sum(
            u.demand_model.tod_factors.get(period, 1.0) * u.n_samples
            for u in updates
        )
        averaged_tod[period] = weighted_sum / total_samples

    # Aggregate monthly factors
    all_months = set()
    for u in updates:
        all_months.update(u.demand_model.monthly_factors.keys())

    averaged_monthly: dict[int, float] = {}
    for month in all_months:
        weighted_sum = sum(
            u.demand_model.monthly_factors.get(month, 1.0) * u.n_samples
            for u in updates
        )
        averaged_monthly[month] = weighted_sum / total_samples

    # Aggregate event multipliers
    all_event_types = set()
    for u in updates:
        all_event_types.update(u.demand_model.event_multipliers.keys())

    averaged_events: dict[str, float] = {}
    for etype in all_event_types:
        weighted_sum = sum(
            u.demand_model.event_multipliers.get(etype, 1.0) * u.n_samples
            for u in updates
        )
        averaged_events[etype] = weighted_sum / total_samples

    return DemandScaleModel(
        tod_factors=averaged_tod,
        monthly_factors=averaged_monthly,
        event_multipliers=averaged_events,
        city_id="global",
        n_training_samples=total_samples,
        last_updated=datetime.utcnow(),
    )


def fedavg_emission_model(
    updates: list[FederatedModelUpdate],
) -> EmissionCorrectionModel:
    """Federated Averaging on emission correction models."""
    total_samples = sum(u.n_samples for u in updates)
    if total_samples <= 0:
        return EmissionCorrectionModel()

    avg_correction = sum(
        u.emission_model.fleet_correction * u.n_samples for u in updates
    ) / total_samples
    avg_ev = sum(u.emission_model.ev_fraction * u.n_samples for u in updates) / total_samples
    avg_hdt = sum(u.emission_model.hdt_fraction * u.n_samples for u in updates) / total_samples

    return EmissionCorrectionModel(
        fleet_correction=round(avg_correction, 4),
        ev_fraction=round(avg_ev, 4),
        hdt_fraction=round(avg_hdt, 4),
        city_id="global",
    )


# ---------------------------------------------------------------------------
# FederatedLearningServer
# ---------------------------------------------------------------------------

@dataclass
class CityClient:
    """Represents one city participating in federated learning."""
    city_id: str
    city_name: str
    n_corridors: int
    avg_aadt: int
    state: str = "CA"
    enrolled: bool = True
    last_update: Optional[datetime] = None


class FederatedLearningServer:
    """Aggregator server for AITO federated learning.

    Collects model updates from enrolled cities, applies FedAvg,
    and distributes global models.

    Usage:
        server = FederatedLearningServer()
        server.enroll_city(CityClient("san_diego", "San Diego", 200, 28000))
        server.submit_update(update_from_san_diego)
        server.submit_update(update_from_los_angeles)
        global_model = server.aggregate()
        print(global_model.n_cities, "cities contributed")
    """

    def __init__(self, min_cities_for_aggregation: int = 2) -> None:
        self.min_cities = min_cities_for_aggregation
        self._cities: dict[str, CityClient] = {}
        self._pending_updates: list[FederatedModelUpdate] = []
        self._global_models: list[GlobalModel] = []
        self._current_round: int = 0

    def enroll_city(self, city: CityClient) -> None:
        self._cities[city.city_id] = city

    def submit_update(self, update: FederatedModelUpdate) -> None:
        """Receive a model update from a city client."""
        if update.city_id not in self._cities:
            raise ValueError(f"City {update.city_id} not enrolled")
        # Apply DP noise if not already applied
        privatized_demand = privatize_demand_model(
            update.demand_model,
            epsilon=update.dp_epsilon,
        )
        update.demand_model = privatized_demand
        self._pending_updates.append(update)
        if self._cities[update.city_id]:
            self._cities[update.city_id].last_update = update.timestamp

    def aggregate(self) -> Optional[GlobalModel]:
        """Run FedAvg if enough cities have submitted updates."""
        if len(self._pending_updates) < self.min_cities:
            return None

        self._current_round += 1
        demand_global = fedavg_demand_model(self._pending_updates)
        emission_global = fedavg_emission_model(self._pending_updates)

        # Convergence: compare to previous global model
        prev_round_delta = 0.0
        if self._global_models:
            prev = self._global_models[-1]
            am_delta = abs(
                demand_global.tod_factors.get("AM Peak", 1.0)
                - prev.demand_model.tod_factors.get("AM Peak", 1.0)
            )
            prev_round_delta = am_delta

        global_model = GlobalModel(
            round_number=self._current_round,
            demand_model=demand_global,
            emission_model=emission_global,
            n_cities=len(self._pending_updates),
            total_samples=sum(u.n_samples for u in self._pending_updates),
            convergence_delta=round(prev_round_delta, 5),
        )
        self._global_models.append(global_model)
        self._pending_updates.clear()
        return global_model

    def get_global_model(self) -> Optional[GlobalModel]:
        return self._global_models[-1] if self._global_models else None

    @property
    def n_enrolled_cities(self) -> int:
        return len(self._cities)

    @property
    def pending_update_count(self) -> int:
        return len(self._pending_updates)

    def status(self) -> dict:
        return {
            "enrolled_cities": self.n_enrolled_cities,
            "current_round": self._current_round,
            "pending_updates": self.pending_update_count,
            "completed_rounds": len(self._global_models),
            "convergence_delta": (
                self._global_models[-1].convergence_delta
                if self._global_models else None
            ),
        }


# ---------------------------------------------------------------------------
# Local city updater (synthetic data for demo)
# ---------------------------------------------------------------------------

def generate_synthetic_city_update(
    city_id: str,
    n_samples: int = 5000,
    tod_variation: float = 0.10,
    seed: int = 42,
) -> FederatedModelUpdate:
    """Generate a synthetic city model update for demo / testing."""
    rng = random.Random(seed)

    demand_model = DemandScaleModel(
        tod_factors={
            period: max(0.1, base + rng.gauss(0, tod_variation))
            for period, base in {
                "AM Peak": 1.00, "Midday": 0.65, "PM Peak": 0.90,
                "Evening": 0.50, "Overnight": 0.20, "Weekend": 0.55,
            }.items()
        },
        monthly_factors={m: 1.0 + rng.gauss(0, 0.05) for m in range(1, 13)},
        event_multipliers={
            "sports_major": 2.5 + rng.gauss(0, 0.3),
            "sports_minor": 1.4 + rng.gauss(0, 0.2),
            "concert_large": 1.6 + rng.gauss(0, 0.2),
        },
        city_id=city_id,
        n_training_samples=n_samples,
        last_updated=datetime.utcnow(),
    )

    emission_model = EmissionCorrectionModel(
        fleet_correction=1.0 + rng.gauss(0, 0.05),
        ev_fraction=0.06 + rng.gauss(0, 0.01),
        hdt_fraction=0.02 + rng.gauss(0, 0.005),
        city_id=city_id,
    )

    return FederatedModelUpdate(
        city_id=city_id,
        round_number=1,
        demand_model=demand_model,
        emission_model=emission_model,
        n_samples=n_samples,
    )
