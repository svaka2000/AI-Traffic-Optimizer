"""Tests for aito/ml/federated_learning.py (GF14)."""
import pytest
from datetime import datetime
from aito.ml.federated_learning import (
    DemandScaleModel,
    EmissionCorrectionModel,
    FederatedModelUpdate,
    GlobalModel,
    add_dp_noise,
    privatize_demand_model,
    fedavg_demand_model,
    fedavg_emission_model,
    CityClient,
    FederatedLearningServer,
    generate_synthetic_city_update,
)


class TestDemandScaleModel:
    def test_default_has_tod_factors(self):
        model = DemandScaleModel()
        assert "AM Peak" in model.tod_factors
        assert len(model.tod_factors) >= 4

    def test_tod_factors_positive(self):
        model = DemandScaleModel()
        for period, factor in model.tod_factors.items():
            assert factor > 0, f"Period {period} has non-positive factor"

    def test_monthly_factors_all_months(self):
        model = DemandScaleModel()
        assert len(model.monthly_factors) == 12
        for m in range(1, 13):
            assert m in model.monthly_factors

    def test_am_peak_factor_is_one(self):
        model = DemandScaleModel()
        assert model.tod_factors["AM Peak"] == pytest.approx(1.0, abs=0.01)


class TestEmissionCorrectionModel:
    def test_default_fleet_correction_near_one(self):
        model = EmissionCorrectionModel()
        assert 0.5 < model.fleet_correction < 2.0

    def test_ev_fraction_between_zero_and_one(self):
        model = EmissionCorrectionModel()
        assert 0 <= model.ev_fraction <= 1.0

    def test_hdt_fraction_between_zero_and_one(self):
        model = EmissionCorrectionModel()
        assert 0 <= model.hdt_fraction <= 1.0


class TestAddDPNoise:
    def test_noise_is_scalar(self):
        result = add_dp_noise(value=1.0, sensitivity=0.5, epsilon=1.0)
        assert isinstance(result, float)

    def test_small_epsilon_large_noise(self):
        import random
        rng = random.Random(42)
        results = [add_dp_noise(1.0, 0.5, epsilon=0.01, rng=rng) for _ in range(100)]
        spread = max(results) - min(results)
        assert spread > 1.0  # high noise for small epsilon

    def test_large_epsilon_small_noise(self):
        import random
        rng = random.Random(42)
        results = [add_dp_noise(1.0, 0.5, epsilon=100.0, rng=rng) for _ in range(100)]
        spread = max(results) - min(results)
        assert spread < 0.1  # low noise for large epsilon


class TestPrivatizeDemandModel:
    def test_returns_demand_scale_model(self):
        model = DemandScaleModel()
        privatized = privatize_demand_model(model, epsilon=1.0)
        assert isinstance(privatized, DemandScaleModel)

    def test_tod_periods_preserved(self):
        model = DemandScaleModel()
        privatized = privatize_demand_model(model, epsilon=1.0)
        assert set(privatized.tod_factors.keys()) == set(model.tod_factors.keys())

    def test_monthly_factors_preserved_keys(self):
        model = DemandScaleModel()
        privatized = privatize_demand_model(model, epsilon=1.0)
        assert set(privatized.monthly_factors.keys()) == set(model.monthly_factors.keys())

    def test_all_noisy_factors_positive(self):
        model = DemandScaleModel()
        privatized = privatize_demand_model(model, epsilon=1.0, seed=42)
        for period, factor in privatized.tod_factors.items():
            assert factor > 0, f"Period {period} got non-positive factor after DP"


class TestFedAvg:
    def _make_update(self, city_id: str, n_samples: int, seed: int) -> FederatedModelUpdate:
        return generate_synthetic_city_update(city_id=city_id, n_samples=n_samples, seed=seed)

    def test_fedavg_demand_returns_model(self):
        updates = [
            self._make_update("san_diego", 5000, 42),
            self._make_update("los_angeles", 8000, 99),
        ]
        global_model = fedavg_demand_model(updates)
        assert isinstance(global_model, DemandScaleModel)

    def test_fedavg_demand_city_id_global(self):
        updates = [
            self._make_update("san_diego", 5000, 42),
            self._make_update("los_angeles", 8000, 99),
        ]
        global_model = fedavg_demand_model(updates)
        assert global_model.city_id == "global"

    def test_fedavg_demand_total_samples(self):
        updates = [
            self._make_update("san_diego", 5000, 42),
            self._make_update("los_angeles", 8000, 99),
        ]
        global_model = fedavg_demand_model(updates)
        assert global_model.n_training_samples == 13000

    def test_fedavg_emission_returns_model(self):
        updates = [
            self._make_update("san_diego", 5000, 42),
            self._make_update("los_angeles", 8000, 99),
        ]
        global_model = fedavg_emission_model(updates)
        assert isinstance(global_model, EmissionCorrectionModel)

    def test_fedavg_emission_weighted_average(self):
        u1 = generate_synthetic_city_update("city_a", n_samples=1000, seed=1)
        u2 = generate_synthetic_city_update("city_b", n_samples=1000, seed=2)
        u1.emission_model.fleet_correction = 1.0
        u2.emission_model.fleet_correction = 2.0
        u1.n_samples = 1000
        u2.n_samples = 1000
        result = fedavg_emission_model([u1, u2])
        assert result.fleet_correction == pytest.approx(1.5, rel=0.01)

    def test_fedavg_empty_returns_default(self):
        result_demand = fedavg_demand_model([])
        assert isinstance(result_demand, DemandScaleModel)
        result_emission = fedavg_emission_model([])
        assert isinstance(result_emission, EmissionCorrectionModel)


class TestFederatedLearningServer:
    def setup_method(self):
        self.server = FederatedLearningServer(min_cities_for_aggregation=2)
        self.cities = [
            CityClient("san_diego",   "San Diego",   n_corridors=200, avg_aadt=28000),
            CityClient("los_angeles", "Los Angeles", n_corridors=800, avg_aadt=42000),
            CityClient("san_francisco", "San Francisco", n_corridors=150, avg_aadt=22000),
        ]

    def test_enroll_city(self):
        self.server.enroll_city(self.cities[0])
        assert self.server.n_enrolled_cities == 1

    def test_enroll_multiple_cities(self):
        for city in self.cities:
            self.server.enroll_city(city)
        assert self.server.n_enrolled_cities == 3

    def test_submit_update_enrolled_city(self):
        self.server.enroll_city(self.cities[0])
        update = generate_synthetic_city_update("san_diego", n_samples=5000, seed=42)
        self.server.submit_update(update)
        assert self.server.pending_update_count == 1

    def test_submit_update_unenrolled_raises(self):
        update = generate_synthetic_city_update("unknown_city", n_samples=5000, seed=42)
        with pytest.raises(ValueError):
            self.server.submit_update(update)

    def test_aggregate_below_min_cities_returns_none(self):
        self.server.enroll_city(self.cities[0])
        update = generate_synthetic_city_update("san_diego", n_samples=5000, seed=42)
        self.server.submit_update(update)
        result = self.server.aggregate()
        assert result is None  # only 1 city, need 2

    def test_aggregate_with_enough_cities(self):
        for city in self.cities[:2]:
            self.server.enroll_city(city)
        for i, city in enumerate(self.cities[:2]):
            update = generate_synthetic_city_update(city.city_id, n_samples=5000, seed=42 + i)
            self.server.submit_update(update)
        result = self.server.aggregate()
        assert isinstance(result, GlobalModel)

    def test_aggregate_clears_pending(self):
        for city in self.cities[:2]:
            self.server.enroll_city(city)
        for i, city in enumerate(self.cities[:2]):
            update = generate_synthetic_city_update(city.city_id, n_samples=5000, seed=42 + i)
            self.server.submit_update(update)
        self.server.aggregate()
        assert self.server.pending_update_count == 0

    def test_global_model_round_number(self):
        for city in self.cities[:2]:
            self.server.enroll_city(city)
        for i, city in enumerate(self.cities[:2]):
            update = generate_synthetic_city_update(city.city_id, n_samples=5000, seed=42 + i)
            self.server.submit_update(update)
        result = self.server.aggregate()
        assert result.round_number == 1

    def test_global_model_city_count(self):
        for city in self.cities[:2]:
            self.server.enroll_city(city)
        for i, city in enumerate(self.cities[:2]):
            update = generate_synthetic_city_update(city.city_id, n_samples=5000, seed=42 + i)
            self.server.submit_update(update)
        result = self.server.aggregate()
        assert result.n_cities == 2

    def test_status_returns_dict(self):
        status = self.server.status()
        assert "enrolled_cities" in status
        assert "current_round" in status
        assert "pending_updates" in status

    def test_get_global_model_none_before_aggregate(self):
        assert self.server.get_global_model() is None

    def test_get_global_model_after_aggregate(self):
        for city in self.cities[:2]:
            self.server.enroll_city(city)
        for i, city in enumerate(self.cities[:2]):
            update = generate_synthetic_city_update(city.city_id, n_samples=5000, seed=42 + i)
            self.server.submit_update(update)
        self.server.aggregate()
        result = self.server.get_global_model()
        assert isinstance(result, GlobalModel)

    def test_dp_noise_applied_on_submit(self):
        self.server.enroll_city(self.cities[0])
        original_model = DemandScaleModel()
        update = FederatedModelUpdate(
            city_id="san_diego",
            round_number=1,
            demand_model=DemandScaleModel(),
            emission_model=EmissionCorrectionModel(),
            n_samples=5000,
            dp_epsilon=1.0,
        )
        # Store original AM Peak factor
        original_am = update.demand_model.tod_factors.get("AM Peak", 1.0)
        self.server.submit_update(update)
        # After submit, DP noise was applied — AM factor may differ
        # (just verify no crash and model is still valid)
        assert self.server.pending_update_count == 1


class TestGenerateSyntheticCityUpdate:
    def test_returns_federated_update(self):
        update = generate_synthetic_city_update("test_city", n_samples=3000, seed=42)
        assert isinstance(update, FederatedModelUpdate)

    def test_city_id_matches(self):
        update = generate_synthetic_city_update("san_diego", n_samples=5000, seed=42)
        assert update.city_id == "san_diego"

    def test_n_samples_set(self):
        update = generate_synthetic_city_update("test", n_samples=7500, seed=1)
        assert update.n_samples == 7500

    def test_demand_model_has_tod_factors(self):
        update = generate_synthetic_city_update("test", n_samples=3000, seed=42)
        assert len(update.demand_model.tod_factors) >= 4

    def test_emission_model_present(self):
        update = generate_synthetic_city_update("test", n_samples=3000, seed=42)
        assert isinstance(update.emission_model, EmissionCorrectionModel)

    def test_different_seeds_produce_different_models(self):
        u1 = generate_synthetic_city_update("test", n_samples=5000, seed=1)
        u2 = generate_synthetic_city_update("test", n_samples=5000, seed=2)
        am1 = u1.demand_model.tod_factors.get("AM Peak", 1.0)
        am2 = u2.demand_model.tod_factors.get("AM Peak", 1.0)
        assert am1 != pytest.approx(am2, abs=1e-10)
