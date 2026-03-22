"""EPA-based emissions and fuel consumption calculator.

Uses official EPA idle emission factors for light-duty vehicles to compute
CO2, NOx, and fuel metrics from traffic simulation data. Factors sourced from:
- EPA AP-42 Section 13.2.1 (Light-Duty Gasoline Vehicles)
- EPA MOVES3 model emission rates for urban driving
- FHWA Highway Statistics Series
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class EmissionsReport:
    """Comprehensive emissions report from a simulation run."""
    total_co2_kg: float
    total_nox_g: float
    total_fuel_gallons: float
    total_fuel_liters: float
    co2_per_vehicle_kg: float
    fuel_per_vehicle_gallons: float
    idle_hours: float
    # Annualised projections
    annual_co2_tons: float
    annual_fuel_gallons: float
    annual_fuel_cost_usd: float
    # Comparison metrics
    co2_saved_vs_baseline_kg: float
    fuel_saved_vs_baseline_gallons: float
    trees_equivalent: float  # CO2 offset equivalent in trees per year
    homes_powered_equivalent: float  # CO2 equivalent in homes' electricity


# EPA emission factors (light-duty gasoline vehicle, idling)
EPA_FACTORS = {
    # CO2 emission rate for idling: ~8.887 kg CO2 per gallon of gasoline
    # Average idle fuel consumption: ~0.16 gallons/hour (EPA MOVES3)
    "idle_fuel_gph": 0.16,         # gallons per hour per vehicle, idling
    "idle_fuel_gps": 0.16 / 3600,  # gallons per second
    "co2_per_gallon_kg": 8.887,    # kg CO2 per gallon gasoline (EPA)
    "nox_per_gallon_g": 1.39,      # g NOx per gallon (EPA Tier 3 avg)
    "fuel_price_per_gallon": 4.89, # San Diego avg gas price 2024-2025
    "co2_per_tree_kg_year": 21.77, # kg CO2 absorbed per tree per year (EPA)
    "co2_per_home_kg_year": 7300,  # kg CO2 from avg US household electricity/yr
}


class EmissionsCalculator:
    """Calculate emissions from traffic simulation queue data.

    Input: time series of queue lengths (vehicles idling) at each step.
    Output: comprehensive emissions and fuel report.
    """

    def __init__(
        self,
        step_seconds: float = 1.0,
        factors: dict[str, float] | None = None,
    ) -> None:
        self.step_seconds = step_seconds
        self.factors = {**EPA_FACTORS, **(factors or {})}
        self._queue_log: list[float] = []
        self._departed_log: list[float] = []

    def record_step(self, total_queue: float, total_departed: float = 0.0) -> None:
        """Record a single simulation step."""
        self._queue_log.append(total_queue)
        self._departed_log.append(total_departed)

    def compute_step_emissions(self, total_queue: float) -> dict[str, float]:
        """Compute emissions for a single step (for live dashboard)."""
        idle_seconds = total_queue * self.step_seconds
        fuel_gal = idle_seconds * self.factors["idle_fuel_gps"]
        co2_kg = fuel_gal * self.factors["co2_per_gallon_kg"]
        nox_g = fuel_gal * self.factors["nox_per_gallon_g"]
        return {
            "co2_kg": co2_kg,
            "nox_g": nox_g,
            "fuel_gallons": fuel_gal,
            "idle_vehicle_seconds": idle_seconds,
        }

    def generate_report(
        self,
        simulation_hours: float | None = None,
        baseline_queue_log: list[float] | None = None,
    ) -> EmissionsReport:
        """Generate a comprehensive emissions report from accumulated data."""
        if not self._queue_log:
            return EmissionsReport(
                total_co2_kg=0, total_nox_g=0, total_fuel_gallons=0,
                total_fuel_liters=0, co2_per_vehicle_kg=0,
                fuel_per_vehicle_gallons=0, idle_hours=0,
                annual_co2_tons=0, annual_fuel_gallons=0,
                annual_fuel_cost_usd=0, co2_saved_vs_baseline_kg=0,
                fuel_saved_vs_baseline_gallons=0, trees_equivalent=0,
                homes_powered_equivalent=0,
            )

        queue_arr = np.array(self._queue_log)
        total_idle_vehicle_seconds = float(queue_arr.sum()) * self.step_seconds
        total_idle_hours = total_idle_vehicle_seconds / 3600.0

        # Fuel and emissions
        total_fuel_gal = total_idle_hours * self.factors["idle_fuel_gph"]
        total_fuel_liters = total_fuel_gal * 3.78541
        total_co2_kg = total_fuel_gal * self.factors["co2_per_gallon_kg"]
        total_nox_g = total_fuel_gal * self.factors["nox_per_gallon_g"]

        total_vehicles = max(sum(self._departed_log), 1)
        co2_per_vehicle = total_co2_kg / total_vehicles
        fuel_per_vehicle = total_fuel_gal / total_vehicles

        # Annualisation
        if simulation_hours is None:
            simulation_hours = len(self._queue_log) * self.step_seconds / 3600.0
        annual_factor = 365 * 16 / max(simulation_hours, 0.01)  # 16 operational hours/day

        annual_co2_tons = total_co2_kg * annual_factor / 1000
        annual_fuel_gal = total_fuel_gal * annual_factor
        annual_fuel_cost = annual_fuel_gal * self.factors["fuel_price_per_gallon"]

        # Baseline comparison
        co2_saved = 0.0
        fuel_saved = 0.0
        if baseline_queue_log is not None:
            baseline_idle = float(np.array(baseline_queue_log).sum()) * self.step_seconds / 3600.0
            baseline_fuel = baseline_idle * self.factors["idle_fuel_gph"]
            baseline_co2 = baseline_fuel * self.factors["co2_per_gallon_kg"]
            co2_saved = baseline_co2 - total_co2_kg
            fuel_saved = baseline_fuel - total_fuel_gal

        # Equivalencies
        trees = annual_co2_tons * 1000 / self.factors["co2_per_tree_kg_year"]
        homes = annual_co2_tons * 1000 / self.factors["co2_per_home_kg_year"]

        return EmissionsReport(
            total_co2_kg=total_co2_kg,
            total_nox_g=total_nox_g,
            total_fuel_gallons=total_fuel_gal,
            total_fuel_liters=total_fuel_liters,
            co2_per_vehicle_kg=co2_per_vehicle,
            fuel_per_vehicle_gallons=fuel_per_vehicle,
            idle_hours=total_idle_hours,
            annual_co2_tons=annual_co2_tons,
            annual_fuel_gallons=annual_fuel_gal,
            annual_fuel_cost_usd=annual_fuel_cost,
            co2_saved_vs_baseline_kg=co2_saved,
            fuel_saved_vs_baseline_gallons=fuel_saved,
            trees_equivalent=trees,
            homes_powered_equivalent=homes,
        )

    def reset(self) -> None:
        self._queue_log = []
        self._departed_log = []
