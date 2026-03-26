"""traffic_ai/simulation_engine/emissions.py

Per-step fuel consumption and CO₂ emissions estimator.

Uses EPA MOVES3 idle fuel consumption factors to convert simulation queue
lengths into real-world fuel and CO₂ equivalents. Designed to be called once
per simulation step by the engine so that ``StepMetrics`` carries accurate
environmental data.

EPA References
--------------
- Idle fuel: 0.16 gal/hr/vehicle (EPA MOVES3, light-duty gasoline)
- CO₂ per gallon: 8.887 kg (EPA)
- NOx per gallon: 1.39 g (EPA Tier 3 average)
- Stop-start fuel penalty: 0.0022 gal/stop-event per vehicle
- Moving fuel at ~30 mph: 0.04 gal/mile
"""
from __future__ import annotations


class EmissionsCalculator:
    """Compute fuel and CO₂ emissions from simulation step data.

    All constants are sourced from EPA MOVES3 and EPA greenhouse gas
    equivalency calculator (2024 edition).

    Parameters
    ----------
    idle_fuel_rate_gal_hr:
        Gallons per hour of idle fuel consumption per vehicle.
    stop_penalty_gal:
        Additional fuel (gallons) per stop-start cycle per vehicle.
    co2_per_gallon_kg:
        Kilograms of CO₂ per gallon of gasoline combusted.
    moving_fuel_rate_gal_mile:
        Gallons per mile for a vehicle moving through an intersection.
    intersection_distance_miles:
        Approximate distance of an intersection zone in miles (short segment).
    """

    IDLE_FUEL_RATE: float = 0.16           # gal/hr/vehicle
    STOP_PENALTY_FUEL: float = 0.0022      # extra gal per stop event
    CO2_PER_GALLON: float = 8.887          # kg CO₂/gallon gasoline
    MOVING_FUEL_RATE: float = 0.04         # gal/mile at ~30 mph
    INTERSECTION_DISTANCE_MILES: float = 0.01  # ~50 feet

    def __init__(
        self,
        idle_fuel_rate_gal_hr: float = 0.16,
        stop_penalty_gal: float = 0.0022,
        co2_per_gallon_kg: float = 8.887,
        moving_fuel_rate_gal_mile: float = 0.04,
        intersection_distance_miles: float = 0.01,
    ) -> None:
        self.idle_fuel_rate = idle_fuel_rate_gal_hr
        self.stop_penalty = stop_penalty_gal
        self.co2_per_gallon = co2_per_gallon_kg
        self.moving_fuel_rate = moving_fuel_rate_gal_mile
        self.intersection_dist = intersection_distance_miles

    def compute_step(
        self,
        total_queue: float,
        departures: float,
        phase_changes: int,
        step_seconds: float,
    ) -> tuple[float, float]:
        """Compute fuel consumption and CO₂ for one simulation step.

        Parameters
        ----------
        total_queue:
            Number of vehicles currently queued (idle) across all lanes.
        departures:
            Number of vehicles that departed (moved) this step.
        phase_changes:
            Number of signal phase changes this step (stop-start events).
        step_seconds:
            Duration of the simulation step in seconds.

        Returns
        -------
        (fuel_gallons, co2_kg)
            Tuple of total fuel consumed and CO₂ emitted this step.
        """
        # Idle fuel: vehicles in queue × hours idling × idle rate
        idle_hours = total_queue * (step_seconds / 3600.0)
        idle_fuel = idle_hours * self.idle_fuel_rate

        # Stop-start penalty: phase changes cause acceleration events
        # Approximate fraction of queued vehicles affected per phase change
        stop_fuel = phase_changes * self.stop_penalty * max(total_queue * 0.1, 0.0)

        # Moving fuel: vehicles that cleared the intersection
        moving_fuel = departures * self.moving_fuel_rate * self.intersection_dist

        total_fuel_gallons = idle_fuel + stop_fuel + moving_fuel
        co2_kg = total_fuel_gallons * self.co2_per_gallon

        return total_fuel_gallons, co2_kg

    def annualise(
        self,
        total_fuel_gallons: float,
        simulation_steps: int,
        step_seconds: float,
        operational_hours_per_day: float = 16.0,
    ) -> dict[str, float]:
        """Scale simulation totals to annual real-world estimates.

        Parameters
        ----------
        total_fuel_gallons:
            Fuel consumed over the entire simulation run.
        simulation_steps:
            Total number of steps in the simulation.
        step_seconds:
            Seconds per step.
        operational_hours_per_day:
            Hours per day the intersection is active.

        Returns
        -------
        dict with keys:
            annual_fuel_gallons, annual_co2_tons, annual_cost_usd,
            trees_equivalent, homes_powered_equivalent
        """
        sim_hours = simulation_steps * step_seconds / 3600.0
        if sim_hours < 1e-9:
            return {}
        scale = (365.0 * operational_hours_per_day) / sim_hours
        annual_fuel = total_fuel_gallons * scale
        annual_co2_kg = annual_fuel * self.co2_per_gallon
        annual_co2_tons = annual_co2_kg / 1000.0
        annual_cost = annual_fuel * 3.50  # $3.50/gallon
        trees_equiv = annual_co2_kg / 22.0          # ~22 kg/tree/year
        homes_equiv = annual_co2_kg / 7_300.0       # ~7,300 kg/household/year
        return {
            "annual_fuel_gallons": annual_fuel,
            "annual_co2_tons": annual_co2_tons,
            "annual_cost_usd": annual_cost,
            "trees_equivalent": trees_equiv,
            "homes_powered_equivalent": homes_equiv,
        }
