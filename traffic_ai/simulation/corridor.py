"""Corridor simulation: 5-intersection linear corridor (e.g., El Camino Real, San Diego).

Models green wave propagation, emergency vehicle preemption, and realistic
San Diego corridor geometry with configurable intersection spacing.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


DIRECTIONS = ["N", "S", "E", "W"]
_OPPOSITE = {"N": "S", "S": "N", "E": "W", "W": "E"}

# San Diego corridor reference data
SD_CORRIDORS = {
    "el_camino_real": {
        "name": "El Camino Real",
        "intersections": [
            "El Camino Real & Carmel Valley Rd",
            "El Camino Real & Del Mar Heights Rd",
            "El Camino Real & Via de la Valle",
            "El Camino Real & Manchester Ave",
            "El Camino Real & Birmingham Dr",
        ],
        "spacing_m": [420, 380, 510, 350],  # meters between intersections
        "speed_limit_mph": 40,
        "aadt": 32000,  # annual average daily traffic
    },
    "university_ave": {
        "name": "University Ave Corridor",
        "intersections": [
            "University & I-15 Ramps",
            "University & Fairmount Ave",
            "University & Euclid Ave",
            "University & 54th St",
            "University & College Ave",
        ],
        "spacing_m": [480, 390, 420, 360],
        "speed_limit_mph": 35,
        "aadt": 28000,
    },
    "i5_corridor": {
        "name": "I-5 Corridor (Surface)",
        "intersections": [
            "I-5 & Palomar Airport Rd",
            "I-5 & Poinsettia Ln",
            "I-5 & Cannon Rd",
            "I-5 & Oceanside Blvd",
            "I-5 & Mission Ave",
        ],
        "spacing_m": [1200, 950, 1100, 800],
        "speed_limit_mph": 45,
        "aadt": 45000,
    },
}


@dataclass
class CorridorIntersection:
    """State of a single corridor intersection."""
    node_id: int
    name: str
    lanes: int
    max_queue: int
    queue_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    current_phase: int = 0  # 0 = NS green (corridor direction), 1 = EW green (cross street)
    phase_elapsed: int = 0
    phase_changes: int = 0
    total_arrivals: int = 0
    total_departures: int = 0
    cumulative_wait: float = 0.0
    pending_inflow: dict[str, float] = field(default_factory=dict)
    # Emergency vehicle tracking
    emergency_active: bool = False
    emergency_direction: str = "N"  # direction EV is approaching from
    emergency_countdown: int = 0

    def __post_init__(self) -> None:
        if not self.queue_matrix:
            self.queue_matrix = {
                d: np.zeros(self.lanes, dtype=np.float64) for d in DIRECTIONS
            }

    @property
    def queue_ns(self) -> float:
        return float(self.queue_matrix["N"].sum() + self.queue_matrix["S"].sum())

    @property
    def queue_ew(self) -> float:
        return float(self.queue_matrix["E"].sum() + self.queue_matrix["W"].sum())

    @property
    def total_queue(self) -> float:
        return self.queue_ns + self.queue_ew

    def observe(self, step: int) -> dict[str, float]:
        return {
            "node_id": float(self.node_id),
            "step": float(step),
            "queue_ns": self.queue_ns,
            "queue_ew": self.queue_ew,
            "total_queue": self.total_queue,
            "current_phase": float(self.current_phase),
            "phase_elapsed": float(self.phase_elapsed),
            "emergency_active": 1.0 if self.emergency_active else 0.0,
            "emergency_countdown": float(self.emergency_countdown),
            "avg_speed": max(0.0, 60.0 - self.total_queue * 0.5),
            "lane_occupancy": min(1.0, self.total_queue / max(1.0, self.lanes * 4 * self.max_queue)),
            "wait_time": self.cumulative_wait / max(1.0, self.total_departures),
        }


@dataclass
class EmergencyVehicle:
    """Represents an emergency vehicle traversing the corridor."""
    vehicle_id: str
    direction: str  # "N" or "S" for corridor direction
    current_intersection: int
    target_intersection: int
    speed_factor: float = 1.5  # moves faster than normal traffic
    active: bool = True
    total_delay: float = 0.0


@dataclass
class CorridorMetrics:
    """Step-level metrics for the corridor simulation."""
    step: int
    total_queue: float
    avg_wait_sec: float
    throughput: float
    green_wave_efficiency: float  # fraction of vehicles hitting green
    avg_speed_mph: float
    co2_kg: float
    fuel_gallons: float
    emergency_delay_sec: float
    queue_per_intersection: list[float] = field(default_factory=list)


class CorridorSimulation:
    """5-intersection linear corridor for green wave optimization.

    Simulates a real San Diego corridor with:
    - Realistic intersection spacing and speed limits
    - Platoon dispersion between intersections
    - Emergency vehicle preemption (cascade green)
    - Green wave efficiency measurement
    - EPA-based emissions calculations
    """

    def __init__(
        self,
        corridor_name: str = "el_camino_real",
        n_intersections: int = 5,
        lanes_per_approach: int = 2,
        max_queue: int = 60,
        max_steps: int = 3600,  # 1 hour at 1 step/sec
        step_seconds: float = 1.0,
        seed: int = 42,
    ) -> None:
        self.corridor_data = SD_CORRIDORS.get(corridor_name, SD_CORRIDORS["el_camino_real"])
        self.n_intersections = min(n_intersections, len(self.corridor_data["intersections"]))
        self.lanes = lanes_per_approach
        self.max_queue = max_queue
        self.max_steps = max_steps
        self.step_seconds = step_seconds
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.corridor_name = corridor_name

        # Intersection spacing in seconds of travel time
        speed_mps = self.corridor_data["speed_limit_mph"] * 0.44704
        spacings = self.corridor_data["spacing_m"][: self.n_intersections - 1]
        self.travel_times = [s / speed_mps for s in spacings]

        self._intersections: dict[int, CorridorIntersection] = {}
        self._emergency_vehicles: list[EmergencyVehicle] = []
        self._step_count = 0
        self._metrics_log: list[CorridorMetrics] = []
        self._green_hits = 0
        self._total_arrivals_corridor = 0

        # EPA idle emission factors
        self.co2_per_idle_vehicle_per_sec = 0.00417  # kg CO2 (EPA average light-duty)
        self.fuel_per_idle_vehicle_per_sec = 0.000472  # gallons

        self._build_corridor()

    def reset(self, seed: int | None = None) -> dict[int, dict[str, float]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._metrics_log = []
        self._emergency_vehicles = []
        self._green_hits = 0
        self._total_arrivals_corridor = 0
        self._build_corridor()
        return self._collect_obs()

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, dict[str, float]], float, bool, dict[str, Any]]:
        """Advance corridor simulation by one step."""

        # 1. Apply pending inflow (vehicles arriving from upstream intersection)
        self._apply_pending_inflow()

        # 2. Update signal phases
        for node_id, node in self._intersections.items():
            # Emergency preemption overrides controller
            if node.emergency_active and node.emergency_countdown > 0:
                # Force green in emergency vehicle direction
                ev_phase = 0 if node.emergency_direction in ("N", "S") else 1
                if node.current_phase != ev_phase:
                    node.current_phase = ev_phase
                    node.phase_elapsed = 0
                    node.phase_changes += 1
                else:
                    node.phase_elapsed += 1
                node.emergency_countdown -= 1
                if node.emergency_countdown <= 0:
                    node.emergency_active = False
            else:
                requested = int(actions.get(node_id, node.current_phase))
                if requested != node.current_phase:
                    node.current_phase = requested
                    node.phase_elapsed = 0
                    node.phase_changes += 1
                else:
                    node.phase_elapsed += 1

        # 3. Stochastic arrivals
        hour = self._current_hour()
        for node in self._intersections.values():
            for direction in DIRECTIONS:
                self._sample_arrivals(node, direction, hour)

        # 4. Service and propagate
        self._service_and_propagate()

        # 5. Update emergency vehicles
        self._update_emergency_vehicles()

        # 6. Compute metrics
        metrics = self._compute_step_metrics()
        self._metrics_log.append(metrics)

        total_queue = metrics.total_queue
        reward = -float(total_queue)

        self._step_count += 1
        done = self._step_count >= self.max_steps

        obs = self._collect_obs()
        info = {
            "step": self._step_count,
            "total_queue": total_queue,
            "avg_wait": metrics.avg_wait_sec,
            "throughput": metrics.throughput,
            "co2_kg": metrics.co2_kg,
            "fuel_gallons": metrics.fuel_gallons,
            "green_wave_efficiency": metrics.green_wave_efficiency,
            "avg_speed_mph": metrics.avg_speed_mph,
            "emergency_delay": metrics.emergency_delay_sec,
            "queue_per_intersection": metrics.queue_per_intersection,
        }
        return obs, reward, done, info

    def inject_emergency_vehicle(
        self,
        start_intersection: int = 0,
        end_intersection: int | None = None,
        direction: str = "S",
        preemption_duration: int = 20,
    ) -> str:
        """Inject an emergency vehicle into the corridor with cascading preemption."""
        if end_intersection is None:
            end_intersection = self.n_intersections - 1
        ev_id = f"EV-{len(self._emergency_vehicles):03d}"
        ev = EmergencyVehicle(
            vehicle_id=ev_id,
            direction=direction,
            current_intersection=start_intersection,
            target_intersection=end_intersection,
        )
        self._emergency_vehicles.append(ev)

        # Cascade preemption ahead of the vehicle
        if direction == "S":
            lookahead = range(start_intersection, min(end_intersection + 1, self.n_intersections))
        else:
            lookahead = range(start_intersection, max(end_intersection - 1, -1), -1)

        for i, int_id in enumerate(lookahead):
            node = self._intersections.get(int_id)
            if node:
                delay = int(i * (self.travel_times[0] if self.travel_times else 10) / self.step_seconds)
                node.emergency_active = True
                node.emergency_direction = direction
                node.emergency_countdown = preemption_duration + delay

        return ev_id

    def get_results(self) -> dict[str, Any]:
        """Return comprehensive corridor simulation results."""
        if not self._metrics_log:
            return {}

        total_co2 = sum(m.co2_kg for m in self._metrics_log)
        total_fuel = sum(m.fuel_gallons for m in self._metrics_log)
        avg_wait = np.mean([m.avg_wait_sec for m in self._metrics_log])
        avg_throughput = np.mean([m.throughput for m in self._metrics_log])
        avg_gwe = np.mean([m.green_wave_efficiency for m in self._metrics_log])
        avg_speed = np.mean([m.avg_speed_mph for m in self._metrics_log])
        max_queue = max(m.total_queue for m in self._metrics_log)
        avg_queue = np.mean([m.total_queue for m in self._metrics_log])
        total_ev_delay = sum(ev.total_delay for ev in self._emergency_vehicles)

        return {
            "corridor_name": self.corridor_data["name"],
            "n_intersections": self.n_intersections,
            "total_steps": self._step_count,
            "avg_wait_sec": float(avg_wait),
            "avg_throughput": float(avg_throughput),
            "avg_queue": float(avg_queue),
            "max_queue": float(max_queue),
            "total_co2_kg": float(total_co2),
            "total_fuel_gallons": float(total_fuel),
            "co2_tons_per_year": float(total_co2 * 365 * 24 / max(self._step_count, 1) / 1000),
            "fuel_saved_gallons_per_year": float(total_fuel * 365 * 24 / max(self._step_count, 1)),
            "green_wave_efficiency": float(avg_gwe),
            "avg_speed_mph": float(avg_speed),
            "emergency_total_delay_sec": float(total_ev_delay),
            "metrics_log": self._metrics_log,
        }

    # --- Internal methods ---

    def _build_corridor(self) -> None:
        names = self.corridor_data["intersections"][: self.n_intersections]
        self._intersections = {}
        for i, name in enumerate(names):
            self._intersections[i] = CorridorIntersection(
                node_id=i,
                name=name,
                lanes=self.lanes,
                max_queue=self.max_queue,
            )

    def _current_hour(self) -> float:
        elapsed_s = self._step_count * self.step_seconds
        return (7.0 + elapsed_s / 3600.0) % 24.0  # Start at 7 AM

    def _arrival_rate(self, hour: float, direction: str) -> float:
        """San Diego-calibrated arrival rate using AADT and time-of-day factors."""
        aadt = self.corridor_data["aadt"]
        hourly_base = aadt / 24 / 3600 / (self.lanes * 2)  # veh/sec/lane

        # San Diego time-of-day profile (calibrated from Caltrans PeMS District 11)
        morning = math.exp(-((hour - 7.5) ** 2) / 2.5)
        evening = math.exp(-((hour - 17.0) ** 2) / 3.0)
        midday = 0.3 * math.exp(-((hour - 12.5) ** 2) / 4.0)
        time_factor = 1.0 + 2.0 * max(morning, evening) + midday

        # Directional bias (NS heavier on corridor)
        dir_factor = 1.2 if direction in ("N", "S") else 0.85

        return max(0.02, hourly_base * time_factor * dir_factor) * self.step_seconds

    def _sample_arrivals(self, node: CorridorIntersection, direction: str, hour: float) -> None:
        rate = self._arrival_rate(hour, direction)
        for lane in range(self.lanes):
            n = int(self.rng.poisson(rate))
            available = max(0, node.max_queue - int(node.queue_matrix[direction][lane]))
            actual = min(n, available)
            node.queue_matrix[direction][lane] += actual
            node.total_arrivals += actual
            self._total_arrivals_corridor += actual

    def _service_and_propagate(self) -> None:
        sat_rate = 0.50  # veh/sec/lane (slightly higher than grid for arterial)
        green_dirs = {0: ["N", "S"], 1: ["E", "W"]}

        flow_downstream: dict[tuple[int, str], float] = defaultdict(float)

        for node_id, node in self._intersections.items():
            active_dirs = green_dirs[node.current_phase]
            for direction in DIRECTIONS:
                for lane in range(self.lanes):
                    q = node.queue_matrix[direction][lane]
                    if direction in active_dirs:
                        cap = float(self.rng.poisson(sat_rate * self.step_seconds))
                        moved = min(q, cap)
                        # Track green wave hits
                        if moved > 0 and direction in ("N", "S"):
                            self._green_hits += int(moved)
                    else:
                        moved = 0.0
                    node.queue_matrix[direction][lane] -= moved
                    node.total_departures += int(moved)

                    # Propagate downstream in corridor
                    if moved > 0 and direction in ("N", "S"):
                        # Determine downstream neighbor
                        if direction == "S" and node_id < self.n_intersections - 1:
                            downstream_id = node_id + 1
                        elif direction == "N" and node_id > 0:
                            downstream_id = node_id - 1
                        else:
                            continue
                        # Platoon dispersion: ~60% of vehicles reach next intersection
                        transfer = moved * 0.60
                        incoming_dir = _OPPOSITE[direction]
                        flow_downstream[(downstream_id, incoming_dir)] += transfer

            node.cumulative_wait += node.total_queue * self.step_seconds

        for (nid, direction), volume in flow_downstream.items():
            self._intersections[nid].pending_inflow[direction] = (
                self._intersections[nid].pending_inflow.get(direction, 0.0) + volume
            )

    def _apply_pending_inflow(self) -> None:
        for node in self._intersections.values():
            for direction, incoming in node.pending_inflow.items():
                lane = int(self.rng.integers(0, self.lanes))
                node.queue_matrix[direction][lane] = min(
                    float(node.max_queue),
                    node.queue_matrix[direction][lane] + incoming,
                )
            node.pending_inflow = {}

    def _update_emergency_vehicles(self) -> None:
        for ev in self._emergency_vehicles:
            if not ev.active:
                continue
            node = self._intersections.get(ev.current_intersection)
            if node is None:
                ev.active = False
                continue

            # Check if the intersection is green for EV direction
            ev_phase = 0 if ev.direction in ("N", "S") else 1
            if node.current_phase == ev_phase:
                # Move to next intersection
                if ev.direction == "S":
                    ev.current_intersection += 1
                else:
                    ev.current_intersection -= 1
                if ev.current_intersection < 0 or ev.current_intersection >= self.n_intersections:
                    ev.active = False
            else:
                ev.total_delay += self.step_seconds

    def _compute_step_metrics(self) -> CorridorMetrics:
        total_queue = sum(n.total_queue for n in self._intersections.values())
        total_deps = sum(n.total_departures for n in self._intersections.values())
        total_wait = sum(n.cumulative_wait for n in self._intersections.values())

        avg_wait = total_wait / max(total_deps, 1)
        throughput = total_deps / max(self._step_count + 1, 1)

        # Green wave efficiency
        gwe = self._green_hits / max(self._total_arrivals_corridor, 1)

        # Average speed estimate
        speed_limit = self.corridor_data["speed_limit_mph"]
        congestion_factor = max(0.15, 1.0 - (total_queue / max(self.n_intersections * self.max_queue * self.lanes * 4, 1)))
        avg_speed = speed_limit * congestion_factor

        # EPA emissions
        co2_kg = total_queue * self.co2_per_idle_vehicle_per_sec * self.step_seconds
        fuel_gal = total_queue * self.fuel_per_idle_vehicle_per_sec * self.step_seconds

        # Emergency vehicle delay
        ev_delay = sum(ev.total_delay for ev in self._emergency_vehicles if ev.active)

        queue_per_int = [
            self._intersections[i].total_queue for i in range(self.n_intersections)
        ]

        return CorridorMetrics(
            step=self._step_count,
            total_queue=total_queue,
            avg_wait_sec=avg_wait,
            throughput=throughput,
            green_wave_efficiency=gwe,
            avg_speed_mph=avg_speed,
            co2_kg=co2_kg,
            fuel_gallons=fuel_gal,
            emergency_delay_sec=ev_delay,
            queue_per_intersection=queue_per_int,
        )

    def _collect_obs(self) -> dict[int, dict[str, float]]:
        return {nid: node.observe(self._step_count) for nid, node in self._intersections.items()}
