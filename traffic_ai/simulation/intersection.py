"""traffic_ai/simulation/intersection.py

MultiIntersectionNetwork: configurable N×M grid of traffic intersections with
Gym-compatible step/reset interface, Poisson vehicle arrivals, rush-hour demand
scaling, and vehicle spillback.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------
Direction = str  # "N" | "S" | "E" | "W"
Phase = int  # 0 = NS green, 1 = EW green

DIRECTIONS: list[str] = ["N", "S", "E", "W"]
_OPPOSITE: dict[str, str] = {"N": "S", "S": "N", "E": "W", "W": "E"}
_GREEN_DIRS: dict[int, list[str]] = {0: ["N", "S"], 1: ["E", "W"]}


@dataclass
class IntersectionNode:
    """Mutable state for a single intersection in the network."""

    node_id: int
    lanes: int
    max_queue: int
    # queue_matrix[direction][lane_index] = number of waiting vehicles
    queue_matrix: dict[str, np.ndarray] = field(default_factory=dict)
    current_phase: int = 0
    phase_elapsed: int = 0
    phase_changes: int = 0
    total_arrivals: int = 0
    total_departures: int = 0
    cumulative_wait: float = 0.0
    pending_inflow: dict[str, float] = field(default_factory=dict)

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
            "avg_speed": max(0.0, 60.0 - self.total_queue * 0.5),
            "lane_occupancy": min(1.0, self.total_queue / max(1.0, self.lanes * 4 * self.max_queue)),
            "wait_time": self.cumulative_wait / max(1.0, self.total_departures),
            "arrivals": float(self.total_arrivals),
            "departures": float(self.total_departures),
        }


class MultiIntersectionNetwork:
    """N×M grid of intersections with Gym-compatible interface.

    Parameters
    ----------
    rows, cols:
        Grid dimensions (default 2×2 = 4 intersections).
    lanes_per_approach:
        Number of lanes per direction at each intersection.
    max_queue_length:
        Maximum vehicles per lane before spillback triggers.
    max_steps:
        Episode length in simulation steps.
    step_seconds:
        Real-world seconds represented by each simulation step.
    base_arrival_rate:
        Base Poisson rate (vehicles/second/lane) under normal conditions.
    rush_hour_scale:
        Multiplier applied to arrival rate during rush hours (default 2.5×).
    seed:
        Random seed for reproducibility.
    calibration_data:
        Optional dict mapping hour → mean_vehicle_count for Poisson calibration.
    """

    def __init__(
        self,
        rows: int = 2,
        cols: int = 2,
        lanes_per_approach: int = 2,
        max_queue_length: int = 50,
        max_steps: int = 2_000,
        step_seconds: float = 1.0,
        base_arrival_rate: float = 0.12,
        rush_hour_scale: float = 2.5,
        seed: int = 42,
        calibration_data: dict[int, float] | None = None,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.n_intersections = rows * cols
        self.lanes = lanes_per_approach
        self.max_queue = max_queue_length
        self.max_steps = max_steps
        self.step_seconds = step_seconds
        self.base_rate = base_arrival_rate
        self.rush_scale = rush_hour_scale
        self.seed = seed
        self.calibration = calibration_data or {}

        self.rng = np.random.default_rng(seed)
        self._nodes: dict[int, IntersectionNode] = {}
        self._neighbors: dict[int, dict[str, int | None]] = {}
        self._step_count: int = 0
        self._spillback_events: int = 0
        self._green_switches: int = 0

        self._build_grid()
        self._build_neighbors()

    # ------------------------------------------------------------------
    # Gym-compatible interface
    # ------------------------------------------------------------------

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> dict[int, dict[str, float]]:
        """Reset the environment and return initial observations."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_count = 0
        self._spillback_events = 0
        self._green_switches = 0
        self._build_grid()
        return self._collect_obs()

    def step(
        self, actions: dict[int, int]
    ) -> tuple[dict[int, dict[str, float]], float, bool, dict[str, Any]]:
        """Advance the simulation one step.

        Parameters
        ----------
        actions:
            ``{node_id: phase}`` where phase is 0 (NS green) or 1 (EW green).

        Returns
        -------
        obs:
            Per-intersection observation dicts.
        reward:
            Negative total queue length (minimise congestion).
        done:
            True when episode length is reached.
        info:
            Auxiliary metrics.
        """
        # 1. Apply pending inflow from upstream intersections
        self._apply_pending_inflow()

        # 2. Update phases
        for node_id, node in self._nodes.items():
            requested = int(actions.get(node_id, node.current_phase))
            if requested != node.current_phase:
                node.current_phase = requested
                node.phase_elapsed = 0
                node.phase_changes += 1
                self._green_switches += 1
            else:
                node.phase_elapsed += 1

        # 3. Stochastic arrivals
        hour = self._current_hour()
        for node in self._nodes.values():
            for direction in DIRECTIONS:
                self._sample_arrivals(node, direction, hour)

        # 4. Service vehicles and propagate outflow
        self._service_and_propagate()

        # 5. Compute reward & metrics
        total_queue = sum(n.total_queue for n in self._nodes.values())
        total_wait = sum(n.cumulative_wait for n in self._nodes.values())
        total_departures = sum(n.total_departures for n in self._nodes.values())
        throughput = total_departures / max(self._step_count + 1, 1)
        reward = -float(total_queue)

        self._step_count += 1
        done = self._step_count >= self.max_steps

        obs = self._collect_obs()
        info: dict[str, Any] = {
            "step": self._step_count,
            "total_queue": total_queue,
            "avg_wait": total_wait / max(total_departures, 1),
            "throughput": throughput,
            "spillback_events": self._spillback_events,
            "green_switches": self._green_switches,
            "hour": hour,
        }
        return obs, reward, done, info

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _current_hour(self) -> float:
        elapsed_s = self._step_count * self.step_seconds
        return (elapsed_s / 3600.0) % 24.0

    def _is_rush_hour(self, hour: float) -> bool:
        return (7.0 <= hour < 9.0) or (16.0 <= hour < 19.0)

    def _arrival_rate(self, hour: float) -> float:
        calibrated = self.calibration.get(int(hour), None)
        if calibrated is not None:
            base = calibrated / 3600.0 / self.lanes
        else:
            # Gaussian peaks at 8am and 17.5pm
            morning = math.exp(-((hour - 8.0) ** 2) / 3.0)
            evening = math.exp(-((hour - 17.5) ** 2) / 3.0)
            base = self.base_rate * (1.0 + 1.4 * max(morning, evening))
        if self._is_rush_hour(hour):
            base *= self.rush_scale
        return max(0.01, base) * self.step_seconds

    def _sample_arrivals(
        self, node: IntersectionNode, direction: str, hour: float
    ) -> None:
        rate = self._arrival_rate(hour)
        for lane in range(self.lanes):
            n_arrivals = int(self.rng.poisson(rate))
            available = max(0, node.max_queue - int(node.queue_matrix[direction][lane]))
            if n_arrivals > available:
                self._spillback_events += 1
                n_arrivals = available
            node.queue_matrix[direction][lane] += n_arrivals
            node.total_arrivals += n_arrivals

    def _service_and_propagate(self) -> None:
        sat_rate = 0.45  # veh/s/lane
        flow_to: dict[tuple[int, str], float] = defaultdict(float)

        for node_id, node in self._nodes.items():
            green_dirs = _GREEN_DIRS[node.current_phase]
            for direction in DIRECTIONS:
                for lane in range(self.lanes):
                    q = node.queue_matrix[direction][lane]
                    if direction in green_dirs:
                        cap = float(self.rng.poisson(sat_rate * self.step_seconds))
                        moved = min(q, cap)
                    else:
                        moved = 0.0
                    node.queue_matrix[direction][lane] -= moved
                    node.total_departures += int(moved)
                    if moved > 0:
                        neighbor = self._neighbors[node_id].get(direction)
                        if neighbor is not None:
                            incoming_dir = _OPPOSITE[direction]
                            flow_to[(neighbor, incoming_dir)] += moved * 0.6

            node.cumulative_wait += node.total_queue * self.step_seconds

        # Queue pending inflows for next step
        for (neighbor_id, direction), volume in flow_to.items():
            self._nodes[neighbor_id].pending_inflow[direction] = (
                self._nodes[neighbor_id].pending_inflow.get(direction, 0.0) + volume
            )

    def _apply_pending_inflow(self) -> None:
        for node in self._nodes.values():
            for direction, incoming in node.pending_inflow.items():
                lane = int(self.rng.integers(0, self.lanes))
                overflow = node.queue_matrix[direction][lane] + incoming - node.max_queue
                if overflow > 0:
                    self._spillback_events += 1
                node.queue_matrix[direction][lane] = min(
                    float(node.max_queue),
                    node.queue_matrix[direction][lane] + incoming,
                )
            node.pending_inflow = {}

    def _collect_obs(self) -> dict[int, dict[str, float]]:
        return {nid: node.observe(self._step_count) for nid, node in self._nodes.items()}

    def _build_grid(self) -> None:
        self._nodes = {
            i: IntersectionNode(node_id=i, lanes=self.lanes, max_queue=self.max_queue)
            for i in range(self.n_intersections)
        }

    def _build_neighbors(self) -> None:
        for idx in range(self.n_intersections):
            row, col = divmod(idx, self.cols)
            candidates: dict[str, tuple[int, int]] = {
                "N": (row - 1, col),
                "S": (row + 1, col),
                "W": (row, col - 1),
                "E": (row, col + 1),
            }
            self._neighbors[idx] = {}
            for direction, (r, c) in candidates.items():
                if 0 <= r < self.rows and 0 <= c < self.cols:
                    nid = r * self.cols + c
                    self._neighbors[idx][direction] = nid
                else:
                    self._neighbors[idx][direction] = None

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self.n_intersections

    @property
    def observation_shape(self) -> tuple[int]:
        """Flat observation vector size for a single intersection."""
        return (len(self._nodes[0].observe(0)),)

    @property
    def n_actions(self) -> int:
        return 2  # NS green or EW green
