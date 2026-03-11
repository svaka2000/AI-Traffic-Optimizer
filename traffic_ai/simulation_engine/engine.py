from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from traffic_ai.simulation_engine.demand import DemandModel
from traffic_ai.simulation_engine.types import (
    Direction,
    IntersectionState,
    SignalPhase,
    SimulationResult,
    StepMetrics,
)


class ControllerLike(Protocol):
    name: str

    def reset(self, n_intersections: int) -> None: ...

    def compute_actions(
        self, observations: dict[int, dict[str, float]], step: int
    ) -> dict[int, SignalPhase]: ...


@dataclass(slots=True)
class SimulatorConfig:
    steps: int = 2000
    intersections: int = 4
    lanes_per_direction: int = 2
    step_seconds: float = 1.0
    max_queue_per_lane: int = 60
    demand_profile: str = "rush_hour"
    demand_scale: float = 1.0
    seed: int = 42


class TrafficNetworkSimulator:
    def __init__(self, config: SimulatorConfig) -> None:
        self.config = config
        self.rng = np.random.default_rng(config.seed)
        self.demand = DemandModel(
            profile=config.demand_profile,  # type: ignore[arg-type]
            scale=config.demand_scale,
            step_seconds=config.step_seconds,
        )
        self.intersection_ids = list(range(config.intersections))
        self.neighbors = self._build_directional_neighbors()
        self.states = self._init_intersections()

    def run(self, controller: ControllerLike, steps: int | None = None) -> SimulationResult:
        total_steps = steps or self.config.steps
        controller.reset(len(self.states))
        self.states = self._init_intersections()
        step_logs: list[StepMetrics] = []

        baseline_reference_queue = None
        for step in range(total_steps):
            observations = self._collect_observations(step)
            actions = controller.compute_actions(observations, step)
            self._advance_step(actions, step)
            metrics = self._compute_step_metrics(step)
            if baseline_reference_queue is None:
                baseline_reference_queue = max(metrics.total_queue, 1.0)
            metrics.delay_reduction_pct = max(
                -100.0,
                min(100.0, (baseline_reference_queue - metrics.total_queue) / baseline_reference_queue * 100.0),
            )
            step_logs.append(metrics)

        summaries = self._intersection_summaries()
        aggregate = self._aggregate_metrics(step_logs)
        return SimulationResult(
            controller_name=controller.name,
            step_metrics=step_logs,
            intersection_summaries=summaries,
            aggregate=aggregate,
        )

    def _advance_step(self, actions: dict[int, SignalPhase], step: int) -> None:
        # 1) Apply pending inflow from upstream intersections.
        for state in self.states.values():
            for direction, incoming in state.pending_inflow.items():
                lane_index = int(self.rng.integers(0, self.config.lanes_per_direction))
                state.queue_matrix[direction][lane_index] = min(
                    state.max_queue_per_lane,
                    state.queue_matrix[direction][lane_index] + incoming,
                )
            state.pending_inflow = {}

        # 2) Update phases.
        for intersection_id, state in self.states.items():
            requested = actions.get(intersection_id, state.current_phase)
            if requested != state.current_phase:
                state.current_phase = requested
                state.phase_elapsed = 0
                state.phase_changes += 1
            else:
                state.phase_elapsed += 1

        # 3) Stochastic arrivals.
        for state in self.states.values():
            for direction in ["N", "S", "E", "W"]:
                self._sample_arrivals_for_direction(state, direction, step)

        # 4) Service/departure and network propagation.
        flow_to_neighbors: dict[tuple[int, Direction], float] = defaultdict(float)
        for intersection_id, state in self.states.items():
            departures_by_direction = self._service_intersection(state)
            for direction, departed in departures_by_direction.items():
                neighbor_id = self.neighbors[intersection_id].get(direction)
                if neighbor_id is None:
                    continue
                transfer = departed * 0.65
                if transfer <= 0:
                    continue
                incoming_direction = self._opposite(direction)
                flow_to_neighbors[(neighbor_id, incoming_direction)] += transfer

        for (neighbor_id, direction), volume in flow_to_neighbors.items():
            self.states[neighbor_id].pending_inflow[direction] = (
                self.states[neighbor_id].pending_inflow.get(direction, 0.0) + volume
            )

    def _sample_arrivals_for_direction(
        self, state: IntersectionState, direction: Direction, step: int
    ) -> None:
        rate = self.demand.arrival_rate_per_lane(step, direction)
        for lane in range(self.config.lanes_per_direction):
            arrivals = float(self.rng.poisson(rate * self.config.step_seconds))
            state.queue_matrix[direction][lane] = min(
                state.max_queue_per_lane,
                state.queue_matrix[direction][lane] + arrivals,
            )
            state.total_arrivals += int(arrivals)

    def _service_intersection(self, state: IntersectionState) -> dict[Direction, float]:
        green_directions = ["N", "S"] if state.current_phase == "NS" else ["E", "W"]
        departures: dict[Direction, float] = {d: 0.0 for d in ["N", "S", "E", "W"]}

        saturation_rate_per_lane = 0.45  # veh/s/lane
        for direction in ["N", "S", "E", "W"]:
            for lane in range(self.config.lanes_per_direction):
                queue = state.queue_matrix[direction][lane]
                if direction in green_directions:
                    capacity = float(
                        self.rng.poisson(saturation_rate_per_lane * self.config.step_seconds)
                    )
                    moved = min(queue, capacity)
                else:
                    moved = 0.0
                state.queue_matrix[direction][lane] -= moved
                departures[direction] += moved

        total_queue = state.total_queue
        state.cumulative_wait_sec += total_queue * self.config.step_seconds
        state.cumulative_stopped_vehicles += total_queue
        state.total_departures += int(sum(departures.values()))
        return departures

    def _collect_observations(self, step: int) -> dict[int, dict[str, float]]:
        return {
            intersection_id: state.as_observation(step)
            for intersection_id, state in self.states.items()
        }

    def _compute_step_metrics(self, step: int) -> StepMetrics:
        queues = np.array([state.total_queue for state in self.states.values()], dtype=np.float64)
        total_queue = float(queues.sum())
        total_departures = float(sum(state.total_departures for state in self.states.values()))
        total_wait = float(sum(state.cumulative_wait_sec for state in self.states.values()))
        avg_wait = total_wait / max(total_departures, 1.0)
        throughput = float(
            sum(state.total_departures for state in self.states.values())
        ) / max(step + 1, 1)
        emissions_proxy = total_queue * 0.21 + float(
            sum(state.phase_changes for state in self.states.values()) * 0.8
        )
        fuel_proxy = total_queue * 0.12 + throughput * 0.04
        fairness = self._fairness_score(queues)
        efficiency = throughput / (1.0 + total_queue / max(len(self.states), 1))
        return StepMetrics(
            step=step,
            total_queue=total_queue,
            avg_wait_sec=avg_wait,
            throughput=throughput,
            emissions_proxy=emissions_proxy,
            fuel_proxy=fuel_proxy,
            fairness=fairness,
            efficiency_score=efficiency,
            delay_reduction_pct=0.0,
        )

    @staticmethod
    def _fairness_score(values: np.ndarray) -> float:
        if len(values) == 0:
            return 1.0
        mean = float(values.mean())
        if mean <= 1e-9:
            return 1.0
        diffsum = np.abs(values[:, None] - values[None, :]).sum()
        gini = diffsum / (2.0 * len(values) ** 2 * mean)
        return float(max(0.0, 1.0 - gini))

    def _intersection_summaries(self) -> list[dict[str, float]]:
        summaries: list[dict[str, float]] = []
        for intersection_id, state in self.states.items():
            total_departures = max(state.total_departures, 1)
            summaries.append(
                {
                    "intersection_id": float(intersection_id),
                    "total_arrivals": float(state.total_arrivals),
                    "total_departures": float(state.total_departures),
                    "mean_wait_sec": state.cumulative_wait_sec / total_departures,
                    "queue_ns": state.queue_ns,
                    "queue_ew": state.queue_ew,
                    "phase_changes": float(state.phase_changes),
                }
            )
        return summaries

    def _aggregate_metrics(self, logs: list[StepMetrics]) -> dict[str, float]:
        if not logs:
            return {}
        avg = lambda name: float(np.mean([getattr(item, name) for item in logs]))
        return {
            "average_wait_time": avg("avg_wait_sec"),
            "average_queue_length": avg("total_queue"),
            "average_throughput": avg("throughput"),
            "average_emissions_proxy": avg("emissions_proxy"),
            "average_fuel_proxy": avg("fuel_proxy"),
            "average_fairness": avg("fairness"),
            "average_efficiency_score": avg("efficiency_score"),
            "delay_reduction_pct": avg("delay_reduction_pct"),
            "max_queue_length": float(max(item.total_queue for item in logs)),
        }

    def _init_intersections(self) -> dict[int, IntersectionState]:
        states: dict[int, IntersectionState] = {}
        for intersection_id in self.intersection_ids:
            queue_matrix = {
                "N": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "S": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "E": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
                "W": np.zeros(self.config.lanes_per_direction, dtype=np.float64),
            }
            states[intersection_id] = IntersectionState(
                intersection_id=intersection_id,
                lanes_per_direction=self.config.lanes_per_direction,
                max_queue_per_lane=self.config.max_queue_per_lane,
                queue_matrix=queue_matrix,
                pending_inflow={},
            )
        return states

    def _build_directional_neighbors(self) -> dict[int, dict[Direction, int | None]]:
        side = int(math.ceil(math.sqrt(self.config.intersections)))
        mapping: dict[int, dict[Direction, int | None]] = {}
        for idx in range(self.config.intersections):
            row, col = divmod(idx, side)
            candidates = {
                "N": (row - 1, col),
                "S": (row + 1, col),
                "W": (row, col - 1),
                "E": (row, col + 1),
            }
            mapping[idx] = {}
            for direction, (r, c) in candidates.items():
                if 0 <= r < side and 0 <= c < side:
                    nid = r * side + c
                    mapping[idx][direction] = nid if nid < self.config.intersections else None
                else:
                    mapping[idx][direction] = None
        return mapping

    @staticmethod
    def _opposite(direction: Direction) -> Direction:
        lookup: dict[Direction, Direction] = {"N": "S", "S": "N", "E": "W", "W": "E"}
        return lookup[direction]
