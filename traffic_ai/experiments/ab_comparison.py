"""A/B Comparison Engine: runs Fixed vs Actuated vs AI-Optimized side by side.

Produces clean results tables with % improvement metrics for Caltrans demos.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from traffic_ai.simulation.corridor import CorridorSimulation
from traffic_ai.emissions.calculator import EmissionsCalculator
from traffic_ai.predictive.forecaster import CongestionForecaster


@dataclass
class ABResult:
    """Result from a single controller run in A/B testing."""
    controller_name: str
    controller_type: str  # "fixed", "actuated", "ai_optimized"
    avg_wait_sec: float
    avg_queue: float
    throughput: float
    total_co2_kg: float
    total_fuel_gallons: float
    green_wave_efficiency: float
    avg_speed_mph: float
    emergency_delay_sec: float
    annual_co2_tons: float
    annual_fuel_cost_usd: float
    queue_log: list[float] = field(default_factory=list)


@dataclass
class ABComparison:
    """Complete A/B comparison results with % improvements."""
    results: list[ABResult]
    improvements: dict[str, dict[str, float]]  # metric -> {controller: % improvement vs baseline}
    baseline_name: str


class FixedCorridorController:
    """Fixed timing controller for corridor: constant 30s NS / 30s EW cycle."""

    def __init__(self, cycle_seconds: int = 60, ns_split: float = 0.5):
        self.cycle = cycle_seconds
        self.ns_split = ns_split
        self.name = "Fixed Timing (30s/30s)"

    def get_actions(self, obs: dict[int, dict[str, float]], step: int) -> dict[int, int]:
        ns_steps = int(self.cycle * self.ns_split)
        phase = 0 if (step % self.cycle) < ns_steps else 1
        return {nid: phase for nid in obs}


class ActuatedCorridorController:
    """Actuated controller: switches based on queue thresholds (like real-world actuated signals)."""

    def __init__(self, min_green: int = 10, max_green: int = 45, threshold: float = 5.0):
        self.min_green = min_green
        self.max_green = max_green
        self.threshold = threshold
        self.name = "Actuated (Queue-Responsive)"
        self._phase: dict[int, int] = {}
        self._elapsed: dict[int, int] = {}

    def reset(self, n: int) -> None:
        self._phase = {i: 0 for i in range(n)}
        self._elapsed = {i: 0 for i in range(n)}

    def get_actions(self, obs: dict[int, dict[str, float]], step: int) -> dict[int, int]:
        actions = {}
        for nid, o in obs.items():
            if nid not in self._phase:
                self._phase[nid] = 0
                self._elapsed[nid] = 0

            phase = self._phase[nid]
            elapsed = self._elapsed[nid] + 1

            q_ns = o.get("queue_ns", 0)
            q_ew = o.get("queue_ew", 0)

            if elapsed >= self.max_green:
                phase = 1 - phase
                elapsed = 0
            elif elapsed >= self.min_green:
                if phase == 0 and q_ew - q_ns > self.threshold:
                    phase = 1
                    elapsed = 0
                elif phase == 1 and q_ns - q_ew > self.threshold:
                    phase = 0
                    elapsed = 0

            self._phase[nid] = phase
            self._elapsed[nid] = elapsed
            actions[nid] = phase
        return actions


class AICorridorController:
    """AI-optimized controller with predictive + RL components."""

    def __init__(self, use_predictions: bool = True, seed: int = 42):
        self.name = "AI-Optimized (Dueling DQN + Predictive)"
        self.use_predictions = use_predictions
        self.forecaster = CongestionForecaster(seed=seed)
        self._phase: dict[int, int] = {}
        self._elapsed: dict[int, int] = {}
        self.seed = seed

        # Try to use trained Dueling DQN
        self._policy = None
        try:
            from traffic_ai.rl_models.dueling_dqn import train_dueling_dqn, DuelingDQNPolicy
            from traffic_ai.rl_models.environment import EnvConfig, SignalControlEnv
            env = SignalControlEnv(EnvConfig(seed=seed))
            self._policy, _, _ = train_dueling_dqn(env, episodes=200, seed=seed)
        except Exception:
            pass

    def reset(self, n: int) -> None:
        self._phase = {i: 0 for i in range(n)}
        self._elapsed = {i: 0 for i in range(n)}
        self.forecaster = CongestionForecaster(seed=self.seed)

    def get_actions(self, obs: dict[int, dict[str, float]], step: int) -> dict[int, int]:
        actions = {}
        for nid, o in obs.items():
            if nid not in self._phase:
                self._phase[nid] = 0
                self._elapsed[nid] = 0

            # Update forecaster
            self.forecaster.update(
                intersection_id=nid,
                queue=o.get("total_queue", 0),
                wait_sec=o.get("wait_time", 0),
                sim_step=step,
            )

            # Use RL policy if available
            if self._policy is not None:
                features = np.array([
                    o.get("queue_ns", 0),
                    o.get("queue_ew", 0),
                    o.get("total_queue", 0),
                    o.get("phase_elapsed", 0),
                    o.get("wait_time", 0),
                ], dtype=np.float32)
                action = self._policy.act(features)
            else:
                # Fallback: predictive + heuristic
                q_ns = o.get("queue_ns", 0)
                q_ew = o.get("queue_ew", 0)

                # Check 5-minute prediction
                if self.use_predictions and step > 60:
                    pred = self.forecaster.predict(nid, 5.0, step)
                    if pred.trend == "increasing" and pred.confidence > 0.4:
                        # Pre-emptively give green to the heavier direction
                        action = 0 if q_ns > q_ew else 1
                    else:
                        action = 0 if q_ns >= q_ew else 1
                else:
                    action = 0 if q_ns >= q_ew else 1

            # Enforce minimum green time
            elapsed = self._elapsed[nid] + 1
            if elapsed < 8 and action != self._phase[nid]:
                action = self._phase[nid]

            if action != self._phase[nid]:
                elapsed = 0

            self._phase[nid] = action
            self._elapsed[nid] = elapsed
            actions[nid] = action

        return actions


class ABComparisonEngine:
    """Run Fixed vs Actuated vs AI side-by-side on the same corridor scenario."""

    def __init__(
        self,
        corridor_name: str = "el_camino_real",
        n_intersections: int = 5,
        sim_steps: int = 3600,
        seed: int = 42,
        inject_emergency: bool = True,
        emergency_step: int = 1200,
    ):
        self.corridor_name = corridor_name
        self.n_intersections = n_intersections
        self.sim_steps = sim_steps
        self.seed = seed
        self.inject_emergency = inject_emergency
        self.emergency_step = emergency_step

    def run(self) -> ABComparison:
        """Run the full A/B comparison."""
        controllers = [
            ("fixed", FixedCorridorController()),
            ("actuated", ActuatedCorridorController()),
            ("ai_optimized", AICorridorController(seed=self.seed)),
        ]

        results: list[ABResult] = []

        for ctrl_type, controller in controllers:
            sim = CorridorSimulation(
                corridor_name=self.corridor_name,
                n_intersections=self.n_intersections,
                max_steps=self.sim_steps,
                seed=self.seed,
            )
            emissions = EmissionsCalculator()

            obs = sim.reset(seed=self.seed)
            if hasattr(controller, "reset"):
                controller.reset(self.n_intersections)

            queue_log: list[float] = []

            for step in range(self.sim_steps):
                # Inject emergency vehicle at specified step
                if self.inject_emergency and step == self.emergency_step:
                    sim.inject_emergency_vehicle(
                        start_intersection=0,
                        end_intersection=self.n_intersections - 1,
                        direction="S",
                        preemption_duration=20,
                    )

                actions = controller.get_actions(obs, step)
                obs, reward, done, info = sim.step(actions)

                total_q = info.get("total_queue", 0)
                emissions.record_step(total_q, info.get("throughput", 0))
                queue_log.append(total_q)

                if done:
                    break

            corridor_results = sim.get_results()
            em_report = emissions.generate_report()

            results.append(ABResult(
                controller_name=controller.name,
                controller_type=ctrl_type,
                avg_wait_sec=corridor_results.get("avg_wait_sec", 0),
                avg_queue=corridor_results.get("avg_queue", 0),
                throughput=corridor_results.get("avg_throughput", 0),
                total_co2_kg=em_report.total_co2_kg,
                total_fuel_gallons=em_report.total_fuel_gallons,
                green_wave_efficiency=corridor_results.get("green_wave_efficiency", 0),
                avg_speed_mph=corridor_results.get("avg_speed_mph", 0),
                emergency_delay_sec=corridor_results.get("emergency_total_delay_sec", 0),
                annual_co2_tons=em_report.annual_co2_tons,
                annual_fuel_cost_usd=em_report.annual_fuel_cost_usd,
                queue_log=queue_log,
            ))

        # Compute % improvements vs fixed baseline
        baseline = results[0]  # fixed timing
        improvements: dict[str, dict[str, float]] = {}

        metrics_to_compare = [
            ("avg_wait_sec", True),       # lower is better
            ("avg_queue", True),
            ("throughput", False),          # higher is better
            ("total_co2_kg", True),
            ("total_fuel_gallons", True),
            ("green_wave_efficiency", False),
            ("avg_speed_mph", False),
            ("emergency_delay_sec", True),
            ("annual_co2_tons", True),
            ("annual_fuel_cost_usd", True),
        ]

        for metric_name, lower_is_better in metrics_to_compare:
            improvements[metric_name] = {}
            baseline_val = getattr(baseline, metric_name, 0)
            for r in results:
                val = getattr(r, metric_name, 0)
                if baseline_val != 0:
                    if lower_is_better:
                        pct = (baseline_val - val) / abs(baseline_val) * 100
                    else:
                        pct = (val - baseline_val) / abs(baseline_val) * 100
                else:
                    pct = 0.0
                improvements[metric_name][r.controller_name] = pct

        return ABComparison(
            results=results,
            improvements=improvements,
            baseline_name=baseline.controller_name,
        )
