from __future__ import annotations

from typing import Iterable

from traffic_ai.config.settings import Settings
from traffic_ai.controllers.adaptive_rule import AdaptiveRuleController
from traffic_ai.controllers.base import BaseController
from traffic_ai.controllers.fixed_timing import FixedTimingController
from traffic_ai.controllers.greedy_adaptive import GreedyAdaptiveController
from traffic_ai.controllers.max_pressure import MaxPressureController
from traffic_ai.controllers.ml_controller import SupervisedMLController
from traffic_ai.controllers.rl_controller import RLPolicyController
from traffic_ai.controllers.webster import WebsterController


def build_baseline_controllers(settings: Settings) -> list[BaseController]:
    fixed_cfg    = settings.get("controllers.fixed_timing", {})
    adaptive_cfg = settings.get("controllers.adaptive_rule", {})
    mp_cfg       = settings.get("controllers.max_pressure", {})
    webster_cfg  = settings.get("controllers.webster", {})
    greedy_cfg   = settings.get("controllers.greedy_adaptive", {})
    step_seconds = float(settings.get("simulation.step_seconds", 1.0))
    return [
        FixedTimingController(
            cycle_seconds=int(fixed_cfg.get("cycle_seconds", 60)),
            green_split_ns=float(fixed_cfg.get("green_split_ns", 0.55)),
            step_seconds=step_seconds,
        ),
        AdaptiveRuleController(
            min_green=int(adaptive_cfg.get("min_green", 15)),
            max_green=int(adaptive_cfg.get("max_green", 75)),
            queue_threshold=float(adaptive_cfg.get("queue_threshold", 6.0)),
        ),
        WebsterController(
            recalc_interval=int(webster_cfg.get("recalc_interval", 300)),
            saturation_flow=float(webster_cfg.get("saturation_flow", 0.5)),
            lost_time_per_phase=float(webster_cfg.get("lost_time_per_phase", 6.0)),
            min_cycle=float(webster_cfg.get("min_cycle", 60.0)),
            max_cycle=float(webster_cfg.get("max_cycle", 180.0)),
            step_seconds=step_seconds,
        ),
        GreedyAdaptiveController(
            min_green_steps=int(greedy_cfg.get("min_green_steps", 7)),
            volume_weight=float(greedy_cfg.get("volume_weight", 1.0)),
            delay_weight=float(greedy_cfg.get("delay_weight", 0.2)),
            coordination_weight=float(greedy_cfg.get("coordination_weight", 0.3)),
        ),
        MaxPressureController(
            min_green_sec=int(mp_cfg.get("min_green_sec", 7)),
        ),
    ]


def build_supervised_controllers(models: dict[str, object]) -> list[BaseController]:
    return [
        SupervisedMLController(model=model, min_green=8)
        for _, model in models.items()
    ]


def build_rl_controllers(policies: dict[str, object]) -> list[BaseController]:
    controllers: list[BaseController] = []
    for name, policy in policies.items():
        controllers.append(RLPolicyController(policy=policy, name=f"rl_{name}", min_green=6))
    return controllers


def merge_controller_sets(*controller_sets: Iterable[BaseController]) -> list[BaseController]:
    merged: list[BaseController] = []
    for batch in controller_sets:
        merged.extend(batch)
    return merged

