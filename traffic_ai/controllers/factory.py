from __future__ import annotations

from typing import Iterable

from traffic_ai.config.settings import Settings
from traffic_ai.controllers.adaptive_rule import AdaptiveRuleController
from traffic_ai.controllers.base import BaseController
from traffic_ai.controllers.fixed_timing import FixedTimingController
from traffic_ai.controllers.max_pressure import MaxPressureController
from traffic_ai.controllers.ml_controller import SupervisedMLController
from traffic_ai.controllers.rl_controller import RLPolicyController


def build_baseline_controllers(settings: Settings) -> list[BaseController]:
    fixed_cfg = settings.get("controllers.fixed_timing", {})
    adaptive_cfg = settings.get("controllers.adaptive_rule", {})
    mp_cfg = settings.get("controllers.max_pressure", {})
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

