"""traffic_ai/controllers/__init__.py

AITO controller registry.

Controller Classification
--------------------------
BENCHMARK_CONTROLLERS:
    Used in experiment runner comparisons to establish performance baselines and
    prove RL superiority. These are NOT run in shadow/deployment mode.

DEPLOYED_CONTROLLERS:
    Used in shadow mode, live simulation, and future real-world integration.
    These are the production-grade RL agents.
"""
from traffic_ai.controllers.adaptive_rule import AdaptiveRuleController
from traffic_ai.controllers.base import BaseController
from traffic_ai.controllers.greedy_adaptive import GreedyAdaptiveController
from traffic_ai.controllers.max_pressure import MaxPressureController
from traffic_ai.controllers.webster import WebsterController
from traffic_ai.controllers.factory import (
    build_baseline_controllers,
    build_rl_controllers,
    build_supervised_controllers,
    merge_controller_sets,
)
from traffic_ai.controllers.fixed_timing import FixedTimingController
from traffic_ai.controllers.ml_controller import SupervisedMLController
from traffic_ai.controllers.ml_controllers import (
    GradientBoostingController,
    MLPController,
    RandomForestController,
    XGBoostController,
)
from traffic_ai.controllers.rl_controller import RLPolicyController
from traffic_ai.controllers.rl_controllers import (
    A2CController,
    DQNController,
    PPOController,
    QLearningController,
    RecurrentPPOController,
    SACController,
)

# Alias for clearer display name in BENCHMARK_CONTROLLERS
NeuralNetworkMLPController = MLPController

# ---------------------------------------------------------------------------
# AITO controller registries
# ---------------------------------------------------------------------------

BENCHMARK_CONTROLLERS: list[type[BaseController]] = [
    FixedTimingController,
    AdaptiveRuleController,
    WebsterController,           # Webster (1958) — Econolite/SCATS/MAXTIME industry standard
    GreedyAdaptiveController,    # InSync-style greedy — deployed on Mira Mesa & Rosecrans, San Diego
    MaxPressureController,
    RandomForestController,
    XGBoostController,
    NeuralNetworkMLPController,
]
"""Controllers used only for scientific comparison. NOT deployed in shadow mode."""

DEPLOYED_CONTROLLERS: list[type[BaseController]] = [
    QLearningController,
    DQNController,
    PPOController,
    MaxPressureController,
]
"""Production-grade RL controllers. Run in shadow mode and future real-world integration."""

__all__ = [
    "BaseController",
    "FixedTimingController",
    "AdaptiveRuleController",
    "WebsterController",
    "GreedyAdaptiveController",
    "MaxPressureController",
    "RandomForestController",
    "XGBoostController",
    "GradientBoostingController",
    "MLPController",
    "NeuralNetworkMLPController",
    "QLearningController",
    "DQNController",
    "PPOController",
    "A2CController",
    "SACController",
    "RecurrentPPOController",
    "SupervisedMLController",
    "RLPolicyController",
    "BENCHMARK_CONTROLLERS",
    "DEPLOYED_CONTROLLERS",
    "build_baseline_controllers",
    "build_supervised_controllers",
    "build_rl_controllers",
    "merge_controller_sets",
]
