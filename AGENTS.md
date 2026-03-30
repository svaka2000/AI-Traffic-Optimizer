# AGENTS.md

## Mission
Build and maintain **AITO — AI Traffic Optimization**, a professional-grade engineering platform for traffic signal control research, validation, and deployment readiness.

AITO benchmarks **14 signal controllers** across four families (baseline, adaptive, supervised ML, reinforcement learning) using rigorous statistical methodology, real-world calibrated physics, shadow-mode deployment validation, and sensor fault tolerance. The system is designed for traffic engineers and researchers. No external AI API dependencies.

## Architecture

```
traffic_ai/
├── controllers/          # All signal controllers (BaseController interface)
│   ├── base.py           # Abstract base + select_action/update interface
│   ├── fixed.py          # Fixed timing baseline
│   ├── rule_based.py     # Rule-based adaptive controller
│   ├── ml_controllers.py # RF, XGBoost, GradientBoosting, MLP
│   ├── rl_controllers.py # QLearning, DQN, PPO, A2C, SAC, RecurrentPPO
│   ├── maddpg_controller.py # Multi-agent MADDPG
│   ├── fault_tolerant.py # EWMA-imputing wrapper (Phase 4)
│   └── __init__.py       # BENCHMARK_CONTROLLERS, DEPLOYED_CONTROLLERS registries
├── simulation_engine/    # Physics + types
│   ├── engine.py         # TrafficNetworkSimulator (HCM 7th ed.)
│   ├── types.py          # SignalPhase (6 values), IntersectionState, PHASE_TO_IDX
│   ├── sensor.py         # SensorFaultModel — stuck/noise/dropout (Phase 4)
│   ├── demand.py         # 11 demand profiles
│   └── emissions.py      # EPA MOVES2014b CO₂ model
├── rl_models/            # RL training infrastructure
│   ├── environment.py    # SignalControlEnv — 4-phase MDP, 8-obs, 6-reward
│   ├── dqn.py            # DQNetwork + DQNPolicy + train_dqn
│   ├── q_learning.py     # QLearningPolicy + train_q_learning
│   └── policy_gradient.py# PolicyNet + train_policy_gradient
├── shadow/               # Shadow mode (Phase 5)
│   └── shadow_runner.py  # ShadowModeRunner → artifacts/shadow_report.json
├── explainability/       # Decision explainability (Phase 6)
│   └── explainer.py      # DecisionExplainer — NL explanations + feature importances
├── dashboard/            # Streamlit 6-tab engineering dashboard (Phase 7)
│   └── streamlit_app.py  # Tabs: Overview, Benchmark, Shadow, Training, Data, Export
├── data_pipeline/        # Data ingestion + synthetic generation
├── ml_models/            # Controller model training
├── training/             # ModelTrainer (RL + ML unified)
├── metrics/              # Metrics computation
├── experiments/          # ExperimentRunner, Holm-Bonferroni stats
├── api/                  # FastAPI REST server
└── config/               # Settings + default_config.yaml
```

## Signal Model (4-Phase, AITO Phase 3)

```
action  SignalPhase   Serves
 0      NS_THROUGH    N + S through lanes  (1800 veh/hr/lane, HCM 7th ed.)
 1      EW_THROUGH    E + W through lanes  (1800 veh/hr/lane)
 2      NS_LEFT       N + S left-turn lanes (1200 veh/hr/lane, HCM 7th ed.)
 3      EW_LEFT       E + W left-turn lanes (1200 veh/hr/lane)
```

Legacy "NS"/"EW" strings remain valid `SignalPhase` values for backward compatibility.
`PHASE_TO_IDX` and `IDX_TO_PHASE` in `types.py` handle all conversions.

## Engineering Rules
- Python version: `3.11`.
- Use typed Python (`type hints`) for all public methods.
- Keep modules decoupled; each layer should expose clear interfaces.
- Avoid hard-coded file paths; use config-driven directories.
- All randomness must respect the global seed in `traffic_ai/config/settings.py`.
- Persist experiment outputs to `artifacts/` only.
- No external API dependencies (no Anthropic, OpenAI, or other AI service calls).
- **Never break the `BaseController` interface** — all controllers must implement `reset`, `compute_actions`, `select_action`, `update`.
- Every constant must cite its source (HCM, EPA MOVES2014b, PeMS, etc.).

## Data and Experiment Rules
- Raw datasets go to `data/raw/`, processed data to `data/processed/`.
- Never overwrite raw source files after ingestion.
- Every experiment run must emit:
  - run config JSON
  - metrics CSV
  - model artifacts
  - plots (300 DPI, publication quality)
- Include baseline controllers in every comparative run.

## Quality Rules
- Unit tests live in `tests/`.
- New metrics/controllers require at least one test.
- Run quick validation before finalizing:
  - `python main.py --quick-run`
  - `pytest -q`
  - `python main.py --shadow-mode` (verify shadow pipeline)

## Repo Layout
- Core package: `traffic_ai/`
- CLI entrypoint: `main.py`
- Dashboard: `traffic_ai/dashboard/streamlit_app.py`
- API: `traffic_ai/api/server.py`
- Tests: `tests/`
