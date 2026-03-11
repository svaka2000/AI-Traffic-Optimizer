# PLANS.md

## Objective
Deliver a modular, reproducible AI traffic optimization research platform with classical control, supervised ML, and RL comparisons.

## Phase 1: Core Skeleton — DONE
- [x] Create package structure and artifact directories.
- [x] Add repository operating rules (`AGENTS.md`).
- [x] Add configuration (`traffic_ai/config/settings.py`, `default_config.yaml`) and reproducibility primitives (`traffic_ai/utils/reproducibility.py`).

## Phase 2: Data Platform — DONE
- [x] Kaggle ingestion via API client (`traffic_ai/data/ingestion.py`).
- [x] Public dataset ingestion: Metro Interstate (UCI), synthetic fallback.
- [x] Common schema normalization to 15-column unified schema (`traffic_ai/data/preprocessing.py`).
- [x] Derived features: `is_rush_hour`, `hour_of_day`, `day_of_week`, rolling means (15-min, 60-min).
- [x] Train/validation/test 70/15/15 stratified split; saved to `data/processed/`.

## Phase 3: Simulation Engine — DONE
- [x] Multi-lane intersection simulator (`traffic_ai/simulation_engine/`) — existing engine.
- [x] `MultiIntersectionNetwork` N×M grid with Gym-compatible step/reset (`traffic_ai/simulation/intersection.py`).
- [x] Poisson arrivals with rush-hour 2.5× scaling, vehicle spillback.
- [x] Queue dynamics: max queue cap enforced per lane.
- [x] `EpisodeMetricsTracker` with per-step/per-episode CSV export (`traffic_ai/simulation/metrics.py`).

## Phase 4: Controllers — DONE
- [x] Unified `BaseController` interface with `select_action`, `update`, `compute_actions` (`traffic_ai/controllers/base.py`).
- [x] `FixedTimingController` (30s default cycle) — `traffic_ai/controllers/fixed.py`.
- [x] `RuleBasedController` (queue-balance adaptive) — `traffic_ai/controllers/rule_based.py`.
- [x] Supervised ML controllers: RF, XGBoost, GradientBoosting, MLP (PyTorch 128→64→32), LSTMForecast, ImitationLearning — `traffic_ai/controllers/ml_controllers.py`.
- [x] RL controllers: QLearning (tabular), DQN (replay buffer + target network), PPO (actor-critic) — `traffic_ai/controllers/rl_controllers.py`.

## Phase 5: Research Workflow — DONE
- [x] Experiment runner with cross-validation and ablation sweeps (`traffic_ai/experiments/runner.py`).
- [x] Statistical significance testing (Mann-Whitney U / bootstrap) via `traffic_ai/metrics/statistics.py`.
- [x] New publication-quality plots: controller comparison, feature importance, ablation heatmap, queue over time (`traffic_ai/visualization/plots.py`).

## Phase 6: Interfaces — DONE
- [x] FastAPI service: `POST /run`, `GET /results/{run_id}`, `GET /plots/{run_id}/{plot_name}`, `GET /health` — `traffic_ai/api/server.py`.
- [x] Streamlit dashboard with live grid animation, metrics charts, results panel, controller switcher — `traffic_ai/dashboard/streamlit_app.py`.

## Phase 7: Validation — DONE
- [x] Unit tests: `tests/test_simulation.py` (8 tests), `tests/test_controllers.py` (22 tests), `tests/test_metrics.py` (9 tests).
- [x] Existing test suites: `test_data_pipeline.py`, `test_experiment_runner.py`, `test_simulation_and_controllers.py`.
- [x] End-to-end smoke run: `python main.py --quick-run` passes cleanly.
- [x] `pytest -q` — 66 tests pass.

