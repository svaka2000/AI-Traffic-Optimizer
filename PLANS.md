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

## Phase 8: Research-Grade Upgrades — DONE
### WS1: Upgraded DQN
- [x] Double DQN: online net selects action, target net evaluates (eliminates overestimation bias).
- [x] Dueling Architecture: shared 6→128→128 + V(s) value stream + A(s,a) advantage stream.
- [x] Prioritized Experience Replay (`traffic_ai/rl_models/replay_buffer.py`): |TD error|^α sampling + IS-weight correction + β annealing.
- [x] N-step returns (3-step buffer) before bootstrapping.
- [x] RewardShaper class (`traffic_ai/rl_models/rewards.py`): composite reward with queue/throughput/wait/phase-change/emergency/fairness terms.
- [x] Cosine-annealing LR (3e-4 → 1e-5), gradient clipping (max_norm=1.0), ε: 1.0→0.01 over 80% of training.

### WS2: New RL Controllers
- [x] `A2CController`: GAE(λ=0.95), separate actor/critic optimizers, entropy_coef=0.01.
- [x] `SACController`: twin Q-networks, learnable log_α temperature, Polyak update (τ=0.005), off-policy 50k replay.
- [x] `MADDPGController` (`traffic_ai/controllers/maddpg_controller.py`): per-intersection actors + centralized critics, neighbor-augmented observations, Gumbel-Softmax.
- [x] `RecurrentPPOController`: LSTM actor/critic (hidden=64), SEQ_LEN=16 BPTT, per-intersection hidden states.
- [x] Training functions: `traffic_ai/rl_models/a2c.py`, `sac.py`, `maddpg.py`, `recurrent_ppo.py`.

### WS3: 8 New Demand Profiles
- [x] `DemandModel` extended: weekend, school_zone, event_surge, construction, emergency_priority, high_density_developing, incident_response, weather_degraded.
- [x] `service_rate_multiplier()`, `tick_emergency()`, `tick_incident()`, `noncompliance_rate()` methods.
- [x] `IntersectionState`: emergency_active, emergency_direction, emergency_steps_remaining fields.
- [x] Engine: `_apply_emergency_events()`, `_override_emergency_actions()`, demand side-effect ticks.

### WS4: Environmental Impact Tracking
- [x] `EmissionsCalculator` (`traffic_ai/simulation_engine/emissions.py`): EPA MOVES3 idle fuel (0.16 gal/hr), CO₂ (8.887 kg/gal), stop-start penalty, annualise().
- [x] `StepMetrics`: fuel_gallons + co2_kg fields.
- [x] Engine computes fuel/CO₂ per step; aggregate includes total_fuel_gallons + total_co2_kg.
- [x] Dashboard Environmental Impact tab: per-controller fuel/CO₂ bars, tree-years equivalent KPI.

### WS5: Controller Info Cards
- [x] 14 controller expandable glassmorphism cards in dashboard (architecture, strengths, weaknesses).
- [x] `CONTROLLER_INFO` dict wired into `_render_controller_cards()`.

### WS6: Enhanced Statistics
- [x] `statistics.py`: Holm-Bonferroni step-down correction, Cohen's d, `bootstrap_median_difference()`, `statistical_power_analysis()`, median bootstrap CI.
- [x] Dashboard Statistics tab: live correction method switcher, bootstrap CI table with median.

### WS7: UI/UX Overhaul
- [x] Hero: animated pulse-border, updated subtitle with all new features, 8 pills.
- [x] Glassmorphism `ctrl-info-card` with hover + fadeSlideIn animation.
- [x] Environmental metric cards with green-tinted glassmorphism.
- [x] All 11 demand profiles in Live Simulation selectbox with descriptions.
- [x] 4 new RL controllers (A2C, SAC, MADDPG, RecurrentPPO) wired into Live Simulation.
- [x] Fuel/CO₂/tree-years KPI row in Live Simulation results.
- [x] `CONTROLLER_DISPLAY_NAMES` updated for 14 controllers.
- [x] `plotly>=5.20.0` added to requirements.txt.

## Phase 10: Technical Overhaul — DONE

### Engine Unification (Problem 1)
- [x] `TrafficNetworkSimulator` in `simulation_engine/engine.py` is the single canonical engine.
- [x] Added `reset_env()` / `step_env()` Gym-compatible interface to canonical engine.
- [x] Rewrote `simulation/intersection.py` (`MultiIntersectionNetwork`) as a thin wrapper delegating all physics to engine.
- [x] Rewrote `rl_models/environment.py` (`SignalControlEnv`) as a thin wrapper over engine; eliminated duplicate queue/arrival logic.

### Caltrans PeMS Connector (Problem 2)
- [x] `traffic_ai/data_pipeline/pems_connector.py`: `PeMSConnector(station_id, api_key, cache_dir)` fetches detector data from PeMS API.
- [x] Falls back to synthetic Gaussian-mixture calibrated to I-5 San Diego (station 400456) when `PEMS_API_KEY` env var absent; emits `UserWarning`.
- [x] `--pems-station STATION_ID` CLI flag added to `main.py`; calibration always runs (connector handles fallback).
- [x] `calibration_by_hour(df)` returns `dict[int, float]` mapping hour → mean volume.

### HCM 7th Edition Physics (Problem 3)
- [x] `SimulatorConfig`: `min_green_sec=7` (HCM §19.4.3), `yellow_sec=3` (ITE), `all_red_sec=1`, `saturation_flow_rate=1800.0` veh/hr/lane (HCM Table 19-9), `turning_movement_factor=0.60`.
- [x] Yellow/all-red clearance tracked via `transition_steps_remaining` + `target_phase` in `IntersectionState`; no departures during transitions.
- [x] `default_config.yaml` updated with all new signal timing and network constants.

### Expanded RL Action Space (Problem 4)
- [x] `N_ACTIONS = 16`: 2 phases × 8 durations `[15,20,25,30,35,40,45,60]` s.
- [x] `step()` decodes `phase_idx = action // 8`, `duration = GREEN_DURATIONS[action % 8]`.
- [x] `cycle_length_penalty=0.02` per duration-second added to reward in `SignalControlEnv`.
- [x] Observation space expanded to 6 features: `[phase_elapsed/60, duration/60, queue_ns/120, queue_ew/120, tod_norm, upstream_queue/120]`.
- [x] All RL controllers (`QLearning`, `DQN`, `PolicyGradient`, `A2C`, `SAC`, `RecurrentPPO`) updated; `_action_to_phase(action) = action // 8` preserves `select_action` returning 0 or 1.
- [x] `RLPolicyController` (wrapper) updated to build matching 6-feature vector.

### EPA MOVES2014b Emissions (Problem 5)
- [x] Removed fabricated `emissions_proxy = total_queue * 0.21 + phase_changes * 0.8`.
- [x] `idle_co2_rate_per_sec = 0.000457 kg/s/vehicle` (= 0.0274 kg/min ÷ 60; EPA MOVES2014b Table 3.1).
- [x] `StepMetrics.emissions_co2_kg` added as first-class field; `emissions_proxy` aliased to it for backward compatibility.
- [x] `_aggregate_metrics()` exposes `total_emissions_co2_kg`.

### Imitation Learning Labels (Problem 6)
- [x] `generate_imitation_labels(simulator, rule_controller, n_steps)` added to `ml_controllers.py`: runs `AdaptiveRuleController`, records `(7-feature vector, phase_action)` pairs.
- [x] `controller_models.py` uses `generate_imitation_labels` for all ML training; `_heuristic_fallback()` retained only for exceptions.
- [x] `_extract_features()` used consistently at both training time and inference (`SupervisedMLController`) so feature vectors always match.

### Verification
- [x] `python main.py --quick-run --skip-kaggle` completes cleanly; PeMS synthetic fallback warning as expected.
- [x] `pytest -q` — **132 tests pass**.

## Phase 9: Synthetic Data Studio — DONE

### Backend
- [x] `SyntheticDatasetGenerator` (`traffic_ai/data_pipeline/synthetic_generator.py`): `SyntheticDatasetConfig` (35+ parameters), `SyntheticDatasetResult`, fully vectorised NumPy generation (no Python row-loops over samples), pre-computed DemandModel rate table (≤1,152 calls regardless of n_samples).
- [x] 4 label strategies: `optimal` (1-step simulation comparison), `queue_balance` (heuristic), `fixed` (alternating), `adaptive_rule` (RuleBasedController).
- [x] 5 special-scenario overlays: incidents (queue ×3, speed ×0.4), weather (volume ×1.3, speed ×0.85), event surges (×4 pre, ×3.5 post), school-zone concentration, emergency-vehicle clearance.
- [x] `DatasetStore` (`traffic_ai/data_pipeline/dataset_store.py`): atomic CRUD via `os.replace()`, `save`, `load`, `list_datasets`, `delete`, `rename`, `duplicate`, `export_csv`, `get_config`, `_resolve_dir`, `_safe_name`.
- [x] `ModelTrainer` + `TrainingResult` (`traffic_ai/training/trainer.py`): dispatches RL (EnvConfig parameterised from dataset arrival-rate stats) and ML (feature extraction → `ctrl.fit(X, y_str)`); progress callbacks at each stage.
- [x] `DataIngestor.ingest_all()` extended with `synthetic_dataset_name` parameter; `_ingest_studio_dataset()` loads and copies saved datasets into the pipeline.
- [x] `default_config.yaml`: added `synthetic_datasets_dir: data/synthetic_datasets`.

### Dashboard
- [x] "Data Studio" fourth top-level tab added to `run_dashboard()`.
- [x] 4A — Dataset Manager: glassmorphism `.dataset-card` cards (View / Dup / CSV / Del buttons), empty-state CTA, 3-per-row grid.
- [x] 4B — Generator panel: 7 sections (Basics, Network, Volume, Temporal, Scenarios, Labels, Preview & Generate), estimated generation-time display, live progress bar, toast on completion.
- [x] 4C — Dataset Detail: Plotly line + histogram + hour×day heatmap, Download CSV, Edit & Regenerate (pre-fills sliders), Train Model shortcut.
- [x] 4D — Training Workbench: controller selector with info card, dataset selector, RL/ML-adaptive config, `st.status()` live training, reward-curve plot, evaluation metrics.
- [x] 4E — Model Comparison: bar chart of last result's evaluation metrics.
- [x] `.dataset-card` glassmorphism CSS with `transform: scale(1.02)` hover + `transition: all 0.2s ease`.
- [x] `CONTROLLER_DISPLAY_NAMES` extended with 10 Data Studio trainer keys.
- [x] Sidebar Data Studio shortcut label added.

### Tests
- [x] `tests/test_synthetic_generator.py`: 24 tests — columns, row caps, all 4 label strategies, scenario injection (incidents/weather/events shift distributions), full DatasetStore CRUD roundtrip, ModelTrainer DQN 5-episode smoke, Random Forest smoke.


## AITO Platform Transformation — COMPLETE

### Phase 0: AITO Branding — DONE
- [x] All user-facing surfaces renamed to "AITO — AI Traffic Optimization".
- [x] `main.py`, `api/server.py`, `AGENTS.md` updated.

### Phase 1: Controller Cleanup — DONE
- [x] Removed `LSTMForecastController`, `ImitationLearningController`, timeseries/LogisticRegression model.
- [x] Added `BENCHMARK_CONTROLLERS` and `DEPLOYED_CONTROLLERS` registries in `controllers/__init__.py`.
- [x] Tests updated; 125 tests pass.

### Phase 2: 4-Phase Signal Model — DONE
- [x] `SignalPhase` extended: NS_THROUGH, EW_THROUGH, NS_LEFT, EW_LEFT (+ legacy NS/EW).
- [x] `left_queue_matrix` added to `IntersectionState`; left-turn saturation 1200 veh/hr/lane (HCM 7th ed.).
- [x] All RL controllers updated to use `_PHASE_IDX_TO_STR` for 4-phase output.
- [x] MADDPGController updated to N_ACTIONS=4.
- [x] Tests widened to accept full 4-phase value set.

### Phase 3: Multi-Objective Reward — DONE
- [x] `SignalControlEnv` rewritten: 4-phase actions, 8-feature observations, 6-component reward.
- [x] Components: avg_delay, ped_wait, emissions_co2, switch_penalty, throughput, left_starvation.
- [x] Weights configurable in `default_config.yaml` under `rl.reward_weights`.
- [x] `rl_models/dqn.py`, `q_learning.py`, `policy_gradient.py` updated to STATE_DIM=8, N_ACTIONS=4.

### Phase 4: Sensor Fault Model — DONE
- [x] `traffic_ai/simulation_engine/sensor.py`: `SensorFaultModel` (stuck/noise/dropout; PeMS-calibrated).
- [x] `traffic_ai/controllers/fault_tolerant.py`: `FaultTolerantController` with EWMA imputation (α=0.3).
- [x] 12 tests in `tests/test_sensor_fault.py`.

### Phase 5: Shadow Mode — DONE
- [x] `traffic_ai/shadow/shadow_runner.py`: `ShadowModeRunner` + `ShadowReport`.
- [x] Production controller drives simulation; candidate logs counterfactuals only.
- [x] `--shadow-mode`, `--shadow-production`, `--shadow-candidate` CLI flags in `main.py`.
- [x] Output: `artifacts/shadow_report.json`.
- [x] 7 tests in `tests/test_shadow_mode.py`.

### Phase 6: Explainability Engine — DONE
- [x] `traffic_ai/explainability/explainer.py`: `DecisionExplainer`.
- [x] Sensitivity-based feature importances (Ribeiro et al., 2016 KDD).
- [x] 1-3 sentence natural language explanations for all 4 signal phases.
- [x] 11 tests in `tests/test_explainability.py`.

### Phase 7: Dashboard Rewrite — DONE
- [x] Full rewrite as AITO-branded 6-tab professional engineering dashboard.
- [x] Dark navy/teal/gold color palette; CSS glassmorphism.
- [x] Tabs: Network Overview, Benchmark Lab, Shadow Mode, Controller Training, Data & Calibration, Export.
- [x] DecisionExplainer integrated in Controller Training tab.
- [x] 155 tests pass.

### Phase 8: Real-World Traffic Engineering Upgrade — DONE
Based on technical briefings with City of San Diego Senior Traffic Engineer Steve Celniker
and Caltrans District 11 Division Chief Fariba Ramos.

**Phase 8A — Detailed Signal Plan Model**
- [x] `traffic_ai/simulation_engine/signal_plan.py`: `MovementType`, `PhaseConstraints`, `DetailedSignalState`.
- [x] HCM 7th ed. defaults: min_green=7 s, yellow=4 s, all_red=2 s, min_cycle=60 s, max_cycle=180 s.
- [x] 5 tests in `tests/test_signal_plan.py`.

**Phase 8B — Detection Reliability Model**
- [x] `traffic_ai/simulation_engine/detection.py`: `DetectorType`, `DetectorConfig`, `DetectionSystem`.
- [x] Loop detector: binary fail/repair (failure_rate_per_hour, repair_time_hours).
- [x] Video detector: graded noise degradation during solar glare hours and night.
- [x] NONE detector: always failed → fixed-timing fallback.
- [x] 8 tests in `tests/test_detection.py`.

**Phase 8C — Interconnect and Clock Drift**
- [x] `traffic_ai/simulation_engine/interconnect.py`: `InterconnectType`, `InterconnectConfig`, `InterconnectNetwork`.
- [x] Per-link fail/repair; stale observation substitution when link down.
- [x] Clock drift accumulates at 0.5 s/hr (Celniker estimate); GPS sync on repair.

**Phase 8D — Webster (1958) Controller**
- [x] `traffic_ai/controllers/webster.py`: `WebsterController`.
- [x] C_opt = (1.5×L + 5)/(1−Y); g_i proportional to phase demand.
- [x] Oversaturation (Y ≥ 0.99) fallback to max_cycle; clamping to [min_cycle, max_cycle].
- [x] Models Econolite Centracs, SCATS, SCOOT (industry standard for 60+ years).
- [x] 7 tests in `tests/test_webster.py`.

**Phase 8E — Greedy Adaptive (InSync) Controller**
- [x] `traffic_ai/controllers/greedy_adaptive.py`: `GreedyAdaptiveController`.
- [x] Cost = volume_weight×queue + delay_weight×accumulated_delay − coordination_weight×downstream_pressure.
- [x] No fixed cycle; 10 % hysteresis prevents oscillation; min_green=7 steps.
- [x] Models InSync (Rhythm Engineering) deployed on Mira Mesa Blvd and Rosecrans St, San Diego.
- [x] 5 tests in `tests/test_greedy_adaptive.py`.

**Phase 8F — Priority Event System**
- [x] `traffic_ai/simulation_engine/priority.py`: `PriorityEventType`, `PriorityConfig`, `PriorityEventSystem`.
- [x] Emergency vehicle preemption (hard override, non-negotiable).
- [x] Bus transit signal priority (informational, no hard override).
- [x] Leading Pedestrian Interval (LPI) green reduction at phase changes.
- [x] 6 tests in `tests/test_priority.py`.

**Phase 8G — San Diego Corridor Scenarios**
- [x] `traffic_ai/scenarios/san_diego.py`: `SanDiegoScenario`.
- [x] Scenarios: downtown_grid (16 int), mira_mesa_corridor (8 int), rosecrans_corridor (12 int), mixed_jurisdiction (12 int).
- [x] Rosecrans corridor calibrated to verified 2017 results: 25 % travel time ↓, 53 % stops ↓.
- [x] 6 tests in `tests/test_scenarios.py`.

**Phase 8H/I — Engine and Experiment Runner Integration**
- [x] `engine.py`: DetectionSystem, InterconnectNetwork, PriorityEventSystem integrated into run loop.
- [x] `types.py`: `StepMetrics` extended with clearance_loss_sec, detector_fallback_steps, preemption_events.
- [x] All subsystems default-disabled (backward compatible with all 157 existing tests).
- [x] `controllers/__init__.py` and `factory.py` updated with Webster and GreedyAdaptive.

**Phase 8J — Dashboard Update**
- [x] Phase 8 sidebar panel: San Diego scenario selector, Detector Failures toggle, Priority Events toggles.
- [x] Webster (1958) and Greedy Adaptive (InSync) added to Live Simulation controller dropdown.
- [x] New "Industry Comparison" tab (7th tab): real-world system reference table, methodology comparison, corridor scenarios.
- [x] Industry comparison shows clearance loss, detector uptime %, preemptions from last benchmark.

**Phase 8K — Test Suite**
- [x] 31 new tests across 6 test files (signal_plan, detection, priority, webster, greedy_adaptive, scenarios).
- [x] All 188+ tests pass.

**Phase 8L — Documentation**
- [x] PLANS.md updated with Phase 8 section.
- [x] README.md updated with Real-World Features, San Diego Scenarios, Industry Comparison sections.

### Phase 9: Real Data Validation — DONE

Answers the question: "What real data did you use?" when presenting to
Fariba Ramos (Caltrans District 11) or Steve Celniker (City of San Diego).

**Phase 9A — PeMS CSV Adapter**
- [x] `PeMSConnector.load_from_csv()`: parses real Caltrans PeMS Station 5-Minute
  CSV files; tolerates missing columns, bad rows, messy quoting.
- [x] `PeMSConnector.compute_hourly_demand_profile()`: weekday-filtered,
  quality-filtered (pct_observed ≥ 0.5), fallback to 0.12 if < 3 days data.
- [x] `PeMSConnector.auto_detect_pems_files()`: scans data/raw/ for pems_station_*.csv.
- [x] `PeMSConnector.load_best_available()`: real PeMS or synthetic fallback with
  clear source labeling.

**Phase 9B — Rosecrans Corridor Validator**
- [x] `traffic_ai/validation/rosecrans_validator.py`: `RosecransValidator`,
  `ValidationResult`, `ROSECRANS_BENCHMARK` constants.
- [x] Compares GreedyAdaptive vs FixedTiming on 12-signal Rosecrans corridor.
- [x] Reports gap vs verified 25% real-world improvement (Faulconer 2017).
- [x] Saves `artifacts/validation_report.json`.
- [x] `--validate-rosecrans` CLI flag wired into `main.py`.

**Phase 9C — Demand Model Calibration**
- [x] `DemandModel.calibrate_from_pems_profile()`: uses real PeMS hourly rates
  instead of synthetic Gaussian peaks when calibrated.
- [x] Scale multiplier still applied on top; fully backward compatible.

**Phase 9D — Dashboard Validation Tab**
- [x] New "Validation" tab (7th tab): Rosecrans corridor comparison cards,
  PeMS CSV upload widget, demand profile comparison chart (synthetic vs real).

**Phase 9E — README**
- [x] "Using Real Traffic Data (PeMS)" section with download instructions,
  `--validate-rosecrans` usage, example output.

**Phase 9F — Tests**
- [x] `tests/test_pems_adapter.py`: 7 tests (CSV columns, hourly profile,
  data quality filter, arrival rate formula, auto-detect, load_best_available).
- [x] `tests/test_rosecrans_validator.py`: 5 tests (synthetic fallback, positive
  improvement, JSON written, benchmark constants, summary_lines).
- [x] `tests/test_demand_calibration.py`: 5 tests (PeMS override, synthetic mode,
  rush-hour higher, scale multiplier, replaceable calibration).
- [x] All 212+ tests pass.
