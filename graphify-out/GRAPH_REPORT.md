# Graph Report - .  (2026-04-10)

## Corpus Check
- 123 files · ~142,920 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1548 nodes · 3536 edges · 64 communities detected
- Extraction: 56% EXTRACTED · 44% INFERRED · 0% AMBIGUOUS · INFERRED: 1561 edges (avg confidence: 0.5)
- Token cost: 0 input · 0 output

## God Nodes (most connected - your core abstractions)
1. `TrafficNetworkSimulator` - 119 edges
2. `SimulatorConfig` - 107 edges
3. `DemandModel` - 81 edges
4. `BaseController` - 81 edges
5. `FixedTimingController` - 79 edges
6. `RuleBasedController` - 66 edges
7. `SignalControlEnv` - 52 edges
8. `Settings` - 42 edges
9. `PeMSConnector` - 41 edges
10. `SyntheticDatasetConfig` - 40 edges

## Surprising Connections (you probably didn't know these)
- `tests/test_webster.py  Tests for traffic_ai.controllers.webster.WebsterControlle` --uses--> `WebsterController`  [INFERRED]
  tests/test_webster.py → traffic_ai/controllers/webster.py
- `Webster C_opt = (1.5*L + 5) / (1 - Y) for non-saturated demand.` --uses--> `WebsterController`  [INFERRED]
  tests/test_webster.py → traffic_ai/controllers/webster.py
- `NS green is proportional to NS demand; heavier demand gets longer green.` --uses--> `WebsterController`  [INFERRED]
  tests/test_webster.py → traffic_ai/controllers/webster.py
- `When Y >= 0.99, controller uses max_cycle instead of Webster formula.` --uses--> `WebsterController`  [INFERRED]
  tests/test_webster.py → traffic_ai/controllers/webster.py
- `Very low demand → C_opt < min_cycle → cycle is clamped to min_cycle.` --uses--> `WebsterController`  [INFERRED]
  tests/test_webster.py → traffic_ai/controllers/webster.py

## Communities

### Community 0 - "C0"
Cohesion: 0.02
Nodes (169): traffic_ai/rl_models/a2c.py  Advantage Actor-Critic (A2C) training function.  Us, Train an A2C controller on the traffic simulation.      Parameters     ---------, train_a2c(), AdaptiveRuleController, ControllerTrainingData, generate_controller_training_data(), _heuristic_fallback(), traffic_ai/ml_models/controller_models.py  Supervised controller training using (+161 more)

### Community 1 - "C1"
Cohesion: 0.02
Nodes (145): _config_from_dict(), DatasetStore, traffic_ai/data_pipeline/dataset_store.py  Filesystem-backed CRUD store for save, Returns a list of summary dicts for every saved dataset.          Each summary c, Load a saved dataset by its human-readable or safe name.          Parameters, Delete a dataset directory.          Returns         -------         bool, Rename a dataset.          Returns         -------         bool             True, Duplicate a dataset under a new name.          Returns         -------         b (+137 more)

### Community 2 - "C2"
Cohesion: 0.03
Nodes (86): ABC, BaseController, compute_actions(), Select a phase action index for a single intersection.          Returns, Update internal state/model with a transition. Default: no-op., BaseController, traffic_ai/training — Unified model training interface., MADDPGController (+78 more)

### Community 3 - "C3"
Cohesion: 0.03
Nodes (86): ABComparison, ABComparisonEngine, ABResult, ActuatedCorridorController, AICorridorController, FixedCorridorController, A/B Comparison Engine: runs Fixed vs Actuated vs AI-Optimized side by side.  Pro, AI-optimized controller with predictive + RL components. (+78 more)

### Community 4 - "C4"
Cohesion: 0.05
Nodes (45): DataCleaner, Parses timestamps, uppercases direction codes, and converts numerics safely., Clips numeric sensor columns to the [P1, P99] range; pins occupancy to [0, 1]., Forward-fills then back-fills per location; falls back to column median., Applies the full cleaning pipeline: normalise → ensure → coerce → clip → fill →, Lowercases and strips column names, then applies COLUMN_ALIASES remapping., Guarantees every required column exists before type coercion downstream., FeatureEngineer (+37 more)

### Community 5 - "C5"
Cohesion: 0.03
Nodes (79): A2C Controller, AITO — AI Traffic Optimization Platform, BaseController Interface, DQN: 49.64s avg wait (+143.8% vs fixed, underfitted), Gradient Boosting: 17.32s avg wait, efficiency 0.787, Q-Learning: 19.54s avg wait (−4.0% vs fixed), Random Forest: 15.40s avg wait (−24.4% vs fixed), XGBoost: 17.05s avg wait (−16.3% vs fixed) (+71 more)

### Community 6 - "C6"
Cohesion: 0.05
Nodes (57): FaultTolerantController, traffic_ai/controllers/fault_tolerant.py  Fault-tolerant controller wrapper for, EWMA-imputing wrapper around any BaseController.      Parameters     ----------, Return obs with EWMA-imputed values for suspicious (≤ 0) readings., FixedTimingController, traffic_ai/controllers/fixed.py  FixedTimingController: cycles signal phases on, Cycle signal phases on a fixed schedule.      Parameters     ----------     cycl, traffic_ai/simulation_engine/sensor.py  Sensor fault model for AITO Phase 4.  Si (+49 more)

### Community 7 - "C7"
Cohesion: 0.05
Nodes (45): Enum, InterconnectLink, InterconnectType, traffic_ai/simulation_engine/interconnect.py  Intersection interconnect and cloc, Advance link state by one simulation step., Return stale historical data when interconnect is unavailable., Return (possibly filtered) observation of *to_node* from *from_node*'s perspecti, Bidirectional communication link between two adjacent intersections.      Tracks (+37 more)

### Community 8 - "C8"
Cohesion: 0.09
Nodes (27): EpisodeMetricsTracker, EpisodeSummary, traffic_ai/simulation/metrics.py  Per-step and per-episode metrics tracker for t, Finalise the current episode, compute summary, and write CSV., Snapshot of simulation metrics at a single step., Aggregate metrics over a full episode., Track per-step and per-episode metrics and export to CSV.      Parameters     --, Record metrics for a single simulation step.          Parameters         ------- (+19 more)

### Community 9 - "C9"
Cohesion: 0.08
Nodes (32): Controller: rl_dqn (Deep Q-Network), Controller: rl_policy_gradient, Controller: rl_q_learning, Controller: Webster (fixed-timing baseline), Finding: DQN converges from ~-400 episode reward at episode 0 to stable ~-50 to -75 reward range by episode 750-1000, showing successful RL training, Finding: DQN training plateaus after ~750 episodes with high variance but stable mean reward, indicating policy convergence, Finding: RL controllers (rl_policy_gradient, rl_q_learning) show lowest total queue (darkest purple) in heatmap, Finding: Rule-based controllers (adaptive_rule, fixed_timing, greedy_adaptive, max_pressure) show moderate queue levels (mid-range colors) across simulation (+24 more)

### Community 10 - "C10"
Cohesion: 0.1
Nodes (10): Gumbel-Softmax for MADDPG Discrete Actions, _Actor, _Critic, _MADDPGReplayBuffer, traffic_ai/controllers/maddpg_controller.py  Multi-Agent Deep Deterministic Poli, Stores joint (obs, actions, rewards, next_obs, dones) tuples., Build (n_agents, ACTOR_INPUT_DIM) observation array.          For each agent, ba, Simple local reward: negative of normalized total_queue per agent. (+2 more)

### Community 11 - "C11"
Cohesion: 0.13
Nodes (25): _make_full_obs(), _make_obs(), test_a2c_compute_actions_valid(), test_a2c_runs_multiple_steps(), test_compute_actions_returns_valid_phases(), test_dqn_action_valid(), test_fixed_timing_default_30s_cycle(), test_fixed_timing_switches_phase_at_correct_step() (+17 more)

### Community 12 - "C12"
Cohesion: 0.15
Nodes (21): _make_result(), _minimal_config(), test_events_raise_vehicle_count(), test_generate_columns(), test_generate_metadata(), test_generate_row_count_capped(), test_generate_row_count_non_zero(), test_incidents_change_distributions() (+13 more)

### Community 13 - "C13"
Cohesion: 0.25
Nodes (21): _chart_layout(), _display_name(), _get_controller_class(), _kpi_card(), _load_artifacts_df(), _load_settings(), main(), _render_benchmark_results() (+13 more)

### Community 14 - "C14"
Cohesion: 0.18
Nodes (16): ExplanationResult, traffic_ai/explainability/explainer.py  Decision explainability engine for AITO, Structured explanation for a single controller decision., _make_obs(), tests/test_explainability.py  Tests for AITO Phase 6 explainability engine., test_emergency_explanation_mentions_emergency(), test_explain_action_stored(), test_explain_dominant_feature_in_importances() (+8 more)

### Community 15 - "C15"
Cohesion: 0.17
Nodes (16): _obs(), tests/test_webster.py  Tests for traffic_ai.controllers.webster.WebsterControlle, Very low demand → C_opt < min_cycle → cycle is clamped to min_cycle., Demand near saturation → C_opt → ∞ → cycle is clamped to max_cycle., Timing plan only updates after recalc_interval steps., Webster produces one action per intersection., Webster C_opt = (1.5*L + 5) / (1 - Y) for non-saturated demand., NS green is proportional to NS demand; heavier demand gets longer green. (+8 more)

### Community 16 - "C16"
Cohesion: 0.14
Nodes (10): _controller_color(), plot_ablation(), plot_controller_comparison(), plot_controller_performance(), plot_feature_importance(), plot_queue_over_time(), Grouped bar chart of avg_wait_time and estimated_co2_grams with std-dev error ba, Horizontal bar chart of feature importances for RF and XGBoost. (+2 more)

### Community 17 - "C17"
Cohesion: 0.16
Nodes (14): _apply_correction(), bootstrap_confidence_interval(), bootstrap_median_difference(), _cohens_d(), controller_bootstrap_table(), pairwise_significance_tests(), Bootstrap CI for a sample statistic.      Parameters     ----------     values:, Bootstrap CI for the difference of medians (median_a - median_b).      Unlike th (+6 more)

### Community 18 - "C18"
Cohesion: 0.15
Nodes (12): BaseModel, get_plot(), get_results(), health(), Return a plot PNG for a given run ID and plot name., Health check endpoint., Trigger ExperimentRunner and return artifact paths., Return summary CSV as JSON for a given run ID. (+4 more)

### Community 19 - "C19"
Cohesion: 0.22
Nodes (7): BaseEstimator, ClassifierMixin, ModelEvaluation, _quick_fit_model(), _search_space(), SupervisedModelSuite, TimeSeriesPhaseClassifier

### Community 20 - "C20"
Cohesion: 0.21
Nodes (7): RLPolicyController, --pretrain-only trains RL models and exits without running CV benchmark., RLPolicyController passes upstream_queue as feature[7], normalised by 120., _temp_settings(), test_rl_controller_uses_upstream_queue(), test_runner_ingest_only(), test_runner_pretrain_only()

### Community 21 - "C21"
Cohesion: 0.17
Nodes (11): tests/test_demand_calibration.py  Tests for Phase 9C: DemandModel.calibrate_from, Calling calibrate_from_pems_profile twice replaces the first profile., DemandModel uses PeMS rates when calibrated., DemandModel uses default Poisson rates without calibration., Hour 7-9 arrival rate exceeds hour 2-4 in both synthetic and PeMS modes., Global scale multiplier is applied on top of PeMS calibration., test_arrival_rate_at_rush_hour_is_higher(), test_calibrate_from_pems_overrides_synthetic() (+3 more)

### Community 22 - "C22"
Cohesion: 0.42
Nodes (8): main(), render_ab_comparison(), render_emissions_dashboard(), render_hero(), render_kpi_strip(), render_live_corridor(), render_sd_context(), render_technical_section()

### Community 23 - "C23"
Cohesion: 0.25
Nodes (6): traffic_ai/rl_models/rewards.py  Composite reward shaping for traffic signal RL, Configurable weights for each reward component., Compute a composite, normalised reward signal for a single step.      All inputs, Compute the composite reward for a single simulation step.          Parameters, RewardShaper, RewardWeights

### Community 24 - "C24"
Cohesion: 0.29
Nodes (3): get_scenario(), list_scenarios(), traffic_ai/scenarios/san_diego.py  San Diego corridor scenario configurations.

### Community 25 - "C25"
Cohesion: 0.6
Nodes (5): main(), _maybe_calibrate_from_pems(), parse_args(), _run_pems_calibrate(), _run_shadow_mode()

### Community 26 - "C26"
Cohesion: 0.4
Nodes (2): compute_system_efficiency_score(), _min_max()

### Community 27 - "C27"
Cohesion: 0.33
Nodes (4): EmergencyEvent, traffic_ai/simulation_engine/demand.py  Stochastic vehicle arrival demand model, Progress emergency vehicle simulation for one step.          Called by the engin, Active emergency vehicle event.

### Community 28 - "C28"
Cohesion: 0.33
Nodes (0): 

### Community 29 - "C29"
Cohesion: 0.33
Nodes (4): traffic_ai/rl_models/replay_buffer.py  Prioritized Experience Replay (PER) buffe, Single stored experience., Add a new transition with maximum current priority., Transition

### Community 30 - "C30"
Cohesion: 0.33
Nodes (6): 5-fold Cross-Validation Benchmark Protocol, Bootstrap Confidence Intervals (n=300), Cohen's d Effect Size for Statistical Comparisons, ExperimentRunner (5-fold CV, Holm-Bonferroni), Holm-Bonferroni Multiple Comparison Correction, Mann-Whitney U Test (Non-parametric)

### Community 31 - "C31"
Cohesion: 0.4
Nodes (0): 

### Community 32 - "C32"
Cohesion: 1.0
Nodes (2): _temp_settings(), test_data_pipeline_end_to_end()

### Community 33 - "C33"
Cohesion: 0.67
Nodes (0): 

### Community 34 - "C34"
Cohesion: 0.67
Nodes (3): DecisionExplainer (NL + feature importances), Ribeiro et al. (2016) KDD — Sensitivity-based feature importance, Sensitivity-Based Feature Importance (Ribeiro 2016 KDD)

### Community 35 - "C35"
Cohesion: 1.0
Nodes (0): 

### Community 36 - "C36"
Cohesion: 1.0
Nodes (0): 

### Community 37 - "C37"
Cohesion: 1.0
Nodes (1): True when detection is lost and controller must fall back to fixed timing.

### Community 38 - "C38"
Cohesion: 1.0
Nodes (1): Human-readable detector status string.

### Community 39 - "C39"
Cohesion: 1.0
Nodes (1): Total through-lane queue for NS axis.

### Community 40 - "C40"
Cohesion: 1.0
Nodes (1): Total through-lane queue for EW axis.

### Community 41 - "C41"
Cohesion: 1.0
Nodes (1): All vehicles queued (through + left).

### Community 42 - "C42"
Cohesion: 1.0
Nodes (1): Left-turn queue for NS axis.

### Community 43 - "C43"
Cohesion: 1.0
Nodes (1): Left-turn queue for EW axis.

### Community 44 - "C44"
Cohesion: 1.0
Nodes (1): Total clearance interval = yellow + all-red (seconds).

### Community 45 - "C45"
Cohesion: 1.0
Nodes (1): Lost time per phase change (yellow + all-red, seconds).

### Community 46 - "C46"
Cohesion: 1.0
Nodes (1): 4-Phase Signal Model (NS_THROUGH, EW_THROUGH, NS_LEFT, EW_LEFT)

### Community 47 - "C47"
Cohesion: 1.0
Nodes (1): Varaiya (2013) Trans. Res. Part C — Max pressure control

### Community 48 - "C48"
Cohesion: 1.0
Nodes (1): AGENTS.md — Platform architecture and engineering rules

### Community 49 - "C49"
Cohesion: 1.0
Nodes (1): PLANS.md — Development roadmap (Phases 1-9)

### Community 50 - "C50"
Cohesion: 1.0
Nodes (1): README.md — AITO platform documentation

### Community 51 - "C51"
Cohesion: 1.0
Nodes (1): PROJECT_SUMMARY_GSDSEF.md — Science fair submission

### Community 52 - "C52"
Cohesion: 1.0
Nodes (1): PRESENTATION_NOTES.md — 3-minute speaking notes for GSDSEF

### Community 53 - "C53"
Cohesion: 1.0
Nodes (1): project_summary.txt — Codebase architecture overview

### Community 54 - "C54"
Cohesion: 1.0
Nodes (1): requirements.txt — Python dependencies

### Community 55 - "C55"
Cohesion: 1.0
Nodes (1): docs/TECHNICAL_BRIEF.md — Technical brief for city/state agencies

### Community 56 - "C56"
Cohesion: 1.0
Nodes (1): Controller: adaptive_rule

### Community 57 - "C57"
Cohesion: 1.0
Nodes (1): Controller: fixed_timing

### Community 58 - "C58"
Cohesion: 1.0
Nodes (1): Controller: greedy_adaptive

### Community 59 - "C59"
Cohesion: 1.0
Nodes (1): Controller: max_pressure

### Community 60 - "C60"
Cohesion: 1.0
Nodes (1): Controller: ml_gradientboostingclassifier

### Community 61 - "C61"
Cohesion: 1.0
Nodes (1): Controller: ml_mlpclassifier

### Community 62 - "C62"
Cohesion: 1.0
Nodes (1): Controller: ml_randomforestclassifier

### Community 63 - "C63"
Cohesion: 1.0
Nodes (1): Controller: ml_xgbclassifier

## Knowledge Gaps
- **232 isolated node(s):** `traffic_ai/data_pipeline/pems_connector.py  Caltrans PeMS (Performance Measureme`, `Return the first matching column name (case-insensitive), or None.`, `Convert value to float; return default on any failure.`, `Generate standard PeMS 5-minute column names for n_cols total columns.      PeMS`, `Read a PeMS CSV, auto-detecting whether it has a header row.      PeMS Data Clea` (+227 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `C35`** (2 nodes): `reproducibility.py`, `set_global_seed()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C36`** (1 nodes): `setup.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C37`** (1 nodes): `True when detection is lost and controller must fall back to fixed timing.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C38`** (1 nodes): `Human-readable detector status string.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C39`** (1 nodes): `Total through-lane queue for NS axis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C40`** (1 nodes): `Total through-lane queue for EW axis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C41`** (1 nodes): `All vehicles queued (through + left).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C42`** (1 nodes): `Left-turn queue for NS axis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C43`** (1 nodes): `Left-turn queue for EW axis.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C44`** (1 nodes): `Total clearance interval = yellow + all-red (seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C45`** (1 nodes): `Lost time per phase change (yellow + all-red, seconds).`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C46`** (1 nodes): `4-Phase Signal Model (NS_THROUGH, EW_THROUGH, NS_LEFT, EW_LEFT)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C47`** (1 nodes): `Varaiya (2013) Trans. Res. Part C — Max pressure control`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C48`** (1 nodes): `AGENTS.md — Platform architecture and engineering rules`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C49`** (1 nodes): `PLANS.md — Development roadmap (Phases 1-9)`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C50`** (1 nodes): `README.md — AITO platform documentation`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C51`** (1 nodes): `PROJECT_SUMMARY_GSDSEF.md — Science fair submission`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C52`** (1 nodes): `PRESENTATION_NOTES.md — 3-minute speaking notes for GSDSEF`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C53`** (1 nodes): `project_summary.txt — Codebase architecture overview`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C54`** (1 nodes): `requirements.txt — Python dependencies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C55`** (1 nodes): `docs/TECHNICAL_BRIEF.md — Technical brief for city/state agencies`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C56`** (1 nodes): `Controller: adaptive_rule`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C57`** (1 nodes): `Controller: fixed_timing`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C58`** (1 nodes): `Controller: greedy_adaptive`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C59`** (1 nodes): `Controller: max_pressure`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C60`** (1 nodes): `Controller: ml_gradientboostingclassifier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C61`** (1 nodes): `Controller: ml_mlpclassifier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C62`** (1 nodes): `Controller: ml_randomforestclassifier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `C63`** (1 nodes): `Controller: ml_xgbclassifier`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `TrafficNetworkSimulator` connect `C0` to `C1`, `C2`, `C3`, `C6`?**
  _High betweenness centrality (0.180) - this node is a cross-community bridge._
- **Why does `SimulatorConfig` connect `C0` to `C1`, `C2`, `C3`, `C6`, `C24`?**
  _High betweenness centrality (0.137) - this node is a cross-community bridge._
- **Why does `Settings` connect `C4` to `C1`, `C2`, `C3`?**
  _High betweenness centrality (0.110) - this node is a cross-community bridge._
- **Are the 98 inferred relationships involving `TrafficNetworkSimulator` (e.g. with `tests/test_simulation.py  Tests for traffic_ai.simulation.intersection.MultiInte` and `Reward is negative total queue, so ≤ 0.`) actually correct?**
  _`TrafficNetworkSimulator` has 98 INFERRED edges - model-reasoned connections that need verification._
- **Are the 106 inferred relationships involving `SimulatorConfig` (e.g. with `Attempt PeMS calibration and store the calibration dict in settings.` and `Test PeMS connection and pull calibration data for the given station.`) actually correct?**
  _`SimulatorConfig` has 106 INFERRED edges - model-reasoned connections that need verification._
- **Are the 59 inferred relationships involving `DemandModel` (e.g. with `tests/test_demand_calibration.py  Tests for Phase 9C: DemandModel.calibrate_from` and `DemandModel uses PeMS rates when calibrated.`) actually correct?**
  _`DemandModel` has 59 INFERRED edges - model-reasoned connections that need verification._
- **Are the 75 inferred relationships involving `BaseController` (e.g. with `InstrumentedEnv` and `PhaseCounting`) actually correct?**
  _`BaseController` has 75 INFERRED edges - model-reasoned connections that need verification._