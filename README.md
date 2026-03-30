# AITO — AI Traffic Optimization

<div align="center">

**Professional engineering platform for traffic signal control research and deployment**

*4-phase signal model · Shadow mode deployment · Sensor fault tolerance · EPA MOVES2014b emissions*

[![Python 3.11](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Tests: 155](https://img.shields.io/badge/tests-155%20passing-brightgreen.svg)]()

</div>

---

## Overview

AITO benchmarks **14 traffic signal controllers** across four families — fixed timing, adaptive rule-based, supervised ML (RF, XGBoost, GBM, MLP), and reinforcement learning (Q-Learning, DQN, PPO, A2C, SAC, RecurrentPPO, MADDPG) — on simulated urban intersection networks.

The platform is designed for **traffic engineers**, not just researchers:

- **4-phase signal model**: NS_THROUGH, EW_THROUGH, NS_LEFT, EW_LEFT with HCM 7th edition saturation flows
- **Shadow mode**: evaluate AI controllers without touching live traffic
- **Sensor fault tolerance**: EWMA imputation for stuck/noisy/dropout loop detectors
- **EPA MOVES2014b emissions**: first-class CO₂ tracking, not a proxy
- **Statistical rigor**: Holm-Bonferroni correction across all pairwise comparisons
- **Explainability**: natural language decision explanations for any controller

---

## Architecture

```
traffic_ai/
├── controllers/           # 14 signal controllers (all implement BaseController)
│   ├── base.py            # BaseController: reset, compute_actions, select_action, update
│   ├── fixed.py           # Fixed timing baseline (30s cycle)
│   ├── rule_based.py      # Queue-threshold adaptive
│   ├── ml_controllers.py  # RF, XGBoost, GradientBoosting, MLP
│   ├── rl_controllers.py  # QLearning, DQN, PPO, A2C, SAC, RecurrentPPO
│   ├── maddpg_controller.py # Multi-agent MADDPG (centralized training)
│   └── fault_tolerant.py  # EWMA-imputing fault wrapper
├── simulation_engine/     # Physics + types (HCM 7th ed.)
│   ├── engine.py          # TrafficNetworkSimulator
│   ├── types.py           # SignalPhase (6 values), PHASE_TO_IDX, IntersectionState
│   └── sensor.py          # SensorFaultModel (stuck/noise/dropout)
├── rl_models/             # RL training
│   ├── environment.py     # SignalControlEnv: 4-phase, 8-obs, 6-reward
│   ├── dqn.py             # DQNetwork + train_dqn
│   ├── q_learning.py      # QLearningPolicy + train_q_learning
│   └── policy_gradient.py # PolicyNet + train_policy_gradient
├── shadow/                # Shadow mode deployment
│   └── shadow_runner.py   # ShadowModeRunner → artifacts/shadow_report.json
├── explainability/        # AI transparency
│   └── explainer.py       # DecisionExplainer — NL + feature importances
├── dashboard/             # 6-tab Streamlit engineering dashboard
│   └── streamlit_app.py
├── data_pipeline/         # Ingestion, synthetic generation, PeMS calibration
├── training/              # ModelTrainer (unified ML + RL)
├── experiments/           # ExperimentRunner, Holm-Bonferroni
└── config/                # Settings, default_config.yaml
```

---

## Quickstart

```bash
# Install dependencies
pip install -r requirements.txt

# Run quick benchmark (all controllers, ~2 min)
python main.py --quick-run

# Run full benchmark
python main.py

# Launch the engineering dashboard
streamlit run traffic_ai/dashboard/streamlit_app.py

# Shadow mode: evaluate DQN vs fixed timing without affecting simulation
python main.py --shadow-mode --shadow-production fixed_timing --shadow-candidate dqn

# Run tests
pytest -q
```

---

## CLI Reference

```
python main.py [OPTIONS]

Options:
  --config PATH             YAML configuration file (default: traffic_ai/config/default_config.yaml)
  --quick-run               Reduced budget for fast iteration
  --ingest-only             Run data ingestion only
  --skip-kaggle             Disable Kaggle dataset ingestion
  --skip-public             Disable public dataset ingestion
  --output-dir DIR          Override artifact output directory
  --pems-station STATION_ID Caltrans PeMS station for demand calibration
                            (requires PEMS_API_KEY env var; default: 400456)
  --shadow-mode             Run shadow mode evaluation
  --shadow-production CTRL  Production controller (default: fixed_timing)
  --shadow-candidate CTRL   Candidate AI controller (default: dqn)
```

---

## Signal Model

AITO uses a **4-phase protected signal model** (HCM 7th edition):

| Phase | SignalPhase | Serves | Saturation Flow |
|-------|-------------|--------|----------------|
| 0 | `NS_THROUGH` | N + S through + right | 1800 veh/hr/lane |
| 1 | `EW_THROUGH` | E + W through + right | 1800 veh/hr/lane |
| 2 | `NS_LEFT` | N + S protected left | 1200 veh/hr/lane |
| 3 | `EW_LEFT` | E + W protected left | 1200 veh/hr/lane |

Legacy `"NS"` and `"EW"` strings remain valid for backward compatibility.

---

## Multi-Objective Reward

RL controllers optimize a 6-component reward (configurable weights in `default_config.yaml`):

```
R = -w1·avg_delay - w2·ped_wait - w3·emissions_co2 - switch_cost + w4·throughput - w5·left_starvation

Default: w1=0.12, w2=0.05, w3=0.03, switch_cost=2.0, w4=0.08, w5=0.04
Source: Mannion et al. (2016) AAMAS workshop; Liang et al. (2019) ITSC
```

---

## Shadow Mode

Shadow mode allows safe evaluation of AI controllers **without modifying live traffic**:

```python
from traffic_ai.shadow.shadow_runner import ShadowModeRunner
from traffic_ai.controllers.fixed import FixedTimingController
from traffic_ai.controllers.rl_controllers import DQNController

runner = ShadowModeRunner(
    production=FixedTimingController(),
    candidate=DQNController(),
)
report = runner.run()
# report.agreement_rate, report.estimated_queue_reduction_pct
runner.save_report(report, Path("artifacts/shadow_report.json"))
```

---

## Sensor Fault Tolerance

```python
from traffic_ai.simulation_engine.sensor import SensorFaultModel
from traffic_ai.controllers.fault_tolerant import FaultTolerantController

fault = SensorFaultModel(stuck_prob=0.02, noise_std=0.05, dropout_prob=0.01)
corrupted_obs = fault.apply(raw_obs, step=t, intersection_id=iid)

# Wrap any controller for EWMA-based fault tolerance
ctrl = FaultTolerantController(DQNController(), alpha=0.3)
```

---

## Decision Explainability

```python
from traffic_ai.explainability.explainer import DecisionExplainer

explainer = DecisionExplainer(controller=ctrl)
result = explainer.explain(obs, action=action)
print(result.natural_language)
# "Selected EW_THROUGH (eastbound/westbound green). Key driver: eastbound/westbound
#  through queue (highest feature influence). EW queue (24 veh) exceeds NS queue
#  (8 veh) — EW throughput prioritised."
```

---

## PeMS Calibration

```bash
export PEMS_API_KEY=your_key_here
python main.py --pems-station 400456
```

Falls back to synthetic calibration when `PEMS_API_KEY` is absent.

---

## For Engineers

AITO is structured for deployment validation workflows:

1. **Benchmark**: run `python main.py --quick-run` to establish baseline metrics
2. **Shadow mode**: run `--shadow-mode` to evaluate AI without traffic impact
3. **Explainability**: use `DecisionExplainer` to audit AI decisions
4. **Fault testing**: wrap your controller with `FaultTolerantController` before field deployment
5. **Dashboard**: `streamlit run traffic_ai/dashboard/streamlit_app.py` for live monitoring

---

## Statistical Methodology

All pairwise comparisons use:
- **Mann-Whitney U test** (non-parametric, no normality assumption)
- **Holm-Bonferroni correction** for family-wise error rate
- **Bootstrap confidence intervals** (n=300)
- Significance threshold: α = 0.05

---

## Citations

- HCM 7th Edition (TRB, 2022) — signal timing, saturation flow rates
- EPA MOVES2014b — CO₂ idle emission rates (0.000457 kg/s/vehicle)
- Mannion et al. (2016), AAMAS — multi-objective RL reward weights
- Liang et al. (2019), ITSC — throughput bonus formulation
- Chen et al. (2001), Transport. Res. Part C — sensor fault characterization
- Toth & Ceder (2002), Transport. Res. Part C — EWMA imputation (α=0.3)
- Ribeiro et al. (2016), KDD — sensitivity-based feature importance
