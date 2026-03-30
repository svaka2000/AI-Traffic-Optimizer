"""traffic_ai/dashboard/streamlit_app.py

AITO — AI Traffic Optimization | Professional Engineering Dashboard

6-Tab Engineering Interface:
    1. Network Overview     — Live simulation, KPI cards, real-time metrics
    2. Benchmark Lab        — Multi-controller comparison, statistical significance
    3. Shadow Mode          — AI evaluation without live traffic risk
    4. Controller Training  — Train ML/RL controllers, view learning curves
    5. Data & Calibration   — Synthetic datasets, PeMS calibration, sensor faults
    6. Export               — Download reports, CSVs, shadow mode reports

Color Palette
-------------
    Navy   : #0A1628  (backgrounds)
    Teal   : #00C2CB  (primary accent, highlights)
    Gold   : #F0B429  (secondary accent, KPI values)
    Dark   : #0D1E35  (card backgrounds)
    Light  : #E8F4F8  (text on dark)
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Ensure project root is on path for Streamlit Cloud
_project_root = Path(__file__).resolve().parents[2]
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

import numpy as np
import pandas as pd
import streamlit as st

# ---------------------------------------------------------------------------
# Page config — must be first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="AITO — AI Traffic Optimization",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# AITO Dark Theme CSS
# ---------------------------------------------------------------------------

AITO_CSS = """
<style>
/* Root variables */
:root {
    --navy: #0A1628;
    --dark: #0D1E35;
    --teal: #00C2CB;
    --gold: #F0B429;
    --light: #E8F4F8;
    --muted: #7A9AB5;
    --danger: #FF4B4B;
    --success: #00C851;
}

/* App background */
.stApp { background-color: var(--navy); color: var(--light); }
[data-testid="stSidebar"] { background-color: var(--dark); }
[data-testid="stSidebar"] * { color: var(--light) !important; }

/* Headers */
h1, h2, h3 { color: var(--teal) !important; }
h4, h5, h6 { color: var(--gold) !important; }

/* KPI cards */
.kpi-card {
    background: var(--dark);
    border: 1px solid var(--teal);
    border-radius: 8px;
    padding: 16px 20px;
    text-align: center;
    margin: 4px 0;
}
.kpi-label { color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 1px; }
.kpi-value { color: var(--gold); font-size: 28px; font-weight: 700; margin: 4px 0; }
.kpi-delta { font-size: 12px; }
.kpi-delta.positive { color: var(--success); }
.kpi-delta.negative { color: var(--danger); }

/* Section headers */
.section-header {
    border-left: 4px solid var(--teal);
    padding-left: 12px;
    margin: 16px 0 8px 0;
    color: var(--light);
    font-size: 16px;
    font-weight: 600;
}

/* Badge styles */
.badge {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 12px;
    font-size: 11px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.badge-baseline { background: #2A3F5F; color: #7A9AB5; }
.badge-rl       { background: #0A3D2E; color: var(--success); }
.badge-ml       { background: #2D1B4E; color: #B39DDB; }
.badge-adaptive { background: #2A2A00; color: var(--gold); }

/* Metric table */
.metric-table { width: 100%; border-collapse: collapse; }
.metric-table th { background: #142240; color: var(--teal); padding: 8px 12px; text-align: left; border-bottom: 2px solid var(--teal); }
.metric-table td { padding: 7px 12px; border-bottom: 1px solid #1A2E48; color: var(--light); }
.metric-table tr:hover td { background: #162033; }
.metric-table .best { color: var(--success); font-weight: 700; }

/* Tabs */
[data-testid="stTabs"] button { color: var(--muted) !important; border-radius: 6px 6px 0 0; }
[data-testid="stTabs"] button[aria-selected="true"] { color: var(--teal) !important; border-bottom: 2px solid var(--teal) !important; }

/* Buttons */
.stButton > button {
    background-color: var(--teal) !important;
    color: var(--navy) !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 6px !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* Sidebar logo */
.aito-logo {
    text-align: center;
    padding: 16px 8px 8px 8px;
    border-bottom: 1px solid #1F3350;
    margin-bottom: 12px;
}
.aito-logo-title { color: var(--teal); font-size: 22px; font-weight: 800; letter-spacing: 2px; }
.aito-logo-sub { color: var(--muted); font-size: 10px; letter-spacing: 1.5px; text-transform: uppercase; }

/* Status pill */
.status-pill {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    margin-right: 6px;
}
.status-live   { background: var(--success); box-shadow: 0 0 6px var(--success); }
.status-shadow { background: var(--gold);    box-shadow: 0 0 6px var(--gold); }
.status-idle   { background: var(--muted); }

/* Download button */
.stDownloadButton > button {
    background-color: transparent !important;
    color: var(--teal) !important;
    border: 1px solid var(--teal) !important;
}
</style>
"""
st.markdown(AITO_CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTROLLER_DISPLAY_NAMES: dict[str, str] = {
    "fixed_timing": "Fixed Timing",
    "adaptive_rule": "Adaptive Rule",
    "rule_based": "Rule-Based",
    "ml_randomforestclassifier": "Random Forest",
    "ml_xgbclassifier": "XGBoost",
    "ml_gradientboostingclassifier": "Gradient Boosting",
    "ml_mlpclassifier": "Neural Net MLP",
    "rl_q_learning": "Q-Learning",
    "rl_dqn": "Deep Q-Network",
    "rl_dqn_dueling": "Dueling DQN",
    "rl_policy_gradient": "Policy Gradient",
    "rl_a2c": "A2C",
    "rl_sac": "SAC",
    "rl_maddpg": "MADDPG",
    "rl_recurrent_ppo": "Recurrent PPO",
    "q_learning": "Q-Learning",
    "dqn": "Deep Q-Network",
    "policy_gradient": "Policy Gradient",
    "a2c": "A2C",
    "sac": "SAC",
    "recurrent_ppo": "Recurrent PPO",
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "gradient_boosting": "Gradient Boosting",
    "mlp": "Neural Net MLP",
}

CONTROLLER_BADGES: dict[str, str] = {
    "fixed_timing": "baseline", "adaptive_rule": "adaptive", "rule_based": "adaptive",
    "random_forest": "ml", "gradient_boosting": "ml", "xgboost": "ml", "mlp": "ml",
    "q_learning": "rl", "dqn": "rl", "policy_gradient": "rl", "a2c": "rl",
    "sac": "rl", "maddpg": "rl", "recurrent_ppo": "rl",
}

AITO_TEAL  = "#00C2CB"
AITO_GOLD  = "#F0B429"
AITO_NAVY  = "#0A1628"
AITO_GREEN = "#00C851"
AITO_RED   = "#FF4B4B"

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

@st.cache_resource
def _load_settings():
    from traffic_ai.config.settings import load_settings
    return load_settings()


def _kpi_card(label: str, value: str, delta: str = "", delta_positive: bool = True) -> str:
    delta_class = "positive" if delta_positive else "negative"
    delta_html = f'<div class="kpi-delta {delta_class}">{delta}</div>' if delta else ""
    return f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-value">{value}</div>
        {delta_html}
    </div>
    """


def _display_name(key: str) -> str:
    return CONTROLLER_DISPLAY_NAMES.get(key, key.replace("_", " ").title())


def _load_artifacts_df() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Try to load existing summary and step metrics CSVs from artifacts/."""
    summary_path = Path("artifacts/summary.csv")
    steps_path = Path("artifacts/step_metrics.csv")
    summary = pd.read_csv(summary_path) if summary_path.exists() else None
    steps = pd.read_csv(steps_path) if steps_path.exists() else None
    return summary, steps


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

def _render_sidebar() -> None:
    with st.sidebar:
        st.markdown("""
        <div class="aito-logo">
            <div class="aito-logo-title">AITO</div>
            <div class="aito-logo-sub">AI Traffic Optimization</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Configuration")
        with st.expander("Simulation", expanded=False):
            st.session_state["sim_steps"] = st.slider("Steps", 100, 2000, 500, step=100)
            st.session_state["sim_intersections"] = st.selectbox("Intersections", [2, 4, 9, 16], index=1)
            st.session_state["demand_profile"] = st.selectbox(
                "Demand Profile",
                ["rush_hour", "off_peak", "weekend", "incident_response", "event_surge",
                 "school_zone", "highway_merge", "construction_detour"],
                index=0,
            )
            st.session_state["demand_scale"] = st.slider("Demand Scale", 0.5, 3.0, 1.0, step=0.1)

        with st.expander("RL Reward Weights", expanded=False):
            st.session_state["w_delay"]    = st.slider("Avg Delay",    0.0, 1.0, 0.12, step=0.01)
            st.session_state["w_ped"]      = st.slider("Pedestrian",   0.0, 1.0, 0.05, step=0.01)
            st.session_state["w_co2"]      = st.slider("Emissions CO₂",0.0, 1.0, 0.03, step=0.01)
            st.session_state["w_switch"]   = st.slider("Switch Cost",  0.0, 5.0, 2.0,  step=0.1)
            st.session_state["w_thru"]     = st.slider("Throughput",   0.0, 1.0, 0.08, step=0.01)
            st.session_state["w_starve"]   = st.slider("Left Starve",  0.0, 1.0, 0.04, step=0.01)

        with st.expander("Sensor Faults", expanded=False):
            st.session_state["fault_enabled"] = st.checkbox("Enable Fault Injection", value=False)
            st.session_state["fault_stuck_prob"]   = st.slider("Stuck Prob",   0.0, 0.2, 0.02, step=0.01)
            st.session_state["fault_noise_std"]    = st.slider("Noise Std",    0.0, 0.5, 0.05, step=0.01)
            st.session_state["fault_dropout_prob"] = st.slider("Dropout Prob", 0.0, 0.2, 0.01, step=0.01)

        st.markdown("---")
        st.markdown(
            '<div style="color:#7A9AB5;font-size:11px;text-align:center;">'
            'AITO v2.0 — Engineering Platform<br>'
            '<a href="https://github.com/samarthvaka/AI-Traffic-Optimizer" '
            'style="color:#00C2CB;">GitHub</a>'
            '</div>',
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Tab 1: Network Overview
# ---------------------------------------------------------------------------

def _tab_network_overview() -> None:
    st.markdown("## Network Overview")
    st.markdown(
        "Live simulation of the traffic network. Configure parameters in the sidebar, "
        "then click **Run Simulation** to start."
    )

    # Quick-run KPI section from last artifacts
    summary_df, steps_df = _load_artifacts_df()
    if summary_df is not None and not summary_df.empty:
        st.markdown('<div class="section-header">Last Experiment Results</div>', unsafe_allow_html=True)
        cols = st.columns(4)
        best_row = summary_df.loc[summary_df["avg_queue"].idxmin()] if "avg_queue" in summary_df.columns else None
        if best_row is not None:
            cols[0].markdown(_kpi_card("Best Controller", _display_name(str(best_row.get("controller", "")))), unsafe_allow_html=True)
            cols[1].markdown(_kpi_card("Min Avg Queue", f"{best_row.get('avg_queue', 0):.1f} veh"), unsafe_allow_html=True)
            cols[2].markdown(_kpi_card("Controllers Tested", str(len(summary_df))), unsafe_allow_html=True)
            if "avg_co2_kg" in summary_df.columns:
                cols[3].markdown(_kpi_card("Best CO₂ (kg)", f"{summary_df['avg_co2_kg'].min():.2f}"), unsafe_allow_html=True)

    # Live simulation runner
    st.markdown('<div class="section-header">Run Single Simulation</div>', unsafe_allow_html=True)
    ctrl_options = {
        "Fixed Timing": "fixed_timing",
        "Adaptive Rule": "adaptive_rule",
        "Q-Learning": "q_learning",
        "DQN": "dqn",
        "PPO": "ppo",
    }
    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        ctrl_label = st.selectbox("Controller", list(ctrl_options.keys()), key="ov_ctrl")
    with col2:
        steps = st.session_state.get("sim_steps", 300)
        st.metric("Simulation Steps", steps)
    with col3:
        run_btn = st.button("▶ Run", use_container_width=True, key="ov_run_btn")

    if run_btn:
        ctrl_key = ctrl_options[ctrl_label]
        _run_live_simulation(ctrl_key)


def _run_live_simulation(ctrl_key: str) -> None:
    """Run a simulation and display live step metrics."""
    from traffic_ai.simulation_engine.engine import SimulatorConfig, TrafficNetworkSimulator
    from traffic_ai.controllers.fixed import FixedTimingController
    from traffic_ai.controllers.rule_based import RuleBasedController

    ctrl_map = {
        "fixed_timing": FixedTimingController,
        "adaptive_rule": RuleBasedController,
    }
    try:
        if ctrl_key in ("q_learning", "dqn", "ppo"):
            from traffic_ai.controllers.rl_controllers import (
                QLearningController, DQNController, PPOController,
            )
            ctrl_map.update({"q_learning": QLearningController, "dqn": DQNController, "ppo": PPOController})
    except ImportError:
        pass

    CtrlCls = ctrl_map.get(ctrl_key, FixedTimingController)
    ctrl = CtrlCls()

    steps = st.session_state.get("sim_steps", 300)
    n = st.session_state.get("sim_intersections", 4)
    profile = st.session_state.get("demand_profile", "rush_hour")
    scale = st.session_state.get("demand_scale", 1.0)

    cfg = SimulatorConfig(steps=steps, intersections=n, demand_profile=profile, demand_scale=scale, seed=42)
    engine = TrafficNetworkSimulator(cfg)
    ctrl.reset(n)
    obs = engine.reset_env()

    queue_hist: list[float] = []
    throughput_hist: list[float] = []

    chart_placeholder = st.empty()
    progress = st.progress(0)
    kpi_placeholder = st.empty()

    for step in range(steps):
        actions = ctrl.compute_actions(obs, step)
        obs, metrics, done, _ = engine.step_env(actions)

        total_q = sum(o.get("total_queue", 0.0) for o in obs.values()) / max(n, 1)
        total_tp = sum(o.get("departures", 0.0) for o in obs.values())
        queue_hist.append(total_q)
        throughput_hist.append(total_tp)

        if step % 20 == 0 or done:
            df_plot = pd.DataFrame({
                "Step": range(len(queue_hist)),
                "Avg Queue (veh)": queue_hist,
                "Throughput": throughput_hist,
            })
            with chart_placeholder.container():
                import plotly.graph_objects as go
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=df_plot["Step"], y=df_plot["Avg Queue (veh)"],
                    name="Avg Queue", line=dict(color=AITO_TEAL, width=2)))
                fig.add_trace(go.Scatter(x=df_plot["Step"], y=df_plot["Throughput"],
                    name="Throughput", line=dict(color=AITO_GOLD, width=2), yaxis="y2"))
                fig.update_layout(
                    paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                    font=dict(color="#E8F4F8"),
                    xaxis=dict(title="Step", gridcolor="#1A2E48"),
                    yaxis=dict(title="Avg Queue (veh)", gridcolor="#1A2E48"),
                    yaxis2=dict(title="Throughput", overlaying="y", side="right", gridcolor="#1A2E48"),
                    legend=dict(bgcolor="#0D1E35", bordercolor="#1A2E48"),
                    margin=dict(t=20, b=40, l=40, r=40),
                    height=300,
                )
                st.plotly_chart(fig, use_container_width=True)

            with kpi_placeholder.container():
                c1, c2, c3 = st.columns(3)
                c1.metric("Avg Queue", f"{total_q:.1f} veh")
                c2.metric("Step", f"{step}/{steps}")
                c3.metric("Throughput", f"{total_tp:.0f}")

        progress.progress(min((step + 1) / steps, 1.0))
        if done:
            break

    st.success(f"Simulation complete — {_display_name(ctrl_key)} | {steps} steps | {n} intersections")


# ---------------------------------------------------------------------------
# Tab 2: Benchmark Lab
# ---------------------------------------------------------------------------

def _tab_benchmark_lab() -> None:
    st.markdown("## Benchmark Lab")
    st.markdown(
        "Compare all AITO controllers on identical traffic scenarios. "
        "Results include statistical significance testing (Holm-Bonferroni correction)."
    )

    # Load existing results if available
    summary_df, steps_df = _load_artifacts_df()

    col1, col2 = st.columns([3, 1])
    with col1:
        quick_run = st.checkbox("Quick Run (reduced budget)", value=True)
    with col2:
        run_btn = st.button("▶ Run Benchmark", use_container_width=True, key="bench_run")

    if run_btn:
        with st.spinner("Running benchmark — this may take a minute…"):
            try:
                from traffic_ai.config.settings import load_settings
                from traffic_ai.experiments import ExperimentRunner
                settings = load_settings()
                runner = ExperimentRunner(settings=settings, quick_run=quick_run)
                artifacts = runner.run(ingest_only=False, include_kaggle=False, include_public=False)
                summary_df = pd.read_csv(artifacts.summary_csv) if artifacts.summary_csv.exists() else None
                steps_df = pd.read_csv(artifacts.step_metrics_csv) if artifacts.step_metrics_csv.exists() else None
                st.success("Benchmark complete!")
            except Exception as e:
                st.error(f"Benchmark failed: {e}")

    if summary_df is not None and not summary_df.empty:
        _render_benchmark_results(summary_df, steps_df)
    else:
        st.info("No benchmark results found. Click **Run Benchmark** to generate results, or run `python main.py --quick-run` from the terminal.")


def _render_benchmark_results(summary: pd.DataFrame, steps_df: pd.DataFrame | None) -> None:
    st.markdown('<div class="section-header">Controller Rankings</div>', unsafe_allow_html=True)

    display_cols = [c for c in ["controller", "avg_queue", "avg_wait_sec", "throughput",
                                 "avg_co2_kg", "efficiency_score"] if c in summary.columns]
    if not display_cols:
        st.dataframe(summary, use_container_width=True)
        return

    df = summary[display_cols].copy()
    df["controller"] = df["controller"].apply(_display_name)
    df = df.sort_values("avg_queue") if "avg_queue" in df.columns else df

    # Highlight best per column
    def highlight_best(s: pd.Series) -> list[str]:
        is_best = s == s.min() if "queue" in s.name or "wait" in s.name or "co2" in s.name else s == s.max()
        return [f"color: {AITO_GREEN}; font-weight: bold" if v else "" for v in is_best]

    styled = df.style
    for col in [c for c in display_cols if c != "controller"]:
        styled = styled.apply(highlight_best, subset=[col])

    st.dataframe(styled, use_container_width=True, height=300)

    # KPI summary cards
    if "avg_queue" in summary.columns:
        best = summary.loc[summary["avg_queue"].idxmin()]
        worst = summary.loc[summary["avg_queue"].idxmax()]
        improvement = (worst["avg_queue"] - best["avg_queue"]) / max(worst["avg_queue"], 1.0) * 100
        cols = st.columns(4)
        cols[0].markdown(_kpi_card("Best Controller", _display_name(str(best.get("controller", "")))), unsafe_allow_html=True)
        cols[1].markdown(_kpi_card("Min Avg Queue", f"{best['avg_queue']:.1f} veh"), unsafe_allow_html=True)
        cols[2].markdown(_kpi_card("AI vs Baseline", f"{improvement:.0f}%", delta_positive=True), unsafe_allow_html=True)
        cols[3].markdown(_kpi_card("Controllers", str(len(summary))), unsafe_allow_html=True)

    # Step metrics chart
    if steps_df is not None and not steps_df.empty and "total_queue" in steps_df.columns:
        st.markdown('<div class="section-header">Queue Dynamics by Step</div>', unsafe_allow_html=True)
        ctrl_col = next((c for c in ["controller_name", "controller"] if c in steps_df.columns), None)
        if ctrl_col:
            import plotly.express as px
            fig = px.line(
                steps_df, x="step", y="total_queue",
                color=ctrl_col,
                title="Total Queue Over Time",
                color_discrete_sequence=[AITO_TEAL, AITO_GOLD, "#B39DDB", "#00C851",
                                          "#FF4B4B", "#FFB347", "#87CEEB", "#FF69B4"],
            )
            fig.update_layout(
                paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                font=dict(color="#E8F4F8"),
                legend=dict(bgcolor="#0D1E35", bordercolor="#1A2E48"),
                xaxis=dict(gridcolor="#1A2E48"),
                yaxis=dict(gridcolor="#1A2E48"),
            )
            st.plotly_chart(fig, use_container_width=True)

    # Significance results
    sig_path = Path("artifacts/significance.csv")
    if sig_path.exists():
        st.markdown('<div class="section-header">Statistical Significance (Holm-Bonferroni)</div>', unsafe_allow_html=True)
        sig_df = pd.read_csv(sig_path)
        st.dataframe(sig_df.head(20), use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Shadow Mode
# ---------------------------------------------------------------------------

def _tab_shadow_mode() -> None:
    st.markdown("## Shadow Mode")
    st.markdown(
        "Evaluate a candidate AI controller **without** applying it to live traffic. "
        "Only the production controller's actions affect the simulation. "
        "The candidate's recommendations are logged as counterfactuals."
    )

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        prod_options = ["fixed_timing", "adaptive_rule", "rule_based", "q_learning", "dqn", "ppo"]
        prod_sel = st.selectbox("Production Controller", prod_options,
                                 format_func=_display_name, key="shadow_prod")
    with col2:
        cand_sel = st.selectbox("Candidate Controller", prod_options,
                                 format_func=_display_name, key="shadow_cand",
                                 index=prod_options.index("dqn") if "dqn" in prod_options else 0)
    with col3:
        shadow_steps = st.number_input("Steps", min_value=30, max_value=600, value=100, step=10)

    run_shadow = st.button("▶ Run Shadow Mode", use_container_width=False, key="shadow_run_btn")

    # Load existing report if available
    report_path = Path("artifacts/shadow_report.json")
    existing_report = None
    if report_path.exists():
        try:
            existing_report = json.loads(report_path.read_text())
        except Exception:
            pass

    if run_shadow:
        with st.spinner("Running shadow mode simulation…"):
            try:
                from traffic_ai.shadow.shadow_runner import ShadowModeRunner
                from traffic_ai.simulation_engine.engine import SimulatorConfig
                _ctrl_cls = _get_controller_class
                prod_ctrl = _ctrl_cls(prod_sel)()
                cand_ctrl = _ctrl_cls(cand_sel)()
                cfg = SimulatorConfig(steps=int(shadow_steps), intersections=4, seed=42)
                runner = ShadowModeRunner(production=prod_ctrl, candidate=cand_ctrl, config=cfg)
                report_obj = runner.run()
                Path("artifacts").mkdir(exist_ok=True)
                runner.save_report(report_obj, report_path)
                existing_report = json.loads(report_path.read_text())
                st.success("Shadow run complete!")
            except Exception as e:
                st.error(f"Shadow mode failed: {e}")

    if existing_report:
        _render_shadow_report(existing_report)
    else:
        st.info("No shadow report found. Click **Run Shadow Mode** to generate one.")


def _get_controller_class(key: str):
    from traffic_ai.controllers.fixed import FixedTimingController
    from traffic_ai.controllers.rule_based import RuleBasedController
    mapping = {
        "fixed_timing": FixedTimingController,
        "adaptive_rule": RuleBasedController,
        "rule_based": RuleBasedController,
    }
    try:
        from traffic_ai.controllers.rl_controllers import (
            QLearningController, DQNController, PPOController,
        )
        mapping.update({"q_learning": QLearningController, "dqn": DQNController, "ppo": PPOController})
    except ImportError:
        pass
    return mapping.get(key, FixedTimingController)


def _render_shadow_report(report: dict) -> None:
    st.markdown('<div class="section-header">Shadow Mode Report</div>', unsafe_allow_html=True)

    cols = st.columns(4)
    cols[0].markdown(_kpi_card(
        "Production", _display_name(report.get("production_controller", ""))
    ), unsafe_allow_html=True)
    cols[1].markdown(_kpi_card(
        "Candidate", _display_name(report.get("candidate_controller", ""))
    ), unsafe_allow_html=True)
    agree_pct = report.get("agreement_rate", 0.0) * 100
    cols[2].markdown(_kpi_card(
        "Agreement Rate", f"{agree_pct:.0f}%",
        delta="↑ Stable" if agree_pct > 70 else "↓ Diverging", delta_positive=agree_pct > 70
    ), unsafe_allow_html=True)
    q_red = report.get("estimated_queue_reduction_pct", 0.0)
    cols[3].markdown(_kpi_card(
        "Est. Queue Reduction", f"{q_red:.1f}%",
        delta_positive=q_red > 0
    ), unsafe_allow_html=True)

    # Phase comparison chart
    records = report.get("step_records", [])
    if records:
        df = pd.DataFrame(records)
        if "production_phase" in df.columns:
            import plotly.graph_objects as go
            phase_vals = {"NS": 0, "NS_THROUGH": 0, "EW": 1, "EW_THROUGH": 1, "NS_LEFT": 2, "EW_LEFT": 3}
            df["prod_idx"] = df["production_phase"].map(phase_vals).fillna(0)
            df["cand_idx"] = df["candidate_phase"].map(phase_vals).fillna(0)

            fig = go.Figure()
            df_subset = df[df["step"] < 200].copy()
            fig.add_trace(go.Scatter(x=df_subset["step"], y=df_subset["prod_idx"],
                mode="lines", name="Production", line=dict(color=AITO_TEAL, width=2)))
            fig.add_trace(go.Scatter(x=df_subset["step"], y=df_subset["cand_idx"],
                mode="lines", name="Candidate", line=dict(color=AITO_GOLD, width=2, dash="dash")))
            fig.update_layout(
                title="Phase Decisions: Production vs Candidate",
                paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                font=dict(color="#E8F4F8"),
                xaxis=dict(title="Step", gridcolor="#1A2E48"),
                yaxis=dict(title="Phase Index", tickvals=[0,1,2,3],
                           ticktext=["NS_THROUGH","EW_THROUGH","NS_LEFT","EW_LEFT"],
                           gridcolor="#1A2E48"),
                legend=dict(bgcolor="#0D1E35"),
                height=280,
            )
            st.plotly_chart(fig, use_container_width=True)

        # Queue comparison
        if "production_queue" in df.columns and "candidate_queue_est" in df.columns:
            df_subset = df[df["step"] < 200].copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=df_subset["step"], y=df_subset["production_queue"],
                name="Production Queue", line=dict(color=AITO_TEAL)))
            fig2.add_trace(go.Scatter(x=df_subset["step"], y=df_subset["candidate_queue_est"],
                name="Candidate Est. Queue", line=dict(color=AITO_GOLD, dash="dash")))
            fig2.update_layout(
                title="Queue Comparison",
                paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                font=dict(color="#E8F4F8"),
                xaxis=dict(gridcolor="#1A2E48"),
                yaxis=dict(gridcolor="#1A2E48"),
                legend=dict(bgcolor="#0D1E35"),
                height=240,
            )
            st.plotly_chart(fig2, use_container_width=True)

    gen_at = report.get("generated_at", "")
    if gen_at:
        st.caption(f"Report generated: {gen_at}")


# ---------------------------------------------------------------------------
# Tab 4: Controller Training
# ---------------------------------------------------------------------------

def _tab_controller_training() -> None:
    st.markdown("## Controller Training")
    st.markdown("Train ML and RL controllers on synthetic traffic datasets.")

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1:
        train_ctrl = st.selectbox(
            "Controller Type",
            ["dqn", "q_learning", "ppo", "random_forest", "xgboost", "gradient_boosting", "mlp"],
            format_func=_display_name,
            key="train_ctrl",
        )
    with col2:
        episodes = st.number_input("Episodes", min_value=5, max_value=200, value=20, step=5)
    with col3:
        train_btn = st.button("▶ Train", use_container_width=True, key="train_btn")

    if train_btn:
        with st.spinner(f"Training {_display_name(train_ctrl)}…"):
            try:
                from traffic_ai.training.trainer import ModelTrainer
                from traffic_ai.data_pipeline.synthetic_generator import (
                    SyntheticDatasetGenerator, SyntheticDatasetConfig,
                )
                cfg = SyntheticDatasetConfig(n_samples=500, label_strategy="queue_balance")
                df = SyntheticDatasetGenerator(cfg).generate().dataframe
                settings = _load_settings()
                trainer = ModelTrainer()
                result = trainer.train(
                    controller_type=train_ctrl,
                    dataset=df,
                    config={"episodes": int(episodes), "step_limit": 100},
                    settings=settings,
                )
                st.success(f"Training complete in {result.training_time_seconds:.1f}s")
                col_a, col_b = st.columns(2)
                col_a.metric("Final Metric", f"{result.evaluation_metrics.get('avg_episode_reward', 0):.3f}")
                col_b.metric("Episodes", str(len(result.reward_history or [])))

                if result.reward_history:
                    import plotly.graph_objects as go
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        y=result.reward_history, mode="lines",
                        name="Episode Reward", line=dict(color=AITO_TEAL, width=2)
                    ))
                    fig.update_layout(
                        title="Learning Curve", paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                        font=dict(color="#E8F4F8"), height=280,
                        xaxis=dict(title="Episode", gridcolor="#1A2E48"),
                        yaxis=dict(title="Total Reward", gridcolor="#1A2E48"),
                    )
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Training failed: {e}")

    # Explainability section
    st.markdown('<div class="section-header">Decision Explainability</div>', unsafe_allow_html=True)
    st.markdown("Get a natural language explanation for a controller decision.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        exp_ctrl = st.selectbox("Controller", ["fixed_timing", "rule_based", "dqn", "q_learning"],
                                  format_func=_display_name, key="exp_ctrl")
        exp_q_ns = st.slider("NS Queue", 0.0, 120.0, 15.0, key="exp_qns")
        exp_q_ew = st.slider("EW Queue", 0.0, 120.0, 8.0, key="exp_qew")
    with col_e2:
        exp_q_ns_l = st.slider("NS Left Queue", 0.0, 30.0, 3.0, key="exp_qnsl")
        exp_q_ew_l = st.slider("EW Left Queue", 0.0, 30.0, 2.0, key="exp_qewl")
        exp_elapsed = st.slider("Phase Elapsed (s)", 0, 120, 20, key="exp_elapsed")

    if st.button("Explain Decision", key="explain_btn"):
        try:
            from traffic_ai.explainability.explainer import DecisionExplainer
            ctrl_cls = _get_controller_class(exp_ctrl)
            ctrl = ctrl_cls()
            obs = {
                "queue_ns": exp_q_ns, "queue_ew": exp_q_ew,
                "queue_ns_through": exp_q_ns, "queue_ew_through": exp_q_ew,
                "queue_ns_left": exp_q_ns_l, "queue_ew_left": exp_q_ew_l,
                "total_queue": exp_q_ns + exp_q_ew + exp_q_ns_l + exp_q_ew_l,
                "phase_elapsed": float(exp_elapsed), "step": 0.0,
                "current_phase_idx": 0.0, "time_of_day_normalized": 0.3,
                "upstream_queue": 0.0, "in_transition": 0.0, "emergency_active": 0.0,
            }
            action = ctrl.select_action(obs)
            explainer = DecisionExplainer(controller=ctrl)
            result = explainer.explain(obs, action=action)

            st.info(f"**Action Selected:** {result.phase_label}")
            st.markdown(f"**Explanation:** {result.natural_language}")

            if result.feature_importances:
                feat_df = pd.DataFrame(
                    list(result.feature_importances.items()),
                    columns=["Feature", "Importance"]
                ).head(6)
                import plotly.express as px
                fig = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                               color_discrete_sequence=[AITO_TEAL])
                fig.update_layout(paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                                   font=dict(color="#E8F4F8"), height=220,
                                   xaxis=dict(gridcolor="#1A2E48"),
                                   yaxis=dict(gridcolor="#1A2E48"))
                st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.error(f"Explainability failed: {e}")


# ---------------------------------------------------------------------------
# Tab 5: Data & Calibration
# ---------------------------------------------------------------------------

def _tab_data_calibration() -> None:
    st.markdown("## Data & Calibration")

    tab5a, tab5b, tab5c = st.tabs(["Synthetic Data Studio", "PeMS Calibration", "Sensor Fault Injection"])

    with tab5a:
        st.markdown("### Synthetic Dataset Generator")
        col1, col2 = st.columns(2)
        with col1:
            n_samples = st.slider("Samples", 100, 5000, 1000, step=100)
            label_strategy = st.selectbox(
                "Label Strategy",
                ["queue_balance", "throughput_max", "delay_min", "multi_objective"],
            )
            demand_profile = st.selectbox(
                "Demand Profile",
                ["rush_hour", "off_peak", "weekend", "event_surge", "school_zone"],
            )
        with col2:
            noise_level = st.slider("Noise Level", 0.0, 1.0, 0.1, step=0.05)
            seed = st.number_input("Seed", value=42, min_value=0)

        if st.button("Generate Dataset", key="gen_data_btn"):
            with st.spinner("Generating synthetic dataset…"):
                try:
                    from traffic_ai.data_pipeline.synthetic_generator import (
                        SyntheticDatasetGenerator, SyntheticDatasetConfig,
                    )
                    cfg = SyntheticDatasetConfig(
                        n_samples=n_samples,
                        label_strategy=label_strategy,
                        demand_profile=demand_profile,
                        noise_level=noise_level,
                        seed=seed,
                    )
                    ds = SyntheticDatasetGenerator(cfg).generate()
                    df = ds.dataframe
                    st.success(f"Generated {len(df)} samples")
                    st.dataframe(df.head(20), use_container_width=True)
                    csv_bytes = df.to_csv(index=False).encode()
                    st.download_button(
                        "⬇ Download CSV",
                        csv_bytes,
                        file_name=f"aito_synthetic_{demand_profile}_{n_samples}.csv",
                        mime="text/csv",
                        key="dl_synthetic",
                    )
                except Exception as e:
                    st.error(f"Generation failed: {e}")

    with tab5b:
        st.markdown("### PeMS Calibration")
        st.markdown(
            "Calibrate simulation demand from Caltrans PeMS loop-detector data. "
            "Requires `PEMS_API_KEY` environment variable."
        )
        pems_key_set = bool(os.environ.get("PEMS_API_KEY", ""))
        if pems_key_set:
            st.success("✓ PEMS_API_KEY detected")
        else:
            st.warning("⚠ PEMS_API_KEY not set — will use synthetic fallback data")

        station_id = st.number_input("Station ID", value=400456, min_value=1)
        col_d1, col_d2 = st.columns(2)
        date_from = col_d1.date_input("From", value=pd.Timestamp("2024-01-15"))
        date_to   = col_d2.date_input("To",   value=pd.Timestamp("2024-01-22"))

        if st.button("Calibrate", key="pems_btn"):
            with st.spinner("Fetching PeMS data…"):
                try:
                    from traffic_ai.data_pipeline.pems_connector import PeMSConnector
                    connector = PeMSConnector(station_id=int(station_id))
                    df = connector.fetch(str(date_from), str(date_to))
                    cal = connector.calibration_by_hour(df)
                    st.success(f"Calibration loaded: {len(cal)} hourly profiles")
                    st.json(cal)
                except Exception as e:
                    st.error(f"PeMS calibration failed: {e}")

    with tab5c:
        st.markdown("### Sensor Fault Injection Preview")
        st.markdown("Preview how sensor faults affect observations before enabling them in the sidebar.")

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            stuck_p  = st.slider("Stuck Prob",   0.0, 0.3, 0.05, step=0.01, key="prev_stuck")
            noise_s  = st.slider("Noise Std",    0.0, 0.5, 0.1,  step=0.01, key="prev_noise")
            drop_p   = st.slider("Dropout Prob", 0.0, 0.3, 0.02, step=0.01, key="prev_drop")

        if st.button("Generate Fault Preview", key="fault_prev_btn"):
            from traffic_ai.simulation_engine.sensor import SensorFaultModel
            fault = SensorFaultModel(stuck_prob=stuck_p, noise_std=noise_s, dropout_prob=drop_p, seed=0)
            clean_queue = 20.0
            steps_preview = 50
            clean_vals = [clean_queue] * steps_preview
            dirty_vals = []
            for i in range(steps_preview):
                obs = {"queue_ns": clean_queue, "queue_ew": clean_queue,
                       "queue_ns_through": clean_queue, "queue_ew_through": clean_queue,
                       "queue_ns_left": 5.0, "queue_ew_left": 4.0,
                       "total_queue": 50.0, "upstream_queue": 8.0}
                corrupted = fault.apply(obs, step=i)
                dirty_vals.append(corrupted["queue_ns"])

            import plotly.graph_objects as go
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=clean_vals, name="Clean", line=dict(color=AITO_TEAL)))
            fig.add_trace(go.Scatter(y=dirty_vals, name="Corrupted", line=dict(color=AITO_RED)))
            fig.update_layout(
                title="NS Queue: Clean vs Corrupted",
                paper_bgcolor=AITO_NAVY, plot_bgcolor="#0D1E35",
                font=dict(color="#E8F4F8"), height=260,
                xaxis=dict(title="Step", gridcolor="#1A2E48"),
                yaxis=dict(title="Queue (veh)", gridcolor="#1A2E48"),
                legend=dict(bgcolor="#0D1E35"),
            )
            st.plotly_chart(fig, use_container_width=True)
            fault_rate = sum(1 for v in dirty_vals if abs(v - clean_queue) > 0.5) / steps_preview * 100
            st.metric("Corruption Rate", f"{fault_rate:.0f}%")


# ---------------------------------------------------------------------------
# Tab 6: Export
# ---------------------------------------------------------------------------

def _tab_export() -> None:
    st.markdown("## Export")
    st.markdown("Download experiment results, shadow mode reports, and configuration files.")

    summary_df, steps_df = _load_artifacts_df()

    st.markdown('<div class="section-header">Experiment Artifacts</div>', unsafe_allow_html=True)
    artifacts_dir = Path("artifacts")
    export_files = [
        ("summary.csv",      "Benchmark Summary",      "text/csv"),
        ("step_metrics.csv", "Step Metrics",            "text/csv"),
        ("significance.csv", "Statistical Significance","text/csv"),
        ("shadow_report.json","Shadow Mode Report",     "application/json"),
        ("ablation.csv",     "Ablation Study",          "text/csv"),
    ]

    for fname, label, mime in export_files:
        fpath = artifacts_dir / fname
        if fpath.exists():
            data = fpath.read_bytes()
            st.download_button(
                f"⬇ {label} ({fname})",
                data=data,
                file_name=fname,
                mime=mime,
                key=f"dl_{fname}",
            )
        else:
            st.markdown(f"<span style='color:#7A9AB5'>— {label} not found (run benchmark first)</span>",
                        unsafe_allow_html=True)

    # In-page preview
    st.markdown('<div class="section-header">Results Preview</div>', unsafe_allow_html=True)
    if summary_df is not None:
        st.markdown("**Benchmark Summary**")
        st.dataframe(summary_df, use_container_width=True)
    else:
        st.info("Run a benchmark to see results here.")

    # Config export
    st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
    cfg_path = Path("traffic_ai/config/default_config.yaml")
    if cfg_path.exists():
        st.download_button(
            "⬇ default_config.yaml",
            data=cfg_path.read_bytes(),
            file_name="default_config.yaml",
            mime="text/yaml",
            key="dl_config",
        )


# ---------------------------------------------------------------------------
# Main app
# ---------------------------------------------------------------------------

def main() -> None:
    _render_sidebar()

    st.markdown(
        '<h1 style="color:#00C2CB;font-size:32px;font-weight:800;letter-spacing:2px;">'
        'AITO <span style="color:#7A9AB5;font-size:16px;font-weight:400;letter-spacing:1px;">'
        '— AI Traffic Optimization</span></h1>',
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "🌐 Network Overview",
        "🔬 Benchmark Lab",
        "👁 Shadow Mode",
        "🎓 Controller Training",
        "📊 Data & Calibration",
        "📤 Export",
    ])

    with tab1:
        _tab_network_overview()
    with tab2:
        _tab_benchmark_lab()
    with tab3:
        _tab_shadow_mode()
    with tab4:
        _tab_controller_training()
    with tab5:
        _tab_data_calibration()
    with tab6:
        _tab_export()


if __name__ == "__main__":
    main()
