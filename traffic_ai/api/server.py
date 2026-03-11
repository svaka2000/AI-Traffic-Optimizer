from __future__ import annotations

import uuid
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from traffic_ai.config.settings import load_settings
from traffic_ai.controllers import (
    AdaptiveRuleController,
    FixedTimingController,
)
from traffic_ai.experiments import ExperimentRunner
from traffic_ai.simulation_engine import SimulatorConfig, TrafficNetworkSimulator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class RunConfig(BaseModel):
    quick_run: bool = True
    include_kaggle: bool = False
    include_public: bool = True
    n_episodes: int = Field(default=5, ge=1, le=100)
    grid_rows: int = Field(default=2, ge=1, le=4)
    grid_cols: int = Field(default=2, ge=1, le=4)


class RunExperimentRequest(BaseModel):
    quick_run: bool = True
    include_kaggle: bool = False
    include_public: bool = True


class SimulateRequest(BaseModel):
    controller: Literal["fixed", "adaptive"] = "adaptive"
    steps: int = Field(default=400, ge=100, le=5000)


# ---------------------------------------------------------------------------
# In-memory run registry
# ---------------------------------------------------------------------------
_RUN_REGISTRY: dict[str, dict[str, Any]] = {}

app = FastAPI(title="Traffic AI Optimization API", version="2.0.0")


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health() -> dict[str, str]:
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/run")
def run_experiment(config: RunConfig) -> dict[str, Any]:
    """Trigger ExperimentRunner and return artifact paths."""
    run_id = str(uuid.uuid4())[:8]
    settings = load_settings()
    runner = ExperimentRunner(settings=settings, quick_run=config.quick_run)
    artifacts = runner.run(
        ingest_only=False,
        include_kaggle=config.include_kaggle,
        include_public=config.include_public,
    )
    _RUN_REGISTRY[run_id] = {
        "summary_csv": str(artifacts.summary_csv),
        "step_metrics_csv": str(artifacts.step_metrics_csv),
        "significance_csv": str(artifacts.significance_csv),
        "ablation_csv": str(artifacts.ablation_csv),
        "model_metrics_csv": str(artifacts.model_metrics_csv),
        "generated_plots": [str(p) for p in artifacts.generated_plots],
        "trained_model_dir": str(artifacts.trained_model_dir),
    }
    return {"run_id": run_id, **_RUN_REGISTRY[run_id]}


@app.get("/results/{run_id}")
def get_results(run_id: str) -> dict[str, Any]:
    """Return summary CSV as JSON for a given run ID."""
    if run_id not in _RUN_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    csv_path = Path(_RUN_REGISTRY[run_id]["summary_csv"])
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="Summary CSV not found on disk.")
    df = pd.read_csv(csv_path)
    return {"run_id": run_id, "rows": df.to_dict(orient="records")}


@app.get("/plots/{run_id}/{plot_name}")
def get_plot(run_id: str, plot_name: str) -> FileResponse:
    """Return a plot PNG for a given run ID and plot name."""
    if run_id not in _RUN_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found.")
    # Search for the plot among generated plots
    plots = _RUN_REGISTRY[run_id].get("generated_plots", [])
    for p in plots:
        if Path(p).stem == plot_name or Path(p).name == plot_name:
            if Path(p).exists():
                return FileResponse(p, media_type="image/png")
    # Try artifacts/plots/ directly
    plot_path = Path("artifacts/plots") / f"{plot_name}.png"
    if plot_path.exists():
        return FileResponse(str(plot_path), media_type="image/png")
    raise HTTPException(status_code=404, detail=f"Plot '{plot_name}' not found.")


# ---------------------------------------------------------------------------
# Legacy endpoints (kept for backward compatibility)
# ---------------------------------------------------------------------------

@app.post("/run_experiment")
def run_experiment_legacy(payload: RunExperimentRequest) -> dict[str, str]:
    settings = load_settings()
    runner = ExperimentRunner(settings=settings, quick_run=payload.quick_run)
    artifacts = runner.run(
        ingest_only=False,
        include_kaggle=payload.include_kaggle,
        include_public=payload.include_public,
    )
    return {
        "summary_csv": str(artifacts.summary_csv),
        "step_metrics_csv": str(artifacts.step_metrics_csv),
        "significance_csv": str(artifacts.significance_csv),
        "ablation_csv": str(artifacts.ablation_csv),
    }


@app.post("/simulate")
def simulate(payload: SimulateRequest) -> dict[str, str | float]:
    settings = load_settings()
    config = SimulatorConfig(
        steps=payload.steps,
        intersections=int(settings.get("simulation.intersections", 4)),
        lanes_per_direction=int(settings.get("simulation.lanes_per_direction", 2)),
        step_seconds=float(settings.get("simulation.step_seconds", 1.0)),
        max_queue_per_lane=int(settings.get("simulation.max_queue_per_lane", 60)),
        demand_profile=str(settings.get("simulation.demand_profile", "rush_hour")),
        demand_scale=float(settings.get("simulation.demand_scale", 1.0)),
        seed=settings.seed,
    )
    simulator = TrafficNetworkSimulator(config)
    if payload.controller == "fixed":
        controller = FixedTimingController(step_seconds=config.step_seconds)
    else:
        controller = AdaptiveRuleController()
    result = simulator.run(controller, steps=payload.steps)

    summary = result.aggregate.copy()
    summary["controller"] = controller.name
    out_path = settings.output_dir / "results" / f"api_sim_{controller.name}.csv"
    pd.DataFrame([summary]).to_csv(out_path, index=False)
    return {
        "controller": controller.name,
        "average_wait_time": float(summary["average_wait_time"]),
        "average_queue_length": float(summary["average_queue_length"]),
        "output_csv": str(out_path),
    }

