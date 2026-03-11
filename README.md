# Traffic AI Optimization Platform

Research-grade, reproducible platform for traffic signal optimization across fixed-time, adaptive, supervised ML, and reinforcement-learning controllers.

## Features
- Real-world dataset ingestion (Kaggle + public traffic sources).
- Unified preprocessing and feature engineering pipeline.
- Multi-intersection stochastic traffic simulation engine.
- Controller suite:
  - Fixed timing baseline
  - Rule-based adaptive
  - Supervised ML (RF, GB, XGBoost, MLP, time-series predictor)
  - RL (Q-learning, DQN, policy gradient)
- Reproducible experiment framework with:
  - cross-validation
  - hyperparameter search
  - ablation studies
  - statistical significance testing
- Metrics and publication-quality plots.
- Streamlit dashboard and FastAPI service.

## Quick Start
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py --quick-run
```

## Full Run
```bash
python main.py --output-dir artifacts --config traffic_ai/config/default_config.yaml
```

## Dashboard
```bash
streamlit run traffic_ai/dashboard/streamlit_app.py
```

## API
```bash
uvicorn traffic_ai.api.server:app --reload
```

## Kaggle Setup
1. Create API token from Kaggle account settings.
2. Place `kaggle.json` at:
   - `%USERPROFILE%\\.kaggle\\kaggle.json` (Windows)
3. Run ingestion:
```bash
python main.py --ingest-only
```

## Artifacts
- `artifacts/results/`: CSV and JSON experiment outputs
- `artifacts/models/`: serialized trained models
- `artifacts/plots/`: generated figures

## Testing
```bash
pytest -q
```

