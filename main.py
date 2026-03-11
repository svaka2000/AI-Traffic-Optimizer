from __future__ import annotations

import argparse
import json
from pathlib import Path

from traffic_ai.config.settings import load_settings
from traffic_ai.experiments import ExperimentRunner
from traffic_ai.utils.reproducibility import set_global_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AI Traffic Signal Optimization Research Platform"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="traffic_ai/config/default_config.yaml",
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--quick-run",
        action="store_true",
        help="Use reduced training/simulation budget for faster iteration",
    )
    parser.add_argument(
        "--ingest-only",
        action="store_true",
        help="Run data ingestion and preprocessing only",
    )
    parser.add_argument(
        "--skip-kaggle",
        action="store_true",
        help="Disable Kaggle dataset ingestion",
    )
    parser.add_argument(
        "--skip-public",
        action="store_true",
        help="Disable public dataset ingestion",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory from config",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    if args.output_dir:
        settings.payload["project"]["output_dir"] = args.output_dir
        settings.output_dir.mkdir(parents=True, exist_ok=True)
    set_global_seed(settings.seed)

    runner = ExperimentRunner(settings=settings, quick_run=args.quick_run)
    artifacts = runner.run(
        ingest_only=args.ingest_only,
        include_kaggle=not args.skip_kaggle,
        include_public=not args.skip_public,
    )

    summary = {
        "summary_csv": str(artifacts.summary_csv),
        "step_metrics_csv": str(artifacts.step_metrics_csv),
        "significance_csv": str(artifacts.significance_csv),
        "ablation_csv": str(artifacts.ablation_csv),
        "model_metrics_csv": str(artifacts.model_metrics_csv),
        "generated_plots": [str(p) for p in artifacts.generated_plots],
        "trained_model_dir": str(artifacts.trained_model_dir),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

