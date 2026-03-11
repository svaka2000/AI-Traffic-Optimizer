from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats


def pairwise_significance_tests(
    step_metrics: pd.DataFrame,
    metric_col: str = "avg_wait_sec",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """Mann-Whitney U test for each controller pair (no normality assumption)."""
    rows: list[dict[str, float | str]] = []
    controllers = sorted(step_metrics["controller"].unique().tolist())
    for a, b in combinations(controllers, 2):
        a_values = step_metrics.loc[step_metrics["controller"] == a, metric_col].dropna().to_numpy()
        b_values = step_metrics.loc[step_metrics["controller"] == b, metric_col].dropna().to_numpy()
        if len(a_values) < 2 or len(b_values) < 2:
            continue
        u_stat, p_value = stats.mannwhitneyu(a_values, b_values, alternative="two-sided")
        # Effect size: rank-biserial correlation
        n1, n2 = len(a_values), len(b_values)
        effect_size = 1.0 - (2.0 * u_stat) / (n1 * n2)
        rows.append(
            {
                "controller_a": a,
                "controller_b": b,
                "metric": metric_col,
                "u_stat": float(u_stat),
                "p_value": float(p_value),
                "effect_size_r": float(effect_size),
                "significant": str(bool(p_value < alpha)),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 500,
    ci: float = 0.95,
    seed: int = 42,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    means = samples.mean(axis=1)
    lower = float(np.percentile(means, (1 - ci) / 2 * 100))
    upper = float(np.percentile(means, (1 + ci) / 2 * 100))
    return lower, upper


def controller_bootstrap_table(
    step_df: pd.DataFrame,
    metric: str,
    n_bootstrap: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Bootstrap CIs from real step-level data (not synthetic pseudo-distributions)."""
    rows: list[dict[str, float | str]] = []
    if step_df.empty or "controller" not in step_df.columns or metric not in step_df.columns:
        return pd.DataFrame(rows)
    for controller, group in step_df.groupby("controller"):
        values = group[metric].dropna().to_numpy()
        if len(values) < 2:
            continue
        low, high = bootstrap_confidence_interval(values, n_bootstrap=n_bootstrap, seed=seed)
        rows.append(
            {
                "controller": str(controller),
                f"{metric}_mean": float(values.mean()),
                f"{metric}_ci_low": low,
                f"{metric}_ci_high": high,
            }
        )
    return pd.DataFrame(rows)
