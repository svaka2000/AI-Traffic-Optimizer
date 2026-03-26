from __future__ import annotations

import math
from itertools import combinations
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats


def pairwise_significance_tests(
    step_metrics: pd.DataFrame,
    metric_col: str = "avg_wait_sec",
    alpha: float = 0.05,
    correction: str = "holm",
) -> pd.DataFrame:
    """Mann-Whitney U test for each controller pair with multiple-comparison correction.

    Parameters
    ----------
    step_metrics:
        DataFrame containing columns ``controller`` and the metric column.
    metric_col:
        Name of the metric to compare.
    alpha:
        Family-wise error rate threshold.
    correction:
        Multiple-comparison correction method.  ``"holm"`` applies the
        Holm-Bonferroni step-down procedure (default).  ``"bonferroni"`` uses
        the standard Bonferroni adjustment.  ``"none"`` skips correction.

    Returns
    -------
    pd.DataFrame
        One row per pair with columns: controller_a, controller_b, metric,
        u_stat, p_value, p_adjusted, effect_size_r, cohens_d_equiv,
        significant (after correction).
    """
    rows: list[dict] = []
    controllers = sorted(step_metrics["controller"].unique().tolist())
    for a, b in combinations(controllers, 2):
        a_values = step_metrics.loc[step_metrics["controller"] == a, metric_col].dropna().to_numpy()
        b_values = step_metrics.loc[step_metrics["controller"] == b, metric_col].dropna().to_numpy()
        if len(a_values) < 2 or len(b_values) < 2:
            continue
        u_stat, p_value = stats.mannwhitneyu(a_values, b_values, alternative="two-sided")
        n1, n2 = len(a_values), len(b_values)
        # Rank-biserial correlation effect size (Wendt 1972)
        effect_size_r = 1.0 - (2.0 * u_stat) / (n1 * n2)
        # Cohen's d equivalent via pooled standard deviation
        cohens_d = _cohens_d(a_values, b_values)
        rows.append(
            {
                "controller_a": a,
                "controller_b": b,
                "metric": metric_col,
                "u_stat": float(u_stat),
                "p_value": float(p_value),
                "p_adjusted": float(p_value),   # placeholder, corrected below
                "effect_size_r": float(effect_size_r),
                "cohens_d_equiv": float(cohens_d),
                "significant": False,
            }
        )

    if not rows:
        return pd.DataFrame(rows)

    df = pd.DataFrame(rows)
    df = _apply_correction(df, alpha=alpha, method=correction)
    return df


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d using pooled standard deviation (handles unequal n)."""
    n1, n2 = len(a), len(b)
    pooled_var = ((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2)
    pooled_std = float(np.sqrt(max(pooled_var, 1e-12)))
    return float((a.mean() - b.mean()) / pooled_std)


def _apply_correction(df: pd.DataFrame, alpha: float, method: str) -> pd.DataFrame:
    """Add ``p_adjusted`` and ``significant`` columns using the chosen method."""
    m = len(df)
    p_vals = df["p_value"].to_numpy()

    if method == "none":
        df["p_adjusted"] = p_vals
        df["significant"] = p_vals < alpha
        return df

    if method == "bonferroni":
        p_adj = np.minimum(p_vals * m, 1.0)
        df["p_adjusted"] = p_adj
        df["significant"] = p_adj < alpha
        return df

    # Holm-Bonferroni step-down (default)
    order = np.argsort(p_vals)
    p_adj = np.empty(m, dtype=float)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adjusted = p_vals[idx] * (m - rank)
        running_max = max(running_max, adjusted)
        p_adj[idx] = min(running_max, 1.0)

    df["p_adjusted"] = p_adj
    df["significant"] = p_adj < alpha
    return df


def bootstrap_confidence_interval(
    values: np.ndarray,
    n_bootstrap: int = 2_000,
    ci: float = 0.95,
    seed: int = 42,
    statistic: str = "mean",
) -> tuple[float, float]:
    """Bootstrap CI for a sample statistic.

    Parameters
    ----------
    values:
        1-D array of observations.
    n_bootstrap:
        Number of bootstrap resamples.
    ci:
        Coverage probability (e.g. 0.95 for 95 % CI).
    seed:
        NumPy random seed for reproducibility.
    statistic:
        ``"mean"`` or ``"median"``.

    Returns
    -------
    (lower, upper)
        Percentile-method confidence interval.
    """
    rng = np.random.default_rng(seed)
    samples = rng.choice(values, size=(n_bootstrap, len(values)), replace=True)
    if statistic == "median":
        boot_stats = np.median(samples, axis=1)
    else:
        boot_stats = samples.mean(axis=1)
    lower = float(np.percentile(boot_stats, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_stats, (1 + ci) / 2 * 100))
    return lower, upper


def bootstrap_median_difference(
    a: np.ndarray,
    b: np.ndarray,
    n_bootstrap: int = 2_000,
    ci: float = 0.95,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap CI for the difference of medians (median_a - median_b).

    Unlike the mean-based CI, the median difference is robust to heavy tails
    and outliers common in traffic wait-time distributions.

    Returns
    -------
    dict with keys: observed_diff, ci_low, ci_high, p_value_approx
    """
    rng = np.random.default_rng(seed)
    observed_diff = float(np.median(a) - np.median(b))

    boot_diffs = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        a_boot = rng.choice(a, size=len(a), replace=True)
        b_boot = rng.choice(b, size=len(b), replace=True)
        boot_diffs[i] = np.median(a_boot) - np.median(b_boot)

    lower = float(np.percentile(boot_diffs, (1 - ci) / 2 * 100))
    upper = float(np.percentile(boot_diffs, (1 + ci) / 2 * 100))

    # Approximate two-sided p-value: proportion of bootstrap samples beyond 0
    p_approx = float(2.0 * min(np.mean(boot_diffs <= 0), np.mean(boot_diffs >= 0)))

    return {
        "observed_diff": observed_diff,
        "ci_low": lower,
        "ci_high": upper,
        "p_value_approx": p_approx,
    }


def controller_bootstrap_table(
    step_df: pd.DataFrame,
    metric: str,
    n_bootstrap: int = 2_000,
    seed: int = 42,
    include_median: bool = True,
) -> pd.DataFrame:
    """Bootstrap CIs from step-level data for every controller.

    Parameters
    ----------
    step_df:
        Step-level DataFrame with ``controller`` column.
    metric:
        Metric column to analyse.
    n_bootstrap:
        Resampling iterations.
    seed:
        Random seed.
    include_median:
        If True, also report median and bootstrap CI for the median.

    Returns
    -------
    pd.DataFrame
        One row per controller with mean/median + CIs.
    """
    rows: list[dict] = []
    if step_df.empty or "controller" not in step_df.columns or metric not in step_df.columns:
        return pd.DataFrame(rows)
    for controller, group in step_df.groupby("controller"):
        values = group[metric].dropna().to_numpy()
        if len(values) < 2:
            continue
        low_mean, high_mean = bootstrap_confidence_interval(
            values, n_bootstrap=n_bootstrap, seed=seed, statistic="mean"
        )
        row: dict = {
            "controller": str(controller),
            f"{metric}_mean": float(values.mean()),
            f"{metric}_ci_low": low_mean,
            f"{metric}_ci_high": high_mean,
        }
        if include_median:
            low_med, high_med = bootstrap_confidence_interval(
                values, n_bootstrap=n_bootstrap, seed=seed, statistic="median"
            )
            row[f"{metric}_median"] = float(np.median(values))
            row[f"{metric}_median_ci_low"] = low_med
            row[f"{metric}_median_ci_high"] = high_med
        rows.append(row)
    return pd.DataFrame(rows)


def statistical_power_analysis(
    effect_size_d: float,
    alpha: float = 0.05,
    power: float = 0.80,
    ratio: float = 1.0,
) -> dict[str, float]:
    """Estimate required sample size for a two-sample t-test at given power.

    Uses the Cohen (1988) approximation. Useful for deciding how many
    simulation steps / episodes are needed before statistical comparisons
    are meaningful.

    Parameters
    ----------
    effect_size_d:
        Target Cohen's d (e.g. 0.2 small, 0.5 medium, 0.8 large).
    alpha:
        Type I error rate (two-sided).
    power:
        Desired statistical power (1 - β).
    ratio:
        n2 / n1 ratio (default 1.0 for equal groups).

    Returns
    -------
    dict with keys: n1, n2, total_n, achieved_power, effect_size_d
    """
    from scipy.stats import norm as _norm

    z_alpha = _norm.ppf(1.0 - alpha / 2.0)
    z_beta = _norm.ppf(power)

    if abs(effect_size_d) < 1e-9:
        return {
            "n1": float("inf"),
            "n2": float("inf"),
            "total_n": float("inf"),
            "achieved_power": 0.0,
            "effect_size_d": 0.0,
        }

    # Solve for n1 using the two-sample formula
    n1 = ((z_alpha + z_beta) ** 2 * (1.0 + 1.0 / ratio)) / (effect_size_d ** 2)
    n1 = math.ceil(n1) if n1 != float("inf") else float("inf")
    n2 = math.ceil(n1 * ratio)

    # Verify achieved power with integer n
    se = float(np.sqrt(1.0 / n1 + 1.0 / n2))
    ncp = abs(effect_size_d) / se
    achieved_power = float(1.0 - _norm.cdf(z_alpha - ncp) + _norm.cdf(-z_alpha - ncp))

    return {
        "n1": float(n1),
        "n2": float(n2),
        "total_n": float(n1 + n2),
        "achieved_power": achieved_power,
        "effect_size_d": float(effect_size_d),
    }


