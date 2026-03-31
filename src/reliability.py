"""Extraction reliability — how stable are Claude's signal outputs?

Measures LLM extraction consistency by running multiple extractions on the
same letter and computing variance across runs.

Results are stored in data/reliability/ as JSON files so the dashboard
can display them without re-running expensive API calls.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR
from .schema import flatten_for_timeseries

RELIABILITY_DIR = DATA_DIR / "reliability"

# Ensure directory exists
try:
    RELIABILITY_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    pass  # Read-only filesystem


# =============================================================================
# STORAGE
# =============================================================================


def save_reliability_runs(year: int, runs: list[dict]) -> Path:
    """Save multiple extraction runs for a year."""
    path = RELIABILITY_DIR / f"{year}_runs.json"
    path.write_text(json.dumps(runs, indent=2, default=str))
    return path


def load_reliability_runs(year: int) -> list[dict] | None:
    """Load previously saved reliability runs for a year."""
    path = RELIABILITY_DIR / f"{year}_runs.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def get_available_reliability_years() -> list[int]:
    """Get years with reliability test data."""
    return sorted([
        int(p.stem.replace("_runs", ""))
        for p in RELIABILITY_DIR.glob("*_runs.json")
    ])


# =============================================================================
# ANALYSIS
# =============================================================================


def compute_signal_variance(runs: list[dict]) -> pd.DataFrame:
    """Compute variance statistics across multiple extraction runs.

    Args:
        runs: List of flattened signal dicts (from flatten_for_timeseries)

    Returns:
        DataFrame with columns: signal, mean, std, cv (coefficient of variation),
        min, max, range
    """
    df = pd.DataFrame(runs)

    # Only numeric signal columns
    exclude = ["letter_year", "release_date", "theme_count"]
    numeric_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ["float64", "int64"]
        and not c.startswith("pre_") and not c.startswith("return_")
    ]

    rows = []
    for col in numeric_cols:
        vals = df[col].dropna()
        if len(vals) < 2:
            continue

        mean_val = vals.mean()
        std_val = vals.std()
        cv = (std_val / abs(mean_val)) if abs(mean_val) > 1e-9 else 0.0

        rows.append({
            "signal": col,
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "cv": round(cv, 4),
            "min": round(vals.min(), 4),
            "max": round(vals.max(), 4),
            "range": round(vals.max() - vals.min(), 4),
            "n_runs": len(vals),
        })

    return pd.DataFrame(rows).sort_values("cv", ascending=False).reset_index(drop=True)


def compute_reliability_summary(year: int | None = None) -> dict:
    """Compute overall reliability summary across all tested years.

    Returns:
        Dict with: years_tested, total_runs, avg_cv, median_cv,
        most_stable_signals, least_stable_signals, overall_grade,
        per_signal (DataFrame)
    """
    years = get_available_reliability_years()
    if year:
        years = [y for y in years if y == year]

    if not years:
        return {"available": False}

    all_variances = []
    total_runs = 0

    for y in years:
        runs = load_reliability_runs(y)
        if not runs:
            continue
        total_runs += len(runs)
        var_df = compute_signal_variance(runs)
        var_df["year"] = y
        all_variances.append(var_df)

    if not all_variances:
        return {"available": False}

    combined = pd.concat(all_variances, ignore_index=True)

    # Average CV per signal across years
    per_signal = (
        combined.groupby("signal")
        .agg(avg_cv=("cv", "mean"), avg_std=("std", "mean"), avg_range=("range", "mean"))
        .sort_values("avg_cv")
        .reset_index()
    )

    avg_cv = per_signal["avg_cv"].mean()
    median_cv = per_signal["avg_cv"].median()

    # Grade the overall reliability
    if avg_cv < 0.05:
        grade = "A"
        grade_label = "Highly Consistent"
        detail = "Signals vary by less than 5% on average across runs."
    elif avg_cv < 0.10:
        grade = "B"
        grade_label = "Consistent"
        detail = "Signals vary by 5-10% on average. Minor run-to-run variation."
    elif avg_cv < 0.20:
        grade = "C"
        grade_label = "Moderately Consistent"
        detail = "Signals vary by 10-20%. Some signals show meaningful variation."
    else:
        grade = "D"
        grade_label = "Variable"
        detail = "Significant extraction variance. Results should be interpreted cautiously."

    most_stable = per_signal.head(5)["signal"].tolist()
    least_stable = per_signal.tail(5)["signal"].tolist()

    return {
        "available": True,
        "years_tested": years,
        "total_runs": total_runs,
        "runs_per_year": total_runs // len(years) if years else 0,
        "avg_cv": round(avg_cv, 4),
        "median_cv": round(median_cv, 4),
        "grade": grade,
        "grade_label": grade_label,
        "grade_detail": detail,
        "most_stable": most_stable,
        "least_stable": least_stable,
        "per_signal": per_signal,
    }
