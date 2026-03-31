"""Statistical tests for signal predictive power.

Answers the question: Do extracted signals actually predict forward returns?

Three layers of evidence:
1. Correlation t-tests — Pearson r with t-stat and p-value per signal×horizon
2. Information Coefficients — Spearman rank IC (more robust to outliers)
3. OLS Regression — Multivariate regression of returns on top signals
"""
from __future__ import annotations

import warnings

import pandas as pd
import numpy as np
from scipy import stats as sp_stats

from .dataset import build_modeling_dataset, get_numeric_features


# =============================================================================
# CONSTANTS
# =============================================================================

RETURN_HORIZONS = ["return_fwd_30d", "return_fwd_60d", "return_fwd_90d"]
HORIZON_LABELS = {"return_fwd_30d": "30d", "return_fwd_60d": "60d", "return_fwd_90d": "90d"}


def _signal_columns(df: pd.DataFrame) -> list[str]:
    """Get signal feature columns (exclude market context and returns)."""
    numeric = get_numeric_features(df)
    return [c for c in numeric if not c.startswith("pre_") and not c.startswith("return_")]


# =============================================================================
# 1. CORRELATION T-TESTS
# =============================================================================


def compute_correlation_tstats(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pearson correlation with t-statistic and p-value for each signal×horizon.

    Returns:
        DataFrame with columns: signal, horizon, pearson_r, t_stat, p_value, n, significant
    """
    if df is None:
        df = build_modeling_dataset()
    if df.empty:
        return pd.DataFrame()

    signals = _signal_columns(df)
    rows = []

    for signal in signals:
        for ret_col in RETURN_HORIZONS:
            if ret_col not in df.columns:
                continue
            valid = df[[signal, ret_col]].dropna()
            n = len(valid)
            if n < 4:
                continue

            r, p = sp_stats.pearsonr(valid[signal], valid[ret_col])
            # t = r * sqrt(n-2) / sqrt(1-r^2)
            denom = np.sqrt(1 - r**2) if abs(r) < 1.0 else 1e-12
            t_stat = r * np.sqrt(n - 2) / denom

            rows.append({
                "signal": signal,
                "horizon": HORIZON_LABELS[ret_col],
                "pearson_r": round(r, 3),
                "t_stat": round(t_stat, 2),
                "p_value": round(p, 4),
                "n": n,
                "significant": p < 0.10,
            })

    return pd.DataFrame(rows)


# =============================================================================
# 2. INFORMATION COEFFICIENTS (Rank IC)
# =============================================================================


def compute_information_coefficients(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Spearman rank IC for each signal×horizon.

    IC = Spearman correlation between signal ranks and return ranks.
    More robust than Pearson for small samples and non-linear relationships.

    Returns:
        DataFrame with columns: signal, horizon, rank_ic, p_value, n, significant
    """
    if df is None:
        df = build_modeling_dataset()
    if df.empty:
        return pd.DataFrame()

    signals = _signal_columns(df)
    rows = []

    for signal in signals:
        for ret_col in RETURN_HORIZONS:
            if ret_col not in df.columns:
                continue
            valid = df[[signal, ret_col]].dropna()
            n = len(valid)
            if n < 4:
                continue

            ic, p = sp_stats.spearmanr(valid[signal], valid[ret_col])

            rows.append({
                "signal": signal,
                "horizon": HORIZON_LABELS[ret_col],
                "rank_ic": round(ic, 3),
                "p_value": round(p, 4),
                "n": n,
                "significant": p < 0.10,
            })

    return pd.DataFrame(rows)


def compute_ic_summary(df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Pivot IC results into a signal × horizon matrix.

    Returns:
        DataFrame with signals as rows, horizons as columns, values = rank IC
    """
    ic_df = compute_information_coefficients(df)
    if ic_df.empty:
        return pd.DataFrame()

    pivot = ic_df.pivot(index="signal", columns="horizon", values="rank_ic")
    # Reorder columns
    for col in ["30d", "60d", "90d"]:
        if col not in pivot.columns:
            pivot[col] = np.nan
    pivot = pivot[["30d", "60d", "90d"]]
    pivot["avg_abs_ic"] = pivot.abs().mean(axis=1)
    return pivot.sort_values("avg_abs_ic", ascending=False)


# =============================================================================
# 3. OLS REGRESSION
# =============================================================================


def run_ols_regression(
    df: pd.DataFrame | None = None,
    horizon: str = "return_fwd_30d",
    max_features: int = 5,
) -> dict | None:
    """OLS regression of forward returns on top signals (by absolute IC).

    Selects the top `max_features` signals by average |IC| to avoid overfitting
    with the small sample size.

    Returns:
        Dict with keys: horizon, r_squared, adj_r_squared, f_stat, f_pvalue,
        n, k, coefficients (list of dicts with name, coef, std_err, t_stat, p_value)
        Returns None if insufficient data.
    """
    try:
        import statsmodels.api as sm
    except ImportError:
        return None

    if df is None:
        df = build_modeling_dataset()
    if df.empty or horizon not in df.columns:
        return None

    # Pick top signals by IC
    ic_summary = compute_ic_summary(df)
    if ic_summary.empty:
        return None

    top_signals = ic_summary.head(max_features).index.tolist()

    # Build regression data
    cols = top_signals + [horizon]
    reg_df = df[cols].dropna()
    n = len(reg_df)
    if n < len(top_signals) + 2:
        return None

    y = reg_df[horizon]
    X = sm.add_constant(reg_df[top_signals])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = sm.OLS(y, X).fit()

    coefficients = []
    for name in model.params.index:
        label = name if name != "const" else "Intercept"
        coefficients.append({
            "name": label,
            "coef": round(model.params[name], 6),
            "std_err": round(model.bse[name], 6),
            "t_stat": round(model.tvalues[name], 2),
            "p_value": round(model.pvalues[name], 4),
        })

    return {
        "horizon": HORIZON_LABELS.get(horizon, horizon),
        "r_squared": round(model.rsquared, 3),
        "adj_r_squared": round(model.rsquared_adj, 3),
        "f_stat": round(model.fvalue, 2),
        "f_pvalue": round(model.f_pvalue, 4),
        "n": n,
        "k": len(top_signals),
        "coefficients": coefficients,
    }


def run_all_regressions(
    df: pd.DataFrame | None = None,
    max_features: int = 5,
) -> list[dict]:
    """Run OLS regression for all three return horizons."""
    if df is None:
        df = build_modeling_dataset()
    results = []
    for horizon in RETURN_HORIZONS:
        result = run_ols_regression(df, horizon=horizon, max_features=max_features)
        if result is not None:
            results.append(result)
    return results


# =============================================================================
# 4. OVERALL VERDICT
# =============================================================================


def compute_predictive_verdict(df: pd.DataFrame | None = None) -> dict:
    """Summarize overall statistical evidence for signal predictive power.

    Returns:
        Dict with: total_tests, significant_at_10pct, significant_at_5pct,
        best_signal, best_horizon, best_ic, verdict_text
    """
    ic_df = compute_information_coefficients(df)
    tstat_df = compute_correlation_tstats(df)

    if ic_df.empty:
        return {"verdict_text": "Insufficient data for statistical testing."}

    total = len(ic_df)
    sig_10 = int((ic_df["p_value"] < 0.10).sum())
    sig_5 = int((ic_df["p_value"] < 0.05).sum())

    # Best signal by absolute IC
    best_idx = ic_df["rank_ic"].abs().idxmax()
    best = ic_df.loc[best_idx]

    # Average absolute IC across all signal-horizon pairs
    avg_abs_ic = ic_df["rank_ic"].abs().mean()

    # Verdict logic
    if sig_5 / total > 0.2 and avg_abs_ic > 0.3:
        verdict = "Strong evidence"
        detail = "Multiple signals show statistically significant predictive power."
    elif sig_10 / total > 0.15 or avg_abs_ic > 0.2:
        verdict = "Moderate evidence"
        detail = "Some signals show meaningful correlations with forward returns, though sample size limits confidence."
    elif sig_10 > 0:
        verdict = "Weak evidence"
        detail = "A few signals show marginal predictive power. More data would strengthen conclusions."
    else:
        verdict = "No significant evidence"
        detail = "No signals pass significance thresholds at current sample size."

    return {
        "total_tests": total,
        "significant_at_10pct": sig_10,
        "significant_at_5pct": sig_5,
        "avg_abs_ic": round(avg_abs_ic, 3),
        "best_signal": best["signal"],
        "best_horizon": best["horizon"],
        "best_ic": round(best["rank_ic"], 3),
        "verdict": verdict,
        "verdict_detail": detail,
    }
