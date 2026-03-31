"""Signal efficacy — concrete proof that signals predict returns.

Provides simple, visible metrics that answer "do these signals matter?"
for a non-technical audience:

1. Conditional returns: high-signal vs low-signal cohorts
2. Signal-based strategy Sharpe ratio
3. Best signal spotlight
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dataset import build_modeling_dataset, get_numeric_features


RETURN_HORIZONS = {
    "return_fwd_30d": "30d",
    "return_fwd_60d": "60d",
    "return_fwd_90d": "90d",
}


def _signal_cols(df: pd.DataFrame) -> list[str]:
    numeric = get_numeric_features(df)
    return [c for c in numeric if not c.startswith("pre_") and not c.startswith("return_")]


# =============================================================================
# 1. CONDITIONAL RETURNS: high vs low signal cohorts
# =============================================================================


def compute_conditional_returns(
    df: pd.DataFrame | None = None,
    horizon: str = "return_fwd_90d",
) -> list[dict]:
    """Split each signal at its median and compare average forward returns.

    Returns list of dicts with: signal, high_avg_return, low_avg_return,
    spread, high_n, low_n, horizon
    """
    if df is None:
        df = build_modeling_dataset()
    if df.empty or horizon not in df.columns:
        return []

    signals = _signal_cols(df)
    valid = df.dropna(subset=[horizon])
    if len(valid) < 4:
        return []

    results = []
    for signal in signals:
        sv = valid.dropna(subset=[signal])
        if len(sv) < 4:
            continue

        median_val = sv[signal].median()
        high = sv[sv[signal] >= median_val]
        low = sv[sv[signal] < median_val]

        if len(high) < 1 or len(low) < 1:
            continue

        high_ret = high[horizon].mean()
        low_ret = low[horizon].mean()

        results.append({
            "signal": signal,
            "high_avg_return": round(high_ret, 4),
            "low_avg_return": round(low_ret, 4),
            "spread": round(high_ret - low_ret, 4),
            "high_n": len(high),
            "low_n": len(low),
            "horizon": RETURN_HORIZONS.get(horizon, horizon),
        })

    return sorted(results, key=lambda x: abs(x["spread"]), reverse=True)


# =============================================================================
# 2. SIMPLE STRATEGY METRICS
# =============================================================================


def compute_strategy_metrics(
    df: pd.DataFrame | None = None,
    signal: str = "composite_bullish",
    horizon: str = "return_fwd_90d",
) -> dict | None:
    """Compute a simple long/short strategy: long when signal > median, flat otherwise.

    Returns dict with: signal, horizon, strategy_avg_return, buy_hold_avg_return,
    strategy_hit_rate, n_periods, sharpe_ratio
    """
    if df is None:
        df = build_modeling_dataset()
    if df.empty or horizon not in df.columns or signal not in df.columns:
        return None

    valid = df[[signal, horizon]].dropna()
    if len(valid) < 4:
        return None

    median_val = valid[signal].median()

    # Strategy: invest when signal >= median, sit out otherwise
    invested = valid[valid[signal] >= median_val][horizon]
    all_returns = valid[horizon]

    if len(invested) < 2:
        return None

    strategy_mean = invested.mean()
    strategy_std = invested.std()
    buyhold_mean = all_returns.mean()

    # Annualize Sharpe (approximate: assume ~1 letter/year)
    sharpe = (strategy_mean / strategy_std) if strategy_std > 0 else 0.0

    return {
        "signal": signal,
        "horizon": RETURN_HORIZONS.get(horizon, horizon),
        "strategy_avg_return": round(strategy_mean, 4),
        "buy_hold_avg_return": round(buyhold_mean, 4),
        "excess_return": round(strategy_mean - buyhold_mean, 4),
        "hit_rate": round((invested > 0).mean(), 2),
        "n_periods": len(invested),
        "n_total": len(valid),
        "sharpe_ratio": round(sharpe, 2),
    }


# =============================================================================
# 3. BEST SIGNAL SPOTLIGHT
# =============================================================================


def find_best_signal(
    df: pd.DataFrame | None = None,
    horizon: str = "return_fwd_90d",
) -> dict | None:
    """Find the single best predictive signal for the given horizon.

    Uses absolute spread between high/low cohorts as the ranking metric.
    Returns the full conditional return dict plus strategy metrics for the winner.
    """
    cond = compute_conditional_returns(df, horizon=horizon)
    if not cond:
        return None

    best = cond[0]  # Already sorted by |spread|

    strategy = compute_strategy_metrics(df, signal=best["signal"], horizon=horizon)

    return {
        **best,
        "strategy": strategy,
    }


# =============================================================================
# 4. COMPOSITE EFFICACY SUMMARY
# =============================================================================


def compute_efficacy_summary(df: pd.DataFrame | None = None) -> dict:
    """Top-level summary for the dashboard hero section.

    Returns dict with key metrics across all horizons.
    """
    if df is None:
        df = build_modeling_dataset()

    results = {}
    for ret_col, label in RETURN_HORIZONS.items():
        best = find_best_signal(df, horizon=ret_col)
        if best:
            results[label] = best

    if not results:
        return {"available": False}

    # Pick the horizon with the largest absolute spread
    best_horizon = max(results, key=lambda k: abs(results[k]["spread"]))
    best = results[best_horizon]

    # Top conditional returns across all signals for the best horizon
    ret_col = [k for k, v in RETURN_HORIZONS.items() if v == best_horizon][0]
    all_cond = compute_conditional_returns(df, horizon=ret_col)
    top_signals = all_cond[:5]

    return {
        "available": True,
        "best_horizon": best_horizon,
        "best_signal": best["signal"],
        "best_spread": best["spread"],
        "best_high_return": best["high_avg_return"],
        "best_low_return": best["low_avg_return"],
        "strategy": best.get("strategy"),
        "top_signals": top_signals,
        "by_horizon": results,
    }
