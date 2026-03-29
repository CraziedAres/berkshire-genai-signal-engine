"""Dataset builder for merging signals with market data.

This module creates the final modeling dataset by combining:
1. Extracted signals from letters (src/schema.py)
2. Market features at letter release (src/market.py)
3. Forward returns after letter release

The resulting dataset is suitable for:
- Exploratory analysis
- Signal backtesting
- Feature engineering for ML models
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from .config import DATA_DIR
from .extractor import get_available_signals, load_analysis
from .market import (
    build_market_features,
    get_letter_release_date,
    compute_forward_returns,
    compute_pre_letter_features,
)
from .schema import flatten_for_timeseries


# =============================================================================
# DATASET BUILDER
# =============================================================================


def build_modeling_dataset(
    force_refresh_market: bool = False,
) -> pd.DataFrame:
    """Build the complete modeling dataset.

    Merges:
    - Signal features (from LLM extraction)
    - Pre-letter market context (volatility, momentum at release)
    - Post-letter forward returns (30d, 60d, 90d)

    Returns:
        DataFrame with one row per letter, columns:

        METADATA:
        - letter_year: Fiscal year of the letter
        - release_date: When the letter was released

        SIGNAL FEATURES (from extraction):
        - confidence_overall, confidence_operating, ...
        - uncertainty_overall, uncertainty_macro, ...
        - capital_posture, capital_cash_intent, ...
        - market_regime, market_valuation_concern, ...
        - composite_bullish, composite_defensive, ...

        MARKET CONTEXT (at release):
        - pre_price: BRK-B price at letter release
        - pre_volatility_20d: 20-day rolling vol at release
        - pre_volatility_60d: 60-day rolling vol at release
        - pre_return_20d: 20-day momentum at release
        - pre_return_60d: 60-day momentum at release

        FORWARD RETURNS (after release):
        - return_fwd_30d: 30-day forward return
        - return_fwd_60d: 60-day forward return
        - return_fwd_90d: 90-day forward return
    """
    # Get available signal years
    signal_years = get_available_signals()
    if not signal_years:
        return pd.DataFrame()

    # Build market features
    market_df = build_market_features(force_refresh=force_refresh_market)

    rows = []
    for year in signal_years:
        # Load signal extraction
        try:
            analysis = load_analysis(year)
        except FileNotFoundError:
            continue

        # Flatten signals to dict
        row = flatten_for_timeseries(analysis)

        # Get release date
        release_date = get_letter_release_date(year)
        if release_date is None:
            # Use default estimate: Feb 25 of year + 1
            release_date = date(year + 1, 2, 25)

        row["release_date"] = release_date

        # Add pre-letter market context
        pre_features = compute_pre_letter_features(release_date, market_df)
        row.update(pre_features)

        # Add forward returns
        fwd_returns = compute_forward_returns(release_date, market_df)
        row.update(fwd_returns)

        rows.append(row)

    df = pd.DataFrame(rows)

    # Sort by year
    if "letter_year" in df.columns:
        df = df.sort_values("letter_year").reset_index(drop=True)

    return df


def get_feature_groups() -> dict[str, list[str]]:
    """Get column groupings for the modeling dataset.

    Returns:
        Dict mapping group name to list of column names
    """
    return {
        "metadata": [
            "letter_year",
            "release_date",
        ],
        "confidence_signals": [
            "confidence_overall",
            "confidence_operating",
            "confidence_portfolio",
            "confidence_succession",
        ],
        "uncertainty_signals": [
            "uncertainty_overall",
            "uncertainty_macro",
            "uncertainty_market",
            "uncertainty_operational",
        ],
        "capital_signals": [
            "capital_posture",
            "capital_cash_intent",
            "capital_buyback_enthusiasm",
        ],
        "market_signals": [
            "market_regime",
            "market_valuation_concern",
            "market_opportunity_richness",
            "market_speculation_warning",
        ],
        "insurance_signals": [
            "insurance_float_emphasis",
            "insurance_outlook",
            "insurance_underwriting_discipline",
            "insurance_cat_concern",
        ],
        "acquisition_signals": [
            "acquisition_stance",
            "acquisition_elephant_hunting",
            "acquisition_bolt_on_interest",
            "acquisition_deal_environment",
        ],
        "composite_signals": [
            "composite_bullish",
            "composite_defensive",
        ],
        "market_context": [
            "pre_price",
            "pre_volatility_20d",
            "pre_volatility_60d",
            "pre_return_20d",
            "pre_return_60d",
        ],
        "forward_returns": [
            "return_fwd_30d",
            "return_fwd_60d",
            "return_fwd_90d",
        ],
    }


def get_numeric_features(df: pd.DataFrame) -> list[str]:
    """Get list of numeric feature columns (excludes categoricals and metadata)."""
    exclude = ["letter_year", "release_date"]
    categorical = ["capital_posture", "capital_cash_intent", "market_regime",
                   "insurance_outlook", "acquisition_stance"]

    return [
        col for col in df.columns
        if col not in exclude
        and col not in categorical
        and df[col].dtype in ["float64", "int64"]
    ]


def get_categorical_features(df: pd.DataFrame) -> list[str]:
    """Get list of categorical feature columns."""
    return ["capital_posture", "capital_cash_intent", "market_regime",
            "insurance_outlook", "acquisition_stance"]


# =============================================================================
# ANALYSIS HELPERS
# =============================================================================


def compute_signal_return_correlations(
    df: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compute correlations between signals and forward returns.

    Args:
        df: Modeling dataset (will build if not provided)

    Returns:
        DataFrame with columns: signal, return_window, correlation, p_value, n
    """
    if df is None:
        df = build_modeling_dataset()

    if df.empty:
        return pd.DataFrame()

    numeric_signals = get_numeric_features(df)
    return_cols = ["return_fwd_30d", "return_fwd_60d", "return_fwd_90d"]

    # Filter to signals only (exclude market context and returns)
    signal_cols = [c for c in numeric_signals
                   if not c.startswith("pre_") and not c.startswith("return_")]

    results = []
    for signal in signal_cols:
        for ret_col in return_cols:
            if ret_col not in df.columns:
                continue

            # Drop NaN for correlation
            valid = df[[signal, ret_col]].dropna()
            if len(valid) < 3:
                continue

            corr = valid[signal].corr(valid[ret_col])

            results.append({
                "signal": signal,
                "return_window": ret_col.replace("return_fwd_", ""),
                "correlation": corr,
                "n": len(valid),
            })

    return pd.DataFrame(results)


def summarize_dataset(df: pd.DataFrame | None = None) -> dict:
    """Generate summary statistics for the modeling dataset."""
    if df is None:
        df = build_modeling_dataset()

    if df.empty:
        return {"error": "No data available"}

    summary = {
        "n_letters": len(df),
        "year_range": f"{df['letter_year'].min()}-{df['letter_year'].max()}",
        "signals_available": len(get_numeric_features(df)),
    }

    # Signal summary stats
    numeric = get_numeric_features(df)
    for col in ["confidence_overall", "uncertainty_overall",
                "composite_bullish", "composite_defensive"]:
        if col in df.columns:
            summary[f"mean_{col}"] = df[col].mean()
            summary[f"std_{col}"] = df[col].std()

    # Return summary
    for col in ["return_fwd_30d", "return_fwd_60d", "return_fwd_90d"]:
        if col in df.columns and df[col].notna().any():
            summary[f"mean_{col}"] = df[col].mean()
            summary[f"std_{col}"] = df[col].std()

    return summary


# =============================================================================
# EXPORT
# =============================================================================


def export_dataset(
    path: Path | str | None = None,
    format: str = "csv",
) -> Path:
    """Export the modeling dataset to a file.

    Args:
        path: Output path (defaults to data/modeling_dataset.{format})
        format: "csv" or "parquet"

    Returns:
        Path to exported file
    """
    df = build_modeling_dataset()

    if path is None:
        path = DATA_DIR / f"modeling_dataset.{format}"
    else:
        path = Path(path)

    if format == "csv":
        df.to_csv(path, index=False)
    elif format == "parquet":
        df.to_parquet(path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    return path
