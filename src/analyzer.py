"""Analysis module - combines signals with market data.

This module provides analysis functions on top of the dataset builder.
For the core dataset building, see src/dataset.py.
"""
from __future__ import annotations

import pandas as pd

from .dataset import (
    build_modeling_dataset,
    compute_signal_return_correlations,
    summarize_dataset,
    get_feature_groups,
    get_numeric_features,
    get_categorical_features,
)
from .extractor import load_analysis, get_available_signals
from .schema import LetterExtraction


# Re-export dataset functions
__all__ = [
    "build_modeling_dataset",
    "build_analysis_dataframe",  # Legacy alias
    "compute_signal_return_correlations",
    "get_signal_correlations",  # Legacy alias
    "summarize_dataset",
    "get_signal_summary",  # Legacy alias
    "get_all_analyses",
    "get_themes_over_time",
]


# Legacy aliases for backwards compatibility
build_analysis_dataframe = build_modeling_dataset
get_signal_correlations = compute_signal_return_correlations
get_signal_summary = summarize_dataset


def get_all_analyses() -> list[LetterExtraction]:
    """Load all available letter analyses."""
    return [load_analysis(year) for year in get_available_signals()]


def get_themes_over_time() -> pd.DataFrame:
    """Extract theme trends across years.

    Returns:
        DataFrame with columns: year, theme, prominence, sentiment
    """
    analyses = get_all_analyses()
    rows = []

    for analysis in analyses:
        for theme in analysis.major_themes:
            rows.append({
                "year": analysis.metadata.letter_year,
                "theme": theme.theme,
                "prominence": theme.prominence,
                "sentiment": theme.sentiment,
            })

    return pd.DataFrame(rows)


def get_yearly_signal_comparison(signal_col: str) -> pd.DataFrame:
    """Get a specific signal across all years for comparison.

    Args:
        signal_col: Name of the signal column (e.g., "confidence_overall")

    Returns:
        DataFrame with year and signal value
    """
    df = build_modeling_dataset()
    if df.empty or signal_col not in df.columns:
        return pd.DataFrame()

    return df[["letter_year", signal_col]].copy()
