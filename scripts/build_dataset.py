#!/usr/bin/env python3
"""Build and inspect the modeling dataset.

Usage:
    python scripts/build_dataset.py          # Build and show summary
    python scripts/build_dataset.py --export # Also export to CSV
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import (
    build_modeling_dataset,
    compute_signal_return_correlations,
    summarize_dataset,
    export_dataset,
    get_feature_groups,
)
from src.market import build_market_features


def main():
    print("=" * 70)
    print("BERKSHIRE SIGNAL ENGINE - Dataset Builder")
    print("=" * 70)

    # Step 1: Fetch market data
    print("\n[1] Fetching market data...")
    market_df = build_market_features()
    print(f"    Market data: {len(market_df)} trading days")
    print(f"    Date range: {market_df.index.min().date()} to {market_df.index.max().date()}")
    print(f"    Columns: {list(market_df.columns)}")

    # Step 2: Build modeling dataset
    print("\n[2] Building modeling dataset...")
    df = build_modeling_dataset()

    if df.empty:
        print("    No signals extracted yet!")
        print("    Run: python scripts/extract_all.py")
        return

    print(f"    Rows: {len(df)}")
    print(f"    Columns: {len(df.columns)}")

    # Step 3: Show feature groups
    print("\n[3] Feature groups:")
    groups = get_feature_groups()
    for group, cols in groups.items():
        available = [c for c in cols if c in df.columns]
        print(f"    {group}: {len(available)}/{len(cols)} columns")

    # Step 4: Show summary stats
    print("\n[4] Dataset summary:")
    summary = summarize_dataset(df)
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    # Step 5: Show sample data
    print("\n[5] Sample data (first 3 rows, key columns):")
    key_cols = [
        "letter_year",
        "confidence_overall",
        "uncertainty_overall",
        "composite_bullish",
        "pre_volatility_20d",
        "return_fwd_30d",
    ]
    available_cols = [c for c in key_cols if c in df.columns]
    print(df[available_cols].head(3).to_string(index=False))

    # Step 6: Show correlations
    print("\n[6] Signal-Return correlations:")
    corr_df = compute_signal_return_correlations(df)
    if not corr_df.empty:
        # Show top correlations
        corr_df = corr_df.sort_values("correlation", key=abs, ascending=False)
        print(corr_df.head(10).to_string(index=False))
    else:
        print("    Not enough data for correlations (need 3+ letters with returns)")

    # Step 7: Export if requested
    if "--export" in sys.argv:
        print("\n[7] Exporting dataset...")
        path = export_dataset(format="csv")
        print(f"    Exported to: {path}")

    print("\n" + "=" * 70)
    print("Done!")


if __name__ == "__main__":
    main()
