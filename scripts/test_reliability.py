#!/usr/bin/env python3
"""Test extraction reliability by running Claude multiple times on the same letter.

Usage:
    python scripts/test_reliability.py              # Test most recent letter, 5 runs
    python scripts/test_reliability.py 2024          # Test specific year
    python scripts/test_reliability.py 2024 --runs 10  # Custom run count

Results are saved to data/reliability/{year}_runs.json and displayed
in the dashboard's reliability section.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import extract_signals, load_letter, get_available_letters
from src.schema import flatten_for_timeseries
from src.reliability import (
    save_reliability_runs,
    compute_signal_variance,
    compute_reliability_summary,
)


def main():
    letters = get_available_letters()
    if not letters:
        print("No letters found.")
        return

    # Parse args
    year = letters[-1]  # Default: most recent
    n_runs = 5

    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if args:
        year = int(args[0])

    if "--runs" in sys.argv:
        idx = sys.argv.index("--runs")
        n_runs = int(sys.argv[idx + 1])

    print(f"Reliability test: {year} letter, {n_runs} runs")
    print(f"{'='*60}")

    letter_text = load_letter(year)
    runs = []

    for i in range(n_runs):
        print(f"\n  Run {i+1}/{n_runs}...", end=" ", flush=True)
        try:
            analysis = extract_signals(letter_text)
            flat = flatten_for_timeseries(analysis)
            runs.append(flat)
            print(f"OK (bullish={analysis.composite_bullish_score:.2f}, "
                  f"defensive={analysis.composite_defensive_score:.2f})")
        except Exception as e:
            print(f"FAILED: {e}")

    if len(runs) < 2:
        print("\nNot enough successful runs to compute variance.")
        return

    # Save results
    path = save_reliability_runs(year, runs)
    print(f"\nSaved {len(runs)} runs to {path}")

    # Show variance summary
    print(f"\n{'='*60}")
    print("SIGNAL VARIANCE ACROSS RUNS")
    print(f"{'='*60}")

    var_df = compute_signal_variance(runs)
    print(f"\n{'Signal':<35} {'Mean':>8} {'Std':>8} {'CV':>8} {'Range':>8}")
    print("-" * 75)
    for _, row in var_df.iterrows():
        print(f"{row['signal']:<35} {row['mean']:>8.4f} {row['std']:>8.4f} "
              f"{row['cv']:>8.4f} {row['range']:>8.4f}")

    # Overall summary
    summary = compute_reliability_summary(year)
    if summary.get("available"):
        print(f"\n{'='*60}")
        print(f"OVERALL GRADE: {summary['grade']} — {summary['grade_label']}")
        print(f"  {summary['grade_detail']}")
        print(f"  Avg CV: {summary['avg_cv']:.4f}")
        print(f"  Most stable:  {', '.join(summary['most_stable'][:3])}")
        print(f"  Least stable: {', '.join(summary['least_stable'][:3])}")


if __name__ == "__main__":
    main()
