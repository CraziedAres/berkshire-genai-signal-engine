#!/usr/bin/env python3
"""Extract signals from all available letters."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.extractor import (
    extract_and_save,
    get_available_letters,
    get_available_signals,
)


def main():
    letters = get_available_letters()
    existing = set(get_available_signals())

    if not letters:
        print("No letters found in data/letters/")
        print("Add letter text files named YYYY.txt (e.g., 2023.txt)")
        return

    print(f"Found {len(letters)} letters: {letters}")
    print(f"Already extracted: {sorted(existing)}")

    # Check for --force flag
    force = "--force" in sys.argv
    if force:
        to_extract = letters
    else:
        to_extract = [y for y in letters if y not in existing]

    if not to_extract:
        print("\nAll letters already extracted.")
        print("Use --force to re-extract all.")
        return

    print(f"\nExtracting signals for: {to_extract}")

    for year in to_extract:
        print(f"\n{'='*60}")
        print(f"Extracting {year}...")
        try:
            analysis = extract_and_save(year)

            # Print summary
            print(f"\n  CONFIDENCE")
            print(f"    Overall:    {analysis.confidence.overall_confidence:.2f}")
            print(f"    Operating:  {analysis.confidence.operating_business_confidence:.2f}")
            print(f"    Portfolio:  {analysis.confidence.investment_portfolio_confidence:.2f}")

            print(f"\n  UNCERTAINTY")
            print(f"    Overall:    {analysis.uncertainty.overall_uncertainty:.2f}")
            print(f"    Macro:      {analysis.uncertainty.macro_uncertainty:.2f}")

            print(f"\n  CAPITAL ALLOCATION")
            print(f"    Posture:    {analysis.capital_allocation.posture.value}")
            print(f"    Cash intent:{analysis.capital_allocation.cash_intent.value}")

            print(f"\n  MARKET")
            print(f"    Regime:     {analysis.market_commentary.regime.value}")
            print(f"    Valuation concern: {analysis.market_commentary.valuation_concern:.2f}")

            print(f"\n  ACQUISITIONS")
            print(f"    Stance:     {analysis.acquisitions.stance.value}")
            print(f"    Elephant:   {analysis.acquisitions.elephant_hunting:.2f}")

            print(f"\n  COMPOSITES")
            print(f"    Bullish:    {analysis.composite_bullish_score:.2f}")
            print(f"    Defensive:  {analysis.composite_defensive_score:.2f}")

            print(f"\n  THEMES")
            for theme in analysis.major_themes[:3]:
                print(f"    - {theme.theme} ({theme.prominence:.2f}, {theme.sentiment})")

            print(f"\n  Saved to data/signals/{year}.json")

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*60}")
    print("Done!")


if __name__ == "__main__":
    main()
