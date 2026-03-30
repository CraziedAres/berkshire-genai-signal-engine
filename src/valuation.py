"""Fair value estimation based on extracted signals.

This module computes a signal-based fair value estimate for BRK-B.
The methodology is transparent and designed for demonstration purposes.

IMPORTANT: This is NOT financial advice. The model is illustrative,
showing how extracted signals could inform valuation thinking.

Methodology:
1. Start with current market price
2. Compute a "signal premium/discount" from latest letter signals
3. Apply historical context (where are signals vs. historical range?)
4. Output a fair value range with confidence interval
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date

import pandas as pd

from .extractor import load_analysis, get_available_signals
from .market import fetch_price_data, LETTER_RELEASE_DATES
from .schema import LetterExtraction


# =============================================================================
# CONFIGURATION
# =============================================================================

# Signal weights for fair value adjustment (sum to 1.0)
SIGNAL_WEIGHTS = {
    "confidence_overall": 0.20,
    "uncertainty_overall": -0.15,  # Negative: high uncertainty = discount
    "opportunity_richness": 0.15,
    "acquisition_appetite": 0.10,
    "valuation_concern": -0.15,  # Negative: high concern = discount
    "deal_environment": 0.10,
    "insurance_outlook": 0.10,
    "defensive_posture": -0.05,  # Negative: defensive = discount
}

# Maximum adjustment from signals (e.g., 0.15 = +/- 15%)
MAX_SIGNAL_ADJUSTMENT = 0.15

# Market sentiment weight (how much external sentiment affects fair value)
MARKET_SENTIMENT_WEIGHT = 0.25  # 25% weight to market sentiment

# Fair value range width (e.g., 0.10 = +/- 10% around point estimate)
RANGE_WIDTH = 0.10


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class FairValueEstimate:
    """Fair value estimate with supporting data."""

    # Core estimates
    current_price: float
    fair_value: float
    fair_value_low: float
    fair_value_high: float

    # Implied signals
    signal_adjustment: float  # e.g., +0.05 = 5% premium from letter signals
    market_sentiment_adjustment: float  # adjustment from news sentiment
    total_adjustment: float  # combined adjustment
    premium_discount_pct: float  # e.g., -3% = trading at 3% discount

    # Signal breakdown
    signal_contributions: dict[str, float]

    # Market sentiment
    market_sentiment_score: float  # -1 to +1
    market_sentiment_label: str  # bullish/bearish/neutral

    # Metadata
    letter_year: int
    as_of_date: date

    @property
    def recommendation(self) -> str:
        """Simple recommendation based on premium/discount."""
        if self.premium_discount_pct < -10:
            return "Significantly Undervalued"
        elif self.premium_discount_pct < -5:
            return "Undervalued"
        elif self.premium_discount_pct < 5:
            return "Fairly Valued"
        elif self.premium_discount_pct < 10:
            return "Overvalued"
        else:
            return "Significantly Overvalued"

    @property
    def signal_sentiment(self) -> str:
        """Overall signal sentiment."""
        if self.signal_adjustment > 0.05:
            return "Bullish"
        elif self.signal_adjustment > 0:
            return "Slightly Bullish"
        elif self.signal_adjustment > -0.05:
            return "Neutral"
        elif self.signal_adjustment > -0.10:
            return "Slightly Bearish"
        else:
            return "Bearish"


# =============================================================================
# SIGNAL EXTRACTION FOR VALUATION
# =============================================================================

def extract_valuation_signals(analysis: LetterExtraction) -> dict[str, float]:
    """Extract normalized signals relevant to valuation.

    Returns dict of signal_name -> value (all 0-1 scale, higher = more bullish)
    """
    signals = {}

    # Direct confidence signals (higher = bullish)
    signals["confidence_overall"] = analysis.confidence.overall_confidence

    # Uncertainty (invert: lower uncertainty = bullish)
    signals["uncertainty_overall"] = analysis.uncertainty.overall_uncertainty

    # Market signals
    signals["opportunity_richness"] = analysis.market_commentary.opportunity_richness
    signals["valuation_concern"] = analysis.market_commentary.valuation_concern

    # Acquisition signals
    signals["acquisition_appetite"] = analysis.acquisitions.elephant_hunting
    signals["deal_environment"] = analysis.acquisitions.deal_environment

    # Insurance (normalize outlook to 0-1)
    outlook_map = {
        "very_favorable": 1.0,
        "favorable": 0.75,
        "neutral": 0.5,
        "challenging": 0.25,
        "difficult": 0.0,
    }
    signals["insurance_outlook"] = outlook_map.get(
        analysis.insurance_float.outlook.value, 0.5
    )

    # Capital posture (map to defensiveness score)
    posture_map = {
        "aggressive_deploy": 0.0,
        "selective_deploy": 0.25,
        "hold": 0.5,
        "accumulate_cash": 0.75,
        "defensive": 1.0,
    }
    signals["defensive_posture"] = posture_map.get(
        analysis.capital_allocation.posture.value, 0.5
    )

    return signals


def compute_signal_adjustment(signals: dict[str, float]) -> tuple[float, dict[str, float]]:
    """Compute fair value adjustment from signals.

    Returns:
        adjustment: float between -MAX_SIGNAL_ADJUSTMENT and +MAX_SIGNAL_ADJUSTMENT
        contributions: dict showing each signal's contribution
    """
    contributions = {}
    total_adjustment = 0.0

    for signal_name, weight in SIGNAL_WEIGHTS.items():
        if signal_name not in signals:
            continue

        value = signals[signal_name]

        # Normalize to -0.5 to +0.5 (centered on neutral 0.5)
        normalized = value - 0.5

        # Apply weight
        contribution = normalized * weight * 2  # *2 to scale properly
        contributions[signal_name] = contribution
        total_adjustment += contribution

    # Clamp to max adjustment
    total_adjustment = max(-MAX_SIGNAL_ADJUSTMENT,
                          min(MAX_SIGNAL_ADJUSTMENT, total_adjustment))

    return total_adjustment, contributions


# =============================================================================
# FAIR VALUE COMPUTATION
# =============================================================================

def get_current_price() -> tuple[float, date]:
    """Get the most recent BRK-B closing price."""
    df = fetch_price_data()
    latest = df.iloc[-1]
    return latest["Close"], df.index[-1].date()


def compute_fair_value(
    letter_year: int | None = None,
    include_sentiment: bool = True,
) -> FairValueEstimate:
    """Compute fair value estimate based on letter signals and market sentiment.

    Args:
        letter_year: Specific letter year to use, or None for most recent
        include_sentiment: Whether to incorporate market sentiment

    Returns:
        FairValueEstimate with all components
    """
    from .sentiment import get_market_sentiment

    # Get available signals
    available = get_available_signals()
    if not available:
        raise ValueError("No extracted signals available")

    # Use specified year or most recent
    if letter_year is None:
        letter_year = max(available)
    elif letter_year not in available:
        raise ValueError(f"No signals for year {letter_year}")

    # Load analysis
    analysis = load_analysis(letter_year)

    # Extract valuation-relevant signals
    signals = extract_valuation_signals(analysis)

    # Compute letter signal adjustment
    letter_adjustment, contributions = compute_signal_adjustment(signals)

    # Get market sentiment
    sentiment = get_market_sentiment(use_sample=True)
    sentiment_adjustment = 0.0

    if include_sentiment:
        # Market sentiment contributes up to +/- 5% (scaled by confidence)
        max_sentiment_adj = 0.05
        sentiment_adjustment = (
            sentiment.overall_score *
            max_sentiment_adj *
            sentiment.confidence
        )

    # Combine adjustments (weighted)
    # Letter signals: 75%, Market sentiment: 25%
    total_adjustment = (
        letter_adjustment * (1 - MARKET_SENTIMENT_WEIGHT) +
        sentiment_adjustment * MARKET_SENTIMENT_WEIGHT * 3  # Scale up sentiment contrib
    )

    # Clamp total adjustment
    total_adjustment = max(-MAX_SIGNAL_ADJUSTMENT,
                          min(MAX_SIGNAL_ADJUSTMENT, total_adjustment))

    # Get current price
    current_price, as_of_date = get_current_price()

    # Compute fair value
    fair_value = current_price * (1 + total_adjustment)
    fair_value_low = fair_value * (1 - RANGE_WIDTH)
    fair_value_high = fair_value * (1 + RANGE_WIDTH)

    # Premium/discount
    premium_discount = (current_price - fair_value) / fair_value * 100

    return FairValueEstimate(
        current_price=current_price,
        fair_value=fair_value,
        fair_value_low=fair_value_low,
        fair_value_high=fair_value_high,
        signal_adjustment=letter_adjustment,
        market_sentiment_adjustment=sentiment_adjustment,
        total_adjustment=total_adjustment,
        premium_discount_pct=premium_discount,
        signal_contributions=contributions,
        market_sentiment_score=sentiment.overall_score,
        market_sentiment_label=sentiment.overall_label,
        letter_year=letter_year,
        as_of_date=as_of_date,
    )


def get_historical_fair_values() -> pd.DataFrame:
    """Compute fair value for each available letter year.

    Returns DataFrame with year, fair_value, actual_price, etc.
    """
    available = get_available_signals()
    rows = []

    # Get historical prices
    price_df = fetch_price_data()

    for year in available:
        try:
            analysis = load_analysis(year)
            signals = extract_valuation_signals(analysis)
            adjustment, _ = compute_signal_adjustment(signals)

            # Get price at letter release
            release_date = LETTER_RELEASE_DATES.get(year)
            if release_date is None:
                continue

            # Find closest trading day
            mask = price_df.index.date >= release_date
            if not mask.any():
                continue

            release_price = price_df[mask].iloc[0]["Close"]
            fair_value = release_price * (1 + adjustment)

            rows.append({
                "year": year,
                "release_date": release_date,
                "price_at_release": release_price,
                "signal_adjustment": adjustment,
                "fair_value": fair_value,
                "premium_discount_pct": (release_price - fair_value) / fair_value * 100,
            })
        except Exception:
            continue

    return pd.DataFrame(rows)
