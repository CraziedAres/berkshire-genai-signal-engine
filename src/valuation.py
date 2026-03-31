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
from .market import fetch_price_data, LETTER_RELEASE_DATES, TICKER
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


# =============================================================================
# GRAHAM "INTELLIGENT INVESTOR" VALUATION
# =============================================================================

@dataclass
class GrahamValuation:
    """Benjamin Graham valuation based on The Intelligent Investor principles."""

    current_price: float
    as_of_date: date

    # Graham Number: sqrt(22.5 × EPS × BVPS)
    graham_number: float | None

    # Graham Growth Formula: V = EPS × (8.5 + 2g) × 4.4/Y
    graham_growth_value: float | None

    # Net-Net / Asset value: (Current Assets - Total Liabilities) / Shares
    net_current_asset_value: float | None

    # Key Graham metrics
    trailing_pe: float | None
    price_to_book: float | None
    earnings_per_share: float | None
    book_value_per_share: float | None
    earnings_growth_rate: float | None

    # Graham checklist
    pe_passes: bool  # P/E < 15
    pb_passes: bool  # P/B < 1.5
    pe_pb_passes: bool  # P/E × P/B < 22.5
    positive_earnings: bool
    margin_of_safety_price: float | None  # 2/3 of intrinsic value

    @property
    def composite_fair_value(self) -> float | None:
        """Average of available Graham methods."""
        values = [v for v in [self.graham_number, self.graham_growth_value] if v is not None]
        return sum(values) / len(values) if values else None

    @property
    def checklist_score(self) -> str:
        """How many Graham criteria pass."""
        checks = [self.pe_passes, self.pb_passes, self.pe_pb_passes, self.positive_earnings]
        passed = sum(checks)
        return f"{passed}/4"

    @property
    def recommendation(self) -> str:
        fv = self.composite_fair_value
        if fv is None:
            return "Insufficient Data"
        ratio = self.current_price / fv
        if ratio < 0.67:
            return "Strong Buy (Deep Margin of Safety)"
        elif ratio < 0.85:
            return "Buy (Margin of Safety)"
        elif ratio < 1.0:
            return "Fairly Valued"
        elif ratio < 1.15:
            return "Overvalued"
        else:
            return "Significantly Overvalued"


def compute_graham_valuation() -> GrahamValuation:
    """Compute fair value using Benjamin Graham's methods from The Intelligent Investor.

    Methods:
    1. Graham Number = sqrt(22.5 × EPS × BVPS)
       - Combines earnings and asset value; implies max P/E of 15 and P/B of 1.5
    2. Graham Growth Formula = EPS × (8.5 + 2g) × 4.4/Y
       - 8.5 = P/E for no-growth company
       - g = expected 7-10yr growth rate
       - 4.4 = AAA bond yield in 1962 (Graham's baseline)
       - Y = current AAA corporate bond yield
    3. Margin of Safety = buy at 2/3 of intrinsic value
    """
    import yfinance as yf

    ticker = yf.Ticker(TICKER)
    info = ticker.info

    current_price, as_of_date = get_current_price()

    eps = info.get("trailingEps")
    book_value_raw = info.get("bookValue")
    trailing_pe = info.get("trailingPE")
    earnings_growth = info.get("earningsGrowth")

    # BRK-B book value: yfinance sometimes returns BRK-A values
    # Detect and correct: if book value > 10x price, it's likely BRK-A data
    bvps = None
    if book_value_raw is not None:
        if book_value_raw > current_price * 10:
            bvps = book_value_raw / 1500  # Convert BRK-A to BRK-B
        else:
            bvps = book_value_raw

    price_to_book = current_price / bvps if bvps else None

    # --- Graham Number ---
    graham_number = None
    if eps is not None and eps > 0 and bvps is not None and bvps > 0:
        graham_number = (22.5 * eps * bvps) ** 0.5

    # --- Graham Growth Formula ---
    # V = EPS × (8.5 + 2g) × 4.4/Y
    # Use current AAA yield ~5.0% (approximate)
    graham_growth_value = None
    current_aaa_yield = 5.0  # Approximate current AAA corporate bond yield
    growth_rate = None

    if earnings_growth is not None and earnings_growth > 0:
        growth_rate = earnings_growth * 100  # Convert to percentage
    else:
        # Graham's formula uses expected 7-10 year growth, not trailing quarter.
        # Berkshire's long-term book value CAGR is ~10%; use conservative 8%.
        growth_rate = 8.0

    if eps is not None and eps > 0:
        graham_growth_value = eps * (8.5 + 2 * growth_rate) * (4.4 / current_aaa_yield)

    # --- Net Current Asset Value ---
    # Not typically available for financial conglomerates; skip for BRK
    net_current_asset_value = None

    # --- Graham Checklist ---
    pe_passes = trailing_pe is not None and trailing_pe < 15
    pb_passes = price_to_book is not None and price_to_book < 1.5
    pe_pb_product = (trailing_pe or 0) * (price_to_book or 0)
    pe_pb_passes = pe_pb_product > 0 and pe_pb_product < 22.5
    positive_earnings = eps is not None and eps > 0

    # --- Margin of Safety Price ---
    composite = None
    values = [v for v in [graham_number, graham_growth_value] if v is not None]
    if values:
        composite = sum(values) / len(values)
    margin_of_safety_price = composite * 0.67 if composite else None

    return GrahamValuation(
        current_price=current_price,
        as_of_date=as_of_date,
        graham_number=graham_number,
        graham_growth_value=graham_growth_value,
        net_current_asset_value=net_current_asset_value,
        trailing_pe=trailing_pe,
        price_to_book=price_to_book,
        earnings_per_share=eps,
        book_value_per_share=bvps,
        earnings_growth_rate=growth_rate,
        pe_passes=pe_passes,
        pb_passes=pb_passes,
        pe_pb_passes=pe_pb_passes,
        positive_earnings=positive_earnings,
        margin_of_safety_price=margin_of_safety_price,
    )


# =============================================================================
# BUFFETT HISTORICAL DECISIONS VALUATION
# =============================================================================

# Buffett's revealed preferences from 10 years of letters:
# When he buys back aggressively → thinks stock is cheap
# When he accumulates cash → thinks market is expensive
# When he hunts elephants → sees value in private markets
# This model uses the pattern of his ACTIONS to price BRK-B

# Historical buyback price thresholds (approximate, from letters/filings)
# Buffett authorized buybacks below these Book Value multiples:
BUFFETT_BUYBACK_HISTORY = {
    # year: (buyback_enthusiasm from letter, market regime, posture, approx P/B at time)
    2020: {"buyback_enthusiasm": 0.70, "regime": "undervalued", "posture": "selective_deploy", "approx_pb": 1.15},
    2021: {"buyback_enthusiasm": 0.75, "regime": "fair", "posture": "selective_deploy", "approx_pb": 1.35},
    2022: {"buyback_enthusiasm": 0.60, "regime": "fair", "posture": "accumulate_cash", "approx_pb": 1.30},
    2023: {"buyback_enthusiasm": 0.55, "regime": "overvalued", "posture": "accumulate_cash", "approx_pb": 1.45},
    2024: {"buyback_enthusiasm": 0.45, "regime": "overvalued", "posture": "accumulate_cash", "approx_pb": 1.50},
    2025: {"buyback_enthusiasm": 0.52, "regime": "fair", "posture": "selective_deploy", "approx_pb": 1.40},
}


@dataclass
class BuffettDecisionsValuation:
    """Valuation based on Buffett's revealed preferences over 10 years of decisions."""

    current_price: float
    as_of_date: date

    # Core estimate
    fair_value: float
    fair_value_low: float
    fair_value_high: float

    # Buffett's implied valuation signals
    implied_fair_pb: float  # P/B ratio at which Buffett buys back
    current_pb: float | None
    book_value_per_share: float | None

    # Decision pattern analysis
    buyback_signal: str  # "aggressive buyer", "selective buyer", "paused"
    cash_signal: str  # "deploying", "holding", "hoarding"
    acquisition_signal: str  # "hunting", "patient", "inactive"

    # Historical context
    avg_buyback_pb: float  # Average P/B when Buffett bought back
    current_vs_buyback_zone: str  # "below", "in", "above" buyback zone

    @property
    def recommendation(self) -> str:
        ratio = self.current_price / self.fair_value
        if ratio < 0.90:
            return "Buffett Would Likely Buy"
        elif ratio < 1.05:
            return "In Buffett's Fair Value Zone"
        elif ratio < 1.15:
            return "Buffett Would Hold"
        else:
            return "Buffett Would Sell / Accumulate Cash"


def compute_buffett_valuation() -> BuffettDecisionsValuation:
    """Compute fair value based on Buffett's historical decision patterns.

    Methodology:
    1. Analyze buyback behavior: When did Buffett buy back stock aggressively?
       - High buyback enthusiasm + low P/B = he thought it was cheap
    2. Cash accumulation pattern: Record cash = no good opportunities
    3. Acquisition appetite: Hunting elephants = sees value in private markets
    4. Combine to estimate the P/B ratio Buffett considers "fair"
    5. Apply that P/B to current book value
    """
    import yfinance as yf

    current_price, as_of_date = get_current_price()

    # Get book value per share
    ticker = yf.Ticker(TICKER)
    info = ticker.info
    book_value_raw = info.get("bookValue")
    bvps = None
    if book_value_raw is not None:
        if book_value_raw > current_price * 10:
            bvps = book_value_raw / 1500
        else:
            bvps = book_value_raw

    current_pb = current_price / bvps if bvps else None

    # Load latest letter signals
    available = get_available_signals()
    latest_year = max(available)
    latest_analysis = load_analysis(latest_year)

    # --- Analyze buyback pattern ---
    # Weight recent years more heavily
    weights = {2020: 0.05, 2021: 0.10, 2022: 0.15, 2023: 0.20, 2024: 0.25, 2025: 0.25}
    weighted_pb_sum = 0.0
    weighted_enthusiasm_sum = 0.0
    total_weight = 0.0

    for year, data in BUFFETT_BUYBACK_HISTORY.items():
        w = weights.get(year, 0.1)
        # Higher buyback enthusiasm at a given P/B → stronger signal
        # Weight the P/B by how enthusiastic Buffett was about buying
        weighted_pb_sum += data["approx_pb"] * data["buyback_enthusiasm"] * w
        weighted_enthusiasm_sum += data["buyback_enthusiasm"] * w
        total_weight += w

    # Buffett's implied fair P/B: the P/B he's willing to pay, weighted by conviction
    avg_buyback_pb = weighted_pb_sum / weighted_enthusiasm_sum if weighted_enthusiasm_sum > 0 else 1.3

    # Adjust for current letter signals
    buyback_enth = latest_analysis.capital_allocation.buyback_enthusiasm
    cash_intent = latest_analysis.capital_allocation.cash_intent.value
    posture = latest_analysis.capital_allocation.posture.value
    acq_stance = latest_analysis.acquisitions.stance.value

    # Posture adjustment: if Buffett is deploying, he sees value → raise fair P/B
    posture_adj = {
        "aggressive_deploy": 0.15,
        "selective_deploy": 0.08,
        "hold": 0.0,
        "accumulate_cash": -0.05,
        "defensive": -0.10,
    }
    adj = posture_adj.get(posture, 0.0)

    # Acquisition adjustment: hunting = sees private market value
    acq_adj = {
        "hunting": 0.05,
        "opportunistic": 0.03,
        "patient": 0.0,
        "reluctant": -0.03,
    }
    adj += acq_adj.get(acq_stance, 0.0)

    implied_fair_pb = avg_buyback_pb + adj

    # Compute fair value
    if bvps is not None:
        fair_value = bvps * implied_fair_pb
    else:
        # Fallback: use historical average P/B of ~1.4
        fair_value = current_price * (implied_fair_pb / 1.4)

    fair_value_low = fair_value * 0.90
    fair_value_high = fair_value * 1.10

    # Categorize signals
    if buyback_enth > 0.65:
        buyback_signal = "Aggressive Buyer"
    elif buyback_enth > 0.40:
        buyback_signal = "Selective Buyer"
    else:
        buyback_signal = "Paused"

    cash_map = {
        "deploy_soon": "Deploying",
        "ready_to_deploy": "Ready to Deploy",
        "comfortable_holding": "Holding",
        "building_reserves": "Hoarding",
    }
    cash_signal = cash_map.get(cash_intent, "Unknown")

    acq_signal_map = {
        "hunting": "Actively Hunting",
        "opportunistic": "Opportunistic",
        "patient": "Patient",
        "reluctant": "Inactive",
    }
    acquisition_signal = acq_signal_map.get(acq_stance, "Unknown")

    # Buyback zone comparison
    if current_pb is not None:
        if current_pb < avg_buyback_pb * 0.95:
            buyback_zone = "Below Buyback Zone (Cheap)"
        elif current_pb < avg_buyback_pb * 1.10:
            buyback_zone = "In Buyback Zone"
        else:
            buyback_zone = "Above Buyback Zone (Expensive)"
    else:
        buyback_zone = "Unknown"

    return BuffettDecisionsValuation(
        current_price=current_price,
        as_of_date=as_of_date,
        fair_value=fair_value,
        fair_value_low=fair_value_low,
        fair_value_high=fair_value_high,
        implied_fair_pb=implied_fair_pb,
        current_pb=current_pb,
        book_value_per_share=bvps,
        buyback_signal=buyback_signal,
        cash_signal=cash_signal,
        acquisition_signal=acquisition_signal,
        avg_buyback_pb=avg_buyback_pb,
        current_vs_buyback_zone=buyback_zone,
    )
