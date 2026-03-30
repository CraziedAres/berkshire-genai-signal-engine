"""Market data module for Berkshire Hathaway stock data.

Data Source: Yahoo Finance via yfinance
Ticker: BRK-B (Class B shares - more liquid, lower price than BRK-A)

Assumptions:
1. BRK-B is a suitable proxy for Berkshire performance analysis
2. Adjusted close prices account for splits (BRK-B had a 50:1 split in 2010)
3. Trading days are used for all calculations (weekends/holidays excluded)
4. Returns are simple returns, not log returns (more interpretable)
5. Volatility is annualized using sqrt(252) convention
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from .config import STOCK_DIR

# =============================================================================
# CONSTANTS
# =============================================================================

TICKER = "BRK-B"
TRADING_DAYS_PER_YEAR = 252

# Letter release dates (letters cover prior fiscal year, released late Feb)
# Source: Berkshire Hathaway investor relations
LETTER_RELEASE_DATES: dict[int, date] = {
    2014: date(2015, 2, 28),
    2015: date(2016, 2, 27),
    2016: date(2017, 2, 25),
    2017: date(2018, 2, 24),
    2018: date(2019, 2, 23),
    2019: date(2020, 2, 22),
    2020: date(2021, 2, 27),
    2021: date(2022, 2, 26),
    2022: date(2023, 2, 25),
    2023: date(2024, 2, 24),
    2024: date(2025, 2, 22),
    2025: date(2026, 2, 28),
}


# =============================================================================
# DATA FETCHING
# =============================================================================


def fetch_price_data(
    start: date = date(2010, 1, 1),
    end: date | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Fetch BRK-B price data from Yahoo Finance.

    Args:
        start: Start date for data fetch
        end: End date (defaults to today)
        force_refresh: If True, bypass cache and fetch fresh data

    Returns:
        DataFrame with columns: Open, High, Low, Close, Volume, Dividends, Stock Splits
        Index: DatetimeIndex (trading days only)
    """
    end = end or date.today()
    cache_path = STOCK_DIR / "brk_b_prices.csv"

    # Check cache
    try:
        if cache_path.exists() and not force_refresh:
            cached = pd.read_csv(cache_path, index_col=0)
            cached.index = pd.to_datetime(cached.index, utc=True).tz_localize(None)
            last_cached = cached.index.max().date()

            # If cache is recent enough, use it
            if last_cached >= end - timedelta(days=3):
                return cached
    except Exception:
        pass  # Cache read failed, fetch fresh data

    # Fetch from Yahoo Finance
    ticker = yf.Ticker(TICKER)
    df = ticker.history(start=start, end=end + timedelta(days=1))

    # Try to save to cache (may fail on read-only filesystems like Streamlit Cloud)
    try:
        df.to_csv(cache_path)
    except Exception:
        pass  # Cache write failed, that's okay

    return df


# =============================================================================
# RETURN CALCULATIONS
# =============================================================================


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily returns from price data.

    Args:
        prices: DataFrame with 'Close' column

    Returns:
        DataFrame with added columns:
        - daily_return: Simple daily return (P_t / P_{t-1} - 1)
        - log_return: Log return (for reference, not used in analysis)
    """
    df = prices.copy()

    # Simple return: (P_t - P_{t-1}) / P_{t-1}
    df["daily_return"] = df["Close"].pct_change()

    # Log return: ln(P_t / P_{t-1}) - useful for compounding
    df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


def compute_rolling_volatility(
    returns: pd.DataFrame,
    windows: list[int] = [20, 60],
) -> pd.DataFrame:
    """Compute rolling volatility (annualized standard deviation of returns).

    Args:
        returns: DataFrame with 'daily_return' column
        windows: List of rolling window sizes in trading days

    Returns:
        DataFrame with added columns:
        - volatility_{window}d: Annualized rolling volatility for each window

    Note:
        Annualization: vol_annual = vol_daily * sqrt(252)
        252 trading days per year is the convention.
    """
    df = returns.copy()

    for window in windows:
        daily_vol = df["daily_return"].rolling(window=window).std()
        # Annualize: multiply by sqrt of trading days per year
        annual_vol = daily_vol * (TRADING_DAYS_PER_YEAR ** 0.5)
        df[f"volatility_{window}d"] = annual_vol

    return df


def compute_rolling_returns(
    prices: pd.DataFrame,
    windows: list[int] = [5, 20, 60],
) -> pd.DataFrame:
    """Compute rolling cumulative returns (momentum indicators).

    Args:
        prices: DataFrame with 'Close' column
        windows: List of lookback windows in trading days

    Returns:
        DataFrame with added columns:
        - return_{window}d: Cumulative return over past N days
    """
    df = prices.copy()

    for window in windows:
        # Return over past N days: (P_t / P_{t-N}) - 1
        df[f"return_{window}d"] = df["Close"].pct_change(periods=window)

    return df


# =============================================================================
# MARKET DATA PIPELINE
# =============================================================================


def build_market_features(
    start: date = date(2010, 1, 1),
    end: date | None = None,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """Build complete market features DataFrame.

    Pipeline:
    1. Fetch raw price data
    2. Compute daily returns
    3. Compute rolling volatility (20d, 60d)
    4. Compute rolling returns / momentum (5d, 20d, 60d)

    Returns:
        DataFrame indexed by date with columns:
        - Close: Adjusted close price
        - Volume: Trading volume
        - daily_return: Daily simple return
        - volatility_20d, volatility_60d: Annualized rolling vol
        - return_5d, return_20d, return_60d: Rolling returns
    """
    # Fetch prices
    prices = fetch_price_data(start=start, end=end, force_refresh=force_refresh)

    # Compute features
    df = compute_returns(prices)
    df = compute_rolling_volatility(df)
    df = compute_rolling_returns(df)

    # Select final columns
    feature_cols = [
        "Close",
        "Volume",
        "daily_return",
        "volatility_20d",
        "volatility_60d",
        "return_5d",
        "return_20d",
        "return_60d",
    ]

    return df[[c for c in feature_cols if c in df.columns]]


# =============================================================================
# LETTER DATE ALIGNMENT
# =============================================================================


def get_letter_release_date(letter_year: int) -> date | None:
    """Get the release date for a given letter year.

    Args:
        letter_year: The fiscal year the letter covers (not release year)

    Returns:
        Release date or None if unknown
    """
    return LETTER_RELEASE_DATES.get(letter_year)


def find_trading_day(
    target_date: date,
    market_df: pd.DataFrame,
    direction: str = "forward",
) -> date | None:
    """Find the nearest trading day to a target date.

    Args:
        target_date: The date to match
        market_df: DataFrame with DatetimeIndex of trading days
        direction: "forward" (next trading day) or "backward" (previous)

    Returns:
        Nearest trading day or None if not found
    """
    trading_dates = market_df.index.date

    if direction == "forward":
        mask = trading_dates >= target_date
        if mask.any():
            return trading_dates[mask][0]
    else:
        mask = trading_dates <= target_date
        if mask.any():
            return trading_dates[mask][-1]

    return None


def get_market_context_at_date(
    target_date: date,
    market_df: pd.DataFrame,
) -> dict | None:
    """Get market features for a specific date.

    Args:
        target_date: Date to get context for
        market_df: Market features DataFrame

    Returns:
        Dict of market features or None if date not found
    """
    trading_date = find_trading_day(target_date, market_df, direction="forward")
    if trading_date is None:
        return None

    # Find the row
    mask = market_df.index.date == trading_date
    if not mask.any():
        return None

    row = market_df[mask].iloc[0]
    return row.to_dict()


def compute_forward_returns(
    release_date: date,
    market_df: pd.DataFrame,
    windows: list[int] = [30, 60, 90],
) -> dict[str, float]:
    """Compute forward returns after a letter release date.

    Args:
        release_date: Date of letter release
        market_df: Market features DataFrame
        windows: Forward windows in calendar days

    Returns:
        Dict mapping window to return (e.g., {"return_fwd_30d": 0.05})

    Note:
        Returns are from first trading day on/after release to
        last trading day on/before release + window days.
    """
    results = {}

    # Find starting trading day
    start_trading = find_trading_day(release_date, market_df, "forward")
    if start_trading is None:
        return results

    start_mask = market_df.index.date == start_trading
    if not start_mask.any():
        return results

    start_price = market_df.loc[start_mask, "Close"].iloc[0]

    for window in windows:
        end_date = release_date + timedelta(days=window)
        end_trading = find_trading_day(end_date, market_df, "backward")

        if end_trading is None:
            continue

        end_mask = market_df.index.date == end_trading
        if not end_mask.any():
            continue

        end_price = market_df.loc[end_mask, "Close"].iloc[0]
        fwd_return = (end_price - start_price) / start_price

        results[f"return_fwd_{window}d"] = fwd_return

    return results


def compute_pre_letter_features(
    release_date: date,
    market_df: pd.DataFrame,
) -> dict:
    """Compute market features leading up to letter release.

    Captures the market context when investors read the letter.

    Args:
        release_date: Date of letter release
        market_df: Market features DataFrame

    Returns:
        Dict with pre-letter market features:
        - pre_price: Stock price at release
        - pre_volatility_20d: 20-day vol at release
        - pre_volatility_60d: 60-day vol at release
        - pre_return_20d: 20-day momentum at release
        - pre_return_60d: 60-day momentum at release
    """
    context = get_market_context_at_date(release_date, market_df)
    if context is None:
        return {}

    return {
        "pre_price": context.get("Close"),
        "pre_volatility_20d": context.get("volatility_20d"),
        "pre_volatility_60d": context.get("volatility_60d"),
        "pre_return_20d": context.get("return_20d"),
        "pre_return_60d": context.get("return_60d"),
    }
