"""Stock data module - re-exports from market.py for backwards compatibility.

The main implementation is now in src/market.py.
This file exists for backwards compatibility with existing code.
"""

from .market import (
    # Constants
    TICKER,
    TRADING_DAYS_PER_YEAR,
    LETTER_RELEASE_DATES,
    # Data fetching
    fetch_price_data,
    build_market_features,
    # Return calculations
    compute_returns,
    compute_rolling_volatility,
    compute_rolling_returns,
    # Letter alignment
    get_letter_release_date,
    find_trading_day,
    get_market_context_at_date,
    compute_forward_returns,
    compute_pre_letter_features,
)

# Legacy aliases
get_stock_data = fetch_price_data
get_cached_stock_data = build_market_features
calculate_forward_returns = compute_forward_returns
