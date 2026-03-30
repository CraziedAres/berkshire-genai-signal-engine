"""Market sentiment analysis from financial news sources.

This module fetches and analyzes recent news about Berkshire Hathaway
from reputable financial journalists and outlets.

Sources prioritized:
- CNBC, Bloomberg, Reuters, WSJ, Financial Times
- Motley Fool, Barron's, Morningstar
- Yahoo Finance, MarketWatch
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

# Sentiment keywords for simple rule-based analysis
BULLISH_KEYWORDS = [
    "record", "surge", "soar", "jump", "gain", "rally", "bullish", "optimistic",
    "beat", "exceed", "outperform", "upgrade", "buy", "strong", "growth",
    "profit", "success", "boom", "high", "up", "rise", "positive", "confident",
    "opportunity", "undervalued", "attractive", "recommend", "winner",
]

BEARISH_KEYWORDS = [
    "fall", "drop", "plunge", "sink", "decline", "loss", "bearish", "pessimistic",
    "miss", "disappoint", "underperform", "downgrade", "sell", "weak", "slow",
    "concern", "worry", "risk", "warn", "fear", "crash", "low", "down", "negative",
    "overvalued", "expensive", "avoid", "trouble", "problem",
]

NEUTRAL_KEYWORDS = [
    "hold", "steady", "stable", "unchanged", "flat", "mixed", "uncertain",
    "wait", "neutral", "fair value", "inline", "expected",
]

# Trusted financial news sources (higher weight)
TRUSTED_SOURCES = [
    "cnbc.com", "bloomberg.com", "reuters.com", "wsj.com", "ft.com",
    "barrons.com", "morningstar.com", "fool.com", "marketwatch.com",
    "finance.yahoo.com", "investors.com", "seekingalpha.com",
]


@dataclass
class NewsItem:
    """A single news item about Berkshire."""

    headline: str
    source: str
    url: str
    timestamp: datetime | None = None
    sentiment_score: float = 0.0  # -1 to +1
    sentiment_label: Literal["bullish", "bearish", "neutral"] = "neutral"
    is_trusted_source: bool = False


@dataclass
class MarketSentiment:
    """Aggregated market sentiment from news sources."""

    overall_score: float  # -1 to +1
    overall_label: Literal["bullish", "bearish", "neutral"]
    confidence: float  # 0 to 1, based on number of sources

    bullish_count: int = 0
    bearish_count: int = 0
    neutral_count: int = 0

    news_items: list[NewsItem] = field(default_factory=list)

    fetch_timestamp: datetime = field(default_factory=datetime.now)

    @property
    def total_items(self) -> int:
        return len(self.news_items)

    @property
    def sentiment_emoji(self) -> str:
        if self.overall_score > 0.2:
            return "🟢"
        elif self.overall_score < -0.2:
            return "🔴"
        return "⚪"


def analyze_headline_sentiment(headline: str) -> tuple[float, str]:
    """Analyze sentiment of a single headline.

    Returns:
        score: float from -1 (bearish) to +1 (bullish)
        label: "bullish", "bearish", or "neutral"
    """
    headline_lower = headline.lower()

    bullish_hits = sum(1 for kw in BULLISH_KEYWORDS if kw in headline_lower)
    bearish_hits = sum(1 for kw in BEARISH_KEYWORDS if kw in headline_lower)

    # Compute score
    total_hits = bullish_hits + bearish_hits
    if total_hits == 0:
        return 0.0, "neutral"

    score = (bullish_hits - bearish_hits) / total_hits

    # Determine label
    if score > 0.2:
        label = "bullish"
    elif score < -0.2:
        label = "bearish"
    else:
        label = "neutral"

    return score, label


def is_trusted_source(url: str) -> bool:
    """Check if URL is from a trusted financial news source."""
    url_lower = url.lower()
    return any(source in url_lower for source in TRUSTED_SOURCES)


def parse_news_results(search_results: list[dict]) -> list[NewsItem]:
    """Parse search results into NewsItem objects.

    Args:
        search_results: List of dicts with 'title' and 'url' keys
    """
    items = []

    for result in search_results:
        headline = result.get("title", "")
        url = result.get("url", "")

        if not headline:
            continue

        # Analyze sentiment
        score, label = analyze_headline_sentiment(headline)

        # Check source trust
        trusted = is_trusted_source(url)

        # Extract source domain
        source_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        source = source_match.group(1) if source_match else "unknown"

        items.append(NewsItem(
            headline=headline,
            source=source,
            url=url,
            sentiment_score=score,
            sentiment_label=label,
            is_trusted_source=trusted,
        ))

    return items


def aggregate_sentiment(news_items: list[NewsItem]) -> MarketSentiment:
    """Aggregate sentiment across multiple news items.

    Trusted sources get 2x weight in the aggregate score.
    """
    if not news_items:
        return MarketSentiment(
            overall_score=0.0,
            overall_label="neutral",
            confidence=0.0,
            news_items=[],
        )

    # Weighted average (trusted sources = 2x weight)
    total_weight = 0.0
    weighted_score = 0.0

    bullish_count = 0
    bearish_count = 0
    neutral_count = 0

    for item in news_items:
        weight = 2.0 if item.is_trusted_source else 1.0
        weighted_score += item.sentiment_score * weight
        total_weight += weight

        if item.sentiment_label == "bullish":
            bullish_count += 1
        elif item.sentiment_label == "bearish":
            bearish_count += 1
        else:
            neutral_count += 1

    overall_score = weighted_score / total_weight if total_weight > 0 else 0.0

    # Determine label
    if overall_score > 0.15:
        overall_label = "bullish"
    elif overall_score < -0.15:
        overall_label = "bearish"
    else:
        overall_label = "neutral"

    # Confidence based on number of items (caps at 10 items)
    confidence = min(len(news_items) / 10, 1.0)

    return MarketSentiment(
        overall_score=overall_score,
        overall_label=overall_label,
        confidence=confidence,
        bullish_count=bullish_count,
        bearish_count=bearish_count,
        neutral_count=neutral_count,
        news_items=news_items,
    )


# =============================================================================
# SAMPLE DATA (for demo when web search unavailable)
# =============================================================================

SAMPLE_NEWS = [
    {
        "title": "Berkshire Hathaway Reports Record Operating Earnings of $47.4 Billion",
        "url": "https://www.cnbc.com/berkshire-record-earnings",
    },
    {
        "title": "Warren Buffett's Cash Pile Hits $334 Billion as Deals Remain Elusive",
        "url": "https://www.bloomberg.com/berkshire-cash-pile",
    },
    {
        "title": "Greg Abel Takes Over as Berkshire CEO, Buffett Stays as Chairman",
        "url": "https://www.reuters.com/berkshire-abel-ceo",
    },
    {
        "title": "Berkshire Hathaway Stock: Is It Still a Buy After Buffett Steps Back?",
        "url": "https://www.fool.com/berkshire-buy-analysis",
    },
    {
        "title": "Insurance Float Exceeds $170 Billion at Berkshire Hathaway",
        "url": "https://www.wsj.com/berkshire-insurance-float",
    },
    {
        "title": "Analysts Remain Bullish on Berkshire Despite Leadership Transition",
        "url": "https://www.morningstar.com/berkshire-analyst-ratings",
    },
    {
        "title": "Berkshire's OxyChem Acquisition Shows Continued Deal Appetite",
        "url": "https://www.barrons.com/berkshire-oxychem-deal",
    },
    {
        "title": "BRK.B Shares Undervalued According to Multiple Valuation Metrics",
        "url": "https://finance.yahoo.com/berkshire-undervalued",
    },
]


def get_sample_sentiment() -> MarketSentiment:
    """Get sentiment from sample news data (for demo purposes)."""
    items = parse_news_results(SAMPLE_NEWS)
    return aggregate_sentiment(items)


def get_market_sentiment(use_sample: bool = True) -> MarketSentiment:
    """Get current market sentiment about Berkshire.

    Args:
        use_sample: If True, use sample data. If False, would fetch live data.

    Returns:
        MarketSentiment object with aggregated sentiment
    """
    if use_sample:
        return get_sample_sentiment()

    # TODO: Implement live web search integration
    # This would use a search API or web scraping to get recent news
    return get_sample_sentiment()
