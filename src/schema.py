"""Extraction schema for Berkshire letter signals.

This module defines the structured output format for LLM extraction.
All scores use 0.0-1.0 scale with defined anchors.
All categorical fields use fixed vocabularies.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, field_validator


# =============================================================================
# ENUMS - Fixed vocabularies for categorical fields
# =============================================================================


class CapitalPosture(str, Enum):
    """How Berkshire is positioning capital."""

    AGGRESSIVE_DEPLOY = "aggressive_deploy"  # Actively seeking deals, deploying cash
    SELECTIVE_DEPLOY = "selective_deploy"  # Opportunistic, waiting for right price
    HOLD = "hold"  # Maintaining current positions, neutral
    ACCUMULATE_CASH = "accumulate_cash"  # Building cash reserves, defensive
    DEFENSIVE = "defensive"  # Pulling back, concerned about valuations


class MarketRegime(str, Enum):
    """Buffett's characterization of market conditions."""

    EUPHORIC = "euphoric"  # Irrational exuberance, bubble concerns
    OVERVALUED = "overvalued"  # Prices too high, few opportunities
    FAIR = "fair"  # Reasonable valuations, selective opportunities
    UNDERVALUED = "undervalued"  # Good buying opportunities
    DISTRESSED = "distressed"  # Crisis/panic, exceptional opportunities


class AcquisitionStance(str, Enum):
    """Appetite for acquisitions."""

    HUNTING = "hunting"  # Actively seeking, eager
    OPPORTUNISTIC = "opportunistic"  # Open if right deal appears
    PATIENT = "patient"  # Waiting, nothing attractive
    RELUCTANT = "reluctant"  # Environment unfavorable, prices too high


class CashIntent(str, Enum):
    """What Buffett signals about cash usage."""

    DEPLOY_SOON = "deploy_soon"  # Expects to put cash to work
    READY_TO_DEPLOY = "ready_to_deploy"  # Prepared but waiting
    COMFORTABLE_HOLDING = "comfortable_holding"  # Fine with large cash position
    BUILDING_RESERVES = "building_reserves"  # Intentionally accumulating


class InsuranceOutlook(str, Enum):
    """View on insurance/reinsurance business."""

    VERY_FAVORABLE = "very_favorable"  # Hard market, excellent pricing
    FAVORABLE = "favorable"  # Good conditions
    NEUTRAL = "neutral"  # Normal conditions
    CHALLENGING = "challenging"  # Soft market, competitive pressure
    DIFFICULT = "difficult"  # Poor pricing, elevated risks


# =============================================================================
# COMPONENT MODELS
# =============================================================================


class DocumentMetadata(BaseModel):
    """Metadata about the source document."""

    letter_year: int = Field(
        ...,
        ge=1965,
        le=2030,
        description="The fiscal year the letter covers (not release year)",
    )
    release_date: date | None = Field(
        None,
        description="Approximate release date (late Feb of year+1)",
    )
    author: Literal["warren_buffett", "warren_buffett_charlie_munger", "greg_abel"] = Field(
        "warren_buffett",
        description="Primary author(s)",
    )
    word_count_approx: int | None = Field(
        None,
        description="Approximate word count of letter",
    )


class ConfidenceSignals(BaseModel):
    """Management confidence indicators."""

    overall_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Overall management confidence in Berkshire's position (0-1)",
    )
    operating_business_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in operating subsidiaries' performance (0-1)",
    )
    investment_portfolio_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in equity/bond portfolio (0-1)",
    )
    succession_confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence in leadership continuity (0-1)",
    )
    confidence_rationale: str = Field(
        ...,
        max_length=500,
        description="Brief explanation of confidence assessment",
    )


class UncertaintySignals(BaseModel):
    """Uncertainty and risk indicators."""

    overall_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Degree of uncertainty expressed about future (0-1)",
    )
    macro_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncertainty about macroeconomic conditions (0-1)",
    )
    market_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncertainty about market valuations/direction (0-1)",
    )
    operational_uncertainty: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Uncertainty about Berkshire operations (0-1)",
    )
    uncertainty_rationale: str = Field(
        ...,
        max_length=500,
        description="Brief explanation of uncertainty assessment",
    )


class CapitalAllocation(BaseModel):
    """Capital deployment signals."""

    posture: CapitalPosture = Field(
        ...,
        description="Overall capital allocation stance",
    )
    cash_intent: CashIntent = Field(
        ...,
        description="Signaled intention for cash reserves",
    )
    buyback_enthusiasm: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Enthusiasm for share repurchases (0-1)",
    )
    dividend_stance: Literal["maintain", "increase", "not_discussed"] = Field(
        ...,
        description="Stance on dividends",
    )
    capital_rationale: str = Field(
        ...,
        max_length=500,
        description="Explanation of capital allocation signals",
    )


class MarketCommentary(BaseModel):
    """Market and regime assessment."""

    regime: MarketRegime = Field(
        ...,
        description="Buffett's characterization of current market",
    )
    valuation_concern: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Level of concern about market valuations (0-1)",
    )
    opportunity_richness: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Perceived abundance of investment opportunities (0-1)",
    )
    speculation_warning: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Intensity of warnings about speculation (0-1)",
    )
    market_rationale: str = Field(
        ...,
        max_length=500,
        description="Explanation of market assessment",
    )


class InsuranceFloat(BaseModel):
    """Insurance and float commentary."""

    float_emphasis: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How much the letter emphasizes float/insurance (0-1)",
    )
    outlook: InsuranceOutlook = Field(
        ...,
        description="View on insurance market conditions",
    )
    underwriting_discipline: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Emphasis on underwriting discipline (0-1)",
    )
    cat_exposure_concern: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Concern about catastrophe exposure (0-1)",
    )
    insurance_rationale: str = Field(
        ...,
        max_length=500,
        description="Explanation of insurance assessment",
    )


class AcquisitionSignals(BaseModel):
    """M&A and acquisition commentary."""

    stance: AcquisitionStance = Field(
        ...,
        description="Current acquisition appetite",
    )
    elephant_hunting: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Interest in large acquisitions (0-1)",
    )
    bolt_on_interest: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Interest in smaller add-on acquisitions (0-1)",
    )
    deal_environment: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Favorability of deal environment (0=unfavorable, 1=favorable)",
    )
    acquisition_rationale: str = Field(
        ...,
        max_length=500,
        description="Explanation of acquisition signals",
    )


class ThemeTag(BaseModel):
    """A major theme identified in the letter."""

    theme: str = Field(
        ...,
        max_length=50,
        description="Short theme label",
    )
    prominence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="How prominent this theme is (0-1)",
    )
    sentiment: Literal["positive", "negative", "neutral", "mixed"] = Field(
        ...,
        description="Sentiment toward this theme",
    )


class NotableExcerpt(BaseModel):
    """A significant quote from the letter."""

    quote: str = Field(
        ...,
        max_length=1000,
        description="The exact quote from the letter",
    )
    signal_type: str = Field(
        ...,
        max_length=50,
        description="What signal this quote supports (e.g., 'confidence', 'uncertainty')",
    )
    significance: str = Field(
        ...,
        max_length=200,
        description="Why this quote is significant",
    )


# =============================================================================
# MAIN EXTRACTION MODEL
# =============================================================================


class LetterExtraction(BaseModel):
    """Complete structured extraction from a Berkshire shareholder letter.

    This is the top-level model returned by the LLM extraction.
    All numeric scores are 0.0-1.0 with defined anchors.
    All categorical fields use fixed enum vocabularies.
    """

    # Metadata
    metadata: DocumentMetadata

    # Core signals
    confidence: ConfidenceSignals
    uncertainty: UncertaintySignals
    capital_allocation: CapitalAllocation
    market_commentary: MarketCommentary
    insurance_float: InsuranceFloat
    acquisitions: AcquisitionSignals

    # Themes and evidence
    major_themes: list[ThemeTag] = Field(
        ...,
        min_length=3,
        max_length=7,
        description="3-7 major themes from the letter",
    )
    notable_excerpts: list[NotableExcerpt] = Field(
        ...,
        min_length=3,
        max_length=10,
        description="3-10 notable quotes with signal relevance",
    )

    # Summary
    executive_summary: str = Field(
        ...,
        max_length=1000,
        description="2-4 sentence summary of letter's key messages",
    )

    # Composite scores for time series (computed from components)
    @property
    def composite_bullish_score(self) -> float:
        """Composite bullishness indicator (higher = more bullish)."""
        return (
            self.confidence.overall_confidence * 0.3
            + (1 - self.uncertainty.overall_uncertainty) * 0.2
            + self.market_commentary.opportunity_richness * 0.2
            + self.acquisitions.deal_environment * 0.15
            + (1 - self.market_commentary.valuation_concern) * 0.15
        )

    @property
    def composite_defensive_score(self) -> float:
        """Composite defensiveness indicator (higher = more defensive)."""
        posture_map = {
            CapitalPosture.AGGRESSIVE_DEPLOY: 0.0,
            CapitalPosture.SELECTIVE_DEPLOY: 0.25,
            CapitalPosture.HOLD: 0.5,
            CapitalPosture.ACCUMULATE_CASH: 0.75,
            CapitalPosture.DEFENSIVE: 1.0,
        }
        return (
            posture_map[self.capital_allocation.posture] * 0.4
            + self.uncertainty.overall_uncertainty * 0.3
            + self.market_commentary.valuation_concern * 0.3
        )


# =============================================================================
# FLATTENED OUTPUT FOR TIME SERIES
# =============================================================================


def flatten_for_timeseries(extraction: LetterExtraction) -> dict:
    """Flatten extraction to a single dict suitable for DataFrame row."""
    return {
        # Metadata
        "letter_year": extraction.metadata.letter_year,
        "release_date": extraction.metadata.release_date,
        # Confidence
        "confidence_overall": extraction.confidence.overall_confidence,
        "confidence_operating": extraction.confidence.operating_business_confidence,
        "confidence_portfolio": extraction.confidence.investment_portfolio_confidence,
        "confidence_succession": extraction.confidence.succession_confidence,
        # Uncertainty
        "uncertainty_overall": extraction.uncertainty.overall_uncertainty,
        "uncertainty_macro": extraction.uncertainty.macro_uncertainty,
        "uncertainty_market": extraction.uncertainty.market_uncertainty,
        "uncertainty_operational": extraction.uncertainty.operational_uncertainty,
        # Capital allocation
        "capital_posture": extraction.capital_allocation.posture.value,
        "capital_cash_intent": extraction.capital_allocation.cash_intent.value,
        "capital_buyback_enthusiasm": extraction.capital_allocation.buyback_enthusiasm,
        # Market
        "market_regime": extraction.market_commentary.regime.value,
        "market_valuation_concern": extraction.market_commentary.valuation_concern,
        "market_opportunity_richness": extraction.market_commentary.opportunity_richness,
        "market_speculation_warning": extraction.market_commentary.speculation_warning,
        # Insurance
        "insurance_float_emphasis": extraction.insurance_float.float_emphasis,
        "insurance_outlook": extraction.insurance_float.outlook.value,
        "insurance_underwriting_discipline": extraction.insurance_float.underwriting_discipline,
        "insurance_cat_concern": extraction.insurance_float.cat_exposure_concern,
        # Acquisitions
        "acquisition_stance": extraction.acquisitions.stance.value,
        "acquisition_elephant_hunting": extraction.acquisitions.elephant_hunting,
        "acquisition_bolt_on_interest": extraction.acquisitions.bolt_on_interest,
        "acquisition_deal_environment": extraction.acquisitions.deal_environment,
        # Composites
        "composite_bullish": extraction.composite_bullish_score,
        "composite_defensive": extraction.composite_defensive_score,
        # Theme count
        "theme_count": len(extraction.major_themes),
    }
