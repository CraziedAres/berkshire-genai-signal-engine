# Scoring Guide for Berkshire Letter Extraction

This document defines how to interpret and assign scores for each signal field.

---

## Numeric Score Scale (0.0 - 1.0)

All numeric fields use a consistent 0.0 to 1.0 scale with these anchors:

| Score | Interpretation |
|-------|----------------|
| 0.0 - 0.2 | Very low / Absent / Strong negative |
| 0.2 - 0.4 | Low / Minimal / Somewhat negative |
| 0.4 - 0.6 | Moderate / Neutral / Mixed |
| 0.6 - 0.8 | High / Notable / Somewhat positive |
| 0.8 - 1.0 | Very high / Dominant / Strong positive |

**Important**: Use the full scale. Avoid clustering around 0.5. If a signal is clearly present or absent, score accordingly.

---

## Field-by-Field Scoring Guidance

### Confidence Signals

#### `overall_confidence` (0-1)
Management's expressed confidence in Berkshire's competitive position and future.

| Score | Indicators |
|-------|------------|
| 0.0-0.2 | Explicit concerns about Berkshire's position, unusual hedging language |
| 0.2-0.4 | Muted optimism, significant caveats, defensive tone |
| 0.4-0.6 | Balanced, matter-of-fact assessment |
| 0.6-0.8 | Clear optimism about Berkshire's strengths, positive outlook |
| 0.8-1.0 | Strong conviction, emphatic statements about competitive advantages |

**Example high (0.85)**: "Berkshire is now a sprawling conglomerate, constantly trying to sprawl further... our collection of businesses, coupled with our insurance-based balance sheet, is something that can't be replicated."

**Example low (0.25)**: "We made some serious mistakes this year... our competitive position in [segment] has weakened."

#### `operating_business_confidence` (0-1)
Confidence specifically in operating subsidiaries (BNSF, BHE, manufacturing, etc.).

- High: Praise for managers, strong operating metrics, expansion plans
- Low: Disappointing results, competitive pressures, restructuring needs

#### `investment_portfolio_confidence` (0-1)
Confidence in the equity and fixed income portfolio.

- High: Praise for holdings, "wonderful businesses," long-term conviction
- Low: Write-downs mentioned, mistakes acknowledged, defensive positioning

#### `succession_confidence` (0-1)
Confidence in leadership continuity and succession planning.

- High: Explicit praise for successors, clear succession messaging
- Low: Uncertainty about future leadership, key person risk acknowledged

---

### Uncertainty Signals

#### `overall_uncertainty` (0-1)
Degree of uncertainty expressed about the future broadly.

| Score | Indicators |
|-------|------------|
| 0.0-0.2 | Confident predictions, clear outlook, minimal hedging |
| 0.2-0.4 | Some caveats but generally clear direction |
| 0.4-0.6 | Balanced uncertainty, "time will tell" language |
| 0.6-0.8 | Significant uncertainty expressed, multiple scenarios |
| 0.8-1.0 | Extensive uncertainty, "impossible to predict," crisis language |

**Key phrases indicating high uncertainty:**
- "We have no idea..."
- "The range of outcomes is wide..."
- "It's impossible to predict..."
- "We are prepared for any environment..."

**Key phrases indicating low uncertainty:**
- "We are confident that..."
- "It's clear that..."
- "We expect..."

#### `macro_uncertainty` (0-1)
Uncertainty about economy, interest rates, inflation, geopolitics.

#### `market_uncertainty` (0-1)
Uncertainty about stock market direction and valuations.

#### `operational_uncertainty` (0-1)
Uncertainty about Berkshire's own operations (usually low).

---

### Capital Allocation

#### `posture` (enum)
Overall capital deployment stance.

| Value | Definition | Typical indicators |
|-------|------------|-------------------|
| `aggressive_deploy` | Actively seeking deals, putting cash to work | Recent large acquisitions, eager tone about opportunities |
| `selective_deploy` | Opportunistic but disciplined | "Waiting for the right pitch," price-sensitive language |
| `hold` | Maintaining positions, neutral stance | Focus on existing businesses, no urgency |
| `accumulate_cash` | Building reserves, cautious | Cash growing, few attractive opportunities mentioned |
| `defensive` | Pulling back, concerned about environment | Selling positions, strong caution language |

#### `cash_intent` (enum)
What Buffett signals about cash reserves.

| Value | Definition |
|-------|------------|
| `deploy_soon` | Expects to put cash to work in near term |
| `ready_to_deploy` | Prepared but waiting for opportunities |
| `comfortable_holding` | Fine with large cash position for now |
| `building_reserves` | Intentionally accumulating cash |

#### `buyback_enthusiasm` (0-1)
Interest in repurchasing Berkshire shares.

- High (0.8+): Explicit statements that buybacks are attractive at current prices
- Moderate (0.4-0.6): Buybacks mentioned as option, no strong signal
- Low (0.0-0.3): Shares seen as fairly valued, capital better deployed elsewhere

---

### Market Commentary

#### `regime` (enum)
Buffett's characterization of market conditions.

| Value | Definition | Typical Buffett language |
|-------|------------|-------------------------|
| `euphoric` | Bubble conditions, irrational exuberance | "Casino," "speculation has gone wild," bubble warnings |
| `overvalued` | Generally expensive, few opportunities | "Prices are too high," "nothing to buy" |
| `fair` | Reasonable valuations, selective opportunities | Balanced commentary, some opportunities |
| `undervalued` | Good buying opportunities available | "Bargains available," eager acquisition tone |
| `distressed` | Crisis/panic, exceptional opportunities | "Blood in the streets," aggressive buying commentary |

#### `valuation_concern` (0-1)
Level of concern about market valuations.

**High (0.8+)**: Extended discussion of overvaluation, bubble warnings, comparisons to 1999
**Low (0.0-0.3)**: No concern expressed, or statements about fair/attractive prices

#### `opportunity_richness` (0-1)
Perceived abundance of attractive investments.

**High**: "We see many opportunities," "our phone is ringing"
**Low**: "We find very little to buy," "prices are too high everywhere"

#### `speculation_warning` (0-1)
Intensity of warnings about speculation.

**High (0.8+)**: Extended warnings, moral language, casino metaphors
**Low**: No warnings, or balanced market commentary

---

### Insurance/Float

#### `float_emphasis` (0-1)
How much the letter focuses on insurance and float.

**High (0.7+)**: Extended insurance discussion, float strategy prominent
**Low (0.0-0.3)**: Brief mention, focus on other segments

#### `outlook` (enum)
View on insurance market conditions.

| Value | Definition |
|-------|------------|
| `very_favorable` | Hard market, excellent pricing power |
| `favorable` | Good conditions, disciplined competition |
| `neutral` | Normal market conditions |
| `challenging` | Soft market, competitive pressure |
| `difficult` | Poor pricing, elevated risks |

#### `underwriting_discipline` (0-1)
Emphasis on disciplined underwriting.

**High**: "We will not write business at inadequate prices," discipline emphasized
**Low**: No particular emphasis

#### `cat_exposure_concern` (0-1)
Concern about catastrophe exposure.

**High**: Discussion of cat losses, reinsurance challenges, climate risks
**Low**: No particular concern expressed

---

### Acquisitions

#### `stance` (enum)
Current acquisition appetite.

| Value | Definition | Indicators |
|-------|------------|-----------|
| `hunting` | Actively seeking acquisitions | "Our phone is ringing," "we are eager" |
| `opportunistic` | Open to right opportunity | "If the right deal comes along..." |
| `patient` | Waiting, nothing attractive | "We haven't found anything," "prices too high" |
| `reluctant` | Environment unfavorable | Explicit statements about unattractive M&A landscape |

#### `elephant_hunting` (0-1)
Interest in large ($10B+) acquisitions.

**High**: "Elephant gun is loaded," "we can write very large checks"
**Low**: Focus on smaller deals or organic growth

#### `bolt_on_interest` (0-1)
Interest in smaller add-on acquisitions.

**High**: Discussion of subsidiary bolt-on activity, praise for sub managers doing deals
**Low**: No mention of smaller acquisitions

#### `deal_environment` (0-1)
Favorability of M&A environment (prices, competition, availability).

**High (favorable)**: Good prices, motivated sellers, less competition
**Low (unfavorable)**: High prices, intense competition, few willing sellers

---

## Categorical Field Vocabularies

### Capital Posture
```
aggressive_deploy | selective_deploy | hold | accumulate_cash | defensive
```

### Cash Intent
```
deploy_soon | ready_to_deploy | comfortable_holding | building_reserves
```

### Market Regime
```
euphoric | overvalued | fair | undervalued | distressed
```

### Insurance Outlook
```
very_favorable | favorable | neutral | challenging | difficult
```

### Acquisition Stance
```
hunting | opportunistic | patient | reluctant
```

### Theme Sentiment
```
positive | negative | neutral | mixed
```

---

## Composite Scores (Computed)

These are automatically computed from component scores:

### `composite_bullish_score`
```
= (overall_confidence × 0.3)
+ ((1 - overall_uncertainty) × 0.2)
+ (opportunity_richness × 0.2)
+ (deal_environment × 0.15)
+ ((1 - valuation_concern) × 0.15)
```

### `composite_defensive_score`
```
= (posture_numeric × 0.4)  # 0=aggressive, 1=defensive
+ (overall_uncertainty × 0.3)
+ (valuation_concern × 0.3)
```

---

## Evidence Requirements

Every signal section requires a `_rationale` field explaining the score. This should:
1. Reference specific text from the letter
2. Be 1-3 sentences
3. Explain why the score was assigned, not just what it is

**Good rationale**: "Buffett devotes two paragraphs to warning about 'casino-like' speculation and compares current conditions to 1999, warranting a high speculation_warning score of 0.85."

**Bad rationale**: "The market commentary was cautious."
