"""LLM prompts for Berkshire signal extraction.

See docs/SCORING_GUIDE.md for detailed scoring definitions.
"""

SYSTEM_PROMPT = """You are a financial analyst specializing in Berkshire Hathaway.
Your task is to extract structured signals from Warren Buffett's annual shareholder letters.

CRITICAL INSTRUCTIONS:
1. Focus on Berkshire-specific indicators, not generic sentiment
2. Use the FULL 0.0-1.0 scale—avoid clustering around 0.5
3. Base every score on specific textual evidence
4. Use the exact enum values specified for categorical fields
5. Provide rationales that cite specific language from the letter

DOMAIN KNOWLEDGE:
- Float = insurance premiums collected before claims paid (Berkshire's "free" capital)
- Elephant = large acquisition ($5B+)
- Bolt-on = smaller acquisition by a subsidiary
- Cash posture relates to $100B+ cash reserves Berkshire typically holds
- Letters are released late February covering the prior fiscal year

SCORING GUIDANCE:
- 0.0-0.2: Very low / Absent / Strong negative
- 0.2-0.4: Low / Minimal / Somewhat negative
- 0.4-0.6: Moderate / Neutral / Mixed
- 0.6-0.8: High / Notable / Somewhat positive
- 0.8-1.0: Very high / Dominant / Strong positive"""


EXTRACTION_PROMPT = """Analyze this Berkshire Hathaway shareholder letter and extract structured signals.

<letter>
{letter_text}
</letter>

Extract signals following this exact JSON schema:

{{
  "metadata": {{
    "letter_year": <int: fiscal year the letter covers>,
    "release_date": <string|null: "YYYY-MM-DD" approximate release date>,
    "author": <string: "warren_buffett" or "warren_buffett_charlie_munger">,
    "word_count_approx": <int|null: approximate word count>
  }},

  "confidence": {{
    "overall_confidence": <float 0-1: management confidence in Berkshire's position>,
    "operating_business_confidence": <float 0-1: confidence in operating subsidiaries>,
    "investment_portfolio_confidence": <float 0-1: confidence in equity/bond portfolio>,
    "succession_confidence": <float 0-1: confidence in leadership continuity>,
    "confidence_rationale": <string: 1-3 sentences explaining scores with text evidence>
  }},

  "uncertainty": {{
    "overall_uncertainty": <float 0-1: degree of uncertainty about future>,
    "macro_uncertainty": <float 0-1: uncertainty about economy/rates/inflation>,
    "market_uncertainty": <float 0-1: uncertainty about market direction>,
    "operational_uncertainty": <float 0-1: uncertainty about Berkshire operations>,
    "uncertainty_rationale": <string: 1-3 sentences explaining scores>
  }},

  "capital_allocation": {{
    "posture": <string: "aggressive_deploy"|"selective_deploy"|"hold"|"accumulate_cash"|"defensive">,
    "cash_intent": <string: "deploy_soon"|"ready_to_deploy"|"comfortable_holding"|"building_reserves">,
    "buyback_enthusiasm": <float 0-1: enthusiasm for share repurchases>,
    "dividend_stance": <string: "maintain"|"increase"|"not_discussed">,
    "capital_rationale": <string: 1-3 sentences explaining assessment>
  }},

  "market_commentary": {{
    "regime": <string: "euphoric"|"overvalued"|"fair"|"undervalued"|"distressed">,
    "valuation_concern": <float 0-1: concern about market valuations>,
    "opportunity_richness": <float 0-1: perceived investment opportunities>,
    "speculation_warning": <float 0-1: intensity of speculation warnings>,
    "market_rationale": <string: 1-3 sentences with evidence>
  }},

  "insurance_float": {{
    "float_emphasis": <float 0-1: how much letter emphasizes insurance/float>,
    "outlook": <string: "very_favorable"|"favorable"|"neutral"|"challenging"|"difficult">,
    "underwriting_discipline": <float 0-1: emphasis on disciplined underwriting>,
    "cat_exposure_concern": <float 0-1: concern about catastrophe exposure>,
    "insurance_rationale": <string: 1-3 sentences explaining assessment>
  }},

  "acquisitions": {{
    "stance": <string: "hunting"|"opportunistic"|"patient"|"reluctant">,
    "elephant_hunting": <float 0-1: interest in large acquisitions>,
    "bolt_on_interest": <float 0-1: interest in smaller add-ons>,
    "deal_environment": <float 0-1: favorability of M&A environment>,
    "acquisition_rationale": <string: 1-3 sentences explaining signals>
  }},

  "major_themes": [
    {{
      "theme": <string: short theme label, max 50 chars>,
      "prominence": <float 0-1: how prominent in the letter>,
      "sentiment": <string: "positive"|"negative"|"neutral"|"mixed">
    }}
    // Include 3-7 themes, ordered by prominence
  ],

  "notable_excerpts": [
    {{
      "quote": <string: exact quote from letter>,
      "signal_type": <string: which signal this supports>,
      "significance": <string: why this quote matters>
    }}
    // Include 3-10 excerpts that best support your signal assessments
  ],

  "executive_summary": <string: 2-4 sentence summary of letter's key messages>
}}

IMPORTANT:
- Use exact enum values (lowercase with underscores)
- All float scores must be between 0.0 and 1.0
- Quotes must be exact text from the letter
- Return ONLY valid JSON, no other text"""


# Short prompt for quick extraction (fewer tokens, lower cost)
EXTRACTION_PROMPT_MINIMAL = """Analyze this Berkshire letter and extract key signals as JSON.

<letter>
{letter_text}
</letter>

Return JSON with:
- metadata: {{letter_year, author}}
- confidence: {{overall: 0-1, rationale}}
- uncertainty: {{overall: 0-1, rationale}}
- capital_posture: "aggressive_deploy"|"selective_deploy"|"hold"|"accumulate_cash"|"defensive"
- market_regime: "euphoric"|"overvalued"|"fair"|"undervalued"|"distressed"
- acquisition_stance: "hunting"|"opportunistic"|"patient"|"reluctant"
- themes: [{{theme, prominence: 0-1}}] (3-5 items)
- key_quotes: ["..."] (3-5 quotes)
- summary: "2-3 sentences"

JSON only, no other text."""
