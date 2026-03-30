"""Streamlit dashboard for Berkshire Signal Engine."""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

from src.analyzer import (
    build_analysis_dataframe,
    get_all_analyses,
    get_signal_correlations,
    get_themes_over_time,
)
from src.extractor import get_available_signals
from src.valuation import compute_fair_value, get_historical_fair_values, SIGNAL_WEIGHTS
from src.sentiment import get_market_sentiment

st.set_page_config(
    page_title="Berkshire Signal Engine",
    page_icon="📈",
    layout="wide",
)

st.title("Berkshire GenAI Signal Engine")
st.markdown("*Extracting structured signals from Buffett's shareholder letters*")

# Check for data
available = get_available_signals()
if not available:
    st.warning("No signals extracted yet. Run `python scripts/extract_all.py` first.")
    st.stop()

# Load data
df = build_analysis_dataframe()
analyses = get_all_analyses()

# =============================================================================
# SAMPLE ANALYST PRICE TARGETS (for comparison)
# =============================================================================
# These are illustrative - in production would fetch from financial APIs
ANALYST_TARGETS = {
    "Morningstar": {"target": 525.00, "rating": "4 Stars", "rationale": "Intrinsic value based on DCF of operating businesses + investment portfolio"},
    "UBS": {"target": 540.00, "rating": "Buy", "rationale": "Sum-of-parts valuation with conglomerate discount"},
    "Bank of America": {"target": 510.00, "rating": "Neutral", "rationale": "Trading near fair value given succession transition"},
    "JP Morgan": {"target": 550.00, "rating": "Overweight", "rationale": "Strong operating earnings trajectory under Abel"},
}

# =============================================================================
# FAIR VALUE ESTIMATE (HERO SECTION)
# =============================================================================

st.divider()

try:
    fv = compute_fair_value()
    sentiment = get_market_sentiment()

    # Main fair value display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.metric(
            label="📊 Signal-Based Fair Value",
            value=f"${fv.fair_value:,.2f}",
            delta=f"{fv.total_adjustment:+.1%} total adjustment",
        )
        st.caption(f"Range: ${fv.fair_value_low:,.2f} – ${fv.fair_value_high:,.2f}")

    with col2:
        st.metric(
            label="Current Price",
            value=f"${fv.current_price:,.2f}",
            delta=f"{-fv.premium_discount_pct:+.1f}% vs fair value",
            delta_color="inverse",
        )

    with col3:
        sentiment_colors = {
            "Bullish": "🟢",
            "Slightly Bullish": "🟢",
            "Neutral": "⚪",
            "Slightly Bearish": "🟠",
            "Bearish": "🔴",
        }
        st.metric(
            label="Signal Sentiment",
            value=f"{sentiment_colors.get(fv.signal_sentiment, '⚪')} {fv.signal_sentiment}",
        )

    with col4:
        rec_colors = {
            "Significantly Undervalued": "🟢",
            "Undervalued": "🟢",
            "Fairly Valued": "⚪",
            "Overvalued": "🟠",
            "Significantly Overvalued": "🔴",
        }
        st.metric(
            label="Assessment",
            value=f"{rec_colors.get(fv.recommendation, '⚪')} {fv.recommendation}",
        )

    st.divider()

    # =============================================================================
    # VALUATION METHODOLOGY (Clear, prominent section)
    # =============================================================================

    st.header("Valuation Methodology")

    method_col1, method_col2 = st.columns([1, 1])

    with method_col1:
        st.markdown(f"""
        #### How We Calculate Fair Value

        Our model extracts **quantitative signals** from Warren Buffett's shareholder
        letters and combines them with **market sentiment** from financial news.

        **Formula:**
        ```
        Fair Value = Current Price × (1 + Total Adjustment)
        ```

        **Components:**
        | Source | Weight | Adjustment |
        |--------|--------|------------|
        | Letter Signals | 75% | {fv.signal_adjustment:+.2%} |
        | Market Sentiment | 25% | {fv.market_sentiment_adjustment:+.2%} |
        | **Combined** | **100%** | **{fv.total_adjustment:+.2%}** |

        **Data Sources:**
        - **Letter Year:** {fv.letter_year} shareholder letter
        - **Price Date:** {fv.as_of_date}
        - **News Articles:** {sentiment.total_items} analyzed
        """)

    with method_col2:
        st.markdown("#### Signal Weights")
        st.markdown("*Each signal contributes to the total adjustment based on these weights:*")

        # Signal weights table
        weights_data = []
        for signal, weight in SIGNAL_WEIGHTS.items():
            direction = "Bullish" if weight > 0 else "Bearish"
            emoji = "🟢" if weight > 0 else "🔴"
            weights_data.append({
                "Signal": signal.replace("_", " ").title(),
                "Weight": f"{abs(weight):.0%}",
                "Effect": f"{emoji} {direction}",
            })

        weights_df = pd.DataFrame(weights_data)
        st.dataframe(weights_df, hide_index=True, use_container_width=True)

    # Signal contribution chart
    st.markdown("#### Signal Contributions to Fair Value")

    contrib_df = pd.DataFrame([
        {"Signal": k.replace("_", " ").title(), "Contribution": v}
        for k, v in sorted(fv.signal_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    ])

    fig = px.bar(
        contrib_df,
        x="Contribution",
        y="Signal",
        orientation="h",
        color="Contribution",
        color_continuous_scale=["#d32f2f", "#fff9c4", "#388e3c"],
        color_continuous_midpoint=0,
    )
    fig.update_layout(
        height=300,
        showlegend=False,
        xaxis_title="Contribution to Adjustment",
        yaxis_title="",
        xaxis_tickformat=".1%",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # =============================================================================
    # ANALYST COMPARISON
    # =============================================================================

    st.header("Comparison to Analyst Price Targets")

    st.markdown("""
    How does our signal-based fair value compare to Wall Street analyst targets?
    Below we show the agreement and disagreement with major financial institutions.
    """)

    # Build comparison data
    comparison_data = []
    avg_analyst_target = sum(a["target"] for a in ANALYST_TARGETS.values()) / len(ANALYST_TARGETS)

    for analyst, data in ANALYST_TARGETS.items():
        diff_from_model = ((data["target"] - fv.fair_value) / fv.fair_value) * 100
        diff_from_price = ((data["target"] - fv.current_price) / fv.current_price) * 100

        if abs(diff_from_model) < 3:
            agreement = "🟢 Strong Agreement"
        elif abs(diff_from_model) < 7:
            agreement = "🟡 Moderate Agreement"
        else:
            agreement = "🔴 Disagreement"

        comparison_data.append({
            "Analyst": analyst,
            "Target": f"${data['target']:,.2f}",
            "Rating": data["rating"],
            "vs Our Model": f"{diff_from_model:+.1f}%",
            "vs Current Price": f"{diff_from_price:+.1f}%",
            "Agreement": agreement,
        })

    # Add our model row
    model_vs_avg = ((fv.fair_value - avg_analyst_target) / avg_analyst_target) * 100
    comparison_data.append({
        "Analyst": "📊 Our Model",
        "Target": f"${fv.fair_value:,.2f}",
        "Rating": fv.recommendation,
        "vs Our Model": "—",
        "vs Current Price": f"{-fv.premium_discount_pct:+.1f}%",
        "Agreement": "—",
    })

    comp_df = pd.DataFrame(comparison_data)
    st.dataframe(comp_df, hide_index=True, use_container_width=True)

    # Agreement/Disagreement Analysis
    st.markdown("#### Why the Differences?")

    analysis_col1, analysis_col2 = st.columns(2)

    with analysis_col1:
        st.markdown(f"""
        **Our Model Considers:**
        - Buffett's expressed confidence ({fv.signal_contributions.get('confidence_overall', 0):+.1%} impact)
        - Capital allocation posture
        - Market opportunity assessment
        - Acquisition appetite signals
        - Recent news sentiment ({sentiment.overall_label})

        **Consensus:** ${avg_analyst_target:,.2f} (avg analyst target)
        **Our Estimate:** ${fv.fair_value:,.2f}
        **Difference:** {model_vs_avg:+.1f}%
        """)

    with analysis_col2:
        st.markdown("""
        **Analyst Models Typically Use:**
        - Discounted cash flow (DCF) analysis
        - Sum-of-the-parts valuation
        - Price/Book multiples
        - Historical trading ranges
        - Conglomerate discount adjustments

        **Key Insight:** Our signal-based approach captures *qualitative* factors
        that may not be fully reflected in traditional quantitative models.
        """)

    st.divider()

    # =============================================================================
    # NEWS SENTIMENT BREAKDOWN
    # =============================================================================

    st.header("Market Sentiment Analysis")

    sent_col1, sent_col2, sent_col3 = st.columns([1, 1, 2])

    with sent_col1:
        # Sentiment pie chart
        sentiment_counts = pd.DataFrame({
            "Sentiment": ["Bullish", "Bearish", "Neutral"],
            "Count": [sentiment.bullish_count, sentiment.bearish_count, sentiment.neutral_count],
            "Color": ["#388e3c", "#d32f2f", "#9e9e9e"],
        })

        fig = px.pie(
            sentiment_counts,
            values="Count",
            names="Sentiment",
            title="News Article Sentiment",
            color="Sentiment",
            color_discrete_map={"Bullish": "#388e3c", "Bearish": "#d32f2f", "Neutral": "#9e9e9e"},
        )
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=40, b=0))
        st.plotly_chart(fig, use_container_width=True)

    with sent_col2:
        st.metric(
            "Overall Sentiment",
            f"{sentiment.sentiment_emoji} {sentiment.overall_label.title()}",
        )
        st.metric("Confidence", f"{sentiment.confidence:.0%}")
        st.metric("Articles Analyzed", sentiment.total_items)

        st.markdown(f"""
        **Sentiment Score:** {sentiment.overall_score:+.2f}
        *(Scale: -1.0 bearish to +1.0 bullish)*
        """)

    with sent_col3:
        st.markdown("**Recent Headlines:**")
        for item in sentiment.news_items[:6]:
            emoji = "🟢" if item.sentiment_label == "bullish" else "🔴" if item.sentiment_label == "bearish" else "⚪"
            trusted = " ✓" if item.is_trusted_source else ""
            st.markdown(f"- {emoji} {item.headline} — *{item.source}*{trusted}")

    st.caption("""
    ⚠️ **Disclaimer:** This is a demonstration model, not financial advice.
    The fair value estimate combines qualitative signals from shareholder letters
    with news sentiment analysis and should not be used for investment decisions.
    """)

except Exception as e:
    st.warning(f"Could not compute fair value: {e}")
    import traceback
    st.code(traceback.format_exc())

st.divider()

# Sidebar
st.sidebar.header("Historical Analysis Filters")
st.sidebar.markdown("""
*Filter which shareholder letters to include in the trend analysis below.
The fair value estimate always uses the most recent letter.*
""")
selected_years = st.sidebar.multiselect(
    "Include Letters From",
    options=sorted(df["letter_year"].unique()),
    default=sorted(df["letter_year"].unique()),
)

df_filtered = df[df["letter_year"].isin(selected_years)]

st.sidebar.divider()
st.sidebar.markdown(f"""
**Data Summary:**
- Letters available: {len(df)}
- Years: {min(df['letter_year'])} – {max(df['letter_year'])}
- Signals extracted: 25+
""")

# =============================================================================
# HISTORICAL ANALYSIS (Uses sidebar filter)
# =============================================================================

st.header("Historical Letter Analysis")
st.markdown("*Trends from Buffett's shareholder letters over time. Use sidebar to filter years.*")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Letters Analyzed", len(df_filtered))

with col2:
    avg_conf = df_filtered["confidence_overall"].mean()
    st.metric("Avg Confidence", f"{avg_conf:.2f}")

with col3:
    avg_unc = df_filtered["uncertainty_overall"].mean()
    st.metric("Avg Uncertainty", f"{avg_unc:.2f}")

with col4:
    avg_bull = df_filtered["composite_bullish"].mean()
    st.metric("Avg Bullish Score", f"{avg_bull:.2f}")

st.divider()

# =============================================================================
# SIGNAL TRENDS
# =============================================================================

st.header("Signal Trends Over Time")

tab1, tab2, tab3 = st.tabs(["Confidence & Uncertainty", "Capital & Market", "Composites"])

with tab1:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["confidence_overall"],
        name="Overall Confidence",
        mode="lines+markers",
        line=dict(color="#2E7D32"),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["uncertainty_overall"],
        name="Overall Uncertainty",
        mode="lines+markers",
        line=dict(color="#C62828"),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["confidence_operating"],
        name="Operating Confidence",
        mode="lines+markers",
        line=dict(color="#66BB6A", dash="dash"),
    ))
    fig.update_layout(
        xaxis_title="Letter Year",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["market_valuation_concern"],
        name="Valuation Concern",
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["market_opportunity_richness"],
        name="Opportunity Richness",
        mode="lines+markers",
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["capital_buyback_enthusiasm"],
        name="Buyback Enthusiasm",
        mode="lines+markers",
    ))
    fig.update_layout(
        xaxis_title="Letter Year",
        yaxis_title="Score (0-1)",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["composite_bullish"],
        name="Bullish Composite",
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#1976D2"),
    ))
    fig.add_trace(go.Scatter(
        x=df_filtered["letter_year"],
        y=df_filtered["composite_defensive"],
        name="Defensive Composite",
        mode="lines+markers",
        fill="tozeroy",
        line=dict(color="#F57C00"),
    ))
    fig.update_layout(
        xaxis_title="Letter Year",
        yaxis_title="Composite Score (0-1)",
        yaxis_range=[0, 1],
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SIGNALS VS RETURNS
# =============================================================================

st.header("Signals vs Forward Returns")

if df_filtered["return_fwd_30d"].notna().any():
    col1, col2 = st.columns(2)

    with col1:
        fig = px.scatter(
            df_filtered,
            x="composite_bullish",
            y="return_fwd_30d",
            text="letter_year",
            title="Bullish Composite vs 30-Day Return",
            trendline="ols" if len(df_filtered) > 2 else None,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Bullish Composite Score",
            yaxis_title="30-Day Forward Return",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.scatter(
            df_filtered,
            x="composite_defensive",
            y="return_fwd_90d",
            text="letter_year",
            title="Defensive Composite vs 90-Day Return",
            trendline="ols" if len(df_filtered) > 2 else None,
        )
        fig.update_traces(textposition="top center")
        fig.update_layout(
            xaxis_title="Defensive Composite Score",
            yaxis_title="90-Day Forward Return",
            yaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation table
    st.subheader("Signal-Return Correlations")
    corr_df = get_signal_correlations(df_filtered)
    if not corr_df.empty:
        corr_pivot = corr_df.pivot(index="signal", columns="return_window", values="correlation")
        st.dataframe(
            corr_pivot.style.background_gradient(cmap="RdYlGn", vmin=-1, vmax=1),
            use_container_width=True,
        )
else:
    st.info("No return data available yet (need stock data for post-letter periods)")

# =============================================================================
# THEMES
# =============================================================================

st.header("Major Themes")

themes_df = get_themes_over_time()
if not themes_df.empty:
    themes_filtered = themes_df[themes_df["year"].isin(selected_years)]

    # Theme prominence heatmap
    theme_pivot = themes_filtered.pivot_table(
        index="theme",
        columns="year",
        values="prominence",
        aggfunc="first",
    )

    fig = px.imshow(
        theme_pivot,
        color_continuous_scale="Blues",
        title="Theme Prominence by Year",
        aspect="auto",
    )
    st.plotly_chart(fig, use_container_width=True)

# =============================================================================
# LETTER DETAILS
# =============================================================================

st.header("Letter Details")

for analysis in sorted(analyses, key=lambda x: x.metadata.letter_year, reverse=True):
    if analysis.metadata.letter_year not in selected_years:
        continue

    with st.expander(f"📄 {analysis.metadata.letter_year} Letter"):
        # Scores grid
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("**Confidence**")
            st.metric("Overall", f"{analysis.confidence.overall_confidence:.2f}")
            st.metric("Operating", f"{analysis.confidence.operating_business_confidence:.2f}")
            st.metric("Portfolio", f"{analysis.confidence.investment_portfolio_confidence:.2f}")

        with col2:
            st.markdown("**Uncertainty**")
            st.metric("Overall", f"{analysis.uncertainty.overall_uncertainty:.2f}")
            st.metric("Macro", f"{analysis.uncertainty.macro_uncertainty:.2f}")
            st.metric("Market", f"{analysis.uncertainty.market_uncertainty:.2f}")

        with col3:
            st.markdown("**Capital**")
            st.write(f"Posture: `{analysis.capital_allocation.posture.value}`")
            st.write(f"Cash Intent: `{analysis.capital_allocation.cash_intent.value}`")
            st.metric("Buyback", f"{analysis.capital_allocation.buyback_enthusiasm:.2f}")

        with col4:
            st.markdown("**Market**")
            st.write(f"Regime: `{analysis.market_commentary.regime.value}`")
            st.metric("Valuation Concern", f"{analysis.market_commentary.valuation_concern:.2f}")
            st.metric("Opportunity", f"{analysis.market_commentary.opportunity_richness:.2f}")

        st.divider()

        # Composites
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Bullish Composite", f"{analysis.composite_bullish_score:.2f}")
        with col2:
            st.metric("Defensive Composite", f"{analysis.composite_defensive_score:.2f}")

        st.divider()

        # Summary
        st.markdown("**Executive Summary**")
        st.write(analysis.executive_summary)

        # Themes
        st.markdown("**Major Themes**")
        for theme in analysis.major_themes:
            sentiment_emoji = {"positive": "🟢", "negative": "🔴", "neutral": "⚪", "mixed": "🟡"}
            st.write(
                f"{sentiment_emoji.get(theme.sentiment, '⚪')} "
                f"**{theme.theme}** — prominence: {theme.prominence:.2f}"
            )

        # Key quotes
        st.markdown("**Notable Excerpts**")
        for excerpt in analysis.notable_excerpts:
            st.markdown(f"> {excerpt.quote}")
            st.caption(f"*Signal: {excerpt.signal_type} — {excerpt.significance}*")

# =============================================================================
# FOOTER
# =============================================================================

st.divider()
st.caption("Built with Claude API, Streamlit, and yfinance | Berkshire GenAI Signal Engine")
