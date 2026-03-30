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
from src.valuation import compute_fair_value, get_historical_fair_values
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
# FAIR VALUE ESTIMATE (HERO SECTION)
# =============================================================================

st.divider()

try:
    fv = compute_fair_value()

    # Main fair value display
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.metric(
            label="📊 Signal-Based Fair Value",
            value=f"${fv.fair_value:,.2f}",
            delta=f"{fv.signal_adjustment:+.1%} signal adjustment",
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

    # Market sentiment section
    sentiment = get_market_sentiment()

    st.markdown("##### Market Sentiment (News Analysis)")
    sent_col1, sent_col2, sent_col3, sent_col4 = st.columns(4)

    with sent_col1:
        st.metric(
            "News Sentiment",
            f"{sentiment.sentiment_emoji} {sentiment.overall_label.title()}",
        )
    with sent_col2:
        st.metric("Bullish Articles", f"🟢 {sentiment.bullish_count}")
    with sent_col3:
        st.metric("Bearish Articles", f"🔴 {sentiment.bearish_count}")
    with sent_col4:
        st.metric("Neutral Articles", f"⚪ {sentiment.neutral_count}")

    # Signal contribution breakdown
    with st.expander("📈 Fair Value Methodology", expanded=False):
        st.markdown(f"""
        **Based on {fv.letter_year} Letter Signals + Market Sentiment** (as of {fv.as_of_date})

        The fair value combines:
        - **Letter Signals (75%):** Extracted from shareholder letters
        - **Market Sentiment (25%):** Analyzed from recent financial news

        **Formula:** `Fair Value = Current Price × (1 + Total Adjustment)`

        | Component | Adjustment |
        |-----------|------------|
        | Letter Signals | {fv.signal_adjustment:+.1%} |
        | Market Sentiment | {fv.market_sentiment_adjustment:+.1%} |
        | **Total** | **{fv.total_adjustment:+.1%}** |

        **Letter Signal Contributions:**
        """)

        # Show contributions as a bar chart
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
            height=250,
            showlegend=False,
            xaxis_title="Contribution to Adjustment",
            yaxis_title="",
            xaxis_tickformat=".1%",
        )
        st.plotly_chart(fig, use_container_width=True)

        # Show news headlines
        st.markdown("**Recent News Headlines:**")
        for item in sentiment.news_items[:5]:
            emoji = "🟢" if item.sentiment_label == "bullish" else "🔴" if item.sentiment_label == "bearish" else "⚪"
            trusted = "✓" if item.is_trusted_source else ""
            st.markdown(f"- {emoji} {item.headline} — *{item.source}* {trusted}")

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
st.sidebar.header("Filters")
selected_years = st.sidebar.multiselect(
    "Select Years",
    options=sorted(df["letter_year"].unique()),
    default=sorted(df["letter_year"].unique()),
)

df_filtered = df[df["letter_year"].isin(selected_years)]

# =============================================================================
# OVERVIEW METRICS
# =============================================================================

st.header("Overview")

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
# CATEGORICAL DISTRIBUTIONS
# =============================================================================

st.header("Categorical Signal Distribution")

col1, col2, col3 = st.columns(3)

with col1:
    posture_counts = df_filtered["capital_posture"].value_counts()
    fig = px.pie(
        values=posture_counts.values,
        names=posture_counts.index,
        title="Capital Posture",
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    regime_counts = df_filtered["market_regime"].value_counts()
    fig = px.pie(
        values=regime_counts.values,
        names=regime_counts.index,
        title="Market Regime",
        color_discrete_sequence=px.colors.qualitative.Set3,
    )
    st.plotly_chart(fig, use_container_width=True)

with col3:
    stance_counts = df_filtered["acquisition_stance"].value_counts()
    fig = px.pie(
        values=stance_counts.values,
        names=stance_counts.index,
        title="Acquisition Stance",
        color_discrete_sequence=px.colors.qualitative.Pastel,
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
