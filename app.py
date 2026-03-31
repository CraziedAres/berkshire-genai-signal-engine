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
from src.valuation import (
    compute_fair_value,
    get_historical_fair_values,
    SIGNAL_WEIGHTS,
    compute_graham_valuation,
    compute_buffett_valuation,
)
from src.sentiment import get_market_sentiment
from src.statistics import (
    compute_correlation_tstats,
    compute_ic_summary,
    compute_information_coefficients,
    run_all_regressions,
    compute_predictive_verdict,
)
from src.efficacy import compute_efficacy_summary, compute_conditional_returns
from src.reliability import compute_reliability_summary

# Shared Plotly layout defaults for mobile-friendly charts
PLOTLY_MOBILE_LAYOUT = dict(
    autosize=True,
    margin=dict(l=10, r=10, t=40, b=40),
    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.02,
        xanchor="left",
        x=0,
        font=dict(size=11),
    ),
)

st.set_page_config(
    page_title="Berkshire Signal Engine",
    page_icon="📈",
    layout="wide",
)

# --- Responsive CSS for mobile devices ---
st.markdown("""
<style>
/* Stack columns vertically on small screens */
@media (max-width: 768px) {
    /* Make main content use full width */
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
    }

    /* Stack st.columns vertically */
    [data-testid="stHorizontalBlock"] {
        flex-wrap: wrap !important;
    }
    [data-testid="stHorizontalBlock"] > [data-testid="stColumn"] {
        width: 100% !important;
        flex: 1 1 100% !important;
        min-width: 100% !important;
    }

    /* Collapse sidebar by default on mobile */
    [data-testid="stSidebar"] {
        min-width: 0 !important;
        max-width: 0 !important;
        transform: translateX(-100%);
    }
    [data-testid="stSidebar"][aria-expanded="true"] {
        min-width: 80vw !important;
        max-width: 80vw !important;
        transform: translateX(0);
    }

    /* Ensure tables scroll horizontally */
    [data-testid="stDataFrame"] {
        overflow-x: auto !important;
    }

    /* Reduce metric padding */
    [data-testid="stMetric"] {
        padding: 0.25rem 0 !important;
    }

    /* Slightly smaller title */
    h1 {
        font-size: 1.5rem !important;
    }
}
</style>
""", unsafe_allow_html=True)

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
    gv = compute_graham_valuation()
    bv = compute_buffett_valuation()

    # Hero: 2x2 grid (stacks well on mobile via CSS)
    hero1, hero2 = st.columns(2)
    hero3, hero4 = st.columns(2)

    with hero1:
        st.metric(
            label="Current Price",
            value=f"${fv.current_price:,.2f}",
        )

    with hero2:
        st.metric(
            label="📊 Signal-Based",
            value=f"${fv.fair_value:,.2f}",
            delta=f"{-fv.premium_discount_pct:+.1f}% vs current",
            delta_color="normal",
        )

    with hero3:
        if bv.fair_value:
            bv_delta = ((bv.fair_value - fv.current_price) / fv.current_price) * 100
            st.metric(
                label="🦅 Buffett Model",
                value=f"${bv.fair_value:,.2f}",
                delta=f"{bv_delta:+.1f}% vs current",
                delta_color="normal",
            )
        else:
            st.metric(label="🦅 Buffett Model", value="N/A")

    with hero4:
        if gv.composite_fair_value:
            gv_delta = ((gv.composite_fair_value - fv.current_price) / fv.current_price) * 100
            st.metric(
                label="📖 Graham Model",
                value=f"${gv.composite_fair_value:,.2f}",
                delta=f"{gv_delta:+.1f}% vs current",
                delta_color="normal",
            )
        else:
            st.metric(label="📖 Graham Model", value="N/A")

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
        **PLOTLY_MOBILE_LAYOUT,
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
    # GRAHAM & BUFFETT ALTERNATIVE VALUATIONS
    # =============================================================================

    st.header("Alternative Valuation Models")
    st.markdown("*Two additional fair value estimates using fundamentally different approaches.*")

    graham_col, buffett_col = st.columns(2)

    # --- Graham Model ---
    with graham_col:
        try:
            st.subheader("📖 Graham – The Intelligent Investor")

            if gv.composite_fair_value is not None:
                st.metric(
                    "Graham Fair Value",
                    f"${gv.composite_fair_value:,.2f}",
                    delta=f"{((gv.current_price / gv.composite_fair_value) - 1):+.1%} vs current",
                    delta_color="inverse",
                )
            else:
                st.metric("Graham Fair Value", "N/A")

            st.markdown(f"""
            **Graham Number:** {f'${gv.graham_number:,.2f}' if gv.graham_number else 'N/A'}
            *sqrt(22.5 × EPS × Book Value)*

            **Growth Formula:** {f'${gv.graham_growth_value:,.2f}' if gv.graham_growth_value else 'N/A'}
            *EPS × (8.5 + 2g) × 4.4/Y*

            **Margin of Safety Price:** {f'${gv.margin_of_safety_price:,.2f}' if gv.margin_of_safety_price else 'N/A'}
            *Buy below this for 33% margin*
            """)

            st.markdown("**Graham Checklist**")
            checks = {
                f"P/E < 15 ({gv.trailing_pe:.1f})" if gv.trailing_pe else "P/E < 15": gv.pe_passes,
                f"P/B < 1.5 ({gv.price_to_book:.2f})" if gv.price_to_book else "P/B < 1.5": gv.pb_passes,
                "P/E × P/B < 22.5": gv.pe_pb_passes,
                "Positive Earnings": gv.positive_earnings,
            }
            for label, passed in checks.items():
                st.markdown(f"{'✅' if passed else '❌'} {label}")

            st.markdown(f"**Score:** {gv.checklist_score} | **Verdict:** {gv.recommendation}")

            with st.expander("📐 Graham Methodology"):
                st.markdown(f"""
                Benjamin Graham's *The Intelligent Investor* (1949) introduced the
                concept of **intrinsic value** and **margin of safety** — buying stocks
                only when they trade well below what they're worth.

                **Two valuation methods:**

                **1. Graham Number** = √(22.5 × EPS × BVPS)
                - Combines earnings power (P/E ≤ 15) with asset backing (P/B ≤ 1.5)
                - 22.5 = 15 × 1.5 — Graham's maximum acceptable product
                - For BRK-B: √(22.5 × ${gv.earnings_per_share:.2f} × ${gv.book_value_per_share:,.2f}) = **${gv.graham_number:,.2f}**

                **2. Growth Formula** = EPS × (8.5 + 2g) × 4.4/Y
                - 8.5 = fair P/E for a zero-growth company
                - g = {gv.earnings_growth_rate:.1f}% expected annual growth (7-10yr horizon)
                - 4.4 = AAA bond yield when Graham wrote the formula (1962)
                - Y = ~5.0% current AAA corporate bond yield
                - For BRK-B: ${gv.earnings_per_share:.2f} × (8.5 + 2×{gv.earnings_growth_rate:.1f}) × 4.4/5.0 = **${gv.graham_growth_value:,.2f}**

                **Margin of Safety:** Graham insisted on buying at ≤ 67% of intrinsic
                value. At ${gv.margin_of_safety_price:,.2f}, you'd have his full 33% margin.

                **Checklist** tests whether BRK-B meets Graham's defensive investor criteria:
                adequate size, strong financial condition, earnings stability, and
                moderate P/E and P/B ratios.
                """)

        except Exception as e:
            st.warning(f"Graham valuation unavailable: {e}")

    # --- Buffett Decisions Model ---
    with buffett_col:
        try:
            st.subheader("🦅 Buffett – Revealed Preferences")

            st.metric(
                "Buffett Fair Value",
                f"${bv.fair_value:,.2f}",
                delta=f"{((bv.current_price / bv.fair_value) - 1):+.1%} vs current",
                delta_color="inverse",
            )
            st.caption(f"Range: ${bv.fair_value_low:,.2f} – ${bv.fair_value_high:,.2f}")

            st.markdown(f"""
            **Implied Fair P/B:** {bv.implied_fair_pb:.2f}x
            *Weighted from 6 years of buyback decisions*

            **Current P/B:** {f'{bv.current_pb:.2f}x' if bv.current_pb else 'N/A'}

            **Book Value/Share:** {f'${bv.book_value_per_share:,.2f}' if bv.book_value_per_share else 'N/A'}
            """)

            st.markdown("**Decision Pattern Signals**")
            signal_emojis = {
                "Aggressive Buyer": "🟢", "Selective Buyer": "🟡", "Paused": "🔴",
                "Deploying": "🟢", "Ready to Deploy": "🟡", "Holding": "🟠", "Hoarding": "🔴",
                "Actively Hunting": "🟢", "Opportunistic": "🟡", "Patient": "🟠", "Inactive": "🔴",
            }
            st.markdown(f"- {signal_emojis.get(bv.buyback_signal, '⚪')} **Buyback:** {bv.buyback_signal}")
            st.markdown(f"- {signal_emojis.get(bv.cash_signal, '⚪')} **Cash:** {bv.cash_signal}")
            st.markdown(f"- {signal_emojis.get(bv.acquisition_signal, '⚪')} **Acquisitions:** {bv.acquisition_signal}")

            st.markdown(f"**Zone:** {bv.current_vs_buyback_zone}")
            st.markdown(f"**Verdict:** {bv.recommendation}")

            with st.expander("🔍 Buffett Methodology"):
                st.markdown(f"""
                Rather than using a formula, this model asks: **at what price has Buffett
                himself been willing to buy Berkshire stock?**

                Buffett's own buyback decisions are the strongest signal of what he
                considers fair value — he's spending shareholders' money, and he's said
                he only buys back when the price is "below Berkshire's intrinsic value."

                **How it works:**

                **1. Historical Buyback Analysis (2020–2025)**
                - We extract `buyback_enthusiasm` from each year's shareholder letter
                - Cross-reference with the approximate P/B ratio at the time
                - Weight recent years more heavily (2024-2025 = 25% each, earlier years less)
                - Result: a **weighted average P/B** at which Buffett buys = {bv.avg_buyback_pb:.2f}x

                **2. Current Posture Adjustment**
                - Current capital posture (`{bv.cash_signal}`) shifts the implied fair P/B
                - Acquisition stance (`{bv.acquisition_signal}`) provides additional signal
                - If Buffett is deploying capital → he sees value → fair P/B goes up
                - If hoarding cash → market is expensive → fair P/B goes down

                **3. Fair Value Calculation**
                - Implied Fair P/B: **{bv.implied_fair_pb:.2f}x**
                - Book Value/Share: **${bv.book_value_per_share:,.2f}**
                - Fair Value = {bv.implied_fair_pb:.2f} × ${bv.book_value_per_share:,.2f} = **${bv.fair_value:,.2f}**

                **Buyback Zone:** When the current P/B ({bv.current_pb:.2f}x) is near the
                historical buyback P/B ({bv.avg_buyback_pb:.2f}x), we say the stock is
                "in Buffett's buyback zone" — the range where he's historically been a buyer.

                ---

                **⚠️ A Note on Book Value & GAAP Accounting**

                Buffett himself has cautioned against over-relying on book value. In his
                2018 letter, he stopped using book value growth as Berkshire's key metric,
                calling it "far less relevant than it once was." Why?

                - **GAAP understates operating businesses.** Wholly-owned subsidiaries like
                  GEICO are carried at acquisition cost minus depreciation — their true
                  economic value is vastly higher than what the balance sheet shows.
                - **GAAP overstates earnings volatility.** Since 2018, GAAP requires
                  unrealized investment gains/losses in net income (ASC 320). Berkshire's
                  reported earnings can swing by billions based on stock market moves that
                  have nothing to do with business performance. Buffett calls these figures
                  "wildly misleading."
                - **Float is an asset, not a liability.** Insurance float (~$170B) appears
                  as a liability under GAAP, but Berkshire's profitable underwriting means
                  this is effectively costless capital for investment.
                - **"Owner Earnings" > Net Income.** Buffett prefers owner earnings
                  (net income + depreciation - maintenance capex) as the true measure of
                  earning power.

                We use P/B as a *proxy* because it's the metric with the longest track record
                of correlating with Buffett's buyback decisions — even if he'd tell you the
                real math is more nuanced.
                """)

        except Exception as e:
            st.warning(f"Buffett valuation unavailable: {e}")

    # --- Three-Model Comparison ---
    st.divider()
    st.subheader("Three-Model Comparison")

    try:
        gv_val = gv.composite_fair_value if gv.composite_fair_value else None
        bv_val = bv.fair_value if bv.fair_value else None

        comparison_rows = [
            {
                "Model": "📊 Signal-Based (Letters + News)",
                "Fair Value": f"${fv.fair_value:,.2f}",
                "vs Current Price": f"{-fv.premium_discount_pct:+.1f}%",
                "Method": "NLP signal extraction from shareholder letters + market sentiment",
                "Verdict": fv.recommendation,
            },
        ]
        if gv_val:
            comparison_rows.append({
                "Model": "📖 Graham (Intelligent Investor)",
                "Fair Value": f"${gv_val:,.2f}",
                "vs Current Price": f"{((gv_val - fv.current_price) / fv.current_price * 100):+.1f}%",
                "Method": "Graham Number + Growth Formula (earnings × book value)",
                "Verdict": gv.recommendation,
            })
        if bv_val:
            comparison_rows.append({
                "Model": "🦅 Buffett (Revealed Preferences)",
                "Fair Value": f"${bv_val:,.2f}",
                "vs Current Price": f"{((bv_val - fv.current_price) / fv.current_price * 100):+.1f}%",
                "Method": "Historical buyback patterns → implied fair P/B × book value",
                "Verdict": bv.recommendation,
            })

        comp_models_df = pd.DataFrame(comparison_rows)
        st.dataframe(comp_models_df, hide_index=True, use_container_width=True)

        # Visual comparison bar chart
        chart_data = [{"Model": "Signal-Based", "Fair Value": fv.fair_value}]
        if gv_val:
            chart_data.append({"Model": "Graham", "Fair Value": gv_val})
        if bv_val:
            chart_data.append({"Model": "Buffett", "Fair Value": bv_val})
        chart_data.append({"Model": "Current Price", "Fair Value": fv.current_price})

        chart_df = pd.DataFrame(chart_data)
        fig = px.bar(
            chart_df,
            x="Model",
            y="Fair Value",
            color="Model",
            color_discrete_sequence=["#1976D2", "#388e3c", "#F57C00", "#9e9e9e"],
            title="Fair Value Estimates by Model",
        )
        fig.update_layout(
            showlegend=False,
            yaxis_title="Price ($)",
            yaxis_tickformat="$,.0f",
            height=350,
            margin=dict(l=10, r=10, t=40, b=40),
        )
        # Add current price line
        fig.add_hline(
            y=fv.current_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: ${fv.current_price:,.2f}",
        )
        st.plotly_chart(fig, use_container_width=True)
    except Exception:
        pass  # Comparison chart is optional

    st.divider()

    # =============================================================================
    # NEWS SENTIMENT BREAKDOWN
    # =============================================================================

    st.header("Market Sentiment Analysis")

    sent_col1, sent_col2 = st.columns(2)

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

# Sidebar — keep minimal for mobile
st.sidebar.header("Filters")
selected_years = st.sidebar.multiselect(
    "Letters",
    options=sorted(df["letter_year"].unique()),
    default=sorted(df["letter_year"].unique()),
)

df_filtered = df[df["letter_year"].isin(selected_years)]

# =============================================================================
# HISTORICAL ANALYSIS (Uses sidebar filter)
# =============================================================================

st.header("Historical Letter Analysis")
st.markdown("*Trends from Buffett's shareholder letters over time. Use sidebar to filter years.*")

hist_col1, hist_col2 = st.columns(2)
hist_col3, hist_col4 = st.columns(2)

with hist_col1:
    st.metric("Letters Analyzed", len(df_filtered))

with hist_col2:
    avg_conf = df_filtered["confidence_overall"].mean()
    st.metric("Avg Confidence", f"{avg_conf:.2f}")

with hist_col3:
    avg_unc = df_filtered["uncertainty_overall"].mean()
    st.metric("Avg Uncertainty", f"{avg_unc:.2f}")

with hist_col4:
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
        **PLOTLY_MOBILE_LAYOUT,
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
        **PLOTLY_MOBILE_LAYOUT,
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
        **PLOTLY_MOBILE_LAYOUT,
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
# SIGNAL EFFICACY: Do these signals matter?
# =============================================================================

st.header("Do These Signals Actually Matter?")
st.markdown("*The key question: do high-signal years produce better returns than low-signal years?*")

try:
    efficacy = compute_efficacy_summary(df_filtered)

    if efficacy.get("available"):
        best_signal_name = efficacy["best_signal"].replace("_", " ").title()
        best_horizon = efficacy["best_horizon"]

        # Hero metric: the headline finding
        eff_col1, eff_col2 = st.columns(2)

        with eff_col1:
            st.metric(
                f"Best Signal: {best_signal_name}",
                f"{efficacy['best_spread']:+.1%} spread",
                help=f"Difference in avg {best_horizon} return between high-signal and low-signal letters",
            )
            st.markdown(
                f"When **{best_signal_name}** is above its median: "
                f"avg {best_horizon} return = **{efficacy['best_high_return']:+.1%}**. "
                f"Below median: **{efficacy['best_low_return']:+.1%}**."
            )

        with eff_col2:
            strat = efficacy.get("strategy")
            if strat:
                st.metric(
                    "Signal-Based Strategy",
                    f"{strat['strategy_avg_return']:+.1%} avg return",
                    delta=f"{strat['excess_return']:+.1%} vs buy-and-hold",
                    delta_color="normal",
                )
                st.markdown(
                    f"**Hit rate:** {strat['hit_rate']:.0%} of invested periods positive | "
                    f"**Sharpe:** {strat['sharpe_ratio']:.2f}"
                )

        # Conditional returns for top signals
        st.divider()
        st.subheader("High vs Low Signal Returns")
        st.markdown(
            f"*For each signal, we split at the median and compare average "
            f"**{best_horizon}** forward returns.*"
        )

        top_signals = efficacy.get("top_signals", [])
        if top_signals:
            eff_chart_data = []
            for s in top_signals:
                label = s["signal"].replace("_", " ").title()
                eff_chart_data.append({"Signal": label, "Cohort": "High Signal", "Avg Return": s["high_avg_return"]})
                eff_chart_data.append({"Signal": label, "Cohort": "Low Signal", "Avg Return": s["low_avg_return"]})

            eff_chart_df = pd.DataFrame(eff_chart_data)
            fig = px.bar(
                eff_chart_df,
                x="Signal",
                y="Avg Return",
                color="Cohort",
                barmode="group",
                color_discrete_map={"High Signal": "#388e3c", "Low Signal": "#d32f2f"},
                title=f"Average {best_horizon} Return: High vs Low Signal Cohorts",
            )
            fig.update_layout(
                yaxis_tickformat=".1%",
                yaxis_title=f"Avg {best_horizon} Forward Return",
                height=350,
                margin=dict(l=10, r=10, t=40, b=40),
            )
            fig.add_hline(y=0, line_dash="dash", line_color="gray", line_width=0.5)
            st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("Not enough return data to evaluate signal efficacy.")

except Exception as e:
    st.warning(f"Signal efficacy analysis unavailable: {e}")

st.divider()

# =============================================================================
# STATISTICAL TESTS: Do signals predict returns?
# =============================================================================

st.header("Statistical Tests: Do Signals Predict Returns?")
st.markdown("""
*Beyond simple correlations — t-statistics, Information Coefficients (IC), and OLS regression
test whether the extracted signals have genuine predictive power over forward returns.*
""")

try:
    verdict = compute_predictive_verdict(df_filtered)

    if "verdict" in verdict:
        # --- Verdict banner ---
        verdict_colors = {
            "Strong evidence": "🟢",
            "Moderate evidence": "🟡",
            "Weak evidence": "🟠",
            "No significant evidence": "🔴",
        }
        emoji = verdict_colors.get(verdict["verdict"], "⚪")

        verd_col1, verd_col2 = st.columns(2)
        with verd_col1:
            st.metric("Verdict", f"{emoji} {verdict['verdict']}")
            st.caption(verdict["verdict_detail"])
        with verd_col2:
            st.metric("Avg |IC|", f"{verdict['avg_abs_ic']:.3f}")
            st.metric(
                "Significant Tests",
                f"{verdict['significant_at_10pct']}/{verdict['total_tests']} (p<0.10)",
            )

        st.divider()

        # --- Tab layout for the three test types ---
        stat_tab1, stat_tab2, stat_tab3 = st.tabs([
            "Information Coefficients",
            "Correlation T-Tests",
            "OLS Regression",
        ])

        # TAB 1: Information Coefficients
        with stat_tab1:
            st.markdown("""
            **Rank IC** (Spearman) measures monotonic association between signal ranks and
            return ranks. More robust than Pearson for small samples. |IC| > 0.05 is
            noteworthy in quant finance; > 0.10 is strong.
            """)

            ic_summary = compute_ic_summary(df_filtered)
            if not ic_summary.empty:
                # Heatmap
                ic_heatmap = ic_summary[["30d", "60d", "90d"]]

                fig = px.imshow(
                    ic_heatmap,
                    color_continuous_scale=["#d32f2f", "#ffffff", "#388e3c"],
                    color_continuous_midpoint=0,
                    zmin=-1,
                    zmax=1,
                    title="Rank IC: Signal × Return Horizon",
                    aspect="auto",
                    text_auto=".2f",
                )
                fig.update_layout(
                    height=max(250, len(ic_heatmap) * 28),
                    margin=dict(l=10, r=10, t=40, b=10),
                )
                st.plotly_chart(fig, use_container_width=True)

                # Top signals table
                st.markdown("**Top Signals by Average |IC|**")
                top_ic = ic_summary.head(8).copy()
                top_ic.index = top_ic.index.map(lambda x: x.replace("_", " ").title())
                st.dataframe(
                    top_ic.style.background_gradient(
                        cmap="RdYlGn", subset=["30d", "60d", "90d"], vmin=-1, vmax=1,
                    ).format("{:.3f}"),
                    use_container_width=True,
                )
            else:
                st.info("Not enough data to compute Information Coefficients.")

        # TAB 2: Correlation T-Tests
        with stat_tab2:
            st.markdown("""
            **Pearson r** with t-statistic and p-value. Highlighted rows are significant
            at the 10% level. With small samples, even meaningful effects may not reach
            significance — interpret alongside IC and regression results.
            """)

            tstat_df = compute_correlation_tstats(df_filtered)
            if not tstat_df.empty:
                display_df = tstat_df.copy()
                display_df["signal"] = display_df["signal"].str.replace("_", " ").str.title()

                # Highlight significant rows
                def highlight_significant(row):
                    if row["significant"]:
                        return ["background-color: rgba(56, 142, 60, 0.15)"] * len(row)
                    return [""] * len(row)

                st.dataframe(
                    display_df[["signal", "horizon", "pearson_r", "t_stat", "p_value", "n", "significant"]]
                    .style.apply(highlight_significant, axis=1)
                    .format({
                        "pearson_r": "{:.3f}",
                        "t_stat": "{:.2f}",
                        "p_value": "{:.4f}",
                    }),
                    use_container_width=True,
                    height=min(400, 35 * len(display_df) + 40),
                )

                sig_count = display_df["significant"].sum()
                st.caption(
                    f"{sig_count} of {len(display_df)} signal-horizon pairs significant at p < 0.10. "
                    f"N = {display_df['n'].iloc[0]} letters."
                )
            else:
                st.info("Not enough data for t-tests.")

        # TAB 3: OLS Regression
        with stat_tab3:
            st.markdown("""
            **OLS regression** of forward returns on the top signals (selected by |IC| to
            avoid overfitting). Shows whether signals jointly explain return variation.
            With small N, interpret R² cautiously — adjusted R² penalizes for degrees of freedom.
            """)

            regressions = run_all_regressions(df_filtered, max_features=5)
            if regressions:
                for reg in regressions:
                    st.markdown(f"##### {reg['horizon']} Forward Return")

                    reg_met1, reg_met2, reg_met3 = st.columns(3)
                    with reg_met1:
                        st.metric("R²", f"{reg['r_squared']:.3f}")
                    with reg_met2:
                        st.metric("Adj R²", f"{reg['adj_r_squared']:.3f}")
                    with reg_met3:
                        f_sig = "Yes" if reg["f_pvalue"] < 0.10 else "No"
                        st.metric("F-test significant?", f_sig)
                        st.caption(f"F = {reg['f_stat']:.2f}, p = {reg['f_pvalue']:.4f}")

                    # Coefficients table
                    coef_df = pd.DataFrame(reg["coefficients"])
                    coef_df["name"] = coef_df["name"].str.replace("_", " ").str.title()

                    def highlight_coef(row):
                        if row["p_value"] < 0.10:
                            return ["background-color: rgba(56, 142, 60, 0.15)"] * len(row)
                        return [""] * len(row)

                    st.dataframe(
                        coef_df.style.apply(highlight_coef, axis=1).format({
                            "coef": "{:.6f}",
                            "std_err": "{:.6f}",
                            "t_stat": "{:.2f}",
                            "p_value": "{:.4f}",
                        }),
                        use_container_width=True,
                    )
                    st.caption(f"N = {reg['n']}, K = {reg['k']} features")
                    st.divider()
            else:
                st.info("Not enough data to run regressions (need at least K+2 observations).")

    else:
        st.info(verdict.get("verdict_text", "Insufficient data for statistical testing."))

except Exception as e:
    st.warning(f"Statistical tests unavailable: {e}")

st.caption("""
**A note on interpretation:** Signals tend to show stronger predictive power at longer
horizons (e.g. 90-day) than shorter ones. This is expected — Buffett's letters convey
fundamental, structural views on capital allocation, market conditions, and business
quality that take time to manifest in price. A 30-day window captures mostly noise,
while 90 days allows these slow-moving signals to play out. That said, with a small
sample of letters, even significant results should be read as directionally encouraging
rather than definitive.
""")

st.divider()

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
        # Scores grid (2x2 for mobile)
        detail_col1, detail_col2 = st.columns(2)
        detail_col3, detail_col4 = st.columns(2)

        with detail_col1:
            st.markdown("**Confidence**")
            st.metric("Overall", f"{analysis.confidence.overall_confidence:.2f}")
            st.metric("Operating", f"{analysis.confidence.operating_business_confidence:.2f}")
            st.metric("Portfolio", f"{analysis.confidence.investment_portfolio_confidence:.2f}")

        with detail_col2:
            st.markdown("**Uncertainty**")
            st.metric("Overall", f"{analysis.uncertainty.overall_uncertainty:.2f}")
            st.metric("Macro", f"{analysis.uncertainty.macro_uncertainty:.2f}")
            st.metric("Market", f"{analysis.uncertainty.market_uncertainty:.2f}")

        with detail_col3:
            st.markdown("**Capital**")
            st.write(f"Posture: `{analysis.capital_allocation.posture.value}`")
            st.write(f"Cash Intent: `{analysis.capital_allocation.cash_intent.value}`")
            st.metric("Buyback", f"{analysis.capital_allocation.buyback_enthusiasm:.2f}")

        with detail_col4:
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
# EXTRACTION RELIABILITY
# =============================================================================

st.divider()
st.header("Extraction Reliability")
st.markdown("""
*How stable are these signals? If Claude extracts the same letter twice, does it give the
same answer? This section reports on multi-run consistency testing.*
""")

try:
    reliability = compute_reliability_summary()

    if reliability.get("available"):
        rel_col1, rel_col2 = st.columns(2)

        with rel_col1:
            grade_colors = {"A": "🟢", "B": "🟢", "C": "🟡", "D": "🔴"}
            grade_emoji = grade_colors.get(reliability["grade"], "⚪")
            st.metric(
                "Consistency Grade",
                f"{grade_emoji} {reliability['grade']} — {reliability['grade_label']}",
            )
            st.caption(reliability["grade_detail"])

        with rel_col2:
            st.metric("Avg Signal Variance", f"±{reliability['avg_cv']:.1%}")
            st.metric(
                "Test Coverage",
                f"{len(reliability['years_tested'])} letters, "
                f"{reliability['runs_per_year']} runs each",
            )

        # Per-signal variance table
        per_signal = reliability.get("per_signal")
        if per_signal is not None and not per_signal.empty:
            with st.expander("Signal-Level Variance Details"):
                display_rel = per_signal.copy()
                display_rel["signal"] = display_rel["signal"].str.replace("_", " ").str.title()
                display_rel = display_rel.rename(columns={
                    "signal": "Signal",
                    "avg_cv": "Avg CV",
                    "avg_std": "Avg Std Dev",
                    "avg_range": "Avg Range",
                })

                st.dataframe(
                    display_rel.style.background_gradient(
                        cmap="RdYlGn_r", subset=["Avg CV"], vmin=0, vmax=0.3,
                    ).format({"Avg CV": "{:.3f}", "Avg Std Dev": "{:.4f}", "Avg Range": "{:.4f}"}),
                    use_container_width=True,
                )

                st.markdown(
                    f"**Most stable:** {', '.join(s.replace('_', ' ').title() for s in reliability['most_stable'][:3])} | "
                    f"**Least stable:** {', '.join(s.replace('_', ' ').title() for s in reliability['least_stable'][:3])}"
                )

    else:
        st.info(
            "No reliability tests have been run yet. Run "
            "`python scripts/test_reliability.py` to extract the same letter "
            "multiple times and measure signal consistency."
        )

        st.markdown("""
        **Why this matters:** LLM-based extraction introduces a source of variance that
        traditional NLP pipelines don't have. Measuring this is a sign of methodological rigor.

        **What the test does:**
        1. Takes a single shareholder letter
        2. Runs Claude extraction 5+ times with identical prompts
        3. Measures the standard deviation of each numeric signal across runs
        4. Reports a coefficient of variation (CV) — lower is better

        **Expected results:** Structural signals (confidence, uncertainty) tend to be very
        stable (CV < 5%), while nuanced judgments (opportunity richness, speculation warning)
        may show more variation.
        """)

except Exception as e:
    st.warning(f"Reliability analysis unavailable: {e}")

# =============================================================================
# ABOUT THIS PROJECT
# =============================================================================

st.divider()
st.header("About This Project")

st.markdown("""
**This entire project was built by dictating to [Claude Code](https://docs.anthropic.com/en/docs/claude-code)** — Anthropic's
CLI agent for software engineering. No keystrokes were typed to write the code itself;
every line was generated through natural language conversation.
""")

about_col1, about_col2 = st.columns(2)
about_col3, _ = st.columns(2)

with about_col1:
    st.markdown("#### Build Metrics")
    st.metric("Total Build Time", "~4 hours")
    st.metric("Python Lines of Code", "4,830")
    st.metric("Total Project Lines", "~6,200")

with about_col2:
    st.markdown("#### Git Stats")
    st.metric("Commits", "26")
    st.metric("Total Insertions", "6,955")
    st.metric("Source Files", "18 Python + configs")

with about_col3:
    st.markdown("#### Productivity")
    st.metric("Est. Manual Coding Time", "30+ hours")
    st.metric("Speedup vs Manual", "8–10x")
    st.metric("Deployed To", "Streamlit Cloud")

st.markdown("""
#### How It Was Built

| Step | What Happened |
|------|---------------|
| **0:00 – 0:30** | Schema design — dictated the 25+ signal Pydantic models, prompt engineering |
| **0:30 – 1:00** | Extraction pipeline — Claude API integration, letter parsing, dataset builder |
| **1:00 – 1:30** | Fair value model — signal-based valuation, market sentiment from news |
| **1:30 – 2:00** | Dashboard — Streamlit app with Plotly charts, tabs, correlation analysis |
| **2:00 – 2:30** | Polish — analyst comparison, methodology section, deployment to Streamlit Cloud |
| **2:30 – 3:00** | Graham & Buffett valuation models, three-model comparison, methodology docs |
| **3:00 – 3:30** | Mobile-responsive layout, statistical testing (t-stats, IC, OLS regression) |
| **3:30 – 4:00** | Signal efficacy proof (conditional returns, strategy metrics), extraction reliability testing |

#### Tech Stack
`Claude API` · `Streamlit` · `Plotly` · `Pydantic` · `yfinance` · `scipy` · `statsmodels` · `Python 3.9+`

---
*[View source on GitHub](https://github.com/CraziedAres/Project-Berkshire)*
""")
