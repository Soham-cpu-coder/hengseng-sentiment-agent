"""
CrowdWisdom Trading - Hang Seng & Exchange Sentiment Price Prediction Terminal
Institutional Quantitative Sentiment Analytics & Out-of-Sample Price Forecasting Dashboard.
"""

import os
import sys

# Ensure repository root is in search path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

try:
    import streamlit as st
    import altair as alt
except ImportError:
    print("[!] Required dependencies not found. Please run: pip install streamlit altair pandas numpy")
    sys.exit(0)

import config
from scrapers.telegram_scraper import TelegramScraper
from scrapers.lihkg_scraper import LihkgScraper
from scrapers.futu_moomoo_scraper import FutuMoomooScraper
from scrapers.naver_scraper import NaverFinanceScraper
from scrapers.base_scraper import SocialPost
from news.exa_news import ExaNewsIngestor, NewsArticle
from market_data.price_feed import PriceFeedManager, PriceBar
from models.feature_engineering import FeaturePipeline, TickerSummary, DailyFeatureRecord
from models.predictor import SentimentPricePredictor, PredictionOutput

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION & INSTITUTIONAL THEME
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CrowdWisdom | Quantitative Sentiment Terminal",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Institutional CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    code, .stCode, [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Terminal background */
    .stApp {
        background-color: #0b0e14;
        color: #e2e8f0;
    }

    /* Top Ribbon Ticker Box */
    .ribbon-container {
        display: flex;
        gap: 12px;
        overflow-x: auto;
        padding-bottom: 8px;
        margin-bottom: 20px;
    }
    .ribbon-card {
        background: #121721;
        border: 1px solid #1e2638;
        border-radius: 8px;
        padding: 10px 16px;
        min-width: 210px;
        flex: 1;
        transition: transform 0.15s ease, border-color 0.15s ease;
    }
    .ribbon-card:hover {
        border-color: #3b82f6;
        transform: translateY(-2px);
    }
    .ribbon-ticker {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 13px;
        color: #60a5fa;
    }
    .ribbon-price {
        font-size: 16px;
        font-weight: 700;
        margin: 3px 0;
    }
    .ribbon-meta {
        font-size: 11px;
        color: #94a3b8;
        display: flex;
        justify-content: space-between;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        font-family: 'JetBrains Mono', monospace;
    }
    .badge-bullish { background: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-bearish { background: rgba(239, 68, 68, 0.15); color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3); }
    .badge-neutral { background: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }

    .pos-strong-buy { background: #065f46; color: #6ee7b7; padding: 4px 10px; border-radius: 5px; font-weight: 700; }
    .pos-buy { background: #064e3b; color: #a7f3d0; padding: 4px 10px; border-radius: 5px; font-weight: 600; }
    .pos-hold { background: #374151; color: #d1d5db; padding: 4px 10px; border-radius: 5px; font-weight: 600; }
    .pos-sell { background: #7f1d1d; color: #fecaca; padding: 4px 10px; border-radius: 5px; font-weight: 600; }
    .pos-strong-sell { background: #991b1b; color: #fca5a5; padding: 4px 10px; border-radius: 5px; font-weight: 700; }

    /* Terminal Section Cards */
    .terminal-card {
        background: #111724;
        border: 1px solid #1e293b;
        border-radius: 10px;
        padding: 18px;
        margin-bottom: 18px;
    }
    .terminal-header {
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        color: #94a3b8;
        letter-spacing: 0.05em;
        margin-bottom: 12px;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 6px;
    }

    /* Metric Grid */
    .metric-box {
        background: #0f1523;
        border: 1px solid #1a2336;
        border-radius: 6px;
        padding: 10px 12px;
    }
    .metric-box .label {
        font-size: 11px;
        color: #64748b;
        text-transform: uppercase;
    }
    .metric-box .value {
        font-size: 15px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        margin-top: 2px;
    }

    /* Support Resistance Table */
    .sr-pill {
        display: inline-block;
        font-family: 'JetBrains Mono', monospace;
        padding: 3px 6px;
        border-radius: 3px;
        font-size: 12px;
        margin-right: 4px;
    }
    .sr-res { background: rgba(239, 68, 68, 0.15); color: #f87171; border: 1px solid #7f1d1d; }
    .sr-piv { background: rgba(59, 130, 246, 0.15); color: #93c5fd; border: 1px solid #1e3a8a; }
    .sr-sup { background: rgba(16, 185, 129, 0.15); color: #34d399; border: 1px solid #064e3b; }
    .sr-fib { background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid #581c87; }

    /* Feed Items */
    .feed-card {
        background: #0d121d;
        border-left: 3px solid #3b82f6;
        border-radius: 0 6px 6px 0;
        padding: 10px 14px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# PIPELINE EXECUTION & DATA CACHE
# -----------------------------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def load_market_and_sentiment_pipeline(exchange_choice: str, lookback_days: int):
    """
    Execute end-to-end multi-source ingestion, alignment, and prediction.
    Caches results in memory for sub-second UI interactions.
    """
    if exchange_choice == "HKEX":
        target_assets = config.HKEX_ASSETS
    elif exchange_choice == "KRX":
        target_assets = config.KRX_ASSETS
    else:
        target_assets = {**config.HKEX_ASSETS, **config.KRX_ASSETS}

    tg_scraper = TelegramScraper()
    lihkg_scraper = LihkgScraper()
    futu_scraper = FutuMoomooScraper()
    naver_scraper = NaverFinanceScraper()
    news_ingestor = ExaNewsIngestor()
    price_manager = PriceFeedManager()
    pipeline = FeaturePipeline()

    summaries: List[TickerSummary] = []
    predictions: Dict[str, PredictionOutput] = {}
    ticker_records: Dict[str, List[DailyFeatureRecord]] = {}
    ticker_posts: Dict[str, List[SocialPost]] = {}
    ticker_news: Dict[str, List[NewsArticle]] = {}
    ticker_bars: Dict[str, List[PriceBar]] = {}

    for ticker, asset_cfg in target_assets.items():
        # Scrape alternative social data (minimum 2 distinct sources per exchange)
        posts: List[SocialPost] = []
        if asset_cfg.exchange == "KRX":
            posts.extend(tg_scraper.scrape(ticker, days=lookback_days))
            posts.extend(naver_scraper.scrape(ticker, days=lookback_days))
        else:
            posts.extend(tg_scraper.scrape(ticker, days=lookback_days))
            posts.extend(lihkg_scraper.scrape(ticker, days=lookback_days))
            posts.extend(futu_scraper.scrape(ticker, days=lookback_days))

        news = news_ingestor.fetch_news(ticker, days=lookback_days)
        bars = price_manager.get_price_bars(ticker, days=config.PRICE_LOOKBACK_DAYS)

        summary, records = pipeline.process_ticker(ticker, posts, news, bars)

        predictor = SentimentPricePredictor(alpha=2.0)
        pred_out = predictor.train_and_forecast(ticker, records, summary.current_price)

        summary.predicted_next_price = pred_out.predicted_price
        summary.predicted_next_return_pct = pred_out.predicted_return_pct
        summary.prediction_interval_lower = pred_out.confidence_interval_lower
        summary.prediction_interval_upper = pred_out.confidence_interval_upper

        summaries.append(summary)
        predictions[ticker] = pred_out
        ticker_records[ticker] = records
        ticker_posts[ticker] = posts
        ticker_news[ticker] = news
        ticker_bars[ticker] = bars

    return summaries, predictions, ticker_records, ticker_posts, ticker_news, ticker_bars


# -----------------------------------------------------------------------------
# SIDEBAR CONTROLS & API STATUS
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **CrowdWisdom Terminal**")
    st.caption("Hang Seng & Asian Equities Sentiment Prediction")
    st.markdown("---")

    exchange_select = st.selectbox(
        "Exchange Universe",
        ["HKEX", "KRX", "ALL"],
        index=0,
        help="Select target equities exchange: Hong Kong Stock Exchange (HKEX) or Korea Exchange (KRX).",
    )

    lookback_window = st.slider(
        "Social Ingestion Window (Days)",
        min_value=5,
        max_value=15,
        value=config.LOOKBACK_DAYS,
        step=1,
        help="Assessment specification: Ingest social posts and Exa news for the last 10 days.",
    )

    st.markdown("---")
    st.markdown("#### **Data Feed Telemetry**")
    
    # Status badges
    apify_status = "🟢 Live (Apify Token)" if config.APIFY_API_KEY else "🟡 Calibrated Synthetic"
    exa_status = "🟢 Live (Exa Neural)" if config.EXA_API_KEY else "🟡 Calibrated Synthetic"
    
    st.markdown(f"**Apify Actors:** `{apify_status}`")
    st.markdown(f"**Exa.com News:** `{exa_status}`")
    st.markdown(f"**Market Feeds:** `🟢 Direct OHLCV / Yahoo v8`")
    st.markdown(f"**Time Alignment:** `UTC ISO-8601 (Strict)`")

    st.markdown("---")
    st.markdown("#### **Assessment Requirements**")
    st.markdown("""
    - **≥2 Social Sources:** LIHKG, Telegram, Futu, Naver
    - **Exa.com News:** Neural Semantic Ingestion
    - **S&R Levels:** Floor Pivots & Fibonacci
    - **n+1 Forecast:** Ridge with 95% Confidence Bounds
    - **Zero-Leakage OOS:** Expanding Walk-Forward Split
    """)

    st.markdown("---")
    utc_now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    st.caption(f"Engine Timestamp:\n`{utc_now}`")


# -----------------------------------------------------------------------------
# PIPELINE EXECUTION
# -----------------------------------------------------------------------------
with st.spinner("Executing quantitative ingestion pipeline & walk-forward model..."):
    summaries, predictions, ticker_records, ticker_posts, ticker_news, ticker_bars = load_market_and_sentiment_pipeline(
        exchange_select, lookback_window
    )

if config.LIVE_MODE:
    st.markdown('<div style="background: rgba(16, 185, 129, 0.15); border: 1px solid #10b981; color: #34d399; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-weight: bold;">🟢 LIVE MARKET DATA — Operating with real-time price feeds, Apify scrapers, and Exa news</div>', unsafe_allow_html=True)
else:
    st.markdown('<div style="background: rgba(245, 158, 11, 0.15); border: 1px solid #f59e0b; color: #fbbf24; padding: 10px 16px; border-radius: 6px; margin-bottom: 16px; font-weight: bold;">⚠️ SYNTHETIC DEMO DATA — Running in simulated mode for offline verification. Use live mode with API keys for real data.</div>', unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# TOP TICKER RIBBON (REAL-TIME QUOTE STRIP)
# -----------------------------------------------------------------------------
ribbon_html = ['<div class="ribbon-container">']
for s in summaries:
    pred = predictions.get(s.ticker)
    ret_pct = s.price_change_1d_pct
    ret_color = "#10b981" if ret_pct >= 0 else "#ef4444"
    dir_class = f"badge-{s.sentiment_direction.lower()}"
    pred_dir = pred.predicted_direction if pred else "NEUTRAL"
    pred_color = "#10b981" if pred_dir == "BULLISH" else ("#ef4444" if pred_dir == "BEARISH" else "#f59e0b")

    ribbon_html.append(f"""
    <div class="ribbon-card">
        <div class="ribbon-ticker">{s.ticker} <span style="font-size: 11px; color: #64748b;">({s.exchange})</span></div>
        <div class="ribbon-price">{s.currency} {s.current_price:.2f} <span style="color: {ret_color}; font-size: 12px;">{ret_pct:+.2f}%</span></div>
        <div class="ribbon-meta">
            <span class="badge {dir_class}">{s.sentiment_direction}</span>
            <span>n+1: <strong style="color: {pred_color}">{pred.predicted_return_pct if pred else 0.0:+.2f}%</strong></span>
        </div>
    </div>
    """)
ribbon_html.append('</div>')
st.markdown("".join(ribbon_html), unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# MAIN DASHBOARD TABS
# -----------------------------------------------------------------------------
tab_overview, tab_deepdive, tab_social, tab_news, tab_model = st.tabs([
    "📊 Institutional Leaderboard",
    "🔬 Deep-Dive Ticker Terminal",
    "💬 Social Intelligence Stream",
    "📰 Exa.com News Ingestion",
    "📐 Factor Model & OOS Validation",
])


# =============================================================================
# TAB 1: OVERVIEW / LEADERBOARD (Assessment Requirement 3)
# =============================================================================
with tab_overview:
    col_head1, col_head2 = st.columns([3, 1])
    with col_head1:
        st.subheader("Exchange Sentiment & Horizon n+1 Prediction Leaderboard")
        st.caption("Rank assets dynamically by social popularity (volume), sentiment score & direction, or forecast return.")
    with col_head2:
        sort_by = st.selectbox(
            "Sort Leaderboard By:",
            [
                "Social Mention Volume (Buzz)",
                "Sentiment Score (Bullish → Bearish)",
                "Horizon n+1 Expected Return (%)",
                "Buzz Anomaly (Z-Score)",
            ],
            index=0,
        )

    # Sort logic
    if "Social Mention" in sort_by:
        sorted_summaries = sorted(summaries, key=lambda s: s.social_mention_count, reverse=True)
    elif "Sentiment Score" in sort_by:
        sorted_summaries = sorted(summaries, key=lambda s: s.aggregated_sentiment_score, reverse=True)
    elif "Expected Return" in sort_by:
        sorted_summaries = sorted(summaries, key=lambda s: (s.predicted_next_return_pct or 0.0), reverse=True)
    else:
        sorted_summaries = sorted(summaries, key=lambda s: s.buzz_zscore, reverse=True)

    # Construct DataFrame
    table_rows = []
    for s in sorted_summaries:
        p = predictions.get(s.ticker)
        vr = p.validation_report if p else None
        table_rows.append({
            "Ticker": s.ticker,
            "Asset Name": s.asset_name,
            "Exch": s.exchange,
            "Price": f"{s.currency} {s.current_price:.2f}",
            "1D %": f"{s.price_change_1d_pct:+.2f}%",
            "Mentions": s.social_mention_count,
            "Buzz Z": f"{s.buzz_zscore:+.2f}",
            "Sentiment": f"{s.aggregated_sentiment_score:+.3f}",
            "Direction": s.sentiment_direction,
            "Velocity": f"{s.sentiment_velocity:+.3f}",
            "Divergence": s.sentiment_price_divergence,
            "n+1 Target": f"{s.predicted_next_price:.2f} ({s.predicted_next_return_pct:+.2f}%)",
            "95% CI": f"[{s.prediction_interval_lower:.2f}, {s.prediction_interval_upper:.2f}]",
            "OOS Hit %": f"{vr.mean_directional_accuracy * 100:.1f}%" if vr else "N/A",
            "OOS IC": f"{vr.mean_information_coefficient:+.3f}" if vr else "N/A",
            "Position": s.position.status,
        })

    df_table = pd.DataFrame(table_rows)
    st.dataframe(df_table, use_container_width=True, height=220)

    # Visual Comparative Analytics
    st.markdown("#### **Cross-Asset Comparative Diagnostics**")
    c_col1, c_col2 = st.columns(2)

    with c_col1:
        # Chart 1: Social Volume vs Sentiment Score Scatter/Bar
        chart_data = pd.DataFrame([
            {
                "Ticker": s.ticker,
                "Mentions": s.social_mention_count,
                "SentimentScore": s.aggregated_sentiment_score,
                "Direction": s.sentiment_direction,
                "BuzzZ": s.buzz_zscore,
            }
            for s in summaries
        ])

        bars_chart = alt.Chart(chart_data).mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4).encode(
            x=alt.X("Ticker:N", title="Asset Ticker", sort="-y"),
            y=alt.Y("Mentions:Q", title="Total 10-Day Social Mentions"),
            color=alt.Color("Direction:N", scale=alt.Scale(
                domain=["BULLISH", "NEUTRAL", "BEARISH"],
                range=["#10b981", "#f59e0b", "#ef4444"]
            ), title="Sentiment"),
            tooltip=["Ticker", "Mentions", "SentimentScore", "BuzzZ", "Direction"]
        ).properties(
            title="Social Mention Volume & Sentiment Breakdown (Last 10 Days)",
            height=280,
        ).configure_view(strokeWidth=0).configure_title(fontSize=13, color="#94a3b8")

        st.altair_chart(bars_chart, use_container_width=True)

    with c_col2:
        # Chart 2: Predicted Horizon n+1 Return vs Realized 1-Day Return
        ret_data = pd.DataFrame([
            {
                "Ticker": s.ticker,
                "Realized_1D": s.price_change_1d_pct,
                "Forecast_n1": s.predicted_next_return_pct or 0.0,
            }
            for s in summaries
        ])
        ret_melted = pd.melt(ret_data, id_vars=["Ticker"], var_name="Type", value_name="Return_Pct")

        pred_chart = alt.Chart(ret_melted).mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3).encode(
            x=alt.X("Ticker:N", title="Asset"),
            y=alt.Y("Return_Pct:Q", title="Return (%)"),
            color=alt.Color("Type:N", scale=alt.Scale(
                domain=["Realized_1D", "Forecast_n1"],
                range=["#64748b", "#3b82f6"]
            ), title="Horizon"),
            xOffset="Type:N",
            tooltip=["Ticker", "Type", "Return_Pct"]
        ).properties(
            title="Realized 1-Day Return vs Horizon n+1 Forecast Return (%)",
            height=280,
        ).configure_view(strokeWidth=0).configure_title(fontSize=13, color="#94a3b8")

        st.altair_chart(pred_chart, use_container_width=True)


# =============================================================================
# TAB 2: DEEP-DIVE TICKER TERMINAL (Assessment Requirement 2 & 4)
# =============================================================================
with tab_deepdive:
    selected_ticker = st.selectbox(
        "Select Target Equity for Deep-Dive Quantitative Inspection:",
        [s.ticker for s in summaries],
        index=0,
    )
    active_summary = next(s for s in summaries if s.ticker == selected_ticker)
    active_pred = predictions[selected_ticker]
    active_bars = ticker_bars[selected_ticker]
    active_records = ticker_records[selected_ticker]
    sr = active_summary.support_resistance
    pos = active_summary.position
    val_rep = active_pred.validation_report

    # Asset Header Banner
    col_t1, col_t2, col_t3 = st.columns([2, 1, 1])
    with col_t1:
        st.markdown(f"## **{active_summary.asset_name}** (`{active_summary.ticker}`)")
        st.caption(f"Exchange: **{active_summary.exchange}** | Base Currency: **{active_summary.currency}** | Price Feed: UTC Daily Aligned")
    with col_t2:
        st.metric(
            label="Current Market Price",
            value=f"{active_summary.currency} {active_summary.current_price:.2f}",
            delta=f"{active_summary.price_change_1d_pct:+.2f}% (1D)",
        )
    with col_t3:
        st.metric(
            label="Horizon n+1 Price Forecast",
            value=f"{active_summary.currency} {active_pred.predicted_price:.2f}",
            delta=f"{active_pred.predicted_return_pct:+.2f}%",
        )

    st.markdown("---")

    # Layout: Chart + S&R & Position Panel
    col_chart, col_signal = st.columns([5, 3])

    with col_chart:
        st.markdown("#### **Price Action, Support/Resistance Corridors & n+1 Forecast Cone**")
        
        # Prepare historical price DataFrame
        df_bars = pd.DataFrame([
            {
                "Date": b.timestamp_utc[:10],
                "Close": b.close,
                "High": b.high,
                "Low": b.low,
                "Open": b.open,
                "Volume": b.volume,
            }
            for b in active_bars
        ])

        # Base price line
        base_line = alt.Chart(df_bars).mark_line(color="#60a5fa", strokeWidth=2.5).encode(
            x=alt.X("Date:T", title="Date (UTC)"),
            y=alt.Y("Close:Q", scale=alt.Scale(zero=False), title=f"Price ({active_summary.currency})"),
            tooltip=["Date", "Close", "High", "Low", "Volume"]
        )

        points = alt.Chart(df_bars).mark_circle(size=35, color="#3b82f6").encode(
            x="Date:T",
            y="Close:Q",
            tooltip=["Date", "Close"]
        )

        # Horizontal Support & Resistance reference rules
        r2_rule = alt.Chart(pd.DataFrame({"y": [sr.resistance_2]})).mark_rule(color="#ef4444", strokeDash=[4, 4]).encode(y="y:Q")
        r1_rule = alt.Chart(pd.DataFrame({"y": [sr.resistance_1]})).mark_rule(color="#f87171", strokeDash=[2, 2]).encode(y="y:Q")
        piv_rule = alt.Chart(pd.DataFrame({"y": [sr.pivot]})).mark_rule(color="#3b82f6", strokeDash=[6, 6]).encode(y="y:Q")
        s1_rule = alt.Chart(pd.DataFrame({"y": [sr.support_1]})).mark_rule(color="#34d399", strokeDash=[2, 2]).encode(y="y:Q")
        s2_rule = alt.Chart(pd.DataFrame({"y": [sr.support_2]})).mark_rule(color="#10b981", strokeDash=[4, 4]).encode(y="y:Q")

        # Projected n+1 forecast point with confidence interval
        last_date = pd.to_datetime(df_bars["Date"].iloc[-1])
        next_date = last_date + pd.Timedelta(days=1)
        df_forecast = pd.DataFrame([{
            "Date": next_date.strftime("%Y-%m-%d"),
            "PredPrice": active_pred.predicted_price,
            "CILower": active_pred.confidence_interval_lower,
            "CIUpper": active_pred.confidence_interval_upper,
        }])

        forecast_point = alt.Chart(df_forecast).mark_point(
            shape="diamond", size=150, color="#f59e0b", fill="#f59e0b"
        ).encode(
            x="Date:T",
            y="PredPrice:Q",
            tooltip=[
                alt.Tooltip("Date:T", title="Horizon"),
                alt.Tooltip("PredPrice:Q", title="Forecast Price"),
                alt.Tooltip("CILower:Q", title="95% CI Lower"),
                alt.Tooltip("CIUpper:Q", title="95% CI Upper"),
            ]
        )

        error_bars = alt.Chart(df_forecast).mark_errorbar(color="#f59e0b", thickness=2).encode(
            x="Date:T",
            y=alt.Y("CILower:Q", title=""),
            y2="CIUpper:Q"
        )

        composite_chart = (base_line + points + r2_rule + r1_rule + piv_rule + s1_rule + s2_rule + error_bars + forecast_point).properties(
            height=360
        ).configure_view(strokeWidth=0)

        st.altair_chart(composite_chart, use_container_width=True)

        # Legend explanation
        st.markdown("""
        <div style="font-size: 11.5px; color: #94a3b8; display: flex; gap: 14px; flex-wrap: wrap;">
            <span><span class="sr-pill sr-res">R2/R1</span> Resistance</span>
            <span><span class="sr-pill sr-piv">PIVOT</span> Floor Pivot</span>
            <span><span class="sr-pill sr-sup">S1/S2</span> Support</span>
            <span><span class="sr-pill sr-fib">FIB 0.618</span> Golden Ratio</span>
            <span><strong style="color: #f59e0b;">◆ n+1 Forecast</strong> (with 95% Confidence Bounds)</span>
        </div>
        """, unsafe_allow_html=True)

    with col_signal:
        st.markdown("#### **Trade Position & Technical Structure**")
        pos_badge = f"pos-{pos.status.lower().replace('_', '-')}"
        
        st.markdown(f"""
        <div class="terminal-card">
            <div class="terminal-header">Institutional Position Mandate</div>
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <span class="{pos_badge}">{pos.status}</span>
                <span style="font-family: 'JetBrains Mono'; font-size: 13px;">Conviction: <strong>{pos.conviction * 100:.0f}%</strong></span>
            </div>
            <div style="font-size: 12.5px; color: #cbd5e1; margin-bottom: 14px;">{pos.rationale}</div>
            
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 8px;">
                <div class="metric-box">
                    <div class="label">Suggested Entry</div>
                    <div class="value">{active_summary.currency} {pos.suggested_entry:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Stop Loss (Risk)</div>
                    <div class="value" style="color: #ef4444;">{active_summary.currency} {pos.stop_loss:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Take Profit 1</div>
                    <div class="value" style="color: #10b981;">{active_summary.currency} {pos.take_profit_1:.2f}</div>
                </div>
                <div class="metric-box">
                    <div class="label">Risk/Reward Ratio</div>
                    <div class="value" style="color: #60a5fa;">1 : {pos.risk_reward_ratio:.2f}</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Support & Resistance Quantitative Grid
        st.markdown(f"""
        <div class="terminal-card">
            <div class="terminal-header">Computed Key Support & Resistance Levels</div>
            <table style="width: 100%; font-size: 12.5px; font-family: 'JetBrains Mono', monospace;">
                <tr><td style="color: #f87171;">Resistance 2 (R2)</td><td style="text-align: right; font-weight: bold;">{sr.resistance_2:.2f}</td></tr>
                <tr><td style="color: #fca5a5;">Resistance 1 (R1)</td><td style="text-align: right; font-weight: bold;">{sr.resistance_1:.2f}</td></tr>
                <tr><td style="color: #93c5fd;">Classic Floor Pivot</td><td style="text-align: right; font-weight: bold;">{sr.pivot:.2f}</td></tr>
                <tr><td style="color: #a7f3d0;">Support 1 (S1)</td><td style="text-align: right; font-weight: bold;">{sr.support_1:.2f}</td></tr>
                <tr><td style="color: #34d399;">Support 2 (S2)</td><td style="text-align: right; font-weight: bold;">{sr.support_2:.2f}</td></tr>
                <tr><td style="color: #c084fc;">Fibonacci 0.618</td><td style="text-align: right; font-weight: bold;">{sr.fib_618:.2f}</td></tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Advanced Microstructure & Sentiment Features Row
    st.markdown("#### **Quantitative Sentiment Microstructure Features**")
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)

    with m_col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Engagement-Weighted Sentiment</div>
            <div class="value" style="color: {'#10b981' if active_summary.engagement_weighted_sentiment > 0 else '#ef4444'};">
                {active_summary.engagement_weighted_sentiment:+.3f}
            </div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Weighted by log(views + 2×reposts)</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Sentiment Velocity (ΔS / Δt)</div>
            <div class="value" style="color: {'#10b981' if active_summary.sentiment_velocity > 0 else '#ef4444'};">
                {active_summary.sentiment_velocity:+.3f} /day
            </div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">Institutional momentum rate of change</div>
        </div>
        """, unsafe_allow_html=True)

    with m_col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Social Buzz Anomaly (Z-Score)</div>
            <div class="value" style="color: {'#38bdf8' if active_summary.buzz_zscore > 1.5 else '#e2e8f0'};">
                {active_summary.buzz_zscore:+.2f} σ
            </div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
                {'🔥 Viral Retail Frenzy' if active_summary.buzz_zscore >= 2.0 else 'Normal baseline volume'}
            </div>
        </div>
        """, unsafe_allow_html=True)

    with m_col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="label">Sentiment-Price Divergence</div>
            <div class="value" style="color: {'#10b981' if 'BULLISH' in active_summary.sentiment_price_divergence else ('#ef4444' if 'BEARISH' in active_summary.sentiment_price_divergence else '#64748b')};">
                {active_summary.sentiment_price_divergence}
            </div>
            <div style="font-size: 11px; color: #64748b; margin-top: 4px;">
                {'Smart money dip accumulation' if 'BULLISH' in active_summary.sentiment_price_divergence else ('Retail exhaustion distribution' if 'BEARISH' in active_summary.sentiment_price_divergence else 'Trend and sentiment aligned')}
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 3: SOCIAL SENTIMENT STREAM (Assessment Requirement 1)
# =============================================================================
with tab_social:
    st.subheader(f"Alternative Social Stream: {selected_ticker}")
    st.caption("Extracted post text, UTC timestamps, user engagement metrics, and follower weights across Telegram, LIHKG, Futu, and Naver.")

    posts = ticker_posts[selected_ticker]
    
    col_f1, col_f2 = st.columns([1, 3])
    with col_f1:
        source_filter = st.multiselect(
            "Filter Platform:",
            options=list(set(p.source for p in posts)),
            default=list(set(p.source for p in posts)),
        )

    filtered_posts = [p for p in posts if p.source in source_filter]

    st.write(f"Displaying **{len(filtered_posts)}** social observations strictly aligned to UTC:")

    for p in filtered_posts[:15]:
        sent_color = "#10b981" if (p.sentiment_score or 0) > 0.15 else ("#ef4444" if (p.sentiment_score or 0) < -0.15 else "#f59e0b")
        st.markdown(f"""
        <div class="feed-card">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 4px;">
                <span>
                    <strong style="color: #60a5fa;">[{p.source.upper()}]</strong> 
                    <span style="font-family: 'JetBrains Mono'; font-size: 12px; color: #cbd5e1;">{p.author}</span>
                    <span style="font-size: 11px; color: #64748b;">(Reach: {p.author_followers:,} followers)</span>
                </span>
                <span style="font-family: 'JetBrains Mono'; font-size: 11px; color: #94a3b8;">
                    {p.timestamp_utc[:19]} UTC
                </span>
            </div>
            <div style="font-size: 13.5px; color: #f1f5f9; margin: 6px 0;">"{p.text}"</div>
            <div style="display: flex; justify-content: space-between; font-size: 11.5px; color: #94a3b8; font-family: 'JetBrains Mono';">
                <span>Likes: <strong>{p.likes}</strong> | Reposts/Replies: <strong>{p.reposts}</strong> | Views: <strong>{p.views:,}</strong></span>
                <span>Sentiment Score: <strong style="color: {sent_color};">{p.sentiment_score:+.2f} ({p.sentiment_label})</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 4: EXA.COM NEWS INGESTION (Assessment Requirement 1 & 2)
# =============================================================================
with tab_news:
    st.subheader(f"Exa.com Neural News Intelligence: {selected_ticker}")
    st.caption("Recent financial press releases, broker ratings, and macro developments ingested via Exa neural search to complement social sentiment.")

    news_list = ticker_news[selected_ticker]
    st.write(f"Retrieved **{len(news_list)}** targeted news articles over the last 10 days:")

    for n in news_list:
        n_sent_color = "#10b981" if (n.sentiment_score or 0) > 0.15 else ("#ef4444" if (n.sentiment_score or 0) < -0.15 else "#f59e0b")
        st.markdown(f"""
        <div class="terminal-card">
            <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 6px;">
                <div>
                    <span style="background: #1e3a8a; color: #93c5fd; padding: 2px 6px; border-radius: 4px; font-size: 11px; font-weight: 600;">{n.source_name}</span>
                    <span style="font-size: 14.5px; font-weight: 600; color: #f8fafc; margin-left: 6px;">{n.title}</span>
                </div>
                <span style="font-family: 'JetBrains Mono'; font-size: 11px; color: #64748b;">{n.published_utc[:19]} UTC</span>
            </div>
            <div style="font-size: 13px; color: #cbd5e1; margin: 8px 0; line-height: 1.5;">{n.summary}</div>
            <div style="display: flex; justify-content: space-between; font-size: 11.5px; font-family: 'JetBrains Mono'; border-top: 1px solid #1e293b; padding-top: 6px;">
                <span style="color: #94a3b8;">Neural Relevance: <strong>{n.relevance_score:.2f}</strong></span>
                <span>Sentiment Impact: <strong style="color: {n_sent_color};">{n.sentiment_score:+.2f} ({n.sentiment_label})</strong></span>
            </div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# TAB 5: FACTOR MODEL & WALK-FORWARD OOS VALIDATION (Assessment Requirement 4)
# =============================================================================
with tab_model:
    st.subheader(f"Quantitative Model Architecture & Zero-Leakage Validation: {selected_ticker}")
    st.caption("Strict expanding-window walk-forward cross validation ensuring zero future lookahead bias in price and return prediction.")

    if val_rep:
        # Scorecard metrics
        vm1, vm2, vm3, vm4 = st.columns(4)
        with vm1:
            st.metric("Directional Hit Ratio", f"{val_rep.mean_directional_accuracy * 100:.1f}%", help="Percentage of out-of-sample periods where predicted return direction matched realized price action.")
        with vm2:
            st.metric("Information Coefficient (IC)", f"{val_rep.mean_information_coefficient:+.3f}", help="Pearson correlation between out-of-sample forecast returns and realized next-bar returns.")
        with vm3:
            st.metric("Strategy Sharpe Ratio", f"{val_rep.strategy_sharpe_ratio:.2f}", help="Annualized Sharpe ratio of systematic long/short strategy driven by model signal.")
        with vm4:
            st.metric("Cumulative Strategy Return", f"{val_rep.strategy_cumulative_return_pct:+.2f}%", delta=f"{val_rep.strategy_cumulative_return_pct - val_rep.benchmark_cumulative_return_pct:+.2f}% vs B&H")

        st.markdown("---")

        # Walk-Forward Fold Details Table
        st.markdown("#### **Expanding Window Walk-Forward Fold Breakdown**")
        fold_rows = []
        for f in val_rep.fold_details:
            fold_rows.append({
                "Fold #": f.fold_idx,
                "Train Window": f"Bars [0 → {f.train_end}]",
                "Out-of-Sample Test Window": f"Bars [{f.test_start} → {f.test_end}]",
                "Directional Hit %": f"{f.directional_accuracy * 100:.1f}%",
                "Information Coeff (IC)": f"{f.information_coefficient:+.3f}",
                "RMSE": f"{f.rmse:.5f}",
                "MAE": f"{f.mae:.5f}",
            })
        st.dataframe(pd.DataFrame(fold_rows), use_container_width=True)

    # Feature Importance / Regression Coefficients
    st.markdown("#### **Model Factor Attribution (Standardized Coefficients β)**")
    importance = active_pred.feature_importance
    if importance:
        df_imp = pd.DataFrame([
            {"Feature": k, "Weight": v, "Impact": "Bullish" if v > 0 else "Bearish"}
            for k, v in importance.items()
        ]).sort_values(by="Weight", ascending=False)

        imp_chart = alt.Chart(df_imp).mark_bar().encode(
            x=alt.X("Weight:Q", title="Standardized Regression Coefficient β"),
            y=alt.Y("Feature:N", sort="-x", title="Feature Factor"),
            color=alt.Color("Impact:N", scale=alt.Scale(domain=["Bullish", "Bearish"], range=["#10b981", "#ef4444"])),
            tooltip=["Feature", "Weight", "Impact"]
        ).properties(height=300).configure_view(strokeWidth=0)

        st.altair_chart(imp_chart, use_container_width=True)
