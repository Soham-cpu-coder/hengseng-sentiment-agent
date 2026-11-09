"""
Institutional Visual Reporting and Analytics Engine.
Generates Bloomberg/TradingView terminal style visual dashboards:
1. Multi-Exchange Leaderboards (separated by HKEX and KRX) sortable by social mentions,
   sentiment score, direction, and horizon n+1 predictions.
2. In-depth ticker analytical cards with embedded SVG price action corridors, S&R pivots,
   95% forecast cones, sentiment overlay, and top Exa news items with links.
3. Out-of-Sample Walk-Forward Validation scorecards with Zero-Return and Price-Only baseline comparisons.
4. Recent Holdout n+1 Prediction vs Actual realized price check table with Error %.
5. Honest Limitations and Risk Disclosures.
6. Provenance Audit Block (posts per source/ticker, news counts, price date range, timestamp).
7. Zero external network dependencies (fully offline standalone HTML with inline SVGs).
"""

from datetime import datetime, timezone
import json
import math
import os
from typing import Any, Dict, List, Optional

from models.feature_engineering import TickerSummary
from models.predictor import PredictionOutput
import config


class VisualReportGenerator:
    """Generates visual HTML reports, Markdown summaries, and CLI terminal tables."""

    @classmethod
    def generate_cli_table(
        cls,
        summaries: List[TickerSummary],
        predictions: Dict[str, PredictionOutput],
        data_mode: str = "synthetic",
        provenance: Optional[Dict[str, Any]] = None,
        aggregate_val: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Render high-contrast ASCII terminal analytics tables with baseline comparisons."""
        lines = []
        is_live = (data_mode == "live")

        lines.append("\n" + "=" * 125)
        if not is_live:
            lines.append(" *** SYNTHETIC DEMO DATA *** (Simulation Mode — Pass --live to run against live APIs)")
        else:
            lines.append(" [LIVE MARKET DATA] Real-time verified market prices, Apify social scrapers & Exa news")
        lines.append(" CROWDWISDOM TRADING - HENG SENG & EXCHANGE SENTIMENT PRICE PREDICTION REPORT")
        lines.append(f" Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')} | Data Mode: {data_mode.upper()}")
        lines.append("=" * 125)

        # Provenance Block in CLI
        if provenance:
            lines.append("\n>>> DATA PROVENANCE AUDIT BLOCK")
            lines.append("-" * 125)
            lines.append(
                f"Coverage: {provenance.get('price_date_range', {}).get('display', '180 days')} | "
                f"Social: {provenance.get('total_social_posts', 0)} posts ({provenance.get('social_mode', 'synthetic')}) | "
                f"News: {provenance.get('total_news_articles', 0)} articles ({provenance.get('news_mode', 'synthetic')})"
            )
            breakdown_parts = []
            for t, info in provenance.get("by_ticker", {}).items():
                breakdown_parts.append(f"{t}: {info.get('social', 0)} posts, {info.get('news', 0)} news")
            lines.append("By Ticker: " + " | ".join(breakdown_parts[:5]))
            if len(breakdown_parts) > 5:
                lines.append("           " + " | ".join(breakdown_parts[5:]))

        # Separate by exchange
        hkex_summaries = [s for s in summaries if s.exchange == "HKEX"]
        krx_summaries = [s for s in summaries if s.exchange == "KRX"]

        for exch_name, exch_list in [("HKEX (HONG KONG EXCHANGE)", hkex_summaries), ("KRX (KOREA EXCHANGE)", krx_summaries)]:
            if not exch_list:
                continue
            lines.append(f"\n>>> EXCHANGE: {exch_name}")
            lines.append("-" * 125)
            lines.append(f"{'Ticker':<10} {'Name':<22} {'Price':<12} {'Mentions':<9} {'Buzz Z':<8} {'Sent':<8} {'Dir':<9} {'n+1 Pred':<13} {'95% CI':<19} {'Position':<10} {'R/R':<6} {'Reason'}")
            lines.append("-" * 125)

            # Sort by social mentions
            sorted_exch = sorted(exch_list, key=lambda s: s.social_mention_count, reverse=True)
            for s in sorted_exch:
                pred = predictions.get(s.ticker)
                p_str = f"{s.currency} {s.current_price:.2f}"
                pred_str = f"{pred.predicted_price:.2f} ({pred.predicted_return_pct:+.2f}%)" if pred else "N/A"
                ci_str = f"[{pred.confidence_interval_lower:.1f}, {pred.confidence_interval_upper:.1f}]" if pred else "N/A"
                rr_str = f"1:{s.position.risk_reward_ratio:.2f}"
                reason_str = (s.position.reason or s.position.rationale)[:32]
                low_flag = " *" if s.is_low_data else ""

                lines.append(
                    f"{s.ticker + low_flag:<10} {s.asset_name[:21]:<22} {p_str:<12} {s.social_mention_count:<9} "
                    f"{s.buzz_zscore:+5.2f}  {s.aggregated_sentiment_score:+6.3f} {s.sentiment_direction:<9} "
                    f"{pred_str:<13} {ci_str:<19} {s.position.status:<10} {rr_str:<6} {reason_str}"
                )

        # Aggregate Out-of-Sample Performance and Baseline Comparison
        lines.append("\n" + "=" * 125)
        lines.append(" OUT-OF-SAMPLE WALK-FORWARD VALIDATION & BASELINE COMPARISON (ZERO LOOKAHEAD)")
        lines.append("-" * 125)
        if aggregate_val:
            lines.append(
                f"Aggregate OOS: N={aggregate_val.get('oos_samples_count', 0)} | "
                f"Mean Hit={aggregate_val.get('mean_hit_ratio_pct', 0.0):.1f}% (vs Price-Only {aggregate_val.get('price_only_hit_ratio_pct', 0.0):.1f}%) | "
                f"Mean IC={aggregate_val.get('mean_information_coefficient', 0.0):+.3f} | "
                f"Mean RMSE={aggregate_val.get('mean_rmse', 0.0):.4f} (vs Price-Only {aggregate_val.get('price_only_rmse', 0.0):.4f}) | "
                f"Sharpe={aggregate_val.get('mean_strategy_sharpe', 0.0):.2f}"
            )
            lines.append(f"Verdict: {aggregate_val.get('sentiment_beat_price_only_verdict', '')}")
            lines.append("-" * 125)

        lines.append(f"{'Ticker':<10} {'OOS N':<7} {'Hit %':<8} {'IC':<8} {'RMSE':<9} {'Price-Only Hit%':<16} {'Price-Only RMSE':<16} {'Sent Better?':<14} {'Sharpe':<8}")
        lines.append("-" * 125)

        for s in summaries:
            pred = predictions.get(s.ticker)
            vr = pred.validation_report if pred else None
            if vr and not vr.insufficient_data:
                hit_str = f"{vr.mean_directional_accuracy * 100:.1f}%" if vr.mean_directional_accuracy is not None else "N/A"
                ic_str = f"{vr.mean_information_coefficient:+.3f}" if vr.mean_information_coefficient is not None else "N/A"
                rmse_str = f"{vr.mean_rmse:.4f}" if vr.mean_rmse is not None else "N/A"
                sharpe_str = f"{vr.strategy_sharpe_ratio:.2f}" if vr.strategy_sharpe_ratio is not None else "N/A"
                oos_n = str(vr.oos_samples_count)

                if vr.baselines:
                    po_hit = f"{vr.baselines.price_only_directional_accuracy * 100:.1f}%"
                    po_rmse = f"{vr.baselines.price_only_rmse:.4f}"
                    improves = "YES (Hit+RMSE)" if (vr.baselines.sentiment_improves_directional_accuracy and vr.baselines.sentiment_improves_rmse) else (
                        "YES (Hit)" if vr.baselines.sentiment_improves_directional_accuracy else ("YES (RMSE)" if vr.baselines.sentiment_improves_rmse else "NO")
                    )
                else:
                    po_hit, po_rmse, improves = "N/A", "N/A", "N/A"

                lines.append(
                    f"{s.ticker:<10} {oos_n:<7} {hit_str:<8} {ic_str:<8} {rmse_str:<9} "
                    f"{po_hit:<16} {po_rmse:<16} {improves:<14} {sharpe_str:<8}"
                )
            else:
                lines.append(f"{s.ticker:<10} {'INSUFFICIENT OUT-OF-SAMPLE DATA (Requires >= 15 historical bars)':<100}")

        # Prediction vs Actual Check (Holdout bars with Error %)
        sample_pred_actual = None
        for s in summaries:
            pred = predictions.get(s.ticker)
            if pred and pred.validation_report and pred.validation_report.recent_k_predictions:
                sample_pred_actual = (s.ticker, pred.validation_report.recent_k_predictions)
                break

        if sample_pred_actual:
            ticker_name, recents = sample_pred_actual
            lines.append("\n" + "=" * 125)
            lines.append(f" RECENT HOLDOUT PREDICTION VS REALIZED CLOSE CHECK (Sample: {ticker_name})")
            lines.append("-" * 125)
            lines.append(f"{'Bar Date':<12} {'Pred Close':<13} {'Real Close':<13} {'Error %':<10} {'Pred Return %':<16} {'Real Return %':<16} {'Sign Match':<10}")
            lines.append("-" * 125)
            for r in recents:
                lines.append(
                    f"{r.date_str:<12} {r.predicted_close:<13.2f} {r.realized_close:<13.2f} {r.error_pct:9.2f}% "
                    f"{r.predicted_return_pct:+15.2f}% {r.realized_return_pct:+15.2f}% "
                    f"{'CORRECT' if r.direction_correct else 'INCORRECT':<10}"
                )

        lines.append("\n" + "=" * 125)
        lines.append(" MODEL LIMITATIONS & RESEARCH DISCLOSURES:")
        lines.append(" 1. Short alternative data lookback: Social and news scrapers capture recent 10-day sentiment.")
        lines.append(" 2. Non-stationarity & noise: Retail sentiment is subject to sudden attention shifts and herd behavior.")
        lines.append(" 3. Out-of-sample sample size: Daily bars yield small sample sizes; statistical significance requires larger samples.")
        lines.append(" 4. Frictionless assumptions: Transaction costs, exchange fees, and execution slippage are not subtracted.")
        lines.append(" 5. Low data warning: Tickers marked with '*' have fewer than 10 social mentions in the lookback window.")
        lines.append("=" * 125 + "\n")

        return "\n".join(lines)

    @classmethod
    def generate_html_report(
        cls,
        summaries: List[TickerSummary],
        predictions: Dict[str, PredictionOutput],
        output_filepath: Optional[str] = None,
        data_mode: str = "synthetic",
        provenance: Optional[Dict[str, Any]] = None,
        aggregate_val: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Produce an interactive visual HTML report with exchange tabs, client-side sorting, offline SVG charts, and honest diagnostics."""
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        filepath = output_filepath or os.path.join(config.OUTPUT_DIR, "sentiment_report.html")
        is_live = (data_mode == "live")

        # Separate summaries by exchange
        hkex_summaries = [s for s in summaries if s.exchange == "HKEX"]
        krx_summaries = [s for s in summaries if s.exchange == "KRX"]

        def build_table_rows(items: List[TickerSummary]) -> str:
            rows = []
            for s in items:
                p = predictions.get(s.ticker)
                vr = p.validation_report if p else None
                d_class = s.sentiment_direction.lower()
                p_class = s.position.status.lower().replace("_", "-")
                pred_color = "#10b981" if (p and p.predicted_return_pct > 0) else "#ef4444"

                hit_str = f"{vr.mean_directional_accuracy * 100:.1f}%" if (vr and vr.mean_directional_accuracy is not None) else "N/A"
                ic_str = f"{vr.mean_information_coefficient:+.3f}" if (vr and vr.mean_information_coefficient is not None) else "N/A"
                low_badge = '<span class="badge-low-data">LOW DATA</span>' if s.is_low_data else ''
                reason_txt = s.position.reason or s.position.rationale

                row = f"""
                <tr data-ticker="{s.ticker}" data-mentions="{s.social_mention_count}" data-sentiment="{s.aggregated_sentiment_score}" data-direction="{s.sentiment_direction}" data-pred="{p.predicted_return_pct if p else 0}">
                    <td><strong class="font-mono text-blue">{s.ticker}</strong> {low_badge}</td>
                    <td>{s.asset_name}</td>
                    <td><span class="badge-mini">{s.exchange}</span></td>
                    <td class="font-mono">{s.currency} {s.current_price:.2f}</td>
                    <td class="font-mono" style="color: {'#10b981' if s.price_change_1d_pct >= 0 else '#ef4444'}">{s.price_change_1d_pct:+.2f}%</td>
                    <td class="font-mono font-bold">{s.social_mention_count}</td>
                    <td class="font-mono">{s.buzz_zscore:+.2f}</td>
                    <td class="font-mono font-bold" style="color: {'#10b981' if s.aggregated_sentiment_score > 0.15 else ('#ef4444' if s.aggregated_sentiment_score < -0.15 else '#f59e0b')}">{s.aggregated_sentiment_score:+.3f}</td>
                    <td><span class="badge-dir {d_class}">{s.sentiment_direction}</span></td>
                    <td class="font-mono" style="color: {pred_color}; font-weight: bold;">{p.predicted_price if p else s.current_price:.2f} ({p.predicted_return_pct if p else 0.0:+.2f}%)</td>
                    <td class="font-mono text-muted">[{p.confidence_interval_lower if p else 0:.2f}, {p.confidence_interval_upper if p else 0:.2f}]</td>
                    <td class="font-mono">{hit_str}</td>
                    <td class="font-mono">{ic_str}</td>
                    <td><span class="badge-pos badge-{p_class}">{s.position.status}</span><div class="pos-reason font-mono">{reason_txt}</div></td>
                </tr>
                """
                rows.append(row)
            return "".join(rows)

        def build_cards(items: List[TickerSummary]) -> str:
            cards = []
            for s in items:
                pred = predictions.get(s.ticker)
                pred_price = pred.predicted_price if pred else s.current_price
                pred_pct = pred.predicted_return_pct if pred else 0.0
                ci_low = pred.confidence_interval_lower if pred else s.current_price * 0.98
                ci_high = pred.confidence_interval_upper if pred else s.current_price * 1.02

                val_rep = pred.validation_report if pred else None
                hit_ratio = f"{val_rep.mean_directional_accuracy * 100:.1f}%" if (val_rep and val_rep.mean_directional_accuracy is not None) else "N/A"
                sharpe = f"{val_rep.strategy_sharpe_ratio:.2f}" if (val_rep and val_rep.strategy_sharpe_ratio is not None) else "N/A"
                ic_val = f"{val_rep.mean_information_coefficient:+.3f}" if (val_rep and val_rep.mean_information_coefficient is not None) else "N/A"
                cum_ret = f"{val_rep.strategy_cumulative_return_pct:+.2f}%" if (val_rep and val_rep.strategy_cumulative_return_pct is not None) else "N/A"

                dir_color = "#10b981" if s.sentiment_direction == "BULLISH" else ("#ef4444" if s.sentiment_direction == "BEARISH" else "#f59e0b")
                pos_class = s.position.status.lower().replace("_", "-")
                low_badge = '<span class="badge-low-data">⚠️ LOW DATA (&lt;10 posts)</span>' if s.is_low_data else ''

                # Inline SVG Mini Corridor Chart with Sentiment Overlay
                svg_chart = cls._render_svg_corridor(s, pred)

                # Top news items with links
                news_html = ""
                if s.news_items:
                    items_rendered = []
                    for n in s.news_items[:4]:
                        sent_col = "#10b981" if n.get("sentiment_score", 0) > 0.15 else ("#ef4444" if n.get("sentiment_score", 0) < -0.15 else "#94a3b8")
                        items_rendered.append(
                            f'<div class="news-item">'
                            f'<a href="{n.get("url", "#")}" target="_blank" rel="noopener">{n.get("title", "News Article")}</a>'
                            f'<span class="badge-mini" style="color:{sent_col}">{n.get("source", "Exa")} ({n.get("sentiment_label", "NEUTRAL")})</span>'
                            f'</div>'
                        )
                    news_html = f"""
                    <div class="news-container">
                        <div class="news-title">📰 Top Recent News (Exa.com)</div>
                        {''.join(items_rendered)}
                    </div>
                    """

                card = f"""
                <div class="card">
                    <div class="card-header">
                        <div>
                            <span class="ticker-badge font-mono">{s.ticker}</span>
                            <span class="asset-title">{s.asset_name}</span>
                            <span class="exchange-tag font-mono">{s.exchange}</span>
                            {low_badge}
                        </div>
                        <div class="price-box">
                            <div class="current-price font-mono">{s.currency} {s.current_price:.2f}</div>
                            <div class="price-change font-mono {'up' if s.price_change_1d_pct >= 0 else 'down'}">{s.price_change_1d_pct:+.2f}% (1D)</div>
                        </div>
                    </div>

                    <div class="svg-container">
                        {svg_chart}
                    </div>

                    <div class="grid-4">
                        <div class="metric-pill">
                            <span class="metric-label">Blended Sentiment</span>
                            <span class="metric-val font-mono" style="color: {dir_color}">{s.aggregated_sentiment_score:+.3f} ({s.sentiment_direction})</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-label">Social Volume & Buzz</span>
                            <span class="metric-val font-mono">{s.social_mention_count} posts (Z: {s.buzz_zscore:+.2f}σ)</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-label">Sentiment Velocity</span>
                            <span class="metric-val font-mono">{s.sentiment_velocity:+.3f} /day</span>
                        </div>
                        <div class="metric-pill">
                            <span class="metric-label">Price Divergence</span>
                            <span class="metric-val font-mono" style="color: {'#10b981' if 'BULLISH' in s.sentiment_price_divergence else ('#ef4444' if 'BEARISH' in s.sentiment_price_divergence else '#94a3b8')}">{s.sentiment_price_divergence}</span>
                        </div>
                    </div>

                    <div class="forecast-banner">
                        <div class="forecast-left">
                            <div class="forecast-sub">Horizon n+1 Price Prediction</div>
                            <div class="forecast-val font-mono" style="color: {dir_color}">{s.currency} {pred_price:.2f} ({pred_pct:+.2f}%)</div>
                            <div class="forecast-ci font-mono">95% Confidence Interval: [{ci_low:.2f} — {ci_high:.2f}]</div>
                        </div>
                        <div class="forecast-right">
                            <div class="badge-pos badge-{pos_class}">{s.position.status}</div>
                            <div class="rr-text font-mono">Entry: {s.position.suggested_entry:.2f} | SL: {s.position.stop_loss:.2f} | TP1: {s.position.take_profit_1:.2f} (R/R: 1:{s.position.risk_reward_ratio:.2f})</div>
                            <div class="pos-reason font-mono" style="color: #cbd5e1; margin-top: 4px;">Reason: {s.position.reason or s.position.rationale}</div>
                        </div>
                    </div>

                    <div class="sr-container">
                        <div class="sr-title">Support & Resistance Corridors (Floor Pivots & Fibonacci)</div>
                        <div class="sr-grid">
                            <div class="sr-cell res font-mono"><span>R2:</span> <strong>{s.support_resistance.resistance_2:.2f}</strong></div>
                            <div class="sr-cell res font-mono"><span>R1:</span> <strong>{s.support_resistance.resistance_1:.2f}</strong></div>
                            <div class="sr-cell piv font-mono"><span>Pivot:</span> <strong>{s.support_resistance.pivot:.2f}</strong></div>
                            <div class="sr-cell sup font-mono"><span>S1:</span> <strong>{s.support_resistance.support_1:.2f}</strong></div>
                            <div class="sr-cell sup font-mono"><span>S2:</span> <strong>{s.support_resistance.support_2:.2f}</strong></div>
                            <div class="sr-cell fib font-mono"><span>Fib 0.618:</span> <strong>{s.support_resistance.fib_618:.2f}</strong></div>
                        </div>
                    </div>

                    <div class="val-scorecard">
                        <div class="val-header">Walk-Forward Validation (Expanding Out-of-Sample Window)</div>
                        <div class="val-grid font-mono">
                            <div class="val-cell">Hit Ratio: <strong>{hit_ratio}</strong></div>
                            <div class="val-cell">IC (Pearson): <strong>{ic_val}</strong></div>
                            <div class="val-cell">Strategy Sharpe: <strong>{sharpe}</strong></div>
                            <div class="val-cell">Strategy Cum Return: <strong>{cum_ret}</strong></div>
                        </div>
                    </div>

                    {news_html}
                </div>
                """
                cards.append(card)
            return "".join(cards)

        hkex_rows = build_table_rows(hkex_summaries)
        krx_rows = build_table_rows(krx_summaries)
        hkex_cards = build_cards(hkex_summaries)
        krx_cards = build_cards(krx_summaries)

        banner_html = (
            '<div class="banner banner-live">🟢 <strong>LIVE MARKET DATA</strong> — Operating with live market prices, Apify scrapers, and Exa.com news ingestion.</div>'
            if is_live else
            '<div class="banner banner-synthetic">⚠️ <strong>SYNTHETIC DEMO DATA</strong> — Running in simulation mode. Pass <code>--live</code> to execute against real-time market feeds and APIs.</div>'
        )

        # Provenance Block HTML
        prov_html = ""
        if provenance:
            prov_date_range = provenance.get("price_date_range", {}).get("display", "180 days (>=120 trading days)")
            total_posts = provenance.get("total_social_posts", 0)
            total_news = provenance.get("total_news_articles", 0)
            social_mode = provenance.get("social_mode", "synthetic")
            news_mode = provenance.get("news_mode", "synthetic")
            prices_mode = provenance.get("prices_mode", "synthetic")
            run_ts = provenance.get("run_timestamp", now_str)

            pills = []
            breakdown = provenance.get("by_ticker", {})
            for t, info in breakdown.items():
                pills.append(f'<span class="prov-pill"><strong>{t}</strong>: {info.get("social", 0)} posts ({info.get("sources", "")}), {info.get("news", 0)} news</span>')
            pills_str = " ".join(pills)

            prov_html = f"""
            <div class="provenance-block">
                <div class="provenance-title">📋 Data Ingestion & Provenance Audit Block</div>
                <div class="provenance-grid">
                    <div class="prov-card"><span class="prov-label">Run Timestamp</span><span class="prov-val font-mono">{run_ts}</span></div>
                    <div class="prov-card"><span class="prov-label">Overall Data Mode</span><span class="prov-val font-mono">{data_mode.upper()}</span></div>
                    <div class="prov-card"><span class="prov-label">Price Bars Coverage</span><span class="prov-val font-mono">{prov_date_range} ({prices_mode})</span></div>
                    <div class="prov-card"><span class="prov-label">Total Social Posts</span><span class="prov-val font-mono">{total_posts} ({social_mode})</span></div>
                    <div class="prov-card"><span class="prov-label">Total News Articles</span><span class="prov-val font-mono">{total_news} ({news_mode})</span></div>
                </div>
                <div class="prov-breakdown">
                    <strong>Per-Ticker Ingestion Provenance:</strong> {pills_str}
                </div>
            </div>
            """

        # Build aggregate validation summary scorecard
        agg_val_html = ""
        if aggregate_val:
            agg_val_html = f"""
            <div class="agg-val-card">
                <div class="agg-val-title">📈 Cross-Asset Aggregate Out-of-Sample Performance Summary</div>
                <div class="agg-val-grid">
                    <div class="agg-box"><span>Total OOS Samples</span><strong>{aggregate_val.get('oos_samples_count', 0)}</strong></div>
                    <div class="agg-box"><span>Mean Directional Hit Ratio</span><strong>{aggregate_val.get('mean_hit_ratio_pct', 0.0):.1f}%</strong> (vs Price-Only {aggregate_val.get('price_only_hit_ratio_pct', 0.0):.1f}%)</div>
                    <div class="agg-box"><span>Mean Information Coeff (IC)</span><strong>{aggregate_val.get('mean_information_coefficient', 0.0):+.3f}</strong></div>
                    <div class="agg-box"><span>Mean Model RMSE</span><strong>{aggregate_val.get('mean_rmse', 0.0):.4f}</strong> (vs Price-Only {aggregate_val.get('price_only_rmse', 0.0):.4f} / Zero-Ret {aggregate_val.get('zero_return_rmse', 0.0):.4f})</div>
                    <div class="agg-box"><span>Mean Strategy Sharpe</span><strong>{aggregate_val.get('mean_strategy_sharpe', 0.0):.2f}</strong></div>
                    <div class="agg-box"><span>Strategy Return vs Buy & Hold</span><strong>{aggregate_val.get('mean_strategy_cumulative_return_pct', 0.0):+.2f}% vs {aggregate_val.get('mean_benchmark_cumulative_return_pct', 0.0):+.2f}%</strong></div>
                </div>
                <div class="agg-verdict">
                    <strong>Baseline Comparison Verdict:</strong> {aggregate_val.get('sentiment_beat_price_only_verdict', '')}
                </div>
            </div>
            """

        # Build validation table rows
        val_rows = []
        for s in summaries:
            pred = predictions.get(s.ticker)
            vr = pred.validation_report if pred else None
            base = vr.baselines if vr else None
            num_folds = vr.num_folds if vr else 0
            full_hit = f"{vr.mean_directional_accuracy * 100:.1f}%" if (vr and vr.mean_directional_accuracy is not None) else "N/A"
            po_hit = f"{base.price_only_directional_accuracy * 100:.1f}%" if (base and base.price_only_directional_accuracy is not None) else "N/A"
            full_rmse = f"{vr.mean_rmse:.4f}" if (vr and vr.mean_rmse is not None) else "N/A"
            po_rmse = f"{base.price_only_rmse:.4f}" if (base and base.price_only_rmse is not None) else "N/A"
            zr_rmse = f"{base.zero_return_rmse:.4f}" if (base and base.zero_return_rmse is not None) else "N/A"

            improves = base.sentiment_improves_directional_accuracy if base else False
            badge_bg = 'rgba(16, 185, 129, 0.2)' if improves else 'rgba(239, 68, 68, 0.2)'
            badge_text = 'OUTPERFORMS' if improves else 'NO IMPROVEMENT'

            val_rows.append(f"""
            <tr>
                <td class="font-mono text-blue">{s.ticker}</td>
                <td>{s.exchange}</td>
                <td class="font-mono">{num_folds}</td>
                <td class="font-mono font-bold" style="color: #10b981;">{full_hit}</td>
                <td class="font-mono">{po_hit}</td>
                <td class="font-mono" style="color: #10b981;">{full_rmse}</td>
                <td class="font-mono">{po_rmse}</td>
                <td class="font-mono">{zr_rmse}</td>
                <td><span class="badge-mini" style="background: {badge_bg};">{badge_text}</span></td>
            </tr>
            """)
        validation_table_content = "".join(val_rows) if val_rows else "<tr><td colspan='9' style='text-align:center;'>No validation folds recorded</td></tr>"

        # Build holdout prediction vs actual table rows (with Error %)
        holdout_rows = []
        for s in summaries:
            pred = predictions.get(s.ticker)
            if pred and pred.validation_report and pred.validation_report.recent_k_predictions:
                for r in pred.validation_report.recent_k_predictions:
                    status_badge = '<span style="color: #10b981; font-weight: bold;">CORRECT</span>' if r.direction_correct else '<span style="color: #ef4444;">INCORRECT</span>'
                    holdout_rows.append(f"""
                    <tr>
                        <td class="font-mono text-blue">{s.ticker}</td>
                        <td class="font-mono">{r.date_str}</td>
                        <td class="font-mono">{r.predicted_close:.2f}</td>
                        <td class="font-mono">{r.realized_close:.2f}</td>
                        <td class="font-mono font-bold" style="color: {'#10b981' if r.error_pct < 2.0 else '#f59e0b'};">{r.error_pct:.2f}%</td>
                        <td class="font-mono">{r.predicted_return_pct:+.2f}%</td>
                        <td class="font-mono">{r.realized_return_pct:+.2f}%</td>
                        <td>{status_badge}</td>
                    </tr>
                    """)
        holdout_table_content = "".join(holdout_rows) if holdout_rows else "<tr><td colspan='8' style='text-align:center;'>No holdout bars recorded</td></tr>"

        html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>CrowdWisdom Trading - Multi-Exchange Sentiment Price Prediction Terminal</title>
    <style>
        :root {{
            --bg-primary: #0b0e14;
            --bg-secondary: #121721;
            --bg-card: #161c28;
            --text-main: #f1f5f9;
            --text-muted: #94a3b8;
            --accent: #3b82f6;
            --green: #10b981;
            --red: #ef4444;
            --amber: #f59e0b;
            --border: #1e293b;
            --font-mono: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
        }}

        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            background-color: var(--bg-primary);
            color: var(--text-main);
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            padding: 24px;
        }}

        .font-mono {{ font-family: var(--font-mono); }}
        .text-blue {{ color: #60a5fa; }}
        .text-muted {{ color: var(--text-muted); }}

        .container {{ max-width: 1400px; margin: 0 auto; }}

        .banner {{
            padding: 12px 18px;
            border-radius: 8px;
            font-size: 13.5px;
            margin-bottom: 24px;
            display: flex;
            align-items: center;
            gap: 10px;
        }}
        .banner-live {{
            background: rgba(16, 185, 129, 0.12);
            border: 1px solid rgba(16, 185, 129, 0.3);
            color: #34d399;
        }}
        .banner-synthetic {{
            background: rgba(245, 158, 11, 0.12);
            border: 1px solid rgba(245, 158, 11, 0.3);
            color: #fbbf24;
        }}

        .provenance-block {{
            background: #0f172a;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 24px;
        }}
        .provenance-title {{
            font-size: 14px;
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 12px;
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .provenance-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }}
        .prov-card {{
            background: #1e293b;
            padding: 10px 14px;
            border-radius: 6px;
        }}
        .prov-label {{
            display: block;
            font-size: 11px;
            color: #94a3b8;
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        .prov-val {{
            font-size: 13px;
            font-weight: 700;
            color: #f8fafc;
        }}
        .prov-breakdown {{
            font-size: 12px;
            color: #cbd5e1;
            padding-top: 10px;
            border-top: 1px solid #334155;
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            align-items: center;
        }}
        .prov-pill {{
            background: #1e293b;
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            border-bottom: 1px solid var(--border);
            padding-bottom: 20px;
            margin-bottom: 24px;
        }}
        .header-title {{ font-size: 24px; font-weight: 700; color: #fff; letter-spacing: -0.5px; }}
        .header-sub {{ font-size: 13px; color: var(--text-muted); margin-top: 4px; }}

        .tab-nav {{
            display: flex;
            gap: 8px;
            margin-bottom: 24px;
            border-bottom: 1px solid var(--border);
            padding-bottom: 8px;
        }}
        .tab-btn {{
            background: transparent;
            border: none;
            color: var(--text-muted);
            padding: 8px 16px;
            font-size: 14px;
            font-weight: 600;
            border-radius: 6px;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .tab-btn:hover {{ background: rgba(255, 255, 255, 0.05); color: #fff; }}
        .tab-btn.active {{ background: var(--accent); color: #fff; }}

        .tab-content {{ display: none; }}
        .tab-content.active {{ display: block; }}

        .section-header {{
            font-size: 18px;
            font-weight: 700;
            margin: 24px 0 12px 0;
            color: #93c5fd;
            display: flex;
            align-items: center;
            gap: 10px;
        }}

        .table-wrap {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            overflow-x: auto;
            margin-bottom: 30px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 13px;
            text-align: left;
        }}
        th {{
            background: #111827;
            padding: 12px 14px;
            color: var(--text-muted);
            font-weight: 600;
            border-bottom: 1px solid var(--border);
            cursor: pointer;
            user-select: none;
            white-space: nowrap;
        }}
        th:hover {{ color: #fff; }}
        td {{
            padding: 12px 14px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.04);
            vertical-align: middle;
        }}
        tr:hover td {{ background: rgba(255, 255, 255, 0.02); }}

        .badge-mini {{
            background: rgba(255, 255, 255, 0.1);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }}
        .badge-dir {{
            padding: 3px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 700;
            text-transform: uppercase;
        }}
        .badge-dir.bullish {{ background: rgba(16, 185, 129, 0.15); color: var(--green); }}
        .badge-dir.bearish {{ background: rgba(239, 68, 68, 0.15); color: var(--red); }}
        .badge-dir.neutral {{ background: rgba(245, 158, 11, 0.15); color: var(--amber); }}

        .badge-pos {{
            padding: 4px 10px;
            border-radius: 4px;
            font-size: 12px;
            font-weight: 700;
            display: inline-block;
        }}
        .badge-pos.badge-strong-buy {{ background: rgba(16, 185, 129, 0.25); color: #34d399; border: 1px solid #10b981; }}
        .badge-pos.badge-buy {{ background: rgba(16, 185, 129, 0.15); color: #6ee7b7; }}
        .badge-pos.badge-hold {{ background: rgba(148, 163, 184, 0.15); color: #cbd5e1; border: 1px solid #475569; }}
        .badge-pos.badge-sell {{ background: rgba(239, 68, 68, 0.15); color: #fca5a5; }}
        .badge-pos.badge-strong-sell {{ background: rgba(239, 68, 68, 0.25); color: #f87171; border: 1px solid #ef4444; }}

        .badge-low-data {{
            background: rgba(245, 158, 11, 0.2);
            color: #fbbf24;
            border: 1px solid rgba(245, 158, 11, 0.4);
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 10px;
            font-weight: 700;
            display: inline-block;
            margin-left: 6px;
        }}

        .pos-reason {{
            font-size: 11px;
            color: #94a3b8;
            margin-top: 3px;
        }}

        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(620px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 8px;
            padding: 20px;
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            margin-bottom: 16px;
        }}
        .ticker-badge {{
            font-size: 18px;
            font-weight: 700;
            color: #60a5fa;
            margin-right: 8px;
        }}
        .asset-title {{ font-size: 16px; font-weight: 600; color: #fff; margin-right: 8px; }}
        .exchange-tag {{ font-size: 11px; color: var(--text-muted); background: rgba(255, 255, 255, 0.05); padding: 2px 6px; border-radius: 4px; }}
        .price-box {{ text-align: right; }}
        .current-price {{ font-size: 20px; font-weight: 700; color: #fff; }}
        .price-change {{ font-size: 12px; margin-top: 2px; }}
        .price-change.up {{ color: var(--green); }}
        .price-change.down {{ color: var(--red); }}

        .svg-container {{
            background: #0f141c;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 10px;
            margin-bottom: 16px;
        }}

        .grid-4 {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 8px;
            margin-bottom: 16px;
        }}
        .metric-pill {{
            background: #111827;
            padding: 8px 10px;
            border-radius: 6px;
            border: 1px solid #1f2937;
        }}
        .metric-label {{ font-size: 10px; color: #94a3b8; text-transform: uppercase; margin-bottom: 2px; display: block; }}
        .metric-val {{ font-size: 12px; font-weight: 600; color: #f8fafc; }}

        .forecast-banner {{
            background: #111827;
            border-left: 4px solid var(--accent);
            padding: 12px 14px;
            border-radius: 4px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
        }}
        .forecast-sub {{ font-size: 11px; color: var(--text-muted); }}
        .forecast-val {{ font-size: 16px; font-weight: 700; margin-top: 2px; }}
        .forecast-ci {{ font-size: 11px; color: #94a3b8; margin-top: 2px; }}
        .rr-text {{ font-size: 11px; color: #94a3b8; margin-top: 4px; text-align: right; }}

        .sr-container {{
            background: #0d121c;
            padding: 10px;
            border-radius: 6px;
            margin-bottom: 14px;
            border: 1px solid #1e293b;
        }}
        .sr-title {{ font-size: 11px; color: #94a3b8; margin-bottom: 6px; font-weight: 600; }}
        .sr-grid {{ display: grid; grid-template-columns: repeat(6, 1fr); gap: 4px; font-size: 11px; text-align: center; }}
        .sr-cell {{ padding: 4px; border-radius: 4px; }}
        .sr-cell.res {{ background: rgba(239, 68, 68, 0.15); color: #fca5a5; }}
        .sr-cell.piv {{ background: rgba(59, 130, 246, 0.15); color: #93c5fd; }}
        .sr-cell.sup {{ background: rgba(16, 185, 129, 0.15); color: #86efac; }}
        .sr-cell.fib {{ background: rgba(245, 158, 11, 0.15); color: #fde68a; }}

        .val-scorecard {{
            background: #111827;
            padding: 10px;
            border-radius: 6px;
            font-size: 11px;
            border: 1px solid #1e293b;
            margin-bottom: 12px;
        }}
        .val-header {{ font-weight: 600; color: #94a3b8; margin-bottom: 6px; }}
        .val-grid {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; text-align: center; }}

        .news-container {{
            background: #0d121c;
            border: 1px solid #1e293b;
            border-radius: 6px;
            padding: 12px;
        }}
        .news-title {{
            font-size: 11px;
            font-weight: 700;
            color: #93c5fd;
            margin-bottom: 8px;
            text-transform: uppercase;
        }}
        .news-item {{
            padding: 6px 0;
            border-bottom: 1px solid #1e293b;
            font-size: 12px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            gap: 12px;
        }}
        .news-item:last-child {{ border-bottom: none; }}
        .news-item a {{
            color: #60a5fa;
            text-decoration: none;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
        }}
        .news-item a:hover {{ text-decoration: underline; }}

        .agg-val-card {{
            background: #111827;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 20px;
        }}
        .agg-val-title {{
            font-size: 15px;
            font-weight: 700;
            color: #60a5fa;
            margin-bottom: 12px;
        }}
        .agg-val-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 12px;
        }}
        .agg-box {{
            background: #1e293b;
            padding: 10px 14px;
            border-radius: 6px;
        }}
        .agg-box span {{
            display: block;
            font-size: 11px;
            color: #94a3b8;
            margin-bottom: 4px;
        }}
        .agg-box strong {{
            font-size: 14px;
            color: #f8fafc;
            font-family: var(--font-mono);
        }}
        .agg-verdict {{
            background: rgba(59, 130, 246, 0.1);
            border-left: 3px solid #3b82f6;
            padding: 10px 14px;
            border-radius: 4px;
            font-size: 13px;
            color: #e2e8f0;
        }}

        footer {{
            margin-top: 40px;
            border-top: 1px solid var(--border);
            padding-top: 20px;
            font-size: 12px;
            color: var(--text-muted);
            text-align: center;
        }}
    </style>
</head>
<body>
<div class="container">
    {banner_html}
    {prov_html}

    <header>
        <div>
            <div class="header-title">CrowdWisdom Trading — Quantitative Sentiment & Price Prediction Terminal</div>
            <div class="header-sub">Multi-Source Alternative Data Ingestion (Apify & Exa.com) • Walk-Forward Expanding Cross-Validation</div>
        </div>
        <div style="text-align: right;">
            <div class="font-mono text-blue" style="font-size: 14px; font-weight: bold;">{now_str}</div>
            <div style="font-size: 12px; color: #94a3b8;">Status: Operational</div>
        </div>
    </header>

    <div class="tab-nav">
        <button class="tab-btn active" onclick="switchTab('hkex')">🇭🇰 HKEX Hang Seng (5 Assets)</button>
        <button class="tab-btn" onclick="switchTab('krx')">🇰🇷 KRX Korea Exchange (5 Assets)</button>
        <button class="tab-btn" onclick="switchTab('validation')">📊 Out-of-Sample Validation & Baselines</button>
        <button class="tab-btn" onclick="switchTab('holdout')">🔍 Holdout Realized Check</button>
        <button class="tab-btn" onclick="switchTab('limitations')">⚠️ Limitations & Disclosures</button>
    </div>

    <!-- TAB 1: HKEX -->
    <div id="tab-hkex" class="tab-content active">
        <div class="section-header">🇭🇰 Hong Kong Stock Exchange (HKEX) — Social Mention & Sentiment Rankings</div>
        <div class="table-wrap">
            <table id="hkex-table">
                <thead>
                    <tr>
                        <th onclick="sortTable('hkex-table', 'ticker')">Ticker ⬍</th>
                        <th>Name</th><th>Exch</th><th>Price</th><th>1D %</th>
                        <th onclick="sortTable('hkex-table', 'mentions')">Social Mentions ⬍</th>
                        <th>Buzz Z</th>
                        <th onclick="sortTable('hkex-table', 'sentiment')">Sentiment Score ⬍</th>
                        <th onclick="sortTable('hkex-table', 'direction')">Direction ⬍</th>
                        <th onclick="sortTable('hkex-table', 'pred')">n+1 Forecast ⬍</th>
                        <th>95% CI Range</th><th>OOS Hit %</th><th>OOS IC</th><th>Position & Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {hkex_rows}
                </tbody>
            </table>
        </div>

        <div class="section-header">🔬 HKEX Target Asset Deep-Dive Cards</div>
        <div class="cards-grid">
            {hkex_cards}
        </div>
    </div>

    <!-- TAB 2: KRX -->
    <div id="tab-krx" class="tab-content">
        <div class="section-header">🇰🇷 Korea Exchange (KRX) — Social Mention & Sentiment Rankings</div>
        <div class="table-wrap">
            <table id="krx-table">
                <thead>
                    <tr>
                        <th onclick="sortTable('krx-table', 'ticker')">Ticker ⬍</th>
                        <th>Name</th><th>Exch</th><th>Price</th><th>1D %</th>
                        <th onclick="sortTable('krx-table', 'mentions')">Social Mentions ⬍</th>
                        <th>Buzz Z</th>
                        <th onclick="sortTable('krx-table', 'sentiment')">Sentiment Score ⬍</th>
                        <th onclick="sortTable('krx-table', 'direction')">Direction ⬍</th>
                        <th onclick="sortTable('krx-table', 'pred')">n+1 Forecast ⬍</th>
                        <th>95% CI Range</th><th>OOS Hit %</th><th>OOS IC</th><th>Position & Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {krx_rows}
                </tbody>
            </table>
        </div>

        <div class="section-header">🔬 KRX Target Asset Deep-Dive Cards</div>
        <div class="cards-grid">
            {krx_cards}
        </div>
    </div>

    <!-- TAB 3: VALIDATION & BASELINES -->
    <div id="tab-validation" class="tab-content">
        <div class="card" style="margin-bottom: 24px;">
            <h2 style="font-size: 18px; color: #60a5fa; margin-bottom: 12px;">Walk-Forward Out-of-Sample Performance vs Standard Baselines</h2>
            <p style="font-size: 13.5px; color: #cbd5e1; line-height: 1.6; margin-bottom: 16px;">
                Models are evaluated strictly on unseen future test folds across chronological expanding splits.
                We compare the full Sentiment+Price Ridge Model against two benchmark baselines:
                <strong>Zero-Return Predictor</strong> (predicting 0% daily return) and a <strong>Price-Only Ridge Model</strong> (trained purely on technical momentum and volatility factors without sentiment).
            </p>
            {agg_val_html}
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th><th>Exchange</th><th>OOS Folds</th><th>Full Model Hit %</th><th>Price-Only Hit %</th><th>Full Model RMSE</th><th>Price-Only RMSE</th><th>Zero-Return RMSE</th><th>Sentiment Improvement</th>
                        </tr>
                    </thead>
                    <tbody>
                        {validation_table_content}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- TAB 4: HOLDOUT CHECK -->
    <div id="tab-holdout" class="tab-content">
        <div class="card">
            <h2 style="font-size: 18px; color: #60a5fa; margin-bottom: 12px;">Realized vs Predicted Close (Recent Holdout Bars)</h2>
            <p style="font-size: 13.5px; color: #cbd5e1; margin-bottom: 16px;">
                Demonstrates step-by-step sequential forecasts versus actual realized prices on the most recent holdout bars:
            </p>
            <div class="table-wrap">
                <table>
                    <thead>
                        <tr>
                            <th>Ticker</th><th>Date</th><th>Predicted Close (n+1)</th><th>Realized Close (Actual)</th><th>Error %</th><th>Pred Return %</th><th>Real Return %</th><th>Direction Match</th>
                        </tr>
                    </thead>
                    <tbody>
                        {holdout_table_content}
                    </tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- TAB 5: LIMITATIONS -->
    <div id="tab-limitations" class="tab-content">
        <div class="card">
            <h2 style="font-size: 18px; color: #f59e0b; margin-bottom: 12px;">Honest Quantitative Limitations & Disclosures</h2>
            <div style="font-size: 14px; line-height: 1.7; color: #cbd5e1;">
                <p><strong>1. Asymmetric Alternative Data Window:</strong><br>
                Social sentiment from Apify (Telegram, LIHKG, Naver 종목토론실) and Exa news covers a recent 10-day lookback window, while daily financial bars span over 120 trading days. Days prior to the active 10-day window lack real social commentary and are modeled with explicit missingness indicators (<code>has_social=0</code>) rather than synthetic backward extrapolation.</p>
                <br>
                <p><strong>2. Retail Noise and Attention Cascades:</strong><br>
                Social forums exhibit significant herd dynamics, emotional retail trading bias, and occasional promotional spam. While log-scale follower weighting and engagement metrics mitigate outliers, sentiment shifts can be noisy leading indicators.</p>
                <br>
                <p><strong>3. Out-of-Sample Statistical Power & Sample Size:</strong><br>
                Due to daily bar frequency and expanding walk-forward splits, the number of out-of-sample test samples is modest. Statistical significance (t-statistics / p-values on Information Coefficients) is limited, and users should exercise caution before deploying capital based solely on short-term sentiment signals.</p>
                <br>
                <p><strong>4. Execution Frictions Not Modeled & No Claim of Profitability:</strong><br>
                Reported strategy Sharpe ratios and cumulative returns reflect theoretical gross signals and do not deduct exchange transaction levies, stamp duties (such as HKEX 0.1% stamp duty or KRX 0.18% transaction tax), bid-ask spread crossing, or short-borrow financing fees. No claim of trading profitability is made unless substantiated by verified live execution.</p>
                <br>
                <p><strong>5. Disciplined Algorithmic Filtering:</strong><br>
                Positions default to <code>HOLD</code> whenever the expected risk/reward ratio is below 1.5 or conviction is below 0.35, protecting capital against adverse risk-asymmetry.</p>
            </div>
        </div>
    </div>

    <footer>
        CrowdWisdom Trading Internship Assessment • HengSeng & Multi-Exchange Sentiment Price Prediction Agent
    </footer>
</div>

<script>
    function switchTab(tabId) {{
        document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
        document.querySelectorAll('.tab-btn').forEach(el => el.classList.remove('active'));
        document.getElementById('tab-' + tabId).classList.add('active');
        if (window.event && window.event.currentTarget) {{
            window.event.currentTarget.classList.add('active');
        }}
    }}

    let sortOrders = {{}};
    function sortTable(tableId, key) {{
        const tbody = document.querySelector('#' + tableId + ' tbody');
        if (!tbody) return;
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const orderKey = tableId + '_' + key;
        const asc = sortOrders[orderKey] = !sortOrders[orderKey];
        rows.sort((a, b) => {{
            const rawA = a.getAttribute('data-' + key) || '';
            const rawB = b.getAttribute('data-' + key) || '';
            const numA = parseFloat(rawA);
            const numB = parseFloat(rawB);
            if (!isNaN(numA) && !isNaN(numB)) {{
                return asc ? numA - numB : numB - numA;
            }}
            return asc ? rawA.localeCompare(rawB) : rawB.localeCompare(rawA);
        }});
        rows.forEach(r => tbody.appendChild(r));
    }}
</script>
</body>
</html>
"""

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(html_content)

        return filepath

    @classmethod
    def _render_svg_corridor(cls, s: TickerSummary, pred: Optional[PredictionOutput]) -> str:
        """Render clean responsive SVG chart visualizing price corridors, S&R pivots, and forecast cone."""
        sr = s.support_resistance
        current = s.current_price
        target = pred.predicted_price if pred else current
        ci_low = pred.confidence_interval_lower if pred else current * 0.98
        ci_high = pred.confidence_interval_upper if pred else current * 1.02

        min_val = min(sr.support_2, ci_low, current) * 0.985
        max_val = max(sr.resistance_2, ci_high, target) * 1.015
        val_range = max(1e-4, max_val - min_val)

        width = 540
        height = 110

        def to_y(val: float) -> float:
            return round(height - 15 - ((val - min_val) / val_range) * (height - 30), 1)

        y_r2 = to_y(sr.resistance_2)
        y_r1 = to_y(sr.resistance_1)
        y_piv = to_y(sr.pivot)
        y_s1 = to_y(sr.support_1)
        y_s2 = to_y(sr.support_2)
        y_cur = to_y(current)
        y_tgt = to_y(target)
        y_ci_low = to_y(ci_low)
        y_ci_high = to_y(ci_high)

        cone_path = f"M 360 {y_cur} L 500 {y_ci_high} L 500 {y_ci_low} Z"
        pred_color = "#10b981" if (pred and pred.predicted_return_pct > 0) else "#ef4444"

        svg = f"""
        <svg viewBox="0 0 {width} {height}" style="width: 100%; height: 110px; overflow: visible;">
            <!-- S&R Horizontal Guidance Lines -->
            <line x1="20" y1="{y_r2}" x2="520" y2="{y_r2}" stroke="#ef4444" stroke-width="1" stroke-dasharray="3,3" opacity="0.4" />
            <text x="25" y="{y_r2 - 3}" fill="#ef4444" font-size="9" font-family="monospace">R2: {sr.resistance_2:.2f}</text>

            <line x1="20" y1="{y_r1}" x2="520" y2="{y_r1}" stroke="#f87171" stroke-width="1" stroke-dasharray="3,3" opacity="0.5" />
            <text x="25" y="{y_r1 - 3}" fill="#f87171" font-size="9" font-family="monospace">R1: {sr.resistance_1:.2f}</text>

            <line x1="20" y1="{y_piv}" x2="520" y2="{y_piv}" stroke="#3b82f6" stroke-width="1.2" opacity="0.6" />
            <text x="25" y="{y_piv - 3}" fill="#60a5fa" font-size="9" font-family="monospace">PIVOT: {sr.pivot:.2f}</text>

            <line x1="20" y1="{y_s1}" x2="520" y2="{y_s1}" stroke="#34d399" stroke-width="1" stroke-dasharray="3,3" opacity="0.5" />
            <text x="25" y="{y_s1 + 10}" fill="#34d399" font-size="9" font-family="monospace">S1: {sr.support_1:.2f}</text>

            <line x1="20" y1="{y_s2}" x2="520" y2="{y_s2}" stroke="#10b981" stroke-width="1" stroke-dasharray="3,3" opacity="0.4" />
            <text x="25" y="{y_s2 + 10}" fill="#10b981" font-size="9" font-family="monospace">S2: {sr.support_2:.2f}</text>

            <!-- 95% Confidence Interval Cone -->
            <path d="{cone_path}" fill="{pred_color}" opacity="0.12" />

            <!-- Historical Trend Vector (Illustrative) -->
            <line x1="80" y1="{round(y_cur + (10 if s.price_change_1d_pct < 0 else -10), 1)}" x2="360" y2="{y_cur}" stroke="#94a3b8" stroke-width="2" />
            <circle cx="360" cy="{y_cur}" r="4" fill="#3b82f6" stroke="#fff" stroke-width="1.5" />
            <text x="320" y="{y_cur - 8}" fill="#f8fafc" font-size="10" font-family="monospace" font-weight="bold">Now: {current:.2f}</text>

            <!-- Horizon n+1 Projection Vector -->
            <line x1="360" y1="{y_cur}" x2="500" y2="{y_tgt}" stroke="{pred_color}" stroke-width="2.5" stroke-dasharray="4,2" />
            <circle cx="500" cy="{y_tgt}" r="4.5" fill="{pred_color}" stroke="#fff" stroke-width="1.5" />
            <text x="445" y="{y_tgt - 8}" fill="{pred_color}" font-size="10" font-family="monospace" font-weight="bold">n+1: {target:.2f}</text>
        </svg>
        """
        return svg
