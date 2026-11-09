"""
CrowdWisdom Trading - HengSeng & Exchange Sentiment Price Prediction Agent
Main Pipeline Orchestration Entrypoint.

Usage:
    python main.py [--exchange HKEX|KRX|ALL] [--days 10] [--live] [--output output/sentiment_report.html]
"""

import argparse
from datetime import datetime, timedelta, timezone
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional

# Ensure project root is in sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import config
from scrapers.telegram_scraper import TelegramScraper
from scrapers.lihkg_scraper import LihkgScraper
from scrapers.futu_moomoo_scraper import FutuMoomooScraper
from scrapers.naver_scraper import NaverFinanceScraper
from news.exa_news import ExaNewsIngestor
from market_data.price_feed import PriceFeedManager
from models.feature_engineering import FeaturePipeline, TickerSummary
from models.technical_analysis import TechnicalAnalyzer
from models.predictor import SentimentPricePredictor, PredictionOutput
from reports.visual_report import VisualReportGenerator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("PipelineOrchestrator")


def run_pipeline(
    exchange: str = "HKEX",
    days: int = 10,
    live_mode: bool = False,
    output_html: Optional[str] = None,
    export_json: Optional[str] = None,
):
    """Execute complete end-to-end sentiment ingestion, modeling, and reporting."""
    if live_mode:
        config.LIVE_MODE = True

    mode_label = "LIVE PRODUCTION MODE" if config.LIVE_MODE else "SYNTHETIC DEMO MODE"
    logger.info("=" * 85)
    logger.info(f"STARTING CROWDWISDOM SENTIMENT PREDICTION PIPELINE")
    logger.info(f"Exchange: {exchange} | Lookback: {days}d | Execution Mode: {mode_label}")
    logger.info("=" * 85)

    # 1. Resolve Target Assets
    if exchange.upper() == "HKEX":
        target_assets = config.HKEX_ASSETS
    elif exchange.upper() == "KRX":
        target_assets = config.KRX_ASSETS
    elif exchange.upper() == "ALL":
        target_assets = {**config.HKEX_ASSETS, **config.KRX_ASSETS}
    else:
        logger.warning(f"Unknown exchange '{exchange}', defaulting to HKEX")
        target_assets = config.HKEX_ASSETS

    # 2. Initialize Components
    logger.info("Initializing scrapers, news ingestors, and feature engines...")
    tg_scraper = TelegramScraper()
    lihkg_scraper = LihkgScraper()
    futu_scraper = FutuMoomooScraper()
    naver_scraper = NaverFinanceScraper()
    news_ingestor = ExaNewsIngestor()
    price_manager = PriceFeedManager()
    pipeline = FeaturePipeline()

    summaries: List[TickerSummary] = []
    predictions: Dict[str, PredictionOutput] = {}
    all_raw_posts = []
    all_raw_news = []
    total_price_bars = 0

    # 3. Process Each Target Ticker
    for ticker, asset_cfg in target_assets.items():
        logger.info(f"\n>>> Ingesting alternative data & training model for {ticker} ({asset_cfg.name})...")

        # A. Social Posts Ingestion (Minimum 2 distinct sources per exchange)
        posts = []
        if asset_cfg.exchange == "KRX":
            posts.extend(tg_scraper.scrape(ticker, days=days))
            posts.extend(naver_scraper.scrape(ticker, days=days))
            source_desc = "Telegram & Naver Finance (종목토론실)"
        else:
            posts.extend(tg_scraper.scrape(ticker, days=days))
            posts.extend(lihkg_scraper.scrape(ticker, days=days))
            posts.extend(futu_scraper.scrape(ticker, days=days))
            source_desc = "Telegram, LIHKG, and Futu/Moomoo"

        all_raw_posts.extend([p.to_dict() for p in posts])
        logger.info(f"  • Social Posts Collected: {len(posts)} across {source_desc}")

        # B. News Ingestion (Exa.com)
        news = news_ingestor.fetch_news(ticker, days=days)
        all_raw_news.extend([n.to_dict() for n in news])
        logger.info(f"  • Exa.com News Articles Collected: {len(news)}")

        # C. Market Price Bars (at least 120 trading days)
        bars = price_manager.get_price_bars(ticker, days=config.PRICE_LOOKBACK_DAYS)
        total_price_bars += len(bars)
        logger.info(f"  • Market Price Bars Aligned: {len(bars)} bars (Latest Close: {bars[-1].close:.2f})")

        # D. Feature Engineering & Sentiment Aggregation
        summary, daily_records = pipeline.process_ticker(ticker, posts, news, bars)
        logger.info(f"  • Aggregate Sentiment: {summary.aggregated_sentiment_score:+.3f} ({summary.sentiment_direction})")
        logger.info(f"  • Sentiment Velocity: {summary.sentiment_velocity:+.3f} | Buzz Z-Score: {summary.buzz_zscore:+.2f}")

        # E. Predictive Modeling & Walk-Forward Out-of-Sample Validation
        predictor = SentimentPricePredictor(alpha=0.5)
        pred_out = predictor.train_and_forecast(
            ticker=ticker,
            records=daily_records,
            current_price=summary.current_price,
        )

        # Attach forecast values to summary for reporting
        summary.predicted_next_price = pred_out.predicted_price
        summary.predicted_next_return_pct = pred_out.predicted_return_pct
        summary.prediction_interval_lower = pred_out.confidence_interval_lower
        summary.prediction_interval_upper = pred_out.confidence_interval_upper

        # Re-evaluate position recommendation with forecast n+1 return signal
        summary.position = TechnicalAnalyzer.generate_recommendation(
            current_price=summary.current_price,
            sr=summary.support_resistance,
            sentiment_score=summary.aggregated_sentiment_score,
            tech=summary.technical_indicators,
            predicted_return_pct=pred_out.predicted_return_pct,
        )

        summaries.append(summary)
        predictions[ticker] = pred_out

        logger.info(f"  • Horizon n+1 Forecast: {pred_out.predicted_price:.2f} ({pred_out.predicted_return_pct:+.2f}%) "
                    f"CI: [{pred_out.confidence_interval_lower:.2f}, {pred_out.confidence_interval_upper:.2f}]")
        if pred_out.validation_report and not pred_out.validation_report.insufficient_data:
            vr = pred_out.validation_report
            logger.info(f"  • OOS Validation: Hit Ratio={vr.mean_directional_accuracy * 100:.1f}%, "
                        f"IC={vr.mean_information_coefficient:+.3f}, Strategy Sharpe={vr.strategy_sharpe_ratio:.2f}")

    # Determine Provenance and Data Modes
    social_data_mode = "synthetic" if any(p.get("is_mock", False) for p in all_raw_posts) or not all_raw_posts else "live"
    news_data_mode = news_ingestor.last_data_mode
    prices_data_mode = price_manager.last_data_mode
    overall_data_mode = "live" if (social_data_mode == "live" and news_data_mode == "live" and prices_data_mode == "live") else "synthetic"

    start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    provenance = {
        "run_timestamp": datetime.now(timezone.utc).isoformat(),
        "price_date_range": {
            "start": start_date,
            "end": end_date,
            "lookback_days": config.PRICE_LOOKBACK_DAYS,
            "display": f"{start_date} to {end_date} (~{config.PRICE_LOOKBACK_DAYS} days)",
        },
        "total_social_posts": len(all_raw_posts),
        "total_news_articles": len(all_raw_news),
        "total_price_bars": total_price_bars,
        "social_mode": social_data_mode,
        "news_mode": news_data_mode,
        "prices_mode": prices_data_mode,
        "overall_mode": overall_data_mode,
        "by_ticker": {
            s.ticker: {
                "name": s.asset_name,
                "exchange": s.exchange,
                "social": s.social_mention_count,
                "social_posts": s.social_mention_count,
                "news": len(s.news_items) if hasattr(s, "news_items") else 0,
                "news_articles": len(s.news_items) if hasattr(s, "news_items") else 0,
                "social_sources": s.social_sources_breakdown if hasattr(s, "social_sources_breakdown") else {},
                "is_low_data": s.is_low_data,
            }
            for s in summaries
        },
    }

    # Aggregate cross-asset validation metrics
    valid_reports = [
        predictions[s.ticker].validation_report
        for s in summaries
        if predictions.get(s.ticker) and predictions[s.ticker].validation_report and not predictions[s.ticker].validation_report.insufficient_data
    ]

    if valid_reports:
        total_oos = sum(vr.oos_samples_count for vr in valid_reports)
        mean_hit = sum(vr.mean_directional_accuracy * 100.0 for vr in valid_reports if vr.mean_directional_accuracy is not None) / len(valid_reports)
        po_hits = [vr.baselines.price_only_directional_accuracy * 100.0 for vr in valid_reports if vr.baselines and vr.baselines.price_only_directional_accuracy is not None]
        mean_po_hit = sum(po_hits) / len(po_hits) if po_hits else 50.0

        mean_ic = sum(vr.mean_information_coefficient for vr in valid_reports if vr.mean_information_coefficient is not None) / len(valid_reports)
        mean_rmse = sum(vr.mean_rmse for vr in valid_reports if vr.mean_rmse is not None) / len(valid_reports)
        mean_mae = sum(vr.mean_mae for vr in valid_reports if vr.mean_mae is not None) / len(valid_reports)

        po_rmses = [vr.baselines.price_only_rmse for vr in valid_reports if vr.baselines and vr.baselines.price_only_rmse is not None]
        mean_po_rmse = sum(po_rmses) / len(po_rmses) if po_rmses else 0.0

        zr_rmses = [vr.baselines.zero_return_rmse for vr in valid_reports if vr.baselines and vr.baselines.zero_return_rmse is not None]
        mean_zr_rmse = sum(zr_rmses) / len(zr_rmses) if zr_rmses else 0.0

        sharpes = [vr.strategy_sharpe_ratio for vr in valid_reports if vr.strategy_sharpe_ratio is not None]
        mean_sharpe = sum(sharpes) / len(sharpes) if sharpes else 0.0

        strat_rets = [vr.strategy_cumulative_return_pct for vr in valid_reports if vr.strategy_cumulative_return_pct is not None]
        mean_strat_ret = sum(strat_rets) / len(strat_rets) if strat_rets else 0.0

        bench_rets = [vr.benchmark_cumulative_return_pct for vr in valid_reports if vr.benchmark_cumulative_return_pct is not None]
        mean_bench_ret = sum(bench_rets) / len(bench_rets) if bench_rets else 0.0

        if mean_hit > mean_po_hit and mean_rmse < mean_po_rmse:
            verdict = f"Sentiment model outperformed price-only baseline out-of-sample on both directional hit ratio ({mean_hit:.1f}% vs {mean_po_hit:.1f}%) and RMSE ({mean_rmse:.4f} vs {mean_po_rmse:.4f})."
        elif mean_hit > mean_po_hit:
            verdict = f"Sentiment model achieved higher directional accuracy than price-only baseline ({mean_hit:.1f}% vs {mean_po_hit:.1f}%), with competitive RMSE ({mean_rmse:.4f} vs {mean_po_rmse:.4f})."
        elif mean_rmse < mean_po_rmse:
            verdict = f"Sentiment model reduced forecasting error over price-only baseline (RMSE {mean_rmse:.4f} vs {mean_po_rmse:.4f}), with hit ratio of {mean_hit:.1f}% vs {mean_po_hit:.1f}%."
        else:
            verdict = f"Price-only baseline was competitive with sentiment model (hit ratio {mean_po_hit:.1f}% vs {mean_hit:.1f}%, RMSE {mean_po_rmse:.4f} vs {mean_rmse:.4f})."

        aggregate_val = {
            "oos_samples_count": total_oos,
            "mean_hit_ratio_pct": round(mean_hit, 2),
            "price_only_hit_ratio_pct": round(mean_po_hit, 2),
            "mean_information_coefficient": round(mean_ic, 4),
            "mean_rmse": round(mean_rmse, 5),
            "mean_mae": round(mean_mae, 5),
            "price_only_rmse": round(mean_po_rmse, 5),
            "zero_return_rmse": round(mean_zr_rmse, 5),
            "mean_strategy_sharpe": round(mean_sharpe, 2),
            "mean_strategy_cumulative_return_pct": round(mean_strat_ret, 2),
            "mean_benchmark_cumulative_return_pct": round(mean_bench_ret, 2),
            "sentiment_beat_price_only_verdict": verdict,
        }
    else:
        aggregate_val = {
            "oos_samples_count": 0,
            "mean_hit_ratio_pct": 0.0,
            "price_only_hit_ratio_pct": 0.0,
            "mean_information_coefficient": 0.0,
            "mean_rmse": 0.0,
            "mean_mae": 0.0,
            "price_only_rmse": 0.0,
            "zero_return_rmse": 0.0,
            "mean_strategy_sharpe": 0.0,
            "mean_strategy_cumulative_return_pct": 0.0,
            "mean_benchmark_cumulative_return_pct": 0.0,
            "sentiment_beat_price_only_verdict": "Insufficient out-of-sample data across assets.",
        }

    # 4. Generate Reports
    # A. Terminal Table Output
    cli_report = VisualReportGenerator.generate_cli_table(
        summaries, predictions, data_mode=overall_data_mode, provenance=provenance, aggregate_val=aggregate_val
    )
    print(cli_report)

    # B. Interactive HTML Report
    html_path = output_html or os.path.join(config.OUTPUT_DIR, "sentiment_report.html")
    generated_path = VisualReportGenerator.generate_html_report(
        summaries, predictions, html_path, data_mode=overall_data_mode, provenance=provenance, aggregate_val=aggregate_val
    )
    logger.info(f"Visual HTML Report generated successfully: {generated_path}")

    # C. Markdown Summary
    md_path = os.path.join(config.OUTPUT_DIR, "report_summary.md")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# CrowdWisdom Trading - Sentiment Price Prediction Summary\n\n")
        if overall_data_mode == "synthetic":
            f.write("> **WARNING: SYNTHETIC DEMO DATA** — Simulated alternative data and price paths.\n\n")
        else:
            f.write("> **STATUS: LIVE MARKET DATA** — Real-time verified market prices, Apify social scrapers, and Exa news.\n\n")
        f.write(f"*Generated at: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}*\n\n")
        f.write("## 1. Data Provenance Audit\n\n")
        f.write(f"- **Price Coverage:** {provenance['price_date_range']['display']} ({prices_data_mode})\n")
        f.write(f"- **Social Posts Ingested:** {provenance['total_social_posts']} posts ({social_data_mode})\n")
        f.write(f"- **News Articles Ingested:** {provenance['total_news_articles']} articles ({news_data_mode})\n")
        f.write(f"- **Overall Pipeline Mode:** `{overall_data_mode.upper()}`\n\n")
        f.write("### Per-Ticker Provenance Breakdown\n\n")
        f.write("| Ticker | Exchange | Social Posts | News Articles | Low Data? |\n")
        f.write("|:---|:---|:---:|:---:|:---:|\n")
        for s in summaries:
            low_txt = "⚠️ (<10 posts)" if s.is_low_data else "No"
            f.write(f"| `{s.ticker}` | {s.exchange} | {s.social_mention_count} | {len(s.news_items)} | {low_txt} |\n")
        f.write("\n## 2. Cross-Asset Aggregate Out-of-Sample Performance\n\n")
        f.write(f"- **Total OOS Samples:** {aggregate_val['oos_samples_count']}\n")
        f.write(f"- **Directional Hit Ratio:** **{aggregate_val['mean_hit_ratio_pct']:.1f}%** (Price-Only Baseline: {aggregate_val['price_only_hit_ratio_pct']:.1f}%)\n")
        f.write(f"- **Information Coefficient (IC):** **{aggregate_val['mean_information_coefficient']:+.3f}**\n")
        f.write(f"- **Model RMSE:** **{aggregate_val['mean_rmse']:.4f}** (Price-Only: {aggregate_val['price_only_rmse']:.4f} | Zero-Return: {aggregate_val['zero_return_rmse']:.4f})\n")
        f.write(f"- **Strategy Sharpe Ratio:** **{aggregate_val['mean_strategy_sharpe']:.2f}**\n")
        f.write(f"- **Strategy vs Buy & Hold Return:** **{aggregate_val['mean_strategy_cumulative_return_pct']:+.2f}%** vs **{aggregate_val['mean_benchmark_cumulative_return_pct']:+.2f}%**\n")
        f.write(f"- **Baseline Comparison Verdict:** {aggregate_val['sentiment_beat_price_only_verdict']}\n\n")
        f.write("## 3. Executive Summary & Ticker Breakdown\n\n")
        f.write("```\n" + cli_report + "\n```\n\n")
        f.write("## 4. Limitations & Research Disclosures\n\n")
        f.write("1. **Lookback Window:** Social and news scrapers capture recent 10-day alternative data.\n")
        f.write("2. **Microstructure & Noise:** Social media discussions exhibit sudden attention spikes and retail sentiment noise.\n")
        f.write("3. **Statistical Power:** Daily bars yield compact out-of-sample sample counts; caution should be used before claiming definitive long-term edge.\n")
        f.write("4. **Execution Frictions:** Commission, bid-ask spread, borrowing costs for shorting, and stamp duty are omitted.\n")
        f.write("5. **Low Data Coverage:** Tickers with `< 10` social posts in the 10-day window are flagged with low data warnings.\n")
    logger.info(f"Markdown Summary generated: {md_path}")

    # D. JSON Export
    json_path = export_json or os.path.join(config.OUTPUT_DIR, "pipeline_results.json")
    payload = {
        "metadata": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "exchange": exchange,
            "lookback_days": days,
            "total_assets": len(summaries),
            "data_mode": {
                "social": social_data_mode,
                "news": news_data_mode,
                "prices": prices_data_mode,
                "overall": overall_data_mode,
            },
            "provenance": provenance,
            "aggregate_validation": aggregate_val,
        },
        "summaries": [
            {
                "ticker": s.ticker,
                "asset_name": s.asset_name,
                "exchange": s.exchange,
                "currency": s.currency,
                "current_price": s.current_price,
                "social_mentions": s.social_mention_count,
                "buzz_zscore": s.buzz_zscore,
                "sentiment_score": s.aggregated_sentiment_score,
                "sentiment_direction": s.sentiment_direction,
                "engagement_weighted_sentiment": s.engagement_weighted_sentiment,
                "sentiment_velocity": s.sentiment_velocity,
                "divergence": s.sentiment_price_divergence,
                "is_low_data": s.is_low_data,
                "predicted_next_price": s.predicted_next_price,
                "predicted_next_return_pct": s.predicted_next_return_pct,
                "confidence_interval": [s.prediction_interval_lower, s.prediction_interval_upper],
                "position": {
                    "status": s.position.status,
                    "reason": s.position.reason or s.position.rationale,
                    "conviction": s.position.conviction,
                    "entry": s.position.suggested_entry,
                    "stop_loss": s.position.stop_loss,
                    "take_profit_1": s.position.take_profit_1,
                    "take_profit_2": s.position.take_profit_2,
                    "risk_reward": s.position.risk_reward_ratio,
                },
                "support_resistance": {
                    "pivot": s.support_resistance.pivot,
                    "r1": s.support_resistance.resistance_1,
                    "r2": s.support_resistance.resistance_2,
                    "s1": s.support_resistance.support_1,
                    "s2": s.support_resistance.support_2,
                    "fib_618": s.support_resistance.fib_618,
                    "nearest_support": s.support_resistance.nearest_support,
                    "nearest_resistance": s.support_resistance.nearest_resistance,
                },
                "validation": (
                    {
                        "oos_samples": predictions[s.ticker].validation_report.oos_samples_count,
                        "hit_ratio": predictions[s.ticker].validation_report.mean_directional_accuracy,
                        "information_coefficient": predictions[s.ticker].validation_report.mean_information_coefficient,
                        "strategy_sharpe": predictions[s.ticker].validation_report.strategy_sharpe_ratio,
                        "strategy_cumulative_return_pct": predictions[s.ticker].validation_report.strategy_cumulative_return_pct,
                        "benchmark_cumulative_return_pct": predictions[s.ticker].validation_report.benchmark_cumulative_return_pct,
                        "rmse": predictions[s.ticker].validation_report.mean_rmse,
                        "mae": predictions[s.ticker].validation_report.mean_mae,
                        "insufficient_data": predictions[s.ticker].validation_report.insufficient_data,
                        "baselines": (
                            {
                                "price_only_directional_accuracy": predictions[s.ticker].validation_report.baselines.price_only_directional_accuracy,
                                "price_only_rmse": predictions[s.ticker].validation_report.baselines.price_only_rmse,
                                "zero_return_rmse": predictions[s.ticker].validation_report.baselines.zero_return_rmse,
                                "sentiment_improves_directional_accuracy": predictions[s.ticker].validation_report.baselines.sentiment_improves_directional_accuracy,
                                "sentiment_improves_rmse": predictions[s.ticker].validation_report.baselines.sentiment_improves_rmse,
                            }
                            if predictions[s.ticker].validation_report.baselines
                            else {}
                        ),
                        "predicted_vs_realized": [
                            {
                                "date": r.date_str,
                                "predicted_close": r.predicted_close,
                                "realized_close": r.realized_close,
                                "predicted_return_pct": r.predicted_return_pct,
                                "realized_return_pct": r.realized_return_pct,
                                "direction_correct": r.direction_correct,
                                "error_pct": r.error_pct,
                            }
                            for r in predictions[s.ticker].validation_report.recent_k_predictions
                        ],
                    }
                    if predictions.get(s.ticker) and predictions[s.ticker].validation_report and not predictions[s.ticker].validation_report.insufficient_data
                    else {
                        "insufficient_data": True,
                        "oos_samples": 0,
                        "hit_ratio": None,
                        "information_coefficient": None,
                        "strategy_sharpe": None,
                        "rmse": None,
                        "mae": None,
                        "baselines": {},
                        "predicted_vs_realized": [],
                    }
                ) if predictions.get(s.ticker) else {},
                "news_articles": s.news_items,
            }
            for s in summaries
        ]
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    logger.info(f"JSON Export saved: {json_path}")

    # Cache sample raw data for inspection
    cache_social_path = os.path.join(config.CACHE_DIR, "raw_social_posts.json")
    with open(cache_social_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_posts, f, indent=2)

    cache_news_path = os.path.join(config.CACHE_DIR, "raw_news_articles.json")
    with open(cache_news_path, "w", encoding="utf-8") as f:
        json.dump(all_raw_news, f, indent=2)

    # One-line Data Provenance Summary (Task 3)
    provenance_line = (
        f"Data Provenance: [Prices: {total_price_bars} bars ({prices_data_mode}) | "
        f"Social: {len(all_raw_posts)} posts ({social_data_mode}) | "
        f"News: {len(all_raw_news)} articles ({news_data_mode}) | "
        f"Date Range: {start_date} to {end_date} | Overall: {overall_data_mode.upper()}]"
    )
    print("\n" + provenance_line + "\n")
    logger.info(provenance_line)
    logger.info("PIPELINE EXECUTION COMPLETE. All deliverables generated successfully.\n")


def main():
    parser = argparse.ArgumentParser(
        description="CrowdWisdom Trading - HengSeng & Exchange Sentiment Price Prediction Agent"
    )
    parser.add_argument(
        "--exchange",
        type=str,
        default="HKEX",
        choices=["HKEX", "KRX", "ALL"],
        help="Target exchange to analyze (default: HKEX)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=config.LOOKBACK_DAYS,
        help="Lookback period in days for social & news scraping (default: 10)",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Strict live mode: fail immediately if any scraper, Exa news, or price feed returns no real data. Never falls back to mock data.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path for generated HTML visual report (default: output/sentiment_report.html)",
    )
    parser.add_argument(
        "--json",
        type=str,
        default=None,
        help="Path for exporting results to JSON (default: output/pipeline_results.json)",
    )

    args = parser.parse_args()
    run_pipeline(
        exchange=args.exchange,
        days=args.days,
        live_mode=args.live,
        output_html=args.output,
        export_json=args.json,
    )


if __name__ == "__main__":
    main()
