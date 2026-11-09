"""
Feature Engineering and Alignment Pipeline.
Aligns social posts, news articles, and price bars to UTC daily timestamps.
Computes engagement-weighted sentiment velocity, buzz anomalies (z-score),
sentiment-price divergences, and tabular feature matrices for predictive modeling.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import math
from typing import Any, Dict, List, Optional, Tuple

from scrapers.base_scraper import SocialPost
from news.exa_news import NewsArticle
from market_data.price_feed import PriceBar
from models.sentiment_analyzer import FinancialSentimentAnalyzer
from models.technical_analysis import TechnicalAnalyzer, SupportResistanceLevels, PositionRecommendation
import config


@dataclass
class DailyFeatureRecord:
    """One aligned observation at daily timestamp t."""
    date_str: str                           # "YYYY-MM-DD"
    timestamp_utc: str
    ticker: str
    close_price: float
    open_price: float
    high_price: float
    low_price: float
    volume: float
    returns_1d: float
    has_social: float                       # 1.0 if real social data present, 0.0 if missing
    has_news: float                         # 1.0 if real news data present, 0.0 if missing
    social_mention_count: int
    news_mention_count: int
    raw_sentiment_mean: float               # NaN if no social posts
    engagement_weighted_sentiment: float    # NaN if no social posts
    news_sentiment_mean: float              # NaN if no news articles
    blended_sentiment: float                # Blended sentiment or NaN
    sentiment_velocity: float               # Delta sentiment over prior period (0.0 if missing)
    buzz_zscore: float                      # Z-score of social mention volume (0.0 if missing)
    sentiment_price_divergence: int          # +1 Bullish Div, -1 Bearish Div, 0 None
    rsi_14: float
    sma5_sma20_ratio: float
    volatility: float
    target_next_return: Optional[float] = None  # Return at t+1 (training target)
    target_next_close: Optional[float] = None   # Price at t+1


@dataclass
class TickerSummary:
    """Consolidated quantitative profile for reporting & ranking."""
    ticker: str
    asset_name: str
    exchange: str
    currency: str
    current_price: float
    price_change_1d_pct: float
    social_mention_count: int
    news_mention_count: int
    aggregated_sentiment_score: float
    sentiment_direction: str                 # BULLISH, BEARISH, NEUTRAL
    engagement_weighted_sentiment: float
    sentiment_velocity: float
    buzz_zscore: float
    sentiment_price_divergence: str          # "BULLISH_DIV", "BEARISH_DIV", "CONVERGENT"
    news_social_spread: float
    support_resistance: SupportResistanceLevels
    position: PositionRecommendation
    predicted_next_price: Optional[float] = None
    predicted_next_return_pct: Optional[float] = None
    prediction_interval_lower: Optional[float] = None
    prediction_interval_upper: Optional[float] = None
    is_low_data: bool = False
    news_items: List[Dict[str, Any]] = field(default_factory=list)
    social_sources_breakdown: Dict[str, int] = field(default_factory=dict)
    technical_indicators: Dict[str, float] = field(default_factory=dict)


class FeaturePipeline:
    """Builds aligned feature sets and extracts ticker metrics."""

    def __init__(self):
        self.sentiment_analyzer = FinancialSentimentAnalyzer()

    def process_ticker(
        self,
        ticker: str,
        social_posts: List[SocialPost],
        news_articles: List[NewsArticle],
        price_bars: List[PriceBar],
    ) -> Tuple[TickerSummary, List[DailyFeatureRecord]]:
        """
        Process a single ticker:
        1. Score sentiment on any un-scored posts/news.
        2. Bin social and news by date aligned with price bars.
        3. Extract creative metrics (velocity, buzz z-score, divergence).
        4. Construct ticker summary and daily feature records.
        """
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        asset_name = asset_cfg.name if asset_cfg else ticker
        exchange = asset_cfg.exchange if asset_cfg else "HKEX"
        currency = asset_cfg.currency if asset_cfg else "HKD"

        # 1. Score sentiment for all posts
        scored_posts: List[SocialPost] = []
        for p in social_posts:
            if p.sentiment_score is None:
                score, label = self.sentiment_analyzer.analyze_text(p.text)
                p.sentiment_score = score
                p.sentiment_label = label
            scored_posts.append(p)

        # Score sentiment for all news
        scored_news: List[NewsArticle] = []
        for n in news_articles:
            if n.sentiment_score is None:
                full_text = f"{n.title}. {n.summary}"
                score, label = self.sentiment_analyzer.analyze_text(full_text)
                n.sentiment_score = score
                n.sentiment_label = label
            scored_news.append(n)

        # 2. Bin by Date (YYYY-MM-DD UTC)
        posts_by_date: Dict[str, List[SocialPost]] = {}
        for p in scored_posts:
            d_str = p.timestamp_utc[:10]
            posts_by_date.setdefault(d_str, []).append(p)

        news_by_date: Dict[str, List[NewsArticle]] = {}
        for n in scored_news:
            d_str = n.published_utc[:10]
            news_by_date.setdefault(d_str, []).append(n)

        # 3. Compute baseline volume statistics for buzz z-score
        daily_counts = [len(posts) for posts in posts_by_date.values()] or [1]
        mean_vol = sum(daily_counts) / len(daily_counts)
        var_vol = sum((c - mean_vol) ** 2 for c in daily_counts) / max(1, len(daily_counts))
        std_vol = math.sqrt(var_vol) if var_vol > 0 else 1.0

        # 4. Construct daily aligned records from price bars
        records: List[DailyFeatureRecord] = []
        prior_sentiment: Optional[float] = None

        for i, bar in enumerate(price_bars):
            d_str = bar.timestamp_utc[:10]
            day_posts = posts_by_date.get(d_str, [])
            day_news = news_by_date.get(d_str, [])

            # Compute social sentiment (NaN when no posts, accompanied by has_social feature)
            if day_posts:
                has_social = 1.0
                raw_sent = sum(p.sentiment_score for p in day_posts) / len(day_posts)
                total_eng = sum(p.engagement_score for p in day_posts)
                if total_eng > 0:
                    eng_sent = sum(p.sentiment_score * p.engagement_score for p in day_posts) / total_eng
                else:
                    eng_sent = raw_sent
            else:
                has_social = 0.0
                raw_sent = float("nan")
                eng_sent = float("nan")

            # Compute news sentiment (NaN when no news, accompanied by has_news feature)
            if day_news:
                has_news = 1.0
                news_sent = sum(n.sentiment_score for n in day_news) / len(day_news)
            else:
                has_news = 0.0
                news_sent = float("nan")

            # Blended Sentiment: handle missingness cleanly
            if has_social == 1.0 and has_news == 1.0:
                blended = config.SOCIAL_VS_NEWS_WEIGHT * eng_sent + (1.0 - config.SOCIAL_VS_NEWS_WEIGHT) * news_sent
            elif has_social == 1.0:
                blended = eng_sent
            elif has_news == 1.0:
                blended = news_sent
            else:
                blended = float("nan")

            # Sentiment Velocity: Delta between current blended and prior valid day
            if not math.isnan(blended) and prior_sentiment is not None:
                velocity = blended - prior_sentiment
            else:
                velocity = 0.0

            if not math.isnan(blended):
                prior_sentiment = blended

            # Buzz Z-Score (Social volume anomaly; 0.0 if no social)
            post_cnt = len(day_posts)
            buzz_z = (post_cnt - mean_vol) / std_vol if has_social == 1.0 else 0.0

            # 1-day Return
            ret_1d = (bar.close - price_bars[i - 1].close) / price_bars[i - 1].close if i > 0 else 0.0

            # Sentiment-Price Divergence:
            div_val = 0
            if has_social == 1.0 or has_news == 1.0:
                if ret_1d < -0.005 and velocity > 0.15:
                    div_val = 1   # Bullish divergence (accumulation signal)
                elif ret_1d > 0.005 and velocity < -0.15:
                    div_val = -1  # Bearish divergence (distribution warning)

            # Historical volatility estimate
            vol = abs(ret_1d)

            # Lookback indicators (rolling up to this bar)
            sub_bars = price_bars[:i + 1]
            tech_dict = TechnicalAnalyzer.compute_technical_indicators(sub_bars)
            rsi = tech_dict.get("rsi_14", 50.0)
            sma5 = tech_dict.get("sma_5", bar.close)
            sma20 = tech_dict.get("sma_20", bar.close)
            ratio = (sma5 / sma20) if sma20 > 0 else 1.0

            rec = DailyFeatureRecord(
                date_str=d_str,
                timestamp_utc=bar.timestamp_utc,
                ticker=ticker,
                close_price=bar.close,
                open_price=bar.open,
                high_price=bar.high,
                low_price=bar.low,
                volume=bar.volume,
                returns_1d=round(ret_1d, 5),
                has_social=has_social,
                has_news=has_news,
                social_mention_count=post_cnt,
                news_mention_count=len(day_news),
                raw_sentiment_mean=raw_sent if math.isnan(raw_sent) else round(raw_sent, 4),
                engagement_weighted_sentiment=eng_sent if math.isnan(eng_sent) else round(eng_sent, 4),
                news_sentiment_mean=news_sent if math.isnan(news_sent) else round(news_sent, 4),
                blended_sentiment=blended if math.isnan(blended) else round(blended, 4),
                sentiment_velocity=round(velocity, 4),
                buzz_zscore=round(buzz_z, 2),
                sentiment_price_divergence=div_val,
                rsi_14=rsi,
                sma5_sma20_ratio=round(ratio, 4),
                volatility=round(vol, 4),
            )
            records.append(rec)

        # Attach target (n+1 return & price)
        for i in range(len(records) - 1):
            next_close = records[i + 1].close_price
            curr_close = records[i].close_price
            records[i].target_next_close = next_close
            records[i].target_next_return = round((next_close - curr_close) / curr_close, 5)

        # 5. Build TickerSummary for the current state (latest snapshot)
        current_bar = price_bars[-1]
        prev_bar = price_bars[-2] if len(price_bars) > 1 else current_bar
        price_change_1d = ((current_bar.close - prev_bar.close) / prev_bar.close) * 100.0

        # Total counts across entire lookback
        total_social = len(scored_posts)
        total_news = len(scored_news)

        # Weighted aggregate sentiment
        if scored_posts:
            tot_eng = sum(p.engagement_score for p in scored_posts)
            agg_social = sum(p.sentiment_score * p.engagement_score for p in scored_posts) / max(1e-5, tot_eng)
        else:
            agg_social = 0.0

        agg_news = (sum(n.sentiment_score for n in scored_news) / total_news) if scored_news else agg_social
        agg_blended = round(config.SOCIAL_VS_NEWS_WEIGHT * agg_social + (1.0 - config.SOCIAL_VS_NEWS_WEIGHT) * agg_news, 4)

        if agg_blended > config.SENTIMENT_BULLISH_THRESHOLD:
            direction = "BULLISH"
        elif agg_blended < config.SENTIMENT_BEARISH_THRESHOLD:
            direction = "BEARISH"
        else:
            direction = "NEUTRAL"

        # Latest velocity & buzz
        latest_rec = records[-1] if records else None
        latest_vel = latest_rec.sentiment_velocity if latest_rec else 0.0
        latest_buzz = latest_rec.buzz_zscore if latest_rec else 0.0
        latest_div_code = latest_rec.sentiment_price_divergence if latest_rec else 0
        div_str = "BULLISH_DIV" if latest_div_code > 0 else ("BEARISH_DIV" if latest_div_code < 0 else "CONVERGENT")

        # Technical S&R and Position
        sr_levels = TechnicalAnalyzer.compute_support_resistance(price_bars)
        tech_indicators = TechnicalAnalyzer.compute_technical_indicators(price_bars)
        pos_rec = TechnicalAnalyzer.generate_recommendation(
            current_price=current_bar.close,
            sr=sr_levels,
            sentiment_score=agg_blended,
            tech=tech_indicators,
        )

        # Top news items with links
        top_news = [
            {
                "title": n.title,
                "url": n.url,
                "source": n.source_name,
                "published_utc": n.published_utc,
                "sentiment_label": n.sentiment_label or ("BULLISH" if (n.sentiment_score or 0) > 0.15 else ("BEARISH" if (n.sentiment_score or 0) < -0.15 else "NEUTRAL")),
                "sentiment_score": round(n.sentiment_score or 0.0, 2),
            }
            for n in news_articles[:5]
        ]

        # Social source breakdown
        src_breakdown: Dict[str, int] = {}
        for p in social_posts:
            src_breakdown[p.source] = src_breakdown.get(p.source, 0) + 1

        summary = TickerSummary(
            ticker=ticker,
            asset_name=asset_name,
            exchange=exchange,
            currency=currency,
            current_price=current_bar.close,
            price_change_1d_pct=round(price_change_1d, 2),
            social_mention_count=total_social,
            news_mention_count=total_news,
            aggregated_sentiment_score=agg_blended,
            sentiment_direction=direction,
            engagement_weighted_sentiment=round(agg_social, 4),
            sentiment_velocity=round(latest_vel, 4),
            buzz_zscore=round(latest_buzz, 2),
            sentiment_price_divergence=div_str,
            news_social_spread=round(agg_news - agg_social, 4),
            support_resistance=sr_levels,
            position=pos_rec,
            is_low_data=(total_social < 10),
            news_items=top_news,
            social_sources_breakdown=src_breakdown,
            technical_indicators=tech_indicators,
        )

        return summary, records
