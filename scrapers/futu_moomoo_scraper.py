"""
Futu / Moomoo Community Scraper.
Scrapes retail stock discussion boards integrated into Hong Kong brokerage feeds
for real-time investor sentiment and engagement.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

import config
from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.mock_data_generator import generate_mock_social_posts

logger = logging.getLogger(__name__)


class FutuMoomooScraper(BaseScraper):
    """Scrapes Futu/Moomoo Hong Kong stock community discussions via Apify."""

    def __init__(self, apify_handler: Optional[ApifyHandler] = None):
        super().__init__(name="Futu Moomoo Scraper", source_id="futu_moomoo")
        self.apify = apify_handler or ApifyHandler(config.APIFY_API_KEY)
        self.actor_id = config.APIFY_ACTOR_FUTU

    def scrape(
        self,
        ticker: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[SocialPost]:
        """Scrape Futu community discussions for the ticker."""
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        raw_code = ticker.split(".")[0]  # e.g., '0700' from '0700.HK'

        posts: List[SocialPost] = []

        if self.apify.is_configured:
            logger.info(f"Scraping Futu/Moomoo via Apify actor {self.actor_id} for {ticker}...")
            start_urls = [
                {"url": f"https://www.moomoo.com/community/stock/{raw_code}"},
                {"url": f"https://news.futunn.com/stock/{raw_code}-hk/community"},
            ]

            run_input = {
                "startUrls": start_urls,
                "maxCrawlDepth": 2,
                "maxPagesPerCrawl": 6,
            }

            raw_items = self.apify.run_actor_and_fetch_items(
                actor_id=self.actor_id,
                run_input=run_input,
                timeout_secs=90,
            )

            cutoff_dt = datetime.now(timezone.utc) - timedelta(days=days)
            for item in raw_items:
                parsed = self._parse_apify_item(item, ticker)
                if parsed:
                    try:
                        p_dt = datetime.fromisoformat(parsed.timestamp_utc.replace("Z", "+00:00"))
                        if p_dt < cutoff_dt:
                            continue
                    except Exception:
                        pass
                    posts.append(parsed)

            if posts:
                logger.info(f"Retrieved {len(posts)} live Futu posts for {ticker}")
                return posts

        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Scraper '{self.name}' (Actor: {self.actor_id}) returned no data for ticker {ticker}. "
                f"Apify API key configured: {self.apify.is_configured}. Mock fallbacks are disabled in live mode."
            )

        # Fallback to realistic synthetic data generator
        logger.info(f"Using high-fidelity simulated Futu data for {ticker} (10-day lookback)")
        mock_posts = generate_mock_social_posts(ticker=ticker, days=days)
        return [p for p in mock_posts if p.source == "futu_moomoo"]

    def _parse_apify_item(self, item: Dict[str, Any], ticker: str) -> Optional[SocialPost]:
        """Extract and normalize Futu community post fields."""
        try:
            text = self.clean_text(item.get("content") or item.get("text") or item.get("title") or "")
            if not text:
                return None

            raw_date = item.get("create_time") or item.get("post_time") or item.get("timestamp")
            timestamp_utc = self.normalize_to_utc(raw_date)

            post_id = str(item.get("post_id") or item.get("id") or hash(text))
            likes = int(item.get("like_count") or item.get("likes") or 0)
            reposts = int(item.get("forward_count") or item.get("comments_count") or 0)
            views = int(item["read_count"]) if "read_count" in item and item["read_count"] is not None else (
                int(item["views"]) if "views" in item and item["views"] is not None else None
            )

            author = item.get("user_name") or item.get("nickname") or f"FutuTrader_{post_id[:5]}"
            followers = int(item["follower_count"]) if "follower_count" in item and item["follower_count"] is not None else None

            return SocialPost(
                post_id=f"futu_{post_id}",
                source="futu_moomoo",
                ticker=ticker,
                timestamp_utc=timestamp_utc,
                text=text,
                author=author,
                author_followers=followers,
                likes=likes,
                reposts=reposts,
                views=views,
                url=item.get("url") or f"https://www.moomoo.com/community/post/{post_id}",
                raw_metadata=item,
            )
        except Exception as e:
            logger.debug(f"Failed to parse Futu item: {e}")
            return None
