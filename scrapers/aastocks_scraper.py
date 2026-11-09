"""
AAStocks HK Financial Forum & News Scraper.
Scrapes retail sentiment and market commentary from aastocks.com via Apify Actor bovi/aastocks-scraper.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

import config
from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.mock_data_generator import generate_mock_social_posts

logger = logging.getLogger(__name__)


class AAStocksScraper(BaseScraper):
    """Scrapes AAStocks financial feeds and commentary via Apify."""

    def __init__(self, apify_handler: Optional[ApifyHandler] = None):
        super().__init__(name="AAStocks Scraper", source_id="aastocks")
        self.apify = apify_handler or ApifyHandler(config.APIFY_API_KEY)
        self.actor_id = getattr(config, "APIFY_ACTOR_AASTOCKS", "bovi~aastocks-scraper")

    def scrape(
        self,
        ticker: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[SocialPost]:
        """Scrape AAStocks feeds for ticker mentions over the lookback period."""
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        kw_list = keywords or (asset_cfg.keywords if asset_cfg else [ticker])
        raw_code = ticker.split(".")[0]

        posts: List[SocialPost] = []

        if self.apify.is_configured:
            logger.info(f"Scraping AAStocks via Apify actor {self.actor_id} for {ticker}...")
            run_input = {
                "feed": "latest-news",
                "language": "tc",
                "maxResults": 60,
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
                    # Filter for ticker relevance
                    text_lower = parsed.text.lower()
                    if raw_code in text_lower or any(kw.lower() in text_lower for kw in kw_list):
                        posts.append(parsed)

            if posts:
                logger.info(f"Retrieved {len(posts)} live AAStocks posts for {ticker}")
                return posts

        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Scraper '{self.name}' (Actor: {self.actor_id}) returned no data for ticker {ticker}. "
                f"Apify API key configured: {self.apify.is_configured}. Mock fallbacks are disabled in live mode."
            )

        logger.info(f"Using high-fidelity simulated AAStocks data for {ticker} (10-day lookback)")
        mock_posts = generate_mock_social_posts(ticker=ticker, days=days)
        return [p for p in mock_posts if p.source in ("aastocks", "lihkg")][:15]

    def _parse_apify_item(self, item: Dict[str, Any], ticker: str) -> Optional[SocialPost]:
        """Extract and normalize AAStocks article/post."""
        try:
            text = self.clean_text(item.get("title") or item.get("text") or item.get("headline") or "")
            if not text:
                return None

            raw_date = item.get("date") or item.get("time") or item.get("datetime")
            timestamp_utc = self.normalize_to_utc(raw_date)

            post_id = str(item.get("id") or item.get("newsId") or hash(text))
            author = item.get("source") or item.get("author") or "AAStocks_Analyst"

            return SocialPost(
                post_id=f"aastocks_{post_id}",
                source="aastocks",
                ticker=ticker,
                timestamp_utc=timestamp_utc,
                text=text,
                author=author,
                author_followers=None,  # Not exposed on AAStocks
                likes=int(item.get("likes") or 0) if "likes" in item else None,
                reposts=int(item.get("shares") or 0) if "shares" in item else None,
                views=int(item["views"]) if "views" in item and item["views"] is not None else None,
                url=item.get("url") or f"http://www.aastocks.com/tc/stocks/analysis/stock-arturl.aspx?id={post_id}",
                raw_metadata=item,
            )
        except Exception as e:
            logger.debug(f"Failed to parse AAStocks item: {e}")
            return None
