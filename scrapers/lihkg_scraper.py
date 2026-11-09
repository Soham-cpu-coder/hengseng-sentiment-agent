"""
LIHKG (Hong Kong Retail Forum) Scraper.
Scrapes Channel 5 (財經台 - Financial Discussions) for local sentiment,
upvotes (likes), downvotes, reply counts, and user metrics via Apify.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

import config
from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.mock_data_generator import generate_mock_social_posts

logger = logging.getLogger(__name__)


class LihkgScraper(BaseScraper):
    """Scrapes LIHKG Category 5 (財經台) financial threads and replies via Apify."""

    def __init__(self, apify_handler: Optional[ApifyHandler] = None):
        super().__init__(name="LIHKG Scraper", source_id="lihkg")
        self.apify = apify_handler or ApifyHandler(config.APIFY_API_KEY)
        self.actor_id = config.APIFY_ACTOR_LIHKG

    def scrape(
        self,
        ticker: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[SocialPost]:
        """Scrape LIHKG threads mentioning the ticker."""
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        kw_list = keywords or (asset_cfg.keywords if asset_cfg else [ticker])

        posts: List[SocialPost] = []

        if self.apify.is_configured:
            logger.info(f"Scraping LIHKG via Apify actor {self.actor_id} for {ticker}...")
            primary_kw = kw_list[0] if kw_list else ticker.split(".")[0]

            # 1. If using dedicated LIHKG actor (claude_code_reviewer/lihkg-scraper)
            if "lihkg" in self.actor_id:
                run_input = {
                    "mode": "search",
                    "bulkInput": primary_kw,
                    "catIds": ["5"],  # Category 5 = 財經台 (Finance)
                    "maxItems": 50,
                }
            else:
                # 2. If using generic scraper actor, target LIHKG's JSON API endpoints
                # LIHKG's frontend is a React SPA with hashed class names, so we query the direct API endpoint
                api_url = f"https://lihkg.com/api_v2/thread/search?q={primary_kw}&count=30"
                page_function = """
                async function pageFunction(context) {
                    const { request, log } = context;
                    try {
                        const response = await fetch(request.url, {
                            headers: { 'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)' }
                        });
                        const data = await response.json();
                        const items = (data.response && data.response.items) ? data.response.items : [];
                        return items.map(t => ({
                            thread_id: t.thread_id,
                            title: t.title,
                            like_count: t.like_count,
                            dislike_count: t.dislike_count,
                            reply_count: t.reply_count,
                            timestamp: t.reply_time ? new Date(t.reply_time * 1000).toISOString() : new Date().toISOString(),
                            user_nickname: t.user_nickname,
                            url: 'https://lihkg.com/thread/' + t.thread_id + '/page/1'
                        }));
                    } catch (err) {
                        log.error('LIHKG API fetch failed: ' + err.message);
                        return [];
                    }
                }
                """
                run_input = {
                    "startUrls": [{"url": api_url}],
                    "pageFunction": page_function,
                    "maxPagesPerCrawl": 2,
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
                logger.info(f"Retrieved {len(posts)} live LIHKG posts for {ticker}")
                return posts

        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Scraper '{self.name}' (Actor: {self.actor_id}) returned no data for ticker {ticker}. "
                f"Apify API key configured: {self.apify.is_configured}. Mock fallbacks are disabled in live mode."
            )

        # Fallback to realistic synthetic data generator
        logger.info(f"Using high-fidelity simulated LIHKG data for {ticker} (10-day lookback)")
        mock_posts = generate_mock_social_posts(ticker=ticker, days=days)
        return [p for p in mock_posts if p.source == "lihkg"]

    def _parse_apify_item(self, item: Dict[str, Any], ticker: str) -> Optional[SocialPost]:
        """Extract and normalize LIHKG forum thread fields."""
        try:
            raw_text = (
                item.get("own_text")
                or item.get("content")
                or item.get("title")
                or item.get("text")
                or item.get("msg")
                or ""
            )
            text = self.clean_text(raw_text)
            if not text:
                return None

            raw_date = (
                item.get("created_at")
                or item.get("last_reply_at")
                or item.get("timestamp")
                or item.get("reply_time")
                or item.get("create_time")
            )
            timestamp_utc = self.normalize_to_utc(raw_date)

            post_id = str(item.get("thread_id") or item.get("post_id") or item.get("reply_id") or hash(text))
            likes = int(item.get("like_count") or item.get("likes") or 0)
            dislikes = int(item.get("dislike_count") or item.get("dislikes") or 0)
            replies = int(item.get("reply_count") or item.get("replies") or 0)

            # Net positive sentiment weight for upvotes vs downvotes
            net_likes = max(0, likes - dislikes // 2)

            # LIHKG does not expose views or author followers. Set explicitly to None as required.
            views = None
            author = item.get("author") or item.get("user_nickname") or f"LIHKG_{post_id[:6]}"
            author_followers = None

            return SocialPost(
                post_id=f"lihkg_{post_id}",
                source="lihkg",
                ticker=ticker,
                timestamp_utc=timestamp_utc,
                text=text,
                author=author,
                author_followers=author_followers,
                likes=net_likes,
                reposts=replies,
                views=views,
                url=item.get("thread_url") or item.get("url") or f"https://lihkg.com/thread/{post_id}/page/1",
                raw_metadata={"likes": likes, "dislikes": dislikes, "replies": replies},
            )
        except Exception as e:
            logger.debug(f"Failed to parse LIHKG item: {e}")
            return None
