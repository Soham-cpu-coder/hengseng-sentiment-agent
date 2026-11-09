"""
Telegram scraper integrating Apify Actor (vdrmota/telegram-scraper or apify/telegram-scraper).
Collects post content, timestamps (UTC), views, forwards, reactions, and channel follower counts.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

import config
from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.mock_data_generator import generate_mock_social_posts

logger = logging.getLogger(__name__)


class TelegramScraper(BaseScraper):
    """Scrapes financial discussion channels and groups on Telegram via Apify."""

    def __init__(self, apify_handler: Optional[ApifyHandler] = None):
        super().__init__(name="Telegram Scraper", source_id="telegram")
        self.apify = apify_handler or ApifyHandler(config.APIFY_API_KEY)
        self.actor_id = config.APIFY_ACTOR_TELEGRAM

    def scrape(
        self,
        ticker: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[SocialPost]:
        """
        Scrape Telegram channels for ticker mentions over the lookback period.
        """
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        channels = asset_cfg.telegram_channels if asset_cfg else ["@hkstockgains", "@hkstockdiscussion"]
        kw_list = keywords or (asset_cfg.keywords if asset_cfg else [ticker])

        posts: List[SocialPost] = []

        if self.apify.is_configured:
            logger.info(f"Scraping Telegram via Apify actor {self.actor_id} for {ticker}...")
            # Real Apify input schema for tri_angle/telegram-scraper
            profile_names = [ch.lstrip("@").strip() for ch in channels if ch.strip()]
            run_input = {
                "profiles": profile_names,
                "collectMessages": True,
                "scrapeLastNDays": days,
                # Compatibility fields for alternate telegram actors
                "channels": profile_names,
                "maxMessagesPerChannel": 100,
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
                    # Filter to last N days
                    try:
                        p_dt = datetime.fromisoformat(parsed.timestamp_utc.replace("Z", "+00:00"))
                        if p_dt < cutoff_dt:
                            continue
                    except Exception:
                        pass
                    # Filter for ticker keyword relevance
                    lower_text = parsed.text.lower()
                    if any(kw.lower() in lower_text for kw in kw_list) or any(c.lower() in lower_text for c in (asset_cfg.cashtags if asset_cfg else [])):
                        posts.append(parsed)

            if posts:
                logger.info(f"Retrieved {len(posts)} live Telegram posts for {ticker}")
                return posts

        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Scraper '{self.name}' (Actor: {self.actor_id}) returned no data for ticker {ticker}. "
                f"Apify API key configured: {self.apify.is_configured}. Mock fallbacks are disabled in live mode."
            )

        # Fallback to realistic synthetic data generator if not in live mode
        logger.info(f"Using high-fidelity simulated Telegram data for {ticker} (10-day lookback)")
        mock_posts = generate_mock_social_posts(ticker=ticker, days=days)
        return [p for p in mock_posts if p.source == "telegram"]

    def _parse_apify_item(self, item: Dict[str, Any], ticker: str) -> Optional[SocialPost]:
        """Extract and normalize standard fields from Apify Telegram item."""
        try:
            msg_obj = item.get("message") if isinstance(item.get("message"), dict) else {}
            raw_text = (
                msg_obj.get("description")
                or msg_obj.get("text")
                or (item.get("message") if isinstance(item.get("message"), str) else "")
                or item.get("text")
                or item.get("description")
                or ""
            )
            text = self.clean_text(raw_text)
            if not text:
                return None

            raw_date = (
                msg_obj.get("fulldate")
                or msg_obj.get("date")
                or item.get("date")
                or item.get("datetime")
                or item.get("timestamp")
            )
            timestamp_utc = self.normalize_to_utc(raw_date)

            post_id = str(msg_obj.get("id") or item.get("id") or item.get("messageId") or hash(text))
            channel = item.get("channel", {}) if isinstance(item.get("channel"), dict) else {}
            channel_name = (
                item.get("username")
                or channel.get("username")
                or item.get("channelUsername")
                or item.get("fullName")
                or "TelegramUser"
            )
            followers_raw = item.get("followers") or channel.get("subscribers") or item.get("channelSubscribers")
            followers = int(followers_raw) if followers_raw is not None else None

            views = int(msg_obj.get("views") or item.get("views") or 0)
            forwards = int(msg_obj.get("forwards") or item.get("forwards") or item.get("shares") or 0)

            # Reactions parsing
            reactions = msg_obj.get("reactions") or item.get("reactions", {})
            likes = 0
            if isinstance(reactions, dict):
                likes = sum(int(v) for v in reactions.values() if isinstance(v, (int, float)))
            elif isinstance(reactions, list):
                likes = sum(int(r.get("count", 1)) for r in reactions if isinstance(r, dict))
            else:
                likes = int(reactions or 0)

            return SocialPost(
                post_id=f"tg_{post_id}",
                source="telegram",
                ticker=ticker,
                timestamp_utc=timestamp_utc,
                text=text,
                author=f"@{channel_name}" if not channel_name.startswith("@") else channel_name,
                author_followers=followers,
                likes=likes,
                reposts=forwards,
                views=views,
                url=msg_obj.get("link") or item.get("url") or f"https://t.me/{channel_name.lstrip('@')}/{post_id}",
                raw_metadata=item,
            )
        except Exception as e:
            logger.debug(f"Failed to parse Telegram item: {e}")
            return None
