"""
Naver Finance (KRX Stock Discussion Board - 종목토론실) Scraper.
Extracts retail investor sentiment, upvotes (공감), downvotes (비공감),
views, and author tiers for Korean equities via Apify.
"""

from datetime import datetime, timedelta, timezone
import logging
from typing import Any, Dict, List, Optional

import config
from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.mock_data_generator import generate_mock_social_posts

logger = logging.getLogger(__name__)


class NaverFinanceScraper(BaseScraper):
    """Scrapes Naver Finance retail discussion rooms via Apify."""

    def __init__(self, apify_handler: Optional[ApifyHandler] = None):
        super().__init__(name="Naver Finance Scraper", source_id="naver_finance")
        self.apify = apify_handler or ApifyHandler(config.APIFY_API_KEY)
        self.actor_id = getattr(config, "APIFY_ACTOR_NAVER", "hgservices~naver-stock-discussion-scraper")

    def scrape(
        self,
        ticker: str,
        days: int = 10,
        keywords: Optional[List[str]] = None,
    ) -> List[SocialPost]:
        """Scrape Naver Finance discussion posts for Korean tickers."""
        raw_code = ticker.split(".")[0]  # e.g., '005930' from '005930.KS'
        posts: List[SocialPost] = []

        if self.apify.is_configured:
            logger.info(f"Scraping Naver Finance via Apify actor {self.actor_id} for {ticker}...")
            if "naver-stock-discussion" in self.actor_id or "hgservices" in self.actor_id:
                oldest_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                run_input = {
                    "stocks": [raw_code],
                    "maxResults": 50,
                    "dateFrom": oldest_date,
                }
            elif "parseforge" in self.actor_id:
                run_input = {
                    "stockCodes": [raw_code],
                    "searchTerms": [raw_code],
                    "maxItems": 50,
                }
            else:
                start_urls = [
                    {"url": f"https://finance.naver.com/item/board.naver?code={raw_code}"},
                ]

                page_function = """
                async function pageFunction(context) {
                    const { $, request } = context;
                    const items = [];
                    $('table.type2 tbody tr').each((i, el) => {
                        const titleEl = $(el).find('td.title a');
                        if (!titleEl.length) return;
                        const title = titleEl.text().trim();
                        const href = titleEl.attr('href') || '';
                        const author = $(el).find('td.p11').first().text().trim() || 'NaverUser';
                        const dateStr = $(el).find('td span.tah').first().text().trim();
                        const views = parseInt($(el).find('td span.tah').eq(1).text()) || 0;
                        const likes = parseInt($(el).find('td strong').eq(0).text()) || 0;
                        const dislikes = parseInt($(el).find('td strong').eq(1).text()) || 0;
                        items.push({
                            title,
                            url: href ? 'https://finance.naver.com' + href : request.url,
                            author,
                            date: dateStr,
                            views,
                            likes,
                            dislikes
                        });
                    });
                    return items;
                }
                """

                run_input = {
                    "startUrls": start_urls,
                    "pageFunction": page_function,
                    "maxPagesPerCrawl": 4,
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
                logger.info(f"Retrieved {len(posts)} live Naver Finance posts for {ticker}")
                return posts

        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Scraper '{self.name}' (Actor: {self.actor_id}) returned no data for ticker {ticker}. "
                f"Apify API key configured: {self.apify.is_configured}. Mock fallbacks are disabled in live mode."
            )

        logger.info(f"Using high-fidelity simulated Naver Finance data for {ticker} (10-day lookback)")
        mock_posts = generate_mock_social_posts(ticker=ticker, days=days)
        return [p for p in mock_posts if p.source == "naver_finance"]

    def _parse_apify_item(self, item: Dict[str, Any], ticker: str) -> Optional[SocialPost]:
        """Normalize raw scraped item into standardized SocialPost."""
        try:
            if item.get("type") == "notice":
                return None

            raw_title = item.get("title") or ""
            raw_body = item.get("text") or item.get("content") or ""
            combined_text = f"{raw_title} {raw_body}".strip() if (raw_title and raw_body) else (raw_title or raw_body)
            text = self.clean_text(combined_text)
            if not text:
                return None

            raw_date = item.get("writtenAt") or item.get("date") or item.get("timestamp") or item.get("createdAt")
            timestamp_utc = self.normalize_to_utc(raw_date)

            post_id = str(item.get("postId") or item.get("nid") or item.get("id") or hash(text))
            likes = int(item.get("likes") or item.get("likeCount") or 0)
            dislikes = int(item.get("dislikes") or item.get("dislikeCount") or 0)
            views = int(item["views"]) if "views" in item and item["views"] is not None else None
            net_likes = max(0, likes - (dislikes // 2))

            author = item.get("author") or item.get("writer") or f"NaverTrader_{post_id[:5]}"
            author_followers = None  # Naver discussion boards do not expose follower metrics

            return SocialPost(
                post_id=f"naver_{post_id}",
                source="naver_finance",
                ticker=ticker,
                timestamp_utc=timestamp_utc,
                text=text,
                author=author,
                author_followers=author_followers,
                likes=net_likes,
                reposts=int(item.get("replies") or item.get("commentCount") or 0),
                views=views,
                url=item.get("url") or f"https://m.stock.naver.com/domestic/stock/{ticker.split('.')[0]}/discussion/{post_id}",
                raw_metadata={"likes": likes, "dislikes": dislikes, "isHolderVerified": item.get("isHolderVerified")},
            )
        except Exception as e:
            logger.debug(f"Failed to parse Naver Finance item: {e}")
            return None
