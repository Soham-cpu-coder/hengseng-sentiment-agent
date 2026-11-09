"""
News Ingestion engine integrating Exa.com API (and Tavily fallback).
Retrieves recent financial news, press releases, and earnings commentary
for target tickers over the last 10 days, aligned strictly to UTC.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
import json
import logging
import random
import re
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.parse
import urllib.request

import config

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Standardized financial news article schema."""
    article_id: str
    ticker: str
    title: str
    url: str
    published_utc: str
    source_name: str
    summary: str
    relevance_score: float = 0.8
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    is_mock: bool = False
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# Realistic financial news headlines for 10-day fallback (clean raw strings without escape warnings)
MOCK_NEWS_HEADLINES: Dict[str, List[Dict[str, Any]]] = {
    "0700.HK": [
        {"title": "Tencent Q3 Profit Jumps 47% Driven by Gaming and AI Ad Acceleration", "source": "Bloomberg HK", "sentiment": 0.88, "summary": "Tencent Holdings Ltd. reported stronger-than-expected net profit as new domestic blockbuster titles and AI-enhanced ad targeting drove margins higher."},
        {"title": "Tencent Deepens Share Buyback Program, Repurchasing HKD 1B Daily", "source": "Reuters", "sentiment": 0.72, "summary": "The social and gaming giant continued its aggressive capital return strategy, maintaining consistent daily buybacks on the Hong Kong Stock Exchange."},
        {"title": "Tencent Cloud Expands Global Footprint with New Middle East Availability Zones", "source": "South China Morning Post", "sentiment": 0.65, "summary": "Tencent announced the launch of new data center regions in Riyadh and Dubai to support growing enterprise AI infrastructure demand."},
        {"title": "Goldman Sachs Reaffirms Buy Rating on Tencent, Sets Target Price to HKD 465", "source": "AAStocks Financial", "sentiment": 0.80, "summary": "Goldman Sachs analysts highlighted Tencent's robust free cash flow conversion and defensive moat in competitive mobile gaming."},
        {"title": "Hang Seng Tech Index Edges Lower as Investors Weigh US Monetary Outlook", "source": "HKET", "sentiment": -0.15, "summary": "Hong Kong tech giants traded mixed as broader macro sentiment softened amid persistent rate uncertainty in global bond markets."},
    ],
    "9988.HK": [
        {"title": "Alibaba Cloud Cuts AI Model Pricing to Accelerate Enterprise Adoption", "source": "Caixin Global", "sentiment": 0.60, "summary": "Alibaba's cloud division reduced inference prices across its Tongyi Qianwen large language models to capture burgeoning enterprise AI workloads."},
        {"title": "Alibaba Records Modest Gross Merchandise Volume Growth During Double 11 Festival", "source": "Wall Street Journal", "sentiment": 0.20, "summary": "Taobao and Tmall saw steady active buyer growth during the flagship shopping festival despite cautious consumer spending."},
        {"title": "Morgan Stanley Lowers Alibaba Target on E-Commerce Margin Pressure", "source": "Reuters", "sentiment": -0.55, "summary": "Morgan Stanley adjusted its price target citing intense promotional competition from rival platforms PDD and Douyin."},
        {"title": r"Alibaba Continues $25B Share Repurchase Mandate, Bolstering Valuation Floor", "source": "South China Morning Post", "sentiment": 0.65, "summary": r"Alibaba repurchased $4.1 billion in shares over the latest quarter, maintaining one of the highest shareholder yield profiles in Asian tech."},
    ],
    "3690.HK": [
        {"title": "Meituan's KeeTa Captures #1 Food Delivery Market Share in Saudi Arabia", "source": "Arabian Business", "sentiment": 0.92, "summary": "Meituan's international delivery arm KeeTa surpassed incumbent delivery platforms in Riyadh within months of expansion."},
        {"title": "Meituan Outpaces Estimates in On-Demand Delivery and In-Store Services", "source": "Bloomberg", "sentiment": 0.82, "summary": "Meituan reported solid revenue growth driven by expanding high-margin local lifestyle vouchers and flash-delivery retail orders."},
        {"title": "Brokerages Raise Meituan Target on Operating Leverage and Overseas Traction", "source": "AAStocks", "sentiment": 0.78, "summary": "UBS and Citi updated ratings to Buy, projecting continued margin expansion in domestic food delivery alongside promising international catalysts."},
    ],
    "0005.HK": [
        {"title": r"HSBC Unveils $3 Billion Share Buyback and Confirms Special Dividend Plan", "source": "Financial Times", "sentiment": 0.85, "summary": r"HSBC Holdings announced a fresh $3 billion buyback while reaffirming attractive dividend payouts following strong wealth management inflows."},
        {"title": "HSBC Simplifies Corporate Structure into Four Business Units to Trim Costs", "source": "Reuters", "sentiment": 0.50, "summary": "Chief Executive Pam Kaur initiated an organizational restructuring to streamline operations across Hong Kong, the UK, and international wholesale markets."},
        {"title": "Global Rate Cut Expectations Prompt Analysts to Moderate Bank Net Interest Income", "source": "Bloomberg", "sentiment": -0.35, "summary": "Lower global policy rates are expected to compress net interest margins for large commercial lenders in subsequent quarters."},
    ],
    "1211.HK": [
        {"title": "BYD Surpasses 500,000 Monthly Deliveries in Landmark Milestone for Global EVs", "source": "Electrek", "sentiment": 0.95, "summary": "BYD achieved a historic 500,000 monthly sales milestone, driven by blockbuster demand for its fifth-generation DM dual-mode plug-in hybrids."},
        {"title": "BYD Expands Assembly Footprint in Southeast Asia and South America", "source": "Nikkei Asia", "sentiment": 0.78, "summary": "BYD's overseas manufacturing push is mitigating tariffs while positioning the brand to capture electric vehicle market share across emerging markets."},
        {"title": "European Tariff Uncertainties Cloud Chinese EV Export Valuations", "source": "Caixin", "sentiment": -0.45, "summary": "Ongoing trade negotiations between China and the European Union have introduced short-term volatility to Chinese EV export forecasts."},
    ],
    "005930.KS": [
        {"title": "Samsung Electronics Rebounds as Hyperscalers Accelerate HBM3E Qualification", "source": "Maeil Business", "sentiment": 0.85, "summary": "Samsung Electronics shares gained momentum amid institutional reports of accelerating qualification tests for next-generation AI memory."},
        {"title": "Foreign Outflows From KRX Tech Stocks Moderate After Extended Selling Wave", "source": "Korea Economic Daily", "sentiment": 0.40, "summary": "Net foreign institutional selling slowed across Korean semiconductor heavyweights following valuation resets."},
        {"title": "Samsung Foundry Wins Advanced 2nm Turnkey AI Accelerator Manufacturing Order", "source": "Reuters", "sentiment": 0.80, "summary": "Samsung's contract chipmaking unit secured custom silicon production contracts, narrowing the gap in advanced sub-3nm nodes."},
    ],
    "000660.KS": [
        {"title": "SK Hynix Posts Record High Operating Profit Driven by HBM Dominance", "source": "Bloomberg", "sentiment": 0.95, "summary": "SK Hynix reported record quarterly operating income, solidifying its position as the premier supplier of high-bandwidth memory for AI infrastructure."},
        {"title": "Brokerages Lift SK Hynix Target Prices as AI Memory Demand Outstrips Supply", "source": "Korea Herald", "sentiment": 0.88, "summary": "Major Wall Street investment banks revised price targets higher, projecting tight high-bandwidth memory inventory through late 2026."},
    ],
    "035420.KS": [
        {"title": "NAVER Showcases Enterprise AI Solutions and Monetization in Public Sector", "source": "Yonhap", "sentiment": 0.78, "summary": "NAVER's cloud and generative AI unit reported strong sequential enterprise booking growth across finance and administrative clients."},
        {"title": "Domestic Ad Spend Moderation Softens Growth Horizon for Korean Internet Platforms", "source": "Chosun Biz", "sentiment": -0.30, "summary": "Slower domestic consumer demand weighed on search and display ad revenues across Korean technology portals."},
    ],
    "005380.KS": [
        {"title": r"Hyundai Motor Unveils Record $2.8B Capital Return Program and Special Dividends", "source": "Financial Times", "sentiment": 0.88, "summary": r"Hyundai Motor announced an expanded multi-year share buyback and cancellation program following robust global hybrid margins."},
        {"title": "Automotive Tariff Speculation Weighs on Export-Driven Korean Automakers", "source": "Nikkei Asia", "sentiment": -0.40, "summary": "Potential trade policy revisions prompted defensive positioning among global institutional investors in Korean auto exporters."},
    ],
    "035720.KS": [
        {"title": "Kakao Refocuses Capital Allocation on Core Messenger and Generative AI Services", "source": "Korea Times", "sentiment": 0.65, "summary": "Kakao announced operational restructuring measures to prioritize high-margin communication and commerce platforms."},
        {"title": "Regulatory Scrutiny Persists Across Multi-Platform Tech Conglomerates in Korea", "source": "Reuters", "sentiment": -0.55, "summary": "Antitrust reviews continued to limit immediate expansion into secondary fintech services for dominant platform operators."},
    ],
}


class ExaNewsIngestor:
    """Ingests financial news using Exa.com API with Tavily and mock fallbacks."""

    EXA_SEARCH_URL = "https://api.exa.ai/search"
    TAVILY_SEARCH_URL = "https://api.tavily.com/search"

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or config.EXA_API_KEY
        self.tavily_key = config.TAVILY_API_KEY
        self.last_data_mode: str = "synthetic"  # "live" or "synthetic"
        self._has_sdk = False
        try:
            from exa_py import Exa
            if self.api_key:
                self.client = Exa(api_key=self.api_key)
                self._has_sdk = True
        except ImportError:
            self._has_sdk = False

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and len(self.api_key.strip()) > 5)

    def fetch_news(self, ticker: str, days: int = 10, limit: int = 8) -> List[NewsArticle]:
        """Fetch recent financial news articles for a ticker."""
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        asset_name = asset_cfg.name if asset_cfg else ticker
        query = f"{asset_name} {ticker} stock earnings financial news"

        # 1. Try Exa SDK
        if self._has_sdk:
            try:
                logger.info(f"Fetching news via Exa SDK for {ticker}...")
                start_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
                res = self.client.search_and_contents(
                    query=query,
                    type="neural",
                    use_autoprompt=True,
                    num_results=limit,
                    start_published_date=start_date,
                    text={"max_characters": 1000},
                    highlights=True,
                )
                articles: List[NewsArticle] = []
                for item in getattr(res, "results", []):
                    art = self._parse_exa_item(item, ticker)
                    if art:
                        articles.append(art)
                if articles:
                    self.last_data_mode = "live"
                    return articles
            except Exception as e:
                logger.error(f"Exa SDK search error for {ticker}: {e}")

        # 2. Try Exa REST API via urllib
        if self.is_configured:
            try:
                logger.info(f"Fetching news via Exa REST API for {ticker}...")
                start_date = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
                payload = {
                    "query": query,
                    "type": "neural",
                    "useAutoprompt": True,
                    "numResults": limit,
                    "startPublishedDate": start_date,
                    "contents": {"text": {"maxCharacters": 1000}, "highlights": True},
                }
                data_bytes = json.dumps(payload).encode("utf-8")
                req = urllib.request.Request(
                    self.EXA_SEARCH_URL,
                    data=data_bytes,
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json",
                        "User-Agent": "CrowdWisdomSentimentAgent/1.0",
                    },
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=20) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))

                articles = []
                for item in resp_data.get("results", []):
                    art = self._parse_exa_item(item, ticker)
                    if art:
                        articles.append(art)
                if articles:
                    self.last_data_mode = "live"
                    logger.info(f"Successfully retrieved {len(articles)} live Exa articles for {ticker}")
                    return articles
            except Exception as e:
                logger.error(f"Exa REST API error for {ticker}: {e}")

        # 3. Try Tavily fallback if configured
        if self.tavily_key:
            try:
                logger.info(f"Attempting Tavily news search fallback for {ticker}...")
                tavily_payload = {
                    "api_key": self.tavily_key,
                    "query": query,
                    "search_depth": "advanced",
                    "include_answer": False,
                    "max_results": limit,
                }
                data_bytes = json.dumps(tavily_payload).encode("utf-8")
                req = urllib.request.Request(
                    self.TAVILY_SEARCH_URL,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))

                articles = []
                for idx, item in enumerate(resp_data.get("results", [])):
                    art = NewsArticle(
                        article_id=f"tavily_{ticker}_{idx}",
                        ticker=ticker,
                        title=item.get("title", ""),
                        url=item.get("url", ""),
                        published_utc=datetime.now(timezone.utc).isoformat(),
                        source_name="Tavily Search",
                        summary=item.get("content", ""),
                        relevance_score=float(item.get("score", 0.8)),
                        is_mock=False,
                    )
                    articles.append(art)
                if articles:
                    self.last_data_mode = "live"
                    return articles
            except Exception as e:
                logger.error(f"Tavily search fallback error: {e}")

        # In LIVE mode, fail explicitly if real news is unavailable
        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: News Ingestor (Exa.com) returned no articles for ticker {ticker}. "
                f"Exa API key configured: {self.is_configured}. Mock fallbacks are disabled in live mode."
            )

        # 4. High-Fidelity Synthetic News Fallback
        self.last_data_mode = "synthetic"
        logger.info(f"Using high-fidelity simulated Exa news for {ticker} (10-day lookback)")
        return self._generate_mock_news(ticker, days)

    def _parse_exa_item(self, item: Any, ticker: str) -> Optional[NewsArticle]:
        """Normalize Exa search result."""
        try:
            if isinstance(item, dict):
                title = item.get("title", "")
                url = item.get("url", "")
                published_raw = item.get("publishedDate") or item.get("published_date")
                text = item.get("text") or " ".join(item.get("highlights", []))
                source_name = urllib.parse.urlparse(url).netloc or "Exa News"
                score = float(item.get("score") or 0.85)
                doc_id = item.get("id") or str(hash(url))
            else:
                title = getattr(item, "title", "")
                url = getattr(item, "url", "")
                published_raw = getattr(item, "published_date", None)
                text = getattr(item, "text", "")
                source_name = urllib.parse.urlparse(url).netloc or "Exa News"
                score = getattr(item, "score", 0.85) or 0.85
                doc_id = getattr(item, "id", str(hash(url)))

            if not title:
                return None

            # Standardize timestamp to UTC
            published_utc = self._normalize_utc_date(published_raw)

            return NewsArticle(
                article_id=f"exa_{doc_id}",
                ticker=ticker,
                title=title.strip(),
                url=url,
                published_utc=published_utc,
                source_name=source_name,
                summary=text[:600].strip(),
                relevance_score=round(score, 4),
            )
        except Exception as e:
            logger.debug(f"Failed to parse Exa item: {e}")
            return None

    def _generate_mock_news(self, ticker: str, days: int) -> List[NewsArticle]:
        """Generate timestamped realistic news items for evaluation."""
        now = datetime.now(timezone.utc)
        templates = MOCK_NEWS_HEADLINES.get(ticker, MOCK_NEWS_HEADLINES["0700.HK"])
        articles: List[NewsArticle] = []

        for idx, tpl in enumerate(templates):
            # Spread news across the 10-day lookback
            day_offset = int((idx / len(templates)) * (days - 1))
            pub_time = now - timedelta(days=day_offset, hours=random.randint(1, 10))
            score = tpl.get("sentiment", 0.5)
            label = "BULLISH" if score > 0.15 else ("BEARISH" if score < -0.15 else "NEUTRAL")

            art = NewsArticle(
                article_id=f"exa_mock_{ticker}_{idx}",
                ticker=ticker,
                title=f"[MOCK DEMO] {tpl['title']}",
                url=f"https://finance.example.com/news/{ticker.lower()}_{idx}",
                published_utc=pub_time.isoformat(),
                source_name=tpl["source"],
                summary=tpl["summary"],
                relevance_score=round(random.uniform(0.82, 0.98), 3),
                sentiment_score=score,
                sentiment_label=label,
                is_mock=True,
                raw_metadata={"is_synthetic": True},
            )
            articles.append(art)

        articles.sort(key=lambda a: a.published_utc)
        return articles

    @staticmethod
    def _normalize_utc_date(dt_str: Any) -> str:
        if not dt_str:
            return datetime.now(timezone.utc).isoformat()
        if isinstance(dt_str, datetime):
            return dt_str.astimezone(timezone.utc).isoformat()
        try:
            clean = str(dt_str).replace("Z", "+00:00")
            dt = datetime.fromisoformat(clean)
            return dt.astimezone(timezone.utc).isoformat()
        except Exception:
            return datetime.now(timezone.utc).isoformat()
