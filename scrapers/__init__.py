"""
Social scrapers package integrating Apify actors for Telegram, LIHKG, Futu, and Naver Finance.
"""

from scrapers.base_scraper import BaseScraper, SocialPost
from scrapers.apify_client import ApifyHandler
from scrapers.telegram_scraper import TelegramScraper
from scrapers.lihkg_scraper import LihkgScraper
from scrapers.futu_moomoo_scraper import FutuMoomooScraper
from scrapers.naver_scraper import NaverFinanceScraper
from scrapers.mock_data_generator import generate_mock_social_posts

__all__ = [
    "BaseScraper",
    "SocialPost",
    "ApifyHandler",
    "TelegramScraper",
    "LihkgScraper",
    "FutuMoomooScraper",
    "NaverFinanceScraper",
    "generate_mock_social_posts",
]
