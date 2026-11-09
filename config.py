"""
CrowdWisdom Trading - HengSeng & Exchange Sentiment Price Prediction Agent
Global Configuration & Parameter Specifications
"""

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

# Load .env file if present
def _load_env_file():
    env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_path):
        try:
            with open(env_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        k = k.strip()
                        v = v.strip().strip("'\"")
                        if k and k not in os.environ:
                            os.environ[k] = v
        except Exception:
            pass

_load_env_file()

# ==============================================================================
# RUNTIME MODE SPECIFICATION
# ==============================================================================
# When True (--live flag passed), synthetic fallbacks are strictly prohibited.
# If any live scraper, Exa news call, or price feed returns empty/error, a clear exception is raised.
LIVE_MODE: bool = os.getenv("LIVE_MODE", "0").lower() in ("1", "true", "yes")

# ==============================================================================
# API CREDENTIALS & SERVICE TOKENS
# ==============================================================================
# Priority: Environment variable -> .env -> empty string
APIFY_API_KEY: str = os.getenv("APIFY_API_KEY", os.getenv("APIFY_TOKEN", ""))
EXA_API_KEY: str = os.getenv("EXA_API_KEY", "")
TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")

# Apify Actor IDs for scraping social platforms (verified on Apify Store):
# 1. Telegram: tri_angle/telegram-scraper (Channel messages, view counts, reactions)
APIFY_ACTOR_TELEGRAM: str = os.getenv("APIFY_ACTOR_TELEGRAM", "tri_angle~telegram-scraper")

# 2. LIHKG: claude_code_reviewer/lihkg-scraper (Category 5 財經台 / keyword search)
APIFY_ACTOR_LIHKG: str = os.getenv("APIFY_ACTOR_LIHKG", "claude_code_reviewer~lihkg-scraper")

# 3. Naver Finance: hgservices/naver-stock-discussion-scraper (KRX retail discussion room 종목토론실)
APIFY_ACTOR_NAVER: str = os.getenv("APIFY_ACTOR_NAVER", "hgservices~naver-stock-discussion-scraper")

# 4. AAStocks: bovi/aastocks-scraper (HK market news - note: news headlines, not user forum comments)
APIFY_ACTOR_AASTOCKS: str = os.getenv("APIFY_ACTOR_AASTOCKS", "bovi~aastocks-scraper")

# 5. Futu / Moomoo Community: apify/web-scraper configured with custom pageFunction
APIFY_ACTOR_FUTU: str = os.getenv("APIFY_ACTOR_FUTU", "apify~web-scraper")

# ==============================================================================
# TARGET ASSETS CONFIGURATION (HKEX & KRX)
# ==============================================================================
@dataclass
class AssetConfig:
    ticker: str
    name: str
    chinese_name: str
    exchange: str
    currency: str
    cashtags: List[str]
    keywords: List[str]
    telegram_channels: List[str]
    lihkg_category_id: int = 5  # LIHKG Category 5 = 財經台 (Finance)
    benchmark_price: float = 100.0


# 5 Target Assets for Hong Kong Exchange (HKEX) as required by assessment
HKEX_ASSETS: Dict[str, AssetConfig] = {
    "0700.HK": AssetConfig(
        ticker="0700.HK",
        name="Tencent Holdings Ltd.",
        chinese_name="騰訊控股",
        exchange="HKEX",
        currency="HKD",
        cashtags=["$0700", "$0700.HK", "$TCEHY"],
        keywords=["0700", "0700.HK", "Tencent", "騰訊", "腾讯", "企鵝", "QQ", "微信"],
        telegram_channels=["@hkstockgains", "@hkstockdiscussion", "@alpha_hangseng"],
        benchmark_price=418.60,
    ),
    "9988.HK": AssetConfig(
        ticker="9988.HK",
        name="Alibaba Group Holding Ltd.",
        chinese_name="阿里巴巴",
        exchange="HKEX",
        currency="HKD",
        cashtags=["$9988", "$9988.HK", "$BABA"],
        keywords=["9988", "9988.HK", "Alibaba", "阿里巴巴", "阿里", "淘寶", "雲智能"],
        telegram_channels=["@hkstockgains", "@hkstockdiscussion", "@china_tech_alpha"],
        benchmark_price=88.40,
    ),
    "3690.HK": AssetConfig(
        ticker="3690.HK",
        name="Meituan",
        chinese_name="美團",
        exchange="HKEX",
        currency="HKD",
        cashtags=["$3690", "$3690.HK"],
        keywords=["3690", "3690.HK", "Meituan", "美團", "美团", "外賣", "KeeTa"],
        telegram_channels=["@hkstockgains", "@hkstockdiscussion", "@canton_traders"],
        benchmark_price=178.20,
    ),
    "0005.HK": AssetConfig(
        ticker="0005.HK",
        name="HSBC Holdings plc",
        chinese_name="滙豐控股",
        exchange="HKEX",
        currency="HKD",
        cashtags=["$0005", "$0005.HK", "$HSBC"],
        keywords=["0005", "0005.HK", "HSBC", "滙豐", "汇丰", "大笨象", "獅子銀行"],
        telegram_channels=["@hkstockgains", "@hk_finance_daily", "@dividend_hk"],
        benchmark_price=71.85,
    ),
    "1211.HK": AssetConfig(
        ticker="1211.HK",
        name="BYD Company Limited",
        chinese_name="比亞迪股份",
        exchange="HKEX",
        currency="HKD",
        cashtags=["$1211", "$1211.HK", "$BYDDF"],
        keywords=["1211", "1211.HK", "BYD", "比亞迪", "比亚迪", "電車", "仰望"],
        telegram_channels=["@hkstockgains", "@ev_pulse_china", "@canton_traders"],
        benchmark_price=284.40,
    ),
}

# Optional KRX Assets for Multi-Exchange Capability
KRX_ASSETS: Dict[str, AssetConfig] = {
    "005930.KS": AssetConfig(
        ticker="005930.KS",
        name="Samsung Electronics Co., Ltd.",
        chinese_name="三星電子",
        exchange="KRX",
        currency="KRW",
        cashtags=["$005930", "$SMSN"],
        keywords=["005930", "삼성전자", "삼전", "Samsung", "HBM3E", "파운드리"],
        telegram_channels=["@krx_traders", "@yeouido_insider"],
        benchmark_price=57500.0,
    ),
    "000660.KS": AssetConfig(
        ticker="000660.KS",
        name="SK Hynix Inc.",
        chinese_name="SK海力士",
        exchange="KRX",
        currency="KRW",
        cashtags=["$000660"],
        keywords=["000660", "SK하이닉스", "하닉", "SK Hynix", "HBM", "DRAM"],
        telegram_channels=["@krx_traders", "@yeouido_insider"],
        benchmark_price=183000.0,
    ),
    "035420.KS": AssetConfig(
        ticker="035420.KS",
        name="NAVER Corporation",
        chinese_name="NAVER",
        exchange="KRX",
        currency="KRW",
        cashtags=["$035420"],
        keywords=["035420", "NAVER", "네이버", "하이퍼클로바X"],
        telegram_channels=["@krx_traders"],
        benchmark_price=175000.0,
    ),
    "005380.KS": AssetConfig(
        ticker="005380.KS",
        name="Hyundai Motor Company",
        chinese_name="現代汽車",
        exchange="KRX",
        currency="KRW",
        cashtags=["$005380"],
        keywords=["005380", "현대차", "현대자동차", "Hyundai"],
        telegram_channels=["@krx_traders"],
        benchmark_price=220000.0,
    ),
    "035720.KS": AssetConfig(
        ticker="035720.KS",
        name="Kakao Corp.",
        chinese_name="Kakao",
        exchange="KRX",
        currency="KRW",
        cashtags=["$035720"],
        keywords=["035720", "카카오", "Kakao", "카톡"],
        telegram_channels=["@krx_traders"],
        benchmark_price=36800.0,
    ),
}

# ==============================================================================
# PIPELINE HYPERPARAMETERS
# ==============================================================================
LOOKBACK_DAYS: int = 10               # Ingestion lookback required by prompt (last 10 days)
PRICE_LOOKBACK_DAYS: int = 180        # Price history lookback to ensure >= 120 trading days
PRICE_MIN_TRADING_DAYS: int = 120     # Minimum required trading days for time-series modeling
PRICE_BAR_INTERVAL: str = "1d"        # "1d" (daily) or "1h" (hourly)
TIMEZONE: str = "UTC"                 # Standardized UTC alignment across all sources

# Sentiment Analysis Weights
ENGAGEMENT_WEIGHT_LIKES: float = 1.0
ENGAGEMENT_WEIGHT_REPOSTS: float = 2.0
ENGAGEMENT_WEIGHT_VIEWS: float = 0.05
AUTHOR_WEIGHT_LOG_SCALE: float = 1.0
SOCIAL_VS_NEWS_WEIGHT: float = 0.6    # 60% social sentiment, 40% Exa news sentiment

# Creative Sentiment Feature Thresholds
SENTIMENT_BULLISH_THRESHOLD: float = 0.15
SENTIMENT_BEARISH_THRESHOLD: float = -0.15
BUZZ_SPIKE_ZSCORE_THRESHOLD: float = 2.0

# Predictive Modeling ($n+1$ Price Prediction)
PREDICTION_HORIZON: int = 1           # 1 bar ahead (n+1)
WALK_FORWARD_SPLITS: int = 4          # Number of out-of-sample expanding folds
TEST_SPLIT_RATIO: float = 0.25        # 25% holdout test window
RANDOM_SEED: int = 42

# Report & Cache Directories
BASE_DIR: str = os.path.dirname(os.path.abspath(__file__))
DATA_DIR: str = os.path.join(BASE_DIR, "data")
CACHE_DIR: str = os.path.join(DATA_DIR, "sample_cache")
OUTPUT_DIR: str = os.path.join(BASE_DIR, "output")

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
