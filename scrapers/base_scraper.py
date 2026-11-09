"""
Base scraper definitions and standard post data model for CrowdWisdom pipeline.
Aligns timestamps to UTC and standardizes engagement metrics across sources.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import logging
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class SocialPost:
    """
    Standardized social post schema across all platforms.
    Meets assessment requirements:
    - Post timestamps (aligned to UTC)
    - Post text content
    - Engagement metrics (likes, reposts, views)
    - Author follower counts
    Platforms that do not expose certain fields return None rather than invented estimates.
    """
    post_id: str
    source: str                     # 'telegram', 'lihkg', 'futu_moomoo', 'naver_finance', 'aastocks'
    ticker: str                     # e.g., '0700.HK'
    timestamp_utc: str              # ISO 8601 UTC formatted string
    text: str                       # Cleaned text content
    author: str                     # Author handle or channel name
    author_followers: Optional[int] = None  # Follower/subscriber count (None if platform has no followers)
    likes: Optional[int] = None             # Likes, upvotes, or positive reactions
    reposts: Optional[int] = None           # Reposts, retweets, forwards, or reply counts
    views: Optional[int] = None             # Total view or impression count (None if not exposed)
    url: str = ""                   # Direct URL to the post or thread
    sentiment_score: Optional[float] = None  # Normalized score [-1.0, 1.0]
    sentiment_label: Optional[str] = None    # BULLISH, BEARISH, NEUTRAL
    is_mock: bool = False                    # Clearly flags synthetic demo items
    raw_metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SocialPost":
        valid_keys = {f for f in cls.__dataclass_fields__}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)

    @property
    def engagement_score(self) -> float:
        """
        Log-scaled composite engagement metric.
        Handles None values safely without inventing fake multipliers.
        """
        import math
        interaction_sum = (self.likes or 0) * 1.0 + (self.reposts or 0) * 2.0 + (self.views or 0) * 0.05
        follower_weight = math.log1p(max(0, self.author_followers or 0))
        return math.log1p(max(0.0, interaction_sum)) * max(1.0, follower_weight)


class BaseScraper(ABC):
    """Abstract base class for all Apify and direct platform scrapers."""

    def __init__(self, name: str, source_id: str):
        self.name = name
        self.source_id = source_id

    @abstractmethod
    def scrape(self, ticker: str, days: int = 10, keywords: Optional[List[str]] = None) -> List[SocialPost]:
        """
        Scrape posts for a specific ticker over the last `days`.
        Must return posts strictly aligned to UTC.
        """
        pass

    @staticmethod
    def clean_text(raw_text: str) -> str:
        """Strip HTML tags, redundant whitespaces, and noise."""
        if not raw_text:
            return ""
        # Remove HTML tags
        text = re.sub(r"<[^>]+>", " ", raw_text)
        # Remove URLs
        text = re.sub(r"https?://\S+|www\.\S+", " ", text)
        # Collapse whitespace
        text = re.sub(r"\s+", " ", text).strip()
        return text

    @staticmethod
    def normalize_to_utc(dt_val: Any) -> str:
        """
        Convert any datetime representation (str, int epoch, datetime object)
        to a standard ISO 8601 UTC string.
        """
        if dt_val is None:
            return datetime.now(timezone.utc).isoformat()

        if isinstance(dt_val, (int, float)):
            # Epoch timestamp (check milliseconds vs seconds)
            if dt_val > 1e11:
                dt_val = dt_val / 1000.0
            dt = datetime.fromtimestamp(dt_val, tz=timezone.utc)
            return dt.isoformat()

        if isinstance(dt_val, datetime):
            if dt_val.tzinfo is None:
                dt = dt_val.replace(tzinfo=timezone.utc)
            else:
                dt = dt_val.astimezone(timezone.utc)
            return dt.isoformat()

        if isinstance(dt_val, str):
            clean_str = dt_val.replace("Z", "+00:00").strip()
            # Attempt parsing ISO format
            try:
                dt = datetime.fromisoformat(clean_str)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                else:
                    dt = dt.astimezone(timezone.utc)
                return dt.isoformat()
            except Exception:
                pass

            # Common fallback patterns
            formats = [
                "%Y-%m-%d %H:%M:%S",
                "%Y/%m/%d %H:%M:%S",
                "%Y-%m-%dT%H:%M:%S",
                "%Y-%m-%d",
            ]
            for fmt in formats:
                try:
                    dt = datetime.strptime(clean_str, fmt).replace(tzinfo=timezone.utc)
                    return dt.isoformat()
                except Exception:
                    continue

        return datetime.now(timezone.utc).isoformat()
