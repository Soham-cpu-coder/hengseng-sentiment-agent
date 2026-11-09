"""
Market Data & Price Bar feed for HKEX & KRX equities.
Fetches daily and intraday OHLCV price bars, aligns all timestamps to UTC,
and provides technical indicators and support/resistance baselines.
"""

from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
import logging
import math
import random
from typing import Any, Dict, List, Optional
import urllib.error
import urllib.request

import config

logger = logging.getLogger(__name__)


@dataclass
class PriceBar:
    """Standardized OHLCV price bar with UTC timestamp."""
    timestamp_utc: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    vwap: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PriceFeedManager:
    """Retrieves and manages historical and real-time market price bars."""

    YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"

    def __init__(self):
        self.last_data_mode: str = "synthetic"  # "live" or "synthetic"
        self._has_yfinance = False
        try:
            import yfinance as yf
            self.yf = yf
            self._has_yfinance = True
        except ImportError:
            self._has_yfinance = False

    def get_price_bars(
        self,
        ticker: str,
        days: int = config.PRICE_LOOKBACK_DAYS,
        interval: str = "1d",
    ) -> List[PriceBar]:
        """
        Fetch OHLCV price bars for a ticker over at least 120 trading days.
        Guarantees strictly chronological bars with UTC timestamps.
        Never falls back to synthetic data in live mode.
        """
        days = max(days, config.PRICE_LOOKBACK_DAYS)
        bars: List[PriceBar] = []

        # 1. Try yfinance SDK if available
        if self._has_yfinance:
            try:
                logger.info(f"Fetching market bars for {ticker} via yfinance ({days}d)...")
                df = self.yf.download(ticker, period=f"{days}d", interval=interval, progress=False)
                if df is not None and not df.empty:
                    for idx, row in df.iterrows():
                        dt = idx.to_pydatetime()
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        else:
                            dt = dt.astimezone(timezone.utc)

                        c = float(row["Close"].iloc[0] if hasattr(row["Close"], "iloc") else row["Close"])
                        o = float(row["Open"].iloc[0] if hasattr(row["Open"], "iloc") else row["Open"])
                        h = float(row["High"].iloc[0] if hasattr(row["High"], "iloc") else row["High"])
                        l = float(row["Low"].iloc[0] if hasattr(row["Low"], "iloc") else row["Low"])
                        v = float(row["Volume"].iloc[0] if hasattr(row["Volume"], "iloc") else row["Volume"])
                        vwap = (h + l + c) / 3.0

                        bars.append(PriceBar(
                            timestamp_utc=dt.isoformat(),
                            open=round(o, 2),
                            high=round(h, 2),
                            low=round(l, 2),
                            close=round(c, 2),
                            volume=round(v, 0),
                            vwap=round(vwap, 2),
                        ))
                    if bars:
                        self.last_data_mode = "live"
                        logger.info(f"Retrieved {len(bars)} live bars via yfinance for {ticker}")
                        return bars
            except Exception as e:
                logger.debug(f"yfinance download failed for {ticker}: {e}")

        # 2. Try Yahoo Finance Chart v8 REST API
        try:
            logger.info(f"Attempting Yahoo Finance v8 direct REST API for {ticker} ({days}d)...")
            url = f"{self.YAHOO_CHART_URL.format(ticker=ticker)}?interval={interval}&range={days}d"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"},
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            chart_res = data.get("chart", {}).get("result")
            if chart_res and len(chart_res) > 0:
                res = chart_res[0]
                timestamps = res.get("timestamp", [])
                quote = res.get("indicators", {}).get("quote", [{}])[0]
                opens = quote.get("open", [])
                highs = quote.get("high", [])
                lows = quote.get("low", [])
                closes = quote.get("close", [])
                volumes = quote.get("volume", [])

                for i in range(len(timestamps)):
                    if closes[i] is None:
                        continue
                    dt = datetime.fromtimestamp(timestamps[i], tz=timezone.utc)
                    c = float(closes[i])
                    o = float(opens[i] if opens[i] is not None else c)
                    h = float(highs[i] if highs[i] is not None else c)
                    l = float(lows[i] if lows[i] is not None else c)
                    v = float(volumes[i] if volumes[i] is not None else 100000)

                    bars.append(PriceBar(
                        timestamp_utc=dt.isoformat(),
                        open=round(o, 2),
                        high=round(h, 2),
                        low=round(l, 2),
                        close=round(c, 2),
                        volume=round(v, 0),
                        vwap=round((h + l + c) / 3.0, 2),
                    ))

                if bars:
                    self.last_data_mode = "live"
                    logger.info(f"Retrieved {len(bars)} live bars via direct Yahoo v8 for {ticker}")
                    return bars
        except Exception as e:
            logger.debug(f"Yahoo v8 chart query error for {ticker}: {e}")

        # In LIVE mode, fail explicitly if real data is unavailable
        if config.LIVE_MODE:
            raise RuntimeError(
                f"Live mode error: Price feed returned no real market data for ticker {ticker}. "
                f"Verify network connectivity to Yahoo Finance REST API. Mock fallbacks are disabled in live mode."
            )

        # 3. High-Fidelity Calibrated Synthetic Price Bars
        self.last_data_mode = "synthetic"
        logger.info(f"Generating calibrated realistic market bars for {ticker} ({days} days / >=120 bars)...")
        return self._generate_calibrated_bars(ticker, days)

    def _generate_calibrated_bars(self, ticker: str, days: int) -> List[PriceBar]:
        """
        Generate mathematically realistic geometric brownian motion / mean-reverting
        daily price bars calibrated to the asset's real benchmark price.
        """
        asset_cfg = config.HKEX_ASSETS.get(ticker) or config.KRX_ASSETS.get(ticker)
        base_price = asset_cfg.benchmark_price if asset_cfg else 100.0

        random.seed(42 + sum(ord(c) for c in ticker))
        now = datetime.now(timezone.utc)

        bars: List[PriceBar] = []
        current_close = base_price * random.uniform(0.94, 0.98)  # slight trend upwards to current benchmark
        volatility = 0.018  # ~1.8% daily volatility
        drift = 0.0015      # moderate positive drift

        # Generate from `days` ago up to today
        for i in range(days, -1, -1):
            dt = now - timedelta(days=i)
            # Skip weekends (Saturday=5, Sunday=6)
            if dt.weekday() >= 5:
                continue

            daily_return = drift + random.gauss(0, volatility)
            new_close = current_close * (1.0 + daily_return)

            # Intra-day high/low spread
            intraday_range = abs(random.gauss(0.015, 0.006))
            high = max(current_close, new_close) * (1.0 + intraday_range * 0.6)
            low = min(current_close, new_close) * (1.0 - intraday_range * 0.6)
            open_price = current_close * (1.0 + random.gauss(0, 0.004))
            volume = int(random.lognormvariate(16.5, 0.4))  # Realistic volume distribution
            vwap = (high + low + new_close) / 3.0

            bars.append(PriceBar(
                timestamp_utc=dt.replace(hour=8, minute=0, second=0, microsecond=0).isoformat(),
                open=round(open_price, 2),
                high=round(high, 2),
                low=round(low, 2),
                close=round(new_close, 2),
                volume=round(volume, 0),
                vwap=round(vwap, 2),
            ))
            current_close = new_close

        bars.sort(key=lambda b: b.timestamp_utc)
        return bars
