"""
Technical Analysis and Support/Resistance Level Calculation Engine.
Computes pivot levels, Fibonacci zones, swing extrema, and technical indicators.
Determines trade position recommendations, entry, stop-loss, and take-profit targets.
"""

from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
from market_data.price_feed import PriceBar


@dataclass
class SupportResistanceLevels:
    current_price: float
    pivot: float
    resistance_1: float
    resistance_2: float
    support_1: float
    support_2: float
    swing_high: float
    swing_low: float
    fib_382: float
    fib_618: float
    nearest_support: float = 0.0
    nearest_resistance: float = 0.0


@dataclass
class PositionRecommendation:
    status: str             # STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL
    conviction: float       # 0.0 to 1.0
    suggested_entry: float
    stop_loss: float
    take_profit_1: float
    take_profit_2: float
    risk_reward_ratio: float
    rationale: str
    reason: str = ""


class TechnicalAnalyzer:
    """Computes technical indicators and support/resistance levels from price bars."""

    @staticmethod
    def compute_support_resistance(bars: List[PriceBar]) -> SupportResistanceLevels:
        """
        Calculate key support and resistance levels using:
        1. Classic Floor Pivot Points (from recent bar)
        2. Rolling Swing Highs / Lows
        3. Fibonacci Retracement Levels
        """
        if not bars:
            return SupportResistanceLevels(
                current_price=100.0, pivot=100.0,
                resistance_1=102.0, resistance_2=105.0,
                support_1=98.0, support_2=95.0,
                swing_high=105.0, swing_low=95.0,
                fib_382=101.2, fib_618=98.8,
            )

        recent = bars[-1]
        current_price = recent.close

        # Lookback window for swing levels (up to 20 bars)
        lookback = bars[-min(20, len(bars)):]
        swing_high = max(b.high for b in lookback)
        swing_low = min(b.low for b in lookback)

        # Reference prior bar for classic pivot points
        prior = bars[-2] if len(bars) > 1 else recent
        h = prior.high
        l = prior.low
        c = prior.close

        pivot = (h + l + c) / 3.0
        r1 = 2 * pivot - l
        s1 = 2 * pivot - h
        r2 = pivot + (h - l)
        s2 = pivot - (h - l)

        # Fibonacci retracements from swing
        price_range = max(1e-4, swing_high - swing_low)
        fib_382 = swing_high - (price_range * 0.382)
        fib_618 = swing_high - (price_range * 0.618)

        # Compute nearest support and nearest resistance relative to current price
        all_levels = [r2, r1, pivot, s1, s2, swing_high, swing_low, fib_382, fib_618]
        sup_candidates = [lvl for lvl in all_levels if lvl < current_price - 1e-4]
        res_candidates = [lvl for lvl in all_levels if lvl > current_price + 1e-4]

        nearest_sup = max(sup_candidates) if sup_candidates else (s1 if s1 < current_price else current_price * 0.98)
        nearest_res = min(res_candidates) if res_candidates else (r1 if r1 > current_price else current_price * 1.02)

        return SupportResistanceLevels(
            current_price=round(current_price, 2),
            pivot=round(pivot, 2),
            resistance_1=round(r1, 2),
            resistance_2=round(r2, 2),
            support_1=round(s1, 2),
            support_2=round(s2, 2),
            swing_high=round(swing_high, 2),
            swing_low=round(swing_low, 2),
            fib_382=round(fib_382, 2),
            fib_618=round(fib_618, 2),
            nearest_support=round(nearest_sup, 2),
            nearest_resistance=round(nearest_res, 2),
        )

    @staticmethod
    def compute_technical_indicators(bars: List[PriceBar]) -> Dict[str, float]:
        """Compute core technical indicators: RSI, SMA5, SMA20, ATR, Volatility."""
        if len(bars) < 5:
            return {
                "rsi_14": 50.0, "sma_5": bars[-1].close, "sma_20": bars[-1].close,
                "atr_14": 2.0, "volatility_20": 0.02, "price_change_1d": 0.0,
            }

        closes = [b.close for b in bars]
        highs = [b.high for b in bars]
        lows = [b.low for b in bars]

        # SMA 5 & SMA 20
        sma_5 = sum(closes[-5:]) / 5.0
        sma_20 = sum(closes[-min(20, len(closes)):]) / min(20, len(closes))

        # 1-day Return
        ret_1d = (closes[-1] - closes[-2]) / closes[-2] if len(closes) > 1 else 0.0

        # RSI(14)
        rsi_window = min(14, len(closes) - 1)
        gains = []
        losses = []
        for i in range(len(closes) - rsi_window, len(closes)):
            diff = closes[i] - closes[i - 1]
            if diff >= 0:
                gains.append(diff)
                losses.append(0.0)
            else:
                gains.append(0.0)
                losses.append(abs(diff))

        avg_gain = sum(gains) / len(gains) if gains else 0.0
        avg_loss = sum(losses) / len(losses) if losses else 0.0
        if avg_loss == 0:
            rsi = 100.0 if avg_gain > 0 else 50.0
        else:
            rs = avg_gain / avg_loss
            rsi = 100.0 - (100.0 / (1.0 + rs))

        # ATR(14)
        tr_list = []
        for i in range(1, len(bars)):
            tr = max(
                bars[i].high - bars[i].low,
                abs(bars[i].high - bars[i - 1].close),
                abs(bars[i].low - bars[i - 1].close),
            )
            tr_list.append(tr)
        atr_14 = sum(tr_list[-14:]) / min(14, len(tr_list)) if tr_list else 1.0

        # Historical Volatility
        returns = [(closes[i] - closes[i - 1]) / closes[i - 1] for i in range(1, len(closes))]
        mean_ret = sum(returns) / len(returns) if returns else 0.0
        var = sum((r - mean_ret) ** 2 for r in returns) / len(returns) if returns else 0.0001
        vol_20 = math.sqrt(var)

        return {
            "rsi_14": round(rsi, 2),
            "sma_5": round(sma_5, 2),
            "sma_20": round(sma_20, 2),
            "atr_14": round(atr_14, 2),
            "volatility_20": round(vol_20, 4),
            "price_change_1d": round(ret_1d, 4),
        }

    @staticmethod
    def generate_recommendation(
        current_price: float,
        sr: SupportResistanceLevels,
        sentiment_score: float,
        tech: Dict[str, float],
        predicted_return_pct: Optional[float] = None,
    ) -> PositionRecommendation:
        """
        Generate disciplined position recommendation satisfying Task 4:
        - Stop: just beyond the nearest support (long) or resistance (short) using 0.5 to 1.0 ATR.
        - Targets: next two meaningful resistance (long) or support (short) levels beyond the nearest one,
          skipping levels closer than 1 ATR to entry.
        - Direction from combination of sentiment direction, predicted n+1 return, and technical trend.
        - Conviction score reflecting signal alignment and strength.
        - Strict HOLD rule when Risk/Reward < 1.5 or conviction < 0.35, with clear reason displayed.
        - Strictly maintains ordering invariants:
          BUY:  stop_loss < entry < take_profit_1 < take_profit_2
          SELL: take_profit_2 < take_profit_1 < entry < stop_loss
        """
        entry = round(current_price, 2)
        rsi = tech.get("rsi_14", 50.0)
        atr = max(tech.get("atr_14", entry * 0.02), entry * 0.005)
        sma5 = tech.get("sma_5", entry)
        sma20 = tech.get("sma_20", entry)

        # 1. Technical trend & momentum score (-1.0 to 1.0)
        trend_score = 0.0
        if current_price > sma20:
            trend_score += 0.35
        else:
            trend_score -= 0.35

        if sma5 > sma20:
            trend_score += 0.25
        else:
            trend_score -= 0.25

        if 40.0 <= rsi <= 65.0:
            trend_score += 0.15
        elif rsi > 72.0:
            trend_score -= 0.30  # Overbought exhaustion risk
        elif rsi < 28.0:
            trend_score += 0.25  # Oversold mean-reversion opportunity

        trend_score = max(-1.0, min(1.0, trend_score))

        # 2. Predicted n+1 return signal (-1.0 to 1.0)
        pred_signal = 0.0
        if predicted_return_pct is not None:
            pred_signal = max(-1.0, min(1.0, predicted_return_pct / 2.0))

        # 3. Composite direction signal
        if predicted_return_pct is not None:
            composite_signal = 0.45 * sentiment_score + 0.35 * pred_signal + 0.20 * trend_score
        else:
            composite_signal = 0.60 * sentiment_score + 0.40 * trend_score

        # Conviction score (0.0 to 1.0)
        base_conviction = min(0.95, round(abs(composite_signal) * 1.35, 2))
        # Penalize conviction if sentiment opposes predicted return
        if predicted_return_pct is not None and (sentiment_score * pred_signal < -0.05):
            conviction = round(base_conviction * 0.55, 2)
        else:
            conviction = base_conviction

        # Compile distinct S/R levels
        all_sr_levels = {
            sr.resistance_2, sr.resistance_1, sr.pivot,
            sr.support_1, sr.support_2, sr.swing_high,
            sr.swing_low, sr.fib_382, sr.fib_618,
            sr.nearest_support, sr.nearest_resistance,
        }
        supports_below = sorted([lvl for lvl in all_sr_levels if lvl < entry - 1e-4], reverse=True)
        resistances_above = sorted([lvl for lvl in all_sr_levels if lvl > entry + 1e-4])

        nearest_sup = supports_below[0] if supports_below else (entry - 1.5 * atr)
        nearest_res = resistances_above[0] if resistances_above else (entry + 1.5 * atr)

        # Decide candidate direction
        if composite_signal >= 0.18:
            candidate_status = "STRONG_BUY" if composite_signal >= 0.45 else "BUY"

            # Stop: just beyond nearest support using 0.75 ATR (0.5 to 1.0 ATR)
            sl = round(nearest_sup - 0.75 * atr, 2)
            if sl >= entry:
                sl = round(entry - 1.0 * atr, 2)

            # Targets: next two meaningful resistance levels skipping levels closer than 1 ATR to entry
            meaningful_res = [r for r in resistances_above if r >= entry + 1.0 * atr]
            if len(meaningful_res) >= 2:
                tp1 = round(meaningful_res[0], 2)
                tp2 = round(meaningful_res[1], 2)
            elif len(meaningful_res) == 1:
                tp1 = round(meaningful_res[0], 2)
                tp2 = round(tp1 + max(1.0 * atr, (tp1 - entry) * 0.7), 2)
            else:
                tp1 = round(entry + 2.0 * atr, 2)
                tp2 = round(entry + 3.5 * atr, 2)

            # Ensure invariant: sl < entry < tp1 < tp2
            if tp1 <= entry:
                tp1 = round(entry + 1.5 * atr, 2)
            if tp2 <= tp1:
                tp2 = round(tp1 + 1.0 * atr, 2)

            risk = entry - sl
            reward = tp1 - entry
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

            # Filter with HOLD rule
            if conviction < 0.35:
                status = "HOLD"
                reason = f"Low conviction ({conviction:.2f} < 0.35)"
            elif rr_ratio < 1.5:
                status = "HOLD"
                reason = f"R/R {rr_ratio:.2f} < 1.5"
            else:
                status = candidate_status
                reason = f"Bullish momentum & sentiment (R/R {rr_ratio:.2f} >= 1.5, Conv {conviction:.2f})"

            rationale = reason

        elif composite_signal <= -0.18:
            candidate_status = "STRONG_SELL" if composite_signal <= -0.45 else "SELL"

            # Stop: just beyond nearest resistance using 0.75 ATR (0.5 to 1.0 ATR)
            sl = round(nearest_res + 0.75 * atr, 2)
            if sl <= entry:
                sl = round(entry + 1.0 * atr, 2)

            # Targets: next two meaningful support levels skipping levels closer than 1 ATR to entry
            meaningful_sup = [s for s in supports_below if s <= entry - 1.0 * atr]
            if len(meaningful_sup) >= 2:
                tp1 = round(meaningful_sup[0], 2)
                tp2 = round(meaningful_sup[1], 2)
            elif len(meaningful_sup) == 1:
                tp1 = round(meaningful_sup[0], 2)
                tp2 = round(tp1 - max(1.0 * atr, (entry - tp1) * 0.7), 2)
            else:
                tp1 = round(entry - 2.0 * atr, 2)
                tp2 = round(entry - 3.5 * atr, 2)

            # Ensure invariant: tp2 < tp1 < entry < sl
            if tp1 >= entry:
                tp1 = round(entry - 1.5 * atr, 2)
            if tp2 >= tp1:
                tp2 = round(tp1 - 1.0 * atr, 2)

            risk = sl - entry
            reward = entry - tp1
            rr_ratio = round(reward / risk, 2) if risk > 0 else 0.0

            # Filter with HOLD rule
            if conviction < 0.35:
                status = "HOLD"
                reason = f"Low conviction ({conviction:.2f} < 0.35)"
            elif rr_ratio < 1.5:
                status = "HOLD"
                reason = f"R/R {rr_ratio:.2f} < 1.5"
            else:
                status = candidate_status
                reason = f"Bearish momentum & crowd outflow (R/R {rr_ratio:.2f} >= 1.5, Conv {conviction:.2f})"

            rationale = reason

        else:
            status = "HOLD"
            conviction = max(0.20, conviction)
            sl = round(nearest_sup - 0.75 * atr, 2)
            if sl >= entry:
                sl = round(entry - 1.0 * atr, 2)

            meaningful_res = [r for r in resistances_above if r >= entry + 1.0 * atr]
            tp1 = round(meaningful_res[0], 2) if meaningful_res else round(entry + 1.5 * atr, 2)
            if tp1 <= entry:
                tp1 = round(entry + 1.5 * atr, 2)
            tp2 = round(tp1 + 1.0 * atr, 2)

            risk = max(1e-4, entry - sl)
            reward = max(1e-4, tp1 - entry)
            rr_ratio = round(reward / risk, 2)
            reason = f"Neutral consolidation / conflicting signals (Conv {conviction:.2f})"
            rationale = reason

        return PositionRecommendation(
            status=status,
            conviction=conviction,
            suggested_entry=entry,
            stop_loss=sl,
            take_profit_1=tp1,
            take_profit_2=tp2,
            risk_reward_ratio=rr_ratio,
            rationale=rationale,
            reason=reason,
        )
