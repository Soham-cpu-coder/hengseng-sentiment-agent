"""
Multilingual Financial Sentiment Analysis Engine.
Specialized for Hong Kong and Chinese retail equity trading terminology (Cantonese & Mandarin)
alongside English quantitative financial terminology.
"""

import math
import re
from typing import Dict, List, Optional, Tuple

# Comprehensive Financial Sentiment Lexicons
FINANCIAL_POSITIVE_LEXICON: Dict[str, float] = {
    # English terms
    "bull": 0.7, "bullish": 0.85, "rally": 0.75, "surge": 0.85, "breakout": 0.8,
    "beat": 0.75, "outperform": 0.8, "profit": 0.7, "earnings beat": 0.9,
    "buyback": 0.75, "dividend": 0.65, "upgrade": 0.75, "target raised": 0.8,
    "growth": 0.6, "all-time high": 0.9, "ath": 0.85, "accumulate": 0.65,
    "record high": 0.9, "moon": 0.75, "long": 0.6, "undervalued": 0.65,

    # Traditional Chinese / Cantonese terms (HK retail market)
    "睇好": 0.85, "爆升": 0.95, "破頂": 0.9, "抄底": 0.8, "大升": 0.85,
    "勁賺": 0.9, "增長": 0.65, "利好": 0.8, "回購": 0.75, "買入": 0.8,
    "牛市": 0.85, "升浪": 0.8, "突破": 0.75, "衝高": 0.7, "好消息": 0.7,
    "夾淡倉": 0.85, "護盤": 0.65, "北水買入": 0.8, "大摩看好": 0.75,
    "長線持有": 0.7, "高息": 0.65, "定海神針": 0.75, "逆天": 0.85,
    "超預期": 0.85, "破紀錄": 0.9, "坐穩扶好": 0.7, "牛軍": 0.8,

    # Simplified Chinese terms
    "看多": 0.8, "大涨": 0.85, "暴涨": 0.95, "加仓": 0.75, "利好": 0.8,
    "超买": 0.5, "净流入": 0.7, "重磅": 0.6, "涨停": 0.9, "扭亏为盈": 0.85,

    # Korean financial terms (KRX retail market)
    "매수": 0.8, "상한가": 0.95, "급등": 0.85, "돌파": 0.8, "반등": 0.75,
    "호재": 0.85, "흑자전환": 0.85, "실적호조": 0.8, "외인매수": 0.8, "기관매수": 0.75,
    "순매수": 0.7, "가즈아": 0.85, "신고가": 0.9, "상승": 0.65, "목표가상향": 0.8,
}

FINANCIAL_NEGATIVE_LEXICON: Dict[str, float] = {
    # English terms
    "bear": -0.7, "bearish": -0.85, "crash": -0.95, "dump": -0.85, "breakdown": -0.8,
    "miss": -0.75, "underperform": -0.75, "loss": -0.75, "slump": -0.7,
    "downgrade": -0.8, "selloff": -0.85, "sell": -0.7, "target cut": -0.8,
    "lawsuit": -0.7, "investigation": -0.75, "tariff": -0.65, "overvalued": -0.65,
    "short": -0.65, "scam": -0.9, "bankruptcy": -0.95, "caution": -0.4,

    # Traditional Chinese / Cantonese terms (HK retail market)
    "睇淡": -0.85, "暴跌": -0.95, "穿底": -0.9, "沽空": -0.8, "割肉": -0.85,
    "跌停": -0.9, "虧損": -0.8, "利空": -0.85, "減持": -0.75, "止蝕": -0.7,
    "熊市": -0.85, "暴瀉": -0.95, "破產": -0.95, "下跌": -0.6, "大陰燭": -0.8,
    "廢柴": -0.85, "踩雷": -0.9, "陰跌": -0.75, "垃圾": -0.8, "假突破": -0.7,
    "頂背馳": -0.75, "頂背離": -0.75, "大戶派貨": -0.85, "千祈咪抄底": -0.8,
    "回調": -0.4, "淡友": -0.75, "斬倉": -0.9,

    # Simplified Chinese terms
    "看空": -0.8, "大跌": -0.85, "崩盘": -0.95, "清仓": -0.85, "暴雷": -0.95,
    "净流出": -0.7, "跌破": -0.75, "被套": -0.8,

    # Korean financial terms (KRX retail market)
    "매도": -0.8, "하한가": -0.95, "급락": -0.85, "폭락": -0.95, "손절": -0.85,
    "악재": -0.85, "적자": -0.8, "개미털기": -0.6, "물림": -0.75, "순매도": -0.7,
    "외인매도": -0.8, "하락": -0.6, "신저가": -0.9, "목표가하향": -0.8,
}

INTENSIFIERS: Dict[str, float] = {
    "very": 1.4, "extremely": 1.8, "strongly": 1.6, "super": 1.5,
    "極度": 1.8, "超級": 1.7, "非常": 1.5, "好": 1.3, "勁": 1.6,
    "真係": 1.4, "特別": 1.4, "巨額": 1.5, "严重": 1.6, "极度": 1.8,
}

NEGATORS: List[str] = [
    "not", "no", "never", "dont", "don't", "wont", "won't", "neither",
    "唔", "不", "無", "没有", "咪", "咪去", "未", "無理由",
]


class FinancialSentimentAnalyzer:
    """Calculates granular sentiment scores [-1.0, 1.0] and direction for financial text."""

    def __init__(self):
        self.pos_words = FINANCIAL_POSITIVE_LEXICON
        self.neg_words = FINANCIAL_NEGATIVE_LEXICON
        self.intensifiers = INTENSIFIERS
        self.negators = NEGATORS

    def analyze_text(self, text: str) -> Tuple[float, str]:
        """
        Analyze financial text and return (sentiment_score, direction_label).
        Score is normalized to [-1.0, +1.0].
        Direction is one of 'BULLISH', 'BEARISH', 'NEUTRAL'.
        """
        if not text:
            return 0.0, "NEUTRAL"

        clean_lower = text.lower()
        score = 0.0
        word_count = 0

        # Check positive terms
        for word, val in self.pos_words.items():
            pattern = re.escape(word.lower())
            matches = list(re.finditer(pattern, clean_lower))
            for m in matches:
                multiplier = self._check_modifiers(clean_lower, m.start(), is_positive=True)
                score += val * multiplier
                word_count += 1

        # Check negative terms
        for word, val in self.neg_words.items():
            pattern = re.escape(word.lower())
            matches = list(re.finditer(pattern, clean_lower))
            for m in matches:
                multiplier = self._check_modifiers(clean_lower, m.start(), is_positive=False)
                score += val * multiplier
                word_count += 1

        if word_count == 0:
            # Baseline neutral or subtle bias from general terms
            normalized_score = 0.0
        else:
            # S-curve normalization (hyperbolic tangent scaling)
            normalized_score = math.tanh(score / max(1.0, math.sqrt(word_count)))

        normalized_score = round(max(-1.0, min(1.0, normalized_score)), 4)

        if normalized_score > 0.15:
            label = "BULLISH"
        elif normalized_score < -0.15:
            label = "BEARISH"
        else:
            label = "NEUTRAL"

        return normalized_score, label

    def _check_modifiers(self, text: str, word_start_idx: int, is_positive: bool = True) -> float:
        """Inspect preceding immediate tokens within the same clause for intensifiers or negations."""
        # Only inspect up to 10 characters / words before, and stop at punctuation
        raw_preceding = text[max(0, word_start_idx - 12):word_start_idx]
        # Split on clause punctuation boundaries (, . ! ? ， 。 ； ; !)
        clauses = re.split(r"[,.!?，。；;!\n]+", raw_preceding)
        lookback_window = clauses[-1].strip() if clauses else ""

        multiplier = 1.0

        # Check negation within immediate clause
        for neg in self.negators:
            if re.search(rf"\b{re.escape(neg)}\b", lookback_window) or neg in lookback_window:
                if is_positive:
                    multiplier *= -0.7  # "not good" -> negative
                else:
                    multiplier *= 0.4   # "not bad / not falling" -> neutral/mild
                break

        # Check intensifiers
        for intens, factor in self.intensifiers.items():
            if intens in lookback_window:
                multiplier *= factor
                break

        return multiplier
