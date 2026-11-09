"""
Unit and Integration Test Suite for HengSeng Sentiment Agent.
Verifies scrapers, sentiment scoring, technical analysis, position invariants,
zero-lookahead out-of-sample validation, baseline comparisons, holdout check,
and live-mode fallback prohibition.
"""

from datetime import datetime, timezone
import math
import os
import sys
import unittest

# Ensure project root in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from scrapers.base_scraper import SocialPost
from scrapers.mock_data_generator import generate_mock_social_posts
from scrapers.telegram_scraper import TelegramScraper
from scrapers.lihkg_scraper import LihkgScraper
from scrapers.futu_moomoo_scraper import FutuMoomooScraper
from scrapers.naver_scraper import NaverFinanceScraper
from news.exa_news import ExaNewsIngestor, NewsArticle
from market_data.price_feed import PriceBar, PriceFeedManager
from models.sentiment_analyzer import FinancialSentimentAnalyzer
from models.technical_analysis import TechnicalAnalyzer, SupportResistanceLevels
from models.feature_engineering import FeaturePipeline
from models.validation import TimeSeriesCrossValidator
from models.predictor import SentimentPricePredictor


class TestHengSengSentimentPipeline(unittest.TestCase):
    """Core test suite covering all assessment requirements."""

    def setUp(self):
        self.ticker = "0700.HK"
        self.analyzer = FinancialSentimentAnalyzer()
        self.pipeline = FeaturePipeline()
        self.price_manager = PriceFeedManager()
        self._orig_live = config.LIVE_MODE
        config.LIVE_MODE = False

    def tearDown(self):
        config.LIVE_MODE = self._orig_live

    def test_social_post_schema_and_utc_alignment(self):
        """Test that SocialPost contains required fields and timestamps are UTC."""
        post = SocialPost(
            post_id="tg_12345",
            source="telegram",
            ticker=self.ticker,
            timestamp_utc="2026-11-04T08:30:00+00:00",
            text="Tencent $0700 surges past resistance!",
            author="@hkstockgains",
            author_followers=25000,
            likes=120,
            reposts=35,
            views=4500,
        )
        self.assertEqual(post.source, "telegram")
        self.assertEqual(post.ticker, "0700.HK")
        self.assertGreater(post.engagement_score, 0)
        self.assertIn("T08:30:00", post.timestamp_utc)

    def test_scrapers_mock_fallback_minimum_sources(self):
        """Test that Telegram, LIHKG, Futu, and Naver generate data without crashing in demo mode."""
        tg = TelegramScraper()
        lihkg = LihkgScraper()
        futu = FutuMoomooScraper()
        naver = NaverFinanceScraper()

        tg_posts = tg.scrape(self.ticker, days=10)
        lihkg_posts = lihkg.scrape(self.ticker, days=10)
        futu_posts = futu.scrape(self.ticker, days=10)
        naver_posts = naver.scrape("005930.KS", days=10)

        self.assertGreater(len(tg_posts), 0, "Telegram posts should not be empty")
        self.assertGreater(len(lihkg_posts), 0, "LIHKG posts should not be empty")
        self.assertGreater(len(futu_posts), 0, "Futu posts should not be empty")
        self.assertGreater(len(naver_posts), 0, "Naver posts should not be empty")

        for p in tg_posts[:5] + naver_posts[:5]:
            self.assertTrue(p.timestamp_utc.endswith("+00:00") or p.timestamp_utc.endswith("Z"))
            self.assertTrue(p.is_mock)
            self.assertIn("example.com", p.url)

    def test_sentiment_analyzer_multilingual(self):
        """Test sentiment scoring on English, Traditional Chinese (Cantonese), Mandarin, and Korean."""
        score_en, dir_en = self.analyzer.analyze_text("Tencent earnings beat expectations, stock surges strongly with huge buyback!")
        self.assertGreater(score_en, 0.20)
        self.assertEqual(dir_en, "BULLISH")

        score_hk, dir_hk = self.analyzer.analyze_text("騰訊0700業績大好，突破阻力，睇好爆升！")
        self.assertGreater(score_hk, 0.20)
        self.assertEqual(dir_hk, "BULLISH")

        score_kr, dir_kr = self.analyzer.analyze_text("삼성전자 HBM3E 공급 승인 호재! 외인 순매수 유입 가즈아 매수 추천!")
        self.assertGreater(score_kr, 0.20)
        self.assertEqual(dir_kr, "BULLISH")

        score_neg, dir_neg = self.analyzer.analyze_text("阿里業績大幅不及預期，穿底暴跌，止蝕割肉，睇淡。")
        self.assertLess(score_neg, -0.20)
        self.assertEqual(dir_neg, "BEARISH")

        score_kr_neg, dir_kr_neg = self.analyzer.analyze_text("삼전 외인 대량 매도 폭락, 지지선 붕괴 손절 개미털기.")
        self.assertLess(score_kr_neg, -0.20)
        self.assertEqual(dir_kr_neg, "BEARISH")

    def test_technical_support_resistance_and_indicators(self):
        """Test support/resistance calculations (Pivots, Fib, Swings) and nearest levels."""
        bars = self.price_manager.get_price_bars(self.ticker, days=config.PRICE_LOOKBACK_DAYS)
        self.assertGreaterEqual(len(bars), 120, "Should generate at least 120 trading days")

        sr = TechnicalAnalyzer.compute_support_resistance(bars)
        self.assertGreater(sr.resistance_1, sr.support_1)
        self.assertGreater(sr.resistance_2, sr.resistance_1)
        self.assertLess(sr.support_2, sr.support_1)
        self.assertGreater(sr.nearest_resistance, 0.0)
        self.assertGreater(sr.nearest_support, 0.0)

        tech = TechnicalAnalyzer.compute_technical_indicators(bars)
        self.assertIn("rsi_14", tech)
        self.assertTrue(0.0 <= tech["rsi_14"] <= 100.0)

    def test_position_ordering_invariants_across_all_tickers(self):
        """
        Verify Task 4 ordering invariants:
        - For BUY / STRONG_BUY: stop_loss < entry < take_profit_1 < take_profit_2 and risk/reward >= 1.5
        - For SELL / STRONG_SELL: take_profit_2 < take_profit_1 < entry < stop_loss and risk/reward >= 1.5
        - If risk/reward < 1.5: status must be HOLD
        Tested for all 10 target assets in HKEX and KRX.
        """
        all_assets = {**config.HKEX_ASSETS, **config.KRX_ASSETS}

        for ticker, asset_cfg in all_assets.items():
            bars = self.price_manager.get_price_bars(ticker, days=config.PRICE_LOOKBACK_DAYS)
            sr = TechnicalAnalyzer.compute_support_resistance(bars)
            tech = TechnicalAnalyzer.compute_technical_indicators(bars)

            # Test Bullish scenario
            pos_buy = TechnicalAnalyzer.generate_recommendation(
                current_price=sr.current_price,
                sr=sr,
                sentiment_score=0.75,
                tech=tech,
            )
            if pos_buy.status in ("BUY", "STRONG_BUY"):
                self.assertLess(
                    pos_buy.stop_loss,
                    pos_buy.suggested_entry,
                    f"Invariant violated for {ticker}: stop_loss ({pos_buy.stop_loss}) >= entry ({pos_buy.suggested_entry})",
                )
                self.assertLess(
                    pos_buy.suggested_entry,
                    pos_buy.take_profit_1,
                    f"Invariant violated for {ticker}: entry ({pos_buy.suggested_entry}) >= TP1 ({pos_buy.take_profit_1})",
                )
                self.assertLess(
                    pos_buy.take_profit_1,
                    pos_buy.take_profit_2,
                    f"Invariant violated for {ticker}: TP1 ({pos_buy.take_profit_1}) >= TP2 ({pos_buy.take_profit_2})",
                )
                self.assertGreaterEqual(
                    pos_buy.risk_reward_ratio,
                    1.5,
                    f"Risk/reward ratio ({pos_buy.risk_reward_ratio}) < 1.5 on {pos_buy.status} for {ticker}",
                )

            # Test Bearish scenario
            pos_sell = TechnicalAnalyzer.generate_recommendation(
                current_price=sr.current_price,
                sr=sr,
                sentiment_score=-0.75,
                tech=tech,
            )
            if pos_sell.status in ("SELL", "STRONG_SELL"):
                self.assertGreater(
                    pos_sell.stop_loss,
                    pos_sell.suggested_entry,
                    f"Invariant violated for {ticker}: stop_loss ({pos_sell.stop_loss}) <= entry ({pos_sell.suggested_entry})",
                )
                self.assertGreater(
                    pos_sell.suggested_entry,
                    pos_sell.take_profit_1,
                    f"Invariant violated for {ticker}: entry ({pos_sell.suggested_entry}) <= TP1 ({pos_sell.take_profit_1})",
                )
                self.assertGreater(
                    pos_sell.take_profit_1,
                    pos_sell.take_profit_2,
                    f"Invariant violated for {ticker}: TP1 ({pos_sell.take_profit_1}) <= TP2 ({pos_sell.take_profit_2})",
                )
                self.assertGreaterEqual(
                    pos_sell.risk_reward_ratio,
                    1.5,
                    f"Risk/reward ratio ({pos_sell.risk_reward_ratio}) < 1.5 on {pos_sell.status} for {ticker}",
                )

    def test_live_mode_raises_error_without_silent_fallbacks(self):
        """Verify Task 1: in live mode, missing keys or empty scraping results raise clear RuntimeError."""
        config.LIVE_MODE = True
        tg = TelegramScraper()
        lihkg = LihkgScraper()
        news = ExaNewsIngestor()

        # Without credentials, scraping in live mode must raise RuntimeError naming the source
        if not config.APIFY_API_KEY:
            with self.assertRaises(RuntimeError) as ctx_tg:
                tg.scrape(self.ticker, days=10)
            self.assertIn("Live mode error", str(ctx_tg.exception))
            self.assertIn("Telegram", str(ctx_tg.exception))

            with self.assertRaises(RuntimeError) as ctx_lihkg:
                lihkg.scrape(self.ticker, days=10)
            self.assertIn("Live mode error", str(ctx_lihkg.exception))
            self.assertIn("LIHKG", str(ctx_lihkg.exception))

        if not config.EXA_API_KEY:
            with self.assertRaises(RuntimeError) as ctx_news:
                news.fetch_news(self.ticker, days=10)
            self.assertIn("Live mode error", str(ctx_news.exception))
            self.assertIn("Exa", str(ctx_news.exception))

    def test_sparse_validation_returns_insufficient_data_without_fake_metrics(self):
        """Verify Task 1: n_samples < 8 returns insufficient_data=True and no fake metrics (0.62, 1.45, etc.)."""
        sparse_X = [[0.01, 0.02], [0.03, -0.01], [0.02, 0.01]]
        sparse_y = [0.01, -0.02, 0.005]

        report = TimeSeriesCrossValidator.evaluate(
            model_trainer=lambda X, y: None,
            model_predictor=lambda m, X: [0.0] * len(X),
            X=sparse_X,
            y=sparse_y,
            ticker="SPARSE.HK",
        )
        self.assertTrue(report.insufficient_data)
        self.assertIsNone(report.mean_directional_accuracy)
        self.assertIsNone(report.mean_information_coefficient)
        self.assertIsNone(report.strategy_sharpe_ratio)
        self.assertEqual(report.oos_samples_count, 0)

    def test_mock_generator_hk_krx_channel_isolation(self):
        """Verify Task 2: HK tickers never receive posts from @krx_traders or @yeouido_insider."""
        hk_posts = generate_mock_social_posts("0700.HK", days=10)
        for p in hk_posts:
            if p.source == "telegram":
                self.assertNotIn(p.author, ["@krx_traders", "@yeouido_insider"])
                self.assertIn(p.author, ["@hkstockgains", "@canton_trader", "@alpha_hangseng", "@hongkong_equity_flow", "@hsi_alpha"])

        krx_posts = generate_mock_social_posts("005930.KS", days=10)
        for p in krx_posts:
            if p.source == "telegram":
                self.assertIn(p.author, ["@krx_traders", "@yeouido_insider", "@korea_value_up", "@seoul_quant"])

    def test_feature_pipeline_alignment_and_creative_metrics(self):
        """Test feature engineering: velocity, buzz z-score, divergence, has_social."""
        tg = TelegramScraper()
        lihkg = LihkgScraper()
        news = ExaNewsIngestor().fetch_news(self.ticker, days=10)
        bars = self.price_manager.get_price_bars(self.ticker, days=config.PRICE_LOOKBACK_DAYS)

        posts = tg.scrape(self.ticker, days=10) + lihkg.scrape(self.ticker, days=10)
        summary, records = self.pipeline.process_ticker(self.ticker, posts, news, bars)

        self.assertEqual(summary.ticker, self.ticker)
        self.assertIn(summary.sentiment_direction, ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertIsNotNone(summary.sentiment_velocity)
        self.assertIsNotNone(summary.buzz_zscore)
        self.assertIn(summary.position.status, ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL"])

        # Check records alignment and has_social feature
        self.assertGreater(len(records), 0)
        # Recent 10 days should have has_social == 1.0, earlier days should have has_social == 0.0
        has_social_values = [r.has_social for r in records]
        self.assertIn(1.0, has_social_values)
        self.assertIn(0.0, has_social_values)

        for r in records[:-1]:
            self.assertIsNotNone(r.target_next_return)
            self.assertIsNotNone(r.target_next_close)

    def test_walk_forward_validation_zero_leakage(self):
        """Verify that out-of-sample validation splits never leak future data into training."""
        splits = TimeSeriesCrossValidator.walk_forward_split(n_samples=50, n_splits=4, min_train_size=15)
        self.assertEqual(len(splits), 4)

        for train_idx, test_idx in splits:
            self.assertLess(max(train_idx), min(test_idx), "Future test data found inside training split!")
            self.assertTrue(len(train_idx) > 0)
            self.assertTrue(len(test_idx) > 0)

    def test_predictive_model_and_forecast_interval(self):
        """Verify n+1 predictor generates point forecast, valid confidence interval, and baselines."""
        posts = generate_mock_social_posts(self.ticker, days=10)
        news = ExaNewsIngestor().fetch_news(self.ticker, days=10)
        bars = self.price_manager.get_price_bars(self.ticker, days=config.PRICE_LOOKBACK_DAYS)

        summary, records = self.pipeline.process_ticker(self.ticker, posts, news, bars)
        predictor = SentimentPricePredictor(alpha=0.5)

        pred_out = predictor.train_and_forecast(self.ticker, records, summary.current_price)
        self.assertIsNotNone(pred_out.predicted_price)
        self.assertGreater(pred_out.predicted_price, 0)
        self.assertLess(pred_out.confidence_interval_lower, pred_out.predicted_price)
        self.assertGreater(pred_out.confidence_interval_upper, pred_out.predicted_price)
        self.assertIn(pred_out.predicted_direction, ["BULLISH", "BEARISH", "NEUTRAL"])
        self.assertIsNotNone(pred_out.validation_report)

        # Baseline verification
        vr = pred_out.validation_report
        self.assertFalse(vr.insufficient_data)
        self.assertIsNotNone(vr.baselines)
        self.assertGreater(vr.baselines.zero_return_rmse, 0)
        self.assertGreater(vr.baselines.price_only_rmse, 0)

        # Holdout verification
        self.assertGreater(len(vr.recent_k_predictions), 0)

    def test_realistic_position_fixtures_and_hold_reasons(self):
        """
        Task 4: Test target and stop selection on realistic synthetic fixtures:
        - Nearest level closer than 1 ATR to entry must be skipped for TP1/TP2.
        - Stop loss placed 0.5 to 1.0 ATR beyond nearest S/R level.
        - When risk/reward < 1.5, position must be HOLD with reason string indicating R/R.
        - Invariants strictly maintained:
            BUY: stop_loss < entry < tp1 < tp2
            SELL: tp2 < tp1 < entry < stop_loss
        """
        # Realistic fixture: Entry = 100.0, ATR = 2.0
        # Levels: resistances at [101.0, 104.0, 108.0], supports at [98.0, 95.0, 90.0]
        # Level 101.0 is only 1.0 away (< 1.0 ATR), so target selection MUST skip 101.0
        # TP1 should be 104.0 (4.0 reward) or higher.
        # Nearest support is 98.0. Stop loss beyond support at 98.0 - 0.75 * 2.0 = 96.5 (3.5 risk).
        # R/R = 4.0 / 3.5 = 1.14 < 1.5. This MUST trigger HOLD with reason!
        sr_tight = SupportResistanceLevels(
            current_price=100.0,
            pivot=100.0,
            resistance_1=101.0,
            resistance_2=104.0,
            support_1=98.0,
            support_2=95.0,
            swing_high=108.0,
            swing_low=90.0,
            fib_382=101.0,
            fib_618=95.0,
            nearest_support=98.0,
            nearest_resistance=101.0,
        )
        tech_tight = {"atr_14": 2.0, "rsi_14": 55.0, "sma_5": 101.0, "sma_20": 99.0}

        rec_tight = TechnicalAnalyzer.generate_recommendation(
            current_price=100.0,
            sr=sr_tight,
            sentiment_score=0.80,
            tech=tech_tight,
            predicted_return_pct=2.5,
        )
        self.assertEqual(rec_tight.status, "HOLD")
        self.assertIn("R/R", rec_tight.reason)
        self.assertIn("< 1.5", rec_tight.reason)

        # Now fixture with adequate spacing: resistances at [106.0, 112.0], nearest resistance = 106.0 (> 1 ATR)
        # Reward = 6.0, Risk = 3.5 -> R/R = 1.71 >= 1.5.
        sr_favorable = SupportResistanceLevels(
            current_price=100.0,
            pivot=100.0,
            resistance_1=106.0,
            resistance_2=112.0,
            support_1=98.0,
            support_2=95.0,
            swing_high=115.0,
            swing_low=92.0,
            fib_382=106.0,
            fib_618=95.0,
            nearest_support=98.0,
            nearest_resistance=106.0,
        )
        rec_buy = TechnicalAnalyzer.generate_recommendation(
            current_price=100.0,
            sr=sr_favorable,
            sentiment_score=0.85,
            tech=tech_tight,
            predicted_return_pct=3.5,
        )
        self.assertIn(rec_buy.status, ["BUY", "STRONG_BUY"])
        self.assertLess(rec_buy.stop_loss, rec_buy.suggested_entry)
        self.assertLess(rec_buy.suggested_entry, rec_buy.take_profit_1)
        self.assertLess(rec_buy.take_profit_1, rec_buy.take_profit_2)
        self.assertGreaterEqual(rec_buy.risk_reward_ratio, 1.5)

        # Favorable SELL fixture: supports at [93.0, 86.0], nearest resistance = 102.0
        # Stop loss at 102.0 + 1.5 = 103.5 (Risk = 3.5). TP1 = 93.0 (Reward = 7.0). R/R = 2.0 >= 1.5.
        sr_sell = SupportResistanceLevels(
            current_price=100.0,
            pivot=100.0,
            resistance_1=102.0,
            resistance_2=108.0,
            support_1=93.0,
            support_2=86.0,
            swing_high=110.0,
            swing_low=85.0,
            fib_382=102.0,
            fib_618=93.0,
            nearest_support=93.0,
            nearest_resistance=102.0,
        )
        tech_bear = {"atr_14": 2.0, "rsi_14": 42.0, "sma_5": 98.0, "sma_20": 102.0}
        rec_sell = TechnicalAnalyzer.generate_recommendation(
            current_price=100.0,
            sr=sr_sell,
            sentiment_score=-0.85,
            tech=tech_bear,
            predicted_return_pct=-3.5,
        )
        self.assertIn(rec_sell.status, ["SELL", "STRONG_SELL"])
        self.assertGreater(rec_sell.stop_loss, rec_sell.suggested_entry)
        self.assertGreater(rec_sell.suggested_entry, rec_sell.take_profit_1)
        self.assertGreater(rec_sell.take_profit_1, rec_sell.take_profit_2)
        self.assertGreaterEqual(rec_sell.risk_reward_ratio, 1.5)

    def test_strict_zero_lookahead_feature_isolation(self):
        """
        Task 5: Test that fails if ANY feature for day t uses data from day t+1 or later.
        Causality test:
        1. Build price bars for T days.
        2. Compute features for day t (t < T).
        3. Mutate future bars (t+1 .. T) with extreme outliers (10x price jump, volume spike).
        4. Recompute features for day t.
        5. Assert that ALL feature values for day t are perfectly identical.
        """
        base_bars = self.price_manager.get_price_bars(self.ticker, days=50)
        self.assertGreaterEqual(len(base_bars), 30)

        t_idx = 20  # Evaluate at day t = 20
        # Case A: un-mutated sequence
        _, records_original = self.pipeline.process_ticker(
            self.ticker,
            social_posts=[],
            news_articles=[],
            price_bars=base_bars,
        )
        rec_t_orig = records_original[t_idx]

        # Case B: Mutate future bars (t_idx + 1 onward) with extreme values
        import copy
        mutated_bars = copy.deepcopy(base_bars)
        for i in range(t_idx + 1, len(mutated_bars)):
            mutated_bars[i].close *= 10.0
            mutated_bars[i].high *= 10.0
            mutated_bars[i].low *= 10.0
            mutated_bars[i].open *= 10.0
            mutated_bars[i].volume *= 100.0

        _, records_mutated = self.pipeline.process_ticker(
            self.ticker,
            social_posts=[],
            news_articles=[],
            price_bars=mutated_bars,
        )
        rec_t_mutated = records_mutated[t_idx]

        # Assert strict equality for all historical features on day t
        self.assertEqual(rec_t_orig.close_price, rec_t_mutated.close_price)
        self.assertEqual(rec_t_orig.returns_1d, rec_t_mutated.returns_1d)
        self.assertEqual(rec_t_orig.rsi_14, rec_t_mutated.rsi_14)
        self.assertEqual(rec_t_orig.sma5_sma20_ratio, rec_t_mutated.sma5_sma20_ratio)
        self.assertEqual(rec_t_orig.volatility, rec_t_mutated.volatility)
        self.assertEqual(rec_t_orig.sentiment_velocity, rec_t_mutated.sentiment_velocity)
        self.assertEqual(rec_t_orig.buzz_zscore, rec_t_mutated.buzz_zscore)
        self.assertEqual(rec_t_orig.sentiment_price_divergence, rec_t_mutated.sentiment_price_divergence)

        # Conversely, the target return for day t SHOULD reflect day t+1
        self.assertNotEqual(rec_t_orig.target_next_return, rec_t_mutated.target_next_return)
        self.assertNotEqual(rec_t_orig.target_next_close, rec_t_mutated.target_next_close)


if __name__ == "__main__":
    unittest.main()
