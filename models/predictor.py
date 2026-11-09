"""
Predictive Modeling Engine for Horizon n+1 Return and Price Forecasting.
Fits multivariate regularized linear models (Ridge regression) on engineered
sentiment velocity, buzz anomalies, and technical momentum factors.
Computes point forecasts, factor attribution coefficients, 95% confidence intervals,
baseline comparisons, and holdout prediction vs actual realized returns.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Dict, List, Optional, Tuple

from models.feature_engineering import DailyFeatureRecord
from models.validation import TimeSeriesCrossValidator, ValidationReport, PredictionActualBar
import config


@dataclass
class PredictionOutput:
    ticker: str
    current_price: float
    predicted_return_pct: float
    predicted_price: float
    confidence_interval_lower: float
    confidence_interval_upper: float
    predicted_direction: str                 # "BULLISH", "BEARISH", "NEUTRAL"
    feature_importance: Dict[str, float] = field(default_factory=dict)
    validation_report: Optional[ValidationReport] = None


class SentimentPricePredictor:
    """Predicts next-bar asset return R_{t+1} and price P_{t+1} via regularized Ridge regression."""

    FEATURE_NAMES = [
        "returns_1d",
        "rsi_normalized",
        "sma_ratio_spread",
        "volatility",
        "has_social",
        "has_news",
        "raw_sentiment_mean",
        "engagement_weighted_sentiment",
        "news_sentiment_mean",
        "blended_sentiment",
        "sentiment_velocity",
        "buzz_zscore",
        "sentiment_price_divergence",
    ]
    # Price-only feature indices for baseline comparison
    PRICE_ONLY_INDICES = [0, 1, 2, 3]

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.weights: List[float] = []
        self.intercept: float = 0.0
        self.residual_std: float = 0.015
        self.feature_means: List[float] = []
        self.feature_stds: List[float] = []
        self._sklearn_model = None

    def _extract_feature_vector(self, rec: DailyFeatureRecord) -> List[float]:
        """Convert a single record into standardized feature vector, handling missing sentiment explicitly."""
        rsi_norm = (rec.rsi_14 - 50.0) / 50.0
        sma_spread = rec.sma5_sma20_ratio - 1.0

        raw_sent = 0.0 if (rec.raw_sentiment_mean is None or math.isnan(rec.raw_sentiment_mean)) else rec.raw_sentiment_mean
        eng_sent = 0.0 if (rec.engagement_weighted_sentiment is None or math.isnan(rec.engagement_weighted_sentiment)) else rec.engagement_weighted_sentiment
        news_sent = 0.0 if (rec.news_sentiment_mean is None or math.isnan(rec.news_sentiment_mean)) else rec.news_sentiment_mean
        blended = 0.0 if (rec.blended_sentiment is None or math.isnan(rec.blended_sentiment)) else rec.blended_sentiment

        return [
            rec.returns_1d,
            rsi_norm,
            sma_spread,
            rec.volatility,
            float(rec.has_social),
            float(rec.has_news),
            raw_sent,
            eng_sent,
            news_sent,
            blended,
            rec.sentiment_velocity,
            rec.buzz_zscore,
            float(rec.sentiment_price_divergence),
        ]

    def extract_matrix(
        self,
        records: List[DailyFeatureRecord],
    ) -> Tuple[List[List[float]], List[float]]:
        """Construct feature matrix X and target return vector y from aligned records."""
        X: List[List[float]] = []
        y: List[float] = []

        for rec in records:
            if rec.target_next_return is None:
                continue
            feats = self._extract_feature_vector(rec)
            X.append(feats)
            y.append(rec.target_next_return)

        return X, y

    def extract_latest_feature_vector(self, rec: DailyFeatureRecord) -> List[float]:
        """Extract feature vector for the terminal bar t to forecast t+1."""
        return self._extract_feature_vector(rec)

    def fit(self, X: List[List[float]], y: List[float]) -> "SentimentPricePredictor":
        """
        Fit Ridge regression model on feature matrix X and target return vector y.
        Standardizes features and estimates residual standard error for confidence intervals.
        """
        if not X or not y:
            return self

        n_samples = len(X)
        n_features = len(X[0])

        # Feature standardization statistics
        self.feature_means = [sum(X[i][j] for i in range(n_samples)) / n_samples for j in range(n_features)]
        self.feature_stds = []
        for j in range(n_features):
            var = sum((X[i][j] - self.feature_means[j]) ** 2 for i in range(n_samples)) / max(1, n_samples)
            std = math.sqrt(var)
            self.feature_stds.append(std if std > 1e-6 else 1.0)

        # 1. Attempt scikit-learn Ridge regression
        try:
            from sklearn.linear_model import Ridge
            import numpy as np

            X_arr = np.array(X, dtype=np.float64)
            y_arr = np.array(y, dtype=np.float64)

            means = np.mean(X_arr, axis=0)
            stds = np.std(X_arr, axis=0)
            stds[stds < 1e-6] = 1.0
            X_scaled = (X_arr - means) / stds

            with np.errstate(all="ignore"):
                ridge = Ridge(alpha=max(1.0, self.alpha), fit_intercept=True)
                ridge.fit(X_scaled, y_arr)

            self.intercept = float(ridge.intercept_)
            self.weights = [float(w) for w in ridge.coef_]
            self.feature_means = [float(m) for m in means]
            self.feature_stds = [float(s) for s in stds]
            self._sklearn_model = ridge

            with np.errstate(all="ignore"):
                preds = ridge.predict(X_scaled)
            dof = max(1, n_samples - n_features - 1)
            residuals = y_arr - preds
            self.residual_std = float(math.sqrt(np.sum(residuals ** 2) / dof))
            return self

        except Exception:
            pass

        # 2. Pure Python gradient descent solver fallback
        X_scaled = [
            [(X[i][j] - self.feature_means[j]) / self.feature_stds[j] for j in range(n_features)]
            for i in range(n_samples)
        ]

        mean_y = sum(y) / n_samples
        y_centered = [yi - mean_y for yi in y]
        self.intercept = mean_y

        weights = [0.0] * n_features
        lr = 0.05
        epochs = 350

        for _ in range(epochs):
            grad = [0.0] * n_features
            for i in range(n_samples):
                pred = sum(weights[j] * X_scaled[i][j] for j in range(n_features))
                err = pred - y_centered[i]
                for j in range(n_features):
                    grad[j] += err * X_scaled[i][j]

            for j in range(n_features):
                grad[j] = (grad[j] / n_samples) + (self.alpha * weights[j])
                weights[j] -= lr * grad[j]

        self.weights = weights

        preds = self.predict(X)
        residuals = [yi - pi for yi, pi in zip(y, preds)]
        dof = max(1, n_samples - n_features - 1)
        var_res = sum(r ** 2 for r in residuals) / dof
        self.residual_std = math.sqrt(var_res) if var_res > 0 else 0.015

        return self

    def predict(self, X: List[List[float]]) -> List[float]:
        """Predict target returns for input feature matrix X."""
        if not self.weights or not X:
            return [0.0] * len(X)

        n_features = len(self.weights)
        preds = []
        for row in X:
            scaled_row = [
                (row[j] - self.feature_means[j]) / self.feature_stds[j]
                if j < len(self.feature_means) else row[j]
                for j in range(n_features)
            ]
            pred = self.intercept + sum(w * x for w, x in zip(self.weights, scaled_row))
            preds.append(pred)
        return preds

    def train_and_forecast(
        self,
        ticker: str,
        records: List[DailyFeatureRecord],
        current_price: float,
    ) -> PredictionOutput:
        """
        Orchestrate walk-forward validation, baseline comparison, and final horizon n+1 projection.
        Returns point estimate, factor coefficients, and confidence bounds.
        """
        X, y = self.extract_matrix(records)

        def trainer(X_tr, y_tr):
            m = SentimentPricePredictor(alpha=self.alpha)
            m.fit(X_tr, y_tr)
            return m

        def predictor(model, X_te):
            return model.predict(X_te)

        val_report = TimeSeriesCrossValidator.evaluate(
            model_trainer=trainer,
            model_predictor=predictor,
            X=X,
            y=y,
            ticker=ticker,
            model_name="Ridge-Sentiment-Microstructure",
            n_splits=config.WALK_FORWARD_SPLITS,
            price_feature_indices=self.PRICE_ONLY_INDICES,
        )

        # Hold out the last K bars for prediction vs actual realized check (Task 5)
        # Using the last K bars (K = 5 to 10)
        labeled_records = [r for r in records if r.target_next_return is not None and r.target_next_close is not None]
        k = 10 if len(labeled_records) >= 30 else min(5, max(1, len(labeled_records) - 10))
        recent_k_bars = []
        if len(labeled_records) >= k + 10:
            for i in range(len(labeled_records) - k, len(labeled_records)):
                train_slice = labeled_records[:i]
                eval_rec = labeled_records[i]
                X_tr, y_tr = self.extract_matrix(train_slice)
                if len(X_tr) >= 8:
                    eval_model = SentimentPricePredictor(alpha=self.alpha)
                    eval_model.fit(X_tr, y_tr)
                    eval_x = [self.extract_latest_feature_vector(eval_rec)]
                    pred_ret = eval_model.predict(eval_x)[0]
                    pred_close = eval_rec.close_price * (1.0 + pred_ret)
                    realized_close = eval_rec.target_next_close
                    realized_ret = eval_rec.target_next_return

                    dir_correct = (pred_ret >= 0 and realized_ret >= 0) or (pred_ret < 0 and realized_ret < 0)
                    err_pct = round(abs(pred_close - realized_close) / realized_close * 100.0, 2) if realized_close > 0 else 0.0
                    recent_k_bars.append(PredictionActualBar(
                        date_str=eval_rec.date_str,
                        realized_close=round(realized_close, 2),
                        predicted_close=round(pred_close, 2),
                        realized_return_pct=round(realized_ret * 100.0, 2),
                        predicted_return_pct=round(pred_ret * 100.0, 2),
                        direction_correct=dir_correct,
                        error_pct=err_pct,
                    ))

        if val_report:
            val_report.recent_k_predictions = recent_k_bars

        # Fit model on all available data for final n+1 forecast
        self.fit(X, y)

        latest_rec = records[-1]
        latest_x = [self.extract_latest_feature_vector(latest_rec)]
        predicted_return = self.predict(latest_x)[0]

        predicted_price = current_price * (1.0 + predicted_return)
        ci_margin = 1.96 * max(0.008, self.residual_std) * current_price
        lower_bound = max(0.01, predicted_price - ci_margin)
        upper_bound = predicted_price + ci_margin

        pred_pct = round(predicted_return * 100.0, 2)
        if pred_pct > 0.20:
            pred_dir = "BULLISH"
        elif pred_pct < -0.20:
            pred_dir = "BEARISH"
        else:
            pred_dir = "NEUTRAL"

        importance: Dict[str, float] = {}
        for name, w in zip(self.FEATURE_NAMES, self.weights):
            importance[name] = round(w, 5)

        return PredictionOutput(
            ticker=ticker,
            current_price=round(current_price, 2),
            predicted_return_pct=pred_pct,
            predicted_price=round(predicted_price, 2),
            confidence_interval_lower=round(lower_bound, 2),
            confidence_interval_upper=round(upper_bound, 2),
            predicted_direction=pred_dir,
            feature_importance=importance,
            validation_report=val_report,
        )
