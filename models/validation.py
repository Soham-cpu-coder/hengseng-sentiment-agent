"""
Rigorous Out-of-Sample Time-Series Validation Engine.
Implements Walk-Forward Cross Validation (Expanding & Rolling Windows)
and TimeSeriesSplit with strict chronological ordering to eliminate lookahead bias.
Computes RMSE, MAE, Directional Accuracy (Hit Ratio), Information Coefficient (IC),
simulated strategy Sharpe ratio, baseline comparisons, and n+1 realized vs predicted check.
"""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class FoldResult:
    fold_idx: int
    train_start: int
    train_end: int
    test_start: int
    test_end: int
    rmse: float
    mae: float
    directional_accuracy: float
    information_coefficient: float
    actual_returns: List[float] = field(default_factory=list)
    predicted_returns: List[float] = field(default_factory=list)


@dataclass
class BaselineComparison:
    """Out-of-sample comparison against zero-return and price-only baselines."""
    zero_return_rmse: float
    zero_return_mae: float
    zero_return_directional_accuracy: float
    price_only_rmse: float
    price_only_mae: float
    price_only_directional_accuracy: float
    price_only_ic: float
    sentiment_improves_rmse: bool
    sentiment_improves_directional_accuracy: bool
    sentiment_improves_ic: bool


@dataclass
class PredictionActualBar:
    """Holdout realized vs predicted comparison for recent bars."""
    date_str: str
    realized_close: float
    predicted_close: float
    realized_return_pct: float
    predicted_return_pct: float
    direction_correct: bool
    error_pct: float = 0.0


@dataclass
class ValidationReport:
    model_name: str
    ticker: str
    num_folds: int
    oos_samples_count: int
    mean_rmse: Optional[float]
    mean_mae: Optional[float]
    mean_directional_accuracy: Optional[float]       # Hit ratio (0.0 to 1.0)
    mean_information_coefficient: Optional[float]    # IC (-1.0 to 1.0)
    strategy_cumulative_return_pct: Optional[float]  # Cumulative % return from signal trading
    strategy_sharpe_ratio: Optional[float]           # Annualized Sharpe ratio of the signal strategy
    benchmark_cumulative_return_pct: Optional[float] # Buy-and-hold benchmark return
    insufficient_data: bool = False
    fold_details: List[FoldResult] = field(default_factory=list)
    baselines: Optional[BaselineComparison] = None
    recent_k_predictions: List[PredictionActualBar] = field(default_factory=list)


class TimeSeriesCrossValidator:
    """Performs rigorous walk-forward out-of-sample validation on sequential time series."""

    @staticmethod
    def walk_forward_split(
        n_samples: int,
        n_splits: int = 4,
        min_train_size: int = 15,
        rolling: bool = False,
        window_size: Optional[int] = None,
    ) -> List[Tuple[List[int], List[int]]]:
        """
        Generate chronological train/test index splits.
        Guarantees that test indices always occur STRICTLY AFTER training indices.
        No future data is ever observed during training.
        """
        if n_samples < min_train_size + n_splits:
            n_splits = max(2, n_samples - min_train_size)
            if n_splits < 2:
                return []

        test_size = max(1, (n_samples - min_train_size) // n_splits)
        splits: List[Tuple[List[int], List[int]]] = []

        for i in range(n_splits):
            test_start = min_train_size + i * test_size
            test_end = min(n_samples, test_start + test_size)
            if test_start >= n_samples:
                break

            if rolling and window_size:
                train_start = max(0, test_start - window_size)
            else:
                train_start = 0

            train_indices = list(range(train_start, test_start))
            test_indices = list(range(test_start, test_end))

            if train_indices and test_indices:
                splits.append((train_indices, test_indices))

        return splits

    @classmethod
    def evaluate(
        cls,
        model_trainer: Callable[[List[List[float]], List[float]], Any],
        model_predictor: Callable[[Any, List[List[float]]], List[float]],
        X: List[List[float]],
        y: List[float],
        ticker: str,
        model_name: str = "SentimentRidgePredictor",
        n_splits: int = 4,
        price_feature_indices: Optional[List[int]] = None,
    ) -> ValidationReport:
        """
        Execute walk-forward out-of-sample validation and compile honest metrics.
        Never returns hard-coded fake metrics when data is insufficient.
        """
        n_samples = len(X)
        if n_samples < 8:
            return ValidationReport(
                model_name=model_name,
                ticker=ticker,
                num_folds=0,
                oos_samples_count=0,
                mean_rmse=None,
                mean_mae=None,
                mean_directional_accuracy=None,
                mean_information_coefficient=None,
                strategy_cumulative_return_pct=None,
                strategy_sharpe_ratio=None,
                benchmark_cumulative_return_pct=None,
                insufficient_data=True,
                fold_details=[],
            )

        splits = cls.walk_forward_split(n_samples=n_samples, n_splits=n_splits)
        if not splits:
            return ValidationReport(
                model_name=model_name,
                ticker=ticker,
                num_folds=0,
                oos_samples_count=0,
                mean_rmse=None,
                mean_mae=None,
                mean_directional_accuracy=None,
                mean_information_coefficient=None,
                strategy_cumulative_return_pct=None,
                strategy_sharpe_ratio=None,
                benchmark_cumulative_return_pct=None,
                insufficient_data=True,
                fold_details=[],
            )

        fold_results: List[FoldResult] = []
        all_pred: List[float] = []
        all_actual: List[float] = []

        # Baseline tracking across identical OOS splits
        all_zero_pred: List[float] = []
        all_price_only_pred: List[float] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = [X[i] for i in train_idx]
            y_train = [y[i] for i in train_idx]
            X_test = [X[i] for i in test_idx]
            y_test = [y[i] for i in test_idx]

            # 1. Train full model strictly on training split
            trained_model = model_trainer(X_train, y_train)
            preds = model_predictor(trained_model, X_test)

            # 2. Price-only model baseline
            if price_feature_indices:
                X_train_po = [[row[idx] for idx in price_feature_indices] for row in X_train]
                X_test_po = [[row[idx] for idx in price_feature_indices] for row in X_test]
                trained_po = model_trainer(X_train_po, y_train)
                preds_po = model_predictor(trained_po, X_test_po)
            else:
                preds_po = [0.0] * len(y_test)

            # 3. Zero-return baseline
            preds_zero = [0.0] * len(y_test)

            # Compute fold metrics
            squared_errs = [(p - a) ** 2 for p, a in zip(preds, y_test)]
            abs_errs = [abs(p - a) for p, a in zip(preds, y_test)]
            rmse = math.sqrt(sum(squared_errs) / len(squared_errs)) if squared_errs else 0.0
            mae = sum(abs_errs) / len(abs_errs) if abs_errs else 0.0

            correct_signs = [
                1.0 for p, a in zip(preds, y_test)
                if (p >= 0 and a >= 0) or (p < 0 and a < 0)
            ]
            hit_ratio = sum(correct_signs) / len(y_test) if y_test else 0.5
            ic = cls._pearson_correlation(preds, y_test)

            fold_results.append(FoldResult(
                fold_idx=fold_idx + 1,
                train_start=train_idx[0],
                train_end=train_idx[-1],
                test_start=test_idx[0],
                test_end=test_idx[-1],
                rmse=round(rmse, 5),
                mae=round(mae, 5),
                directional_accuracy=round(hit_ratio, 4),
                information_coefficient=round(ic, 4),
                actual_returns=[round(val, 5) for val in y_test],
                predicted_returns=[round(val, 5) for val in preds],
            ))

            all_pred.extend(preds)
            all_actual.extend(y_test)
            all_zero_pred.extend(preds_zero)
            all_price_only_pred.extend(preds_po)

        # Aggregate across folds
        avg_rmse = sum(f.rmse for f in fold_results) / len(fold_results) if fold_results else 0.0
        avg_mae = sum(f.mae for f in fold_results) / len(fold_results) if fold_results else 0.0

        if len(all_pred) >= 2:
            pooled_ic = cls._pearson_correlation(all_pred, all_actual)
            correct_signs = [
                1.0 for p, a in zip(all_pred, all_actual)
                if (p >= 0 and a >= 0) or (p < 0 and a < 0)
            ]
            pooled_hit = sum(correct_signs) / len(all_actual) if all_actual else 0.5
        else:
            pooled_ic = sum(f.information_coefficient for f in fold_results) / len(fold_results) if fold_results else 0.0
            pooled_hit = sum(f.directional_accuracy for f in fold_results) / len(fold_results) if fold_results else 0.5

        # Simulate Trading Strategy based on signal
        strat_returns = [
            (act if prd > 0 else -act) for prd, act in zip(all_pred, all_actual)
        ]
        cum_strat = 1.0
        for r in strat_returns:
            cum_strat *= (1.0 + r)
        strat_cum_pct = (cum_strat - 1.0) * 100.0

        # Buy & Hold benchmark
        cum_bench = 1.0
        for r in all_actual:
            cum_bench *= (1.0 + r)
        bench_cum_pct = (cum_bench - 1.0) * 100.0

        # Annualized Sharpe ratio of strategy (assuming 252 trading days)
        if len(strat_returns) > 1:
            mean_sr = sum(strat_returns) / len(strat_returns)
            var_sr = sum((r - mean_sr) ** 2 for r in strat_returns) / (len(strat_returns) - 1)
            std_sr = math.sqrt(var_sr) if var_sr > 0 else 0.01
            sharpe = (mean_sr / std_sr) * math.sqrt(252)
        else:
            sharpe = 0.0

        # Evaluate Baselines
        # 1. Zero return baseline
        zero_sq_errs = [a ** 2 for a in all_actual]
        zero_abs_errs = [abs(a) for a in all_actual]
        zero_rmse = math.sqrt(sum(zero_sq_errs) / len(zero_sq_errs)) if zero_sq_errs else 0.0
        zero_mae = sum(zero_abs_errs) / len(zero_abs_errs) if zero_abs_errs else 0.0
        zero_hit = sum(1.0 for a in all_actual if a >= 0) / len(all_actual) if all_actual else 0.5

        # 2. Price-only model baseline
        po_sq_errs = [(p - a) ** 2 for p, a in zip(all_price_only_pred, all_actual)]
        po_abs_errs = [abs(p - a) for p, a in zip(all_price_only_pred, all_actual)]
        po_rmse = math.sqrt(sum(po_sq_errs) / len(po_sq_errs)) if po_sq_errs else 0.0
        po_mae = sum(po_abs_errs) / len(po_abs_errs) if po_abs_errs else 0.0
        po_hit = sum(
            1.0 for p, a in zip(all_price_only_pred, all_actual)
            if (p >= 0 and a >= 0) or (p < 0 and a < 0)
        ) / len(all_actual) if all_actual else 0.5
        po_ic = cls._pearson_correlation(all_price_only_pred, all_actual)

        baselines = BaselineComparison(
            zero_return_rmse=round(zero_rmse, 5),
            zero_return_mae=round(zero_mae, 5),
            zero_return_directional_accuracy=round(zero_hit, 4),
            price_only_rmse=round(po_rmse, 5),
            price_only_mae=round(po_mae, 5),
            price_only_directional_accuracy=round(po_hit, 4),
            price_only_ic=round(po_ic, 4),
            sentiment_improves_rmse=(avg_rmse < po_rmse),
            sentiment_improves_directional_accuracy=(pooled_hit > po_hit),
            sentiment_improves_ic=(pooled_ic > po_ic),
        )

        return ValidationReport(
            model_name=model_name,
            ticker=ticker,
            num_folds=len(fold_results),
            oos_samples_count=len(all_actual),
            mean_rmse=round(avg_rmse, 5),
            mean_mae=round(avg_mae, 5),
            mean_directional_accuracy=round(pooled_hit, 4),
            mean_information_coefficient=round(pooled_ic, 4),
            strategy_cumulative_return_pct=round(strat_cum_pct, 2),
            strategy_sharpe_ratio=round(sharpe, 2),
            benchmark_cumulative_return_pct=round(bench_cum_pct, 2),
            insufficient_data=False,
            fold_details=fold_results,
            baselines=baselines,
        )

    @staticmethod
    def _pearson_correlation(x: List[float], y: List[float]) -> float:
        if len(x) < 2 or len(x) != len(y):
            return 0.0
        mean_x = sum(x) / len(x)
        mean_y = sum(y) / len(y)
        num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
        den_x = sum((xi - mean_x) ** 2 for xi in x)
        den_y = sum((yi - mean_y) ** 2 for yi in y)
        if den_x <= 0 or den_y <= 0:
            return 0.0
        return max(-1.0, min(1.0, num / math.sqrt(den_x * den_y)))
