from __future__ import annotations

import argparse
import math
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, mean_absolute_error, mean_squared_error, roc_auc_score

try:
    from src.data_loader import load_data
    from src.forecasting import rolling_origin_forecasts
except ImportError:
    from data_loader import load_data
    from forecasting import rolling_origin_forecasts


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denominator = np.where(np.abs(y_true) < 1e-8, np.nan, np.abs(y_true))
    ratio = np.abs(y_true - y_pred) / denominator
    return float(np.nanmean(ratio) * 100.0)


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float, float]:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(math.sqrt(mean_squared_error(y_true, y_pred)))
    mape = _mape(y_true, y_pred)
    return mae, rmse, mape


def evaluate_models(
    all_segments_df: pd.DataFrame,
    test_periods: int = 6,
    risk_threshold: float = 0.08,
) -> pd.DataFrame:
    """
    Run per-segment rolling-origin backtests and compare model vs baselines.

    Metrics reported:
    - MAE, RMSE, MAPE for model, naive, and rolling baseline
    - ROC-AUC and confusion-matrix counts for high-delinquency classification
    """
    ordered = all_segments_df.sort_values(["segment_id", "date"]).reset_index(drop=True)
    summary_rows: list[dict[str, Any]] = []

    for segment_id, segment_df in ordered.groupby("segment_id", observed=True):
        segment_df = segment_df.reset_index(drop=True)
        if len(segment_df) < test_periods + 6:
            continue

        start_index = len(segment_df) - test_periods
        backtest = rolling_origin_forecasts(segment_df, start_index=start_index)
        if backtest.empty:
            continue

        y_true = backtest["actual"].to_numpy(dtype=float)
        y_model = backtest["central"].to_numpy(dtype=float)
        y_naive = backtest["naive"].to_numpy(dtype=float)
        y_roll = backtest["baseline"].to_numpy(dtype=float)

        model_mae, model_rmse, model_mape = _forecast_metrics(y_true, y_model)
        naive_mae, naive_rmse, naive_mape = _forecast_metrics(y_true, y_naive)
        roll_mae, roll_rmse, roll_mape = _forecast_metrics(y_true, y_roll)

        y_true_cls = (y_true >= risk_threshold).astype(int)
        y_pred_cls = (y_model >= risk_threshold).astype(int)
        cm = confusion_matrix(y_true_cls, y_pred_cls, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        if len(np.unique(y_true_cls)) > 1:
            roc_auc = float(roc_auc_score(y_true_cls, y_model))
        else:
            roc_auc = float("nan")

        summary_rows.append(
            {
                "segment_id": str(segment_id),
                "n_test_points": int(len(backtest)),
                "model_mae": model_mae,
                "model_rmse": model_rmse,
                "model_mape": model_mape,
                "naive_mae": naive_mae,
                "naive_rmse": naive_rmse,
                "naive_mape": naive_mape,
                "rolling_mae": roll_mae,
                "rolling_rmse": roll_rmse,
                "rolling_mape": roll_mape,
                "roc_auc": roc_auc,
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
            }
        )

    return pd.DataFrame(summary_rows).sort_values("segment_id").reset_index(drop=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backtest delinquency forecasting models.")
    parser.add_argument("--input", required=True, help="Path to CSV data file")
    parser.add_argument("--test-periods", type=int, default=6, help="Number of recent periods for test window")
    parser.add_argument(
        "--risk-threshold",
        type=float,
        default=0.08,
        help="Classification threshold for high delinquency",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    df = load_data(csv_path=args.input)
    metrics = evaluate_models(
        all_segments_df=df,
        test_periods=args.test_periods,
        risk_threshold=args.risk_threshold,
    )

    if metrics.empty:
        print("No segments had enough observations for backtesting.")
        return

    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
