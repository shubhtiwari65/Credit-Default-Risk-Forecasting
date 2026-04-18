from pathlib import Path

from src.data_loader import load_data
from src.evaluation import evaluate_models

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_dataset.csv"


def test_evaluate_models_returns_metrics() -> None:
    data = load_data(csv_path=SAMPLE_PATH)
    metrics = evaluate_models(data, test_periods=4, risk_threshold=0.07)

    assert not metrics.empty
    expected_cols = {
        "segment_id",
        "model_mae",
        "model_rmse",
        "model_mape",
        "naive_mae",
        "rolling_mae",
        "roc_auc",
        "tn",
        "fp",
        "fn",
        "tp",
    }
    assert expected_cols.issubset(metrics.columns)
