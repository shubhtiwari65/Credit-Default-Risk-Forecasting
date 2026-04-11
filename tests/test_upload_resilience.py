from pathlib import Path

import pandas as pd

from src.data_loader import load_data
from src.evaluation import evaluate_models
from src.forecasting import forecast_segment_delinquency


def _build_upload_like_frame() -> pd.DataFrame:
    rows = []
    for i in range(14):
        rows.append(
            {
                "Date": f"2025-{(i % 12) + 1:02d}-28",
                "Segment ID": "upload_segment_a",
                "Repayment Rate": f"{93.0 - 0.2 * i:.2f}%",
                "Delinquency Rate": f"{3.2 + 0.15 * i:.2f}%",
                "Income To Debt Ratio": 2.4 - 0.03 * i,
                "Avg Interest Rate": 10.2 + 0.08 * i,
            }
        )
        rows.append(
            {
                "Date": f"2025-{(i % 12) + 1:02d}-28",
                "Segment ID": "upload_segment_b",
                "Repayment Rate": f"{89.5 - 0.15 * i:.2f}%",
                "Delinquency Rate": f"{5.4 + 0.12 * i:.2f}%",
                "Income To Debt Ratio": 1.9 - 0.02 * i,
                "Avg Interest Rate": 13.1 + 0.05 * i,
            }
        )
    return pd.DataFrame(rows)


def test_upload_like_csv_is_normalized() -> None:
    df = _build_upload_like_frame()
    loaded = load_data(dataframe=df)

    required = {
        "date",
        "segment_id",
        "repayment_rate",
        "delinquency_rate",
        "income_to_debt_ratio",
        "avg_interest_rate",
    }
    assert required.issubset(loaded.columns)
    assert loaded["repayment_rate"].between(0, 1).all()
    assert loaded["delinquency_rate"].between(0, 1).all()


def test_pipeline_smoke_with_upload_like_data() -> None:
    loaded = load_data(dataframe=_build_upload_like_frame())
    seg = loaded[loaded["segment_id"].astype(str) == "upload_segment_a"].sort_values("date")

    result = forecast_segment_delinquency(seg, horizon_weeks=6)
    assert not result.forecast_df.empty

    metrics = evaluate_models(loaded, test_periods=4, risk_threshold=0.05)
    assert not metrics.empty
