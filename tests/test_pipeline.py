from pathlib import Path

import pandas as pd

from src.anomalies import detect_anomalies
from src.data_loader import filter_sparse_segments, load_data
from src.forecasting import forecast_segment_delinquency
from src.scenarios import fit_interest_rate_model, stress_test_interest_rate

SAMPLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_dataset.csv"


def test_forecast_and_scenario_pipeline_runs() -> None:
    data = load_data(csv_path=SAMPLE_PATH)
    filtered = filter_sparse_segments(data, min_observations=12)
    segment_df = filtered[filtered["segment_id"].astype(str) == "age_25_34_personal_loan"].copy()

    result = forecast_segment_delinquency(segment_df, horizon_weeks=6)

    assert not result.forecast_df.empty
    assert {"date", "central", "lower", "upper", "baseline"}.issubset(result.forecast_df.columns)
    assert result.forecast_df["central"].between(0, 1).all()

    model, cols = fit_interest_rate_model(segment_df)
    scenario_df = stress_test_interest_rate(
        segment_features_df=segment_df,
        base_forecast_df=result.forecast_df,
        delta_rate=0.5,
        fitted_model=model,
        feature_columns=cols,
    )

    assert {"baseline_delinquency", "stressed_delinquency", "delta"}.issubset(scenario_df.columns)
    assert scenario_df["stressed_delinquency"].between(0, 1).all()


def test_anomaly_detection_returns_list() -> None:
    data = load_data(csv_path=SAMPLE_PATH)
    segment_df = data[data["segment_id"].astype(str) == "sme_overdraft"].copy()

    anomalies = detect_anomalies(segment_df, test_periods=4, margin=0.001)
    assert isinstance(anomalies, list)
