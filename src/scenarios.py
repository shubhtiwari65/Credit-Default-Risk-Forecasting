from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

DEFAULT_FEATURE_COLUMNS = [
    "avg_interest_rate",
    "income_to_debt_ratio",
    "repayment_rate",
    "unemployment_rate",
    "gdp_growth",
]


def fit_interest_rate_model(
    segment_df: pd.DataFrame,
    feature_columns: Optional[list[str]] = None,
) -> tuple[LinearRegression, list[str]]:
    """Fit a simple linear model for delinquency sensitivity to interest rates."""
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    cols = [col for col in cols if col in segment_df.columns]

    if "avg_interest_rate" not in cols:
        cols = ["avg_interest_rate", *cols]

    cols = list(dict.fromkeys(cols))
    train = segment_df.dropna(subset=cols + ["delinquency_rate"]).copy()

    if len(train) < max(10, len(cols) + 2):
        raise ValueError("Not enough rows to fit regression sensitivity model.")

    model = LinearRegression()
    model.fit(train[cols], train["delinquency_rate"].astype(float))
    return model, cols


def stress_test_interest_rate(
    segment_features_df: pd.DataFrame,
    base_forecast_df: pd.DataFrame,
    delta_rate: float = 0.5,
    fitted_model: Optional[LinearRegression] = None,
    feature_columns: Optional[list[str]] = None,
    elasticity_of_default_wrt_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Recompute delinquency under an interest-rate shock.

    If fitted_model is available, use regression-based adjustment.
    Otherwise, fallback to elasticity scaling for explainability.
    """
    if base_forecast_df.empty:
        raise ValueError("base_forecast_df is empty.")

    scenario = base_forecast_df.copy()
    scenario["baseline_delinquency"] = scenario["central"].astype(float)

    if fitted_model is not None:
        cols = feature_columns or DEFAULT_FEATURE_COLUMNS
        cols = [col for col in cols if col in segment_features_df.columns]
        if "avg_interest_rate" not in cols:
            cols = ["avg_interest_rate", *cols]
        cols = list(dict.fromkeys(cols))

        latest_features = segment_features_df.sort_values("date").tail(1)
        if latest_features.empty:
            raise ValueError("segment_features_df has no rows.")

        feature_block = pd.concat([latest_features[cols]] * len(scenario), ignore_index=True)
        baseline_pred = fitted_model.predict(feature_block)

        shocked_block = feature_block.copy()
        shocked_block["avg_interest_rate"] = shocked_block["avg_interest_rate"].astype(float) + float(delta_rate)
        shocked_pred = fitted_model.predict(shocked_block)

        adjustment = shocked_pred - baseline_pred
        stressed = scenario["baseline_delinquency"].to_numpy() + adjustment
    else:
        stressed = scenario["baseline_delinquency"].to_numpy() * (
            1.0 + float(elasticity_of_default_wrt_rate) * float(delta_rate)
        )

    scenario["stressed_delinquency"] = np.clip(stressed, 0.0, 1.0)
    scenario["delta"] = scenario["stressed_delinquency"] - scenario["baseline_delinquency"]

    return scenario[["date", "baseline_delinquency", "stressed_delinquency", "delta"]]
