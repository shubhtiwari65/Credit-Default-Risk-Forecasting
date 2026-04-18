from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG_CANDIDATES = [
    "income_to_debt_ratio",
    "avg_interest_rate",
    "unemployment_rate",
    "gdp_growth",
]


@dataclass
class ForecastResult:
    segment_id: str
    forecast_df: pd.DataFrame
    model_name: str
    horizon_periods: int
    diagnostics: dict[str, float]


def weeks_to_periods(horizon_weeks: int) -> int:
    """Convert week horizon to month-like periods (4 weeks ~= 1 month)."""
    return max(1, int(math.ceil(horizon_weeks / 4.0)))


def _clip_probability(values: np.ndarray) -> np.ndarray:
    return np.clip(values.astype(float), 0.0, 1.0)


def _future_dates(last_date: pd.Timestamp, periods: int) -> pd.DatetimeIndex:
    return pd.DatetimeIndex(
        [pd.Timestamp(last_date) + pd.offsets.MonthEnd(i) for i in range(1, periods + 1)]
    )


def _baseline_forecast(series: pd.Series, periods: int) -> np.ndarray:
    """Simple exponential smoothing baseline with rolling-mean fallback."""
    clean_series = series.dropna().astype(float)
    if clean_series.empty:
        return np.zeros(periods)

    if len(clean_series) < 3:
        return np.repeat(float(clean_series.iloc[-1]), periods)

    try:
        fitted = SimpleExpSmoothing(clean_series, initialization_method="estimated").fit(optimized=True)
        return _clip_probability(fitted.forecast(periods).to_numpy())
    except Exception:
        rolling_mean = float(clean_series.tail(min(3, len(clean_series))).mean())
        return _clip_probability(np.repeat(rolling_mean, periods))


def _select_exog_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in EXOG_CANDIDATES if col in df.columns and df[col].notna().sum() > 3]


def _fit_sarimax(train_df: pd.DataFrame, exog_cols: Optional[list[str]] = None):
    y = train_df["delinquency_rate"].astype(float)
    exog = train_df[exog_cols].astype(float) if exog_cols else None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        warnings.simplefilter("ignore", category=RuntimeWarning)
        warnings.simplefilter("ignore", category=ValueWarning)
        model = SARIMAX(
            y,
            exog=exog,
            order=(1, 0, 0),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False)
    return fitted


def forecast_segment_delinquency(
    segment_df: pd.DataFrame,
    horizon_weeks: int = 6,
    alpha: float = 0.2,
) -> ForecastResult:
    """
    Forecast delinquency for one segment with baseline and SARIMAX outputs.

    Returns central, lower, upper forecasts and baseline for each future step.
    """
    if segment_df.empty:
        raise ValueError("segment_df is empty.")

    ordered = segment_df.sort_values("date").reset_index(drop=True)
    periods = weeks_to_periods(horizon_weeks)

    segment_id = str(ordered["segment_id"].iloc[-1]) if "segment_id" in ordered.columns else "unknown"
    baseline = _baseline_forecast(ordered["delinquency_rate"], periods)

    exog_cols = _select_exog_columns(ordered)
    forecast_model_name = "sarimax(1,0,0)"
    diagnostics: dict[str, float] = {}

    last_date = pd.Timestamp(ordered["date"].iloc[-1])
    future_dates = _future_dates(last_date, periods)

    future_exog = None
    if exog_cols:
        last_known = ordered[exog_cols].astype(float).tail(1)
        future_exog = pd.concat([last_known] * periods, ignore_index=True)

    try:
        fitted_model = _fit_sarimax(ordered, exog_cols=exog_cols)
        prediction = fitted_model.get_forecast(
            steps=periods,
            exog=future_exog if exog_cols else None,
        )

        central = _clip_probability(np.asarray(prediction.predicted_mean))
        interval = prediction.conf_int(alpha=alpha)
        lower = _clip_probability(interval.iloc[:, 0].to_numpy())
        upper = _clip_probability(interval.iloc[:, 1].to_numpy())

        diagnostics["aic"] = float(getattr(fitted_model, "aic", np.nan))
    except Exception:
        forecast_model_name = "baseline_only"
        central = baseline.copy()
        residual_spread = float(np.std(ordered["delinquency_rate"].tail(min(6, len(ordered))).to_numpy()))
        residual_spread = max(residual_spread, 0.005)
        z_80 = 1.2816
        lower = _clip_probability(central - z_80 * residual_spread)
        upper = _clip_probability(central + z_80 * residual_spread)
        diagnostics["aic"] = float("nan")

    forecast_df = pd.DataFrame(
        {
            "date": future_dates,
            "central": central,
            "lower": lower,
            "upper": upper,
            "baseline": baseline,
        }
    )

    return ForecastResult(
        segment_id=segment_id,
        forecast_df=forecast_df,
        model_name=forecast_model_name,
        horizon_periods=periods,
        diagnostics=diagnostics,
    )


def rolling_origin_forecasts(
    segment_df: pd.DataFrame,
    start_index: int,
    alpha: float = 0.2,
) -> pd.DataFrame:
    """Run one-step-ahead rolling forecasts from start_index to the end of the segment."""
    ordered = segment_df.sort_values("date").reset_index(drop=True)

    if start_index < 3:
        raise ValueError("start_index should be >= 3 to build a meaningful history.")

    rows: list[dict[str, float | pd.Timestamp]] = []

    for idx in range(start_index, len(ordered)):
        history = ordered.iloc[:idx].copy()
        current = ordered.iloc[idx]

        result = forecast_segment_delinquency(history, horizon_weeks=4, alpha=alpha)
        one_step = result.forecast_df.iloc[0]

        rows.append(
            {
                "date": pd.Timestamp(current["date"]),
                "actual": float(current["delinquency_rate"]),
                "central": float(one_step["central"]),
                "lower": float(one_step["lower"]),
                "upper": float(one_step["upper"]),
                "baseline": float(one_step["baseline"]),
                "naive": float(history["delinquency_rate"].iloc[-1]),
            }
        )

    return pd.DataFrame(rows)
