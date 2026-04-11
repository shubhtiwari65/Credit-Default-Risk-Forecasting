from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

try:
    from src.forecasting import rolling_origin_forecasts
except ImportError:
    from forecasting import rolling_origin_forecasts


@dataclass
class Anomaly:
    date: pd.Timestamp
    actual: float
    expected: float
    lower: float
    upper: float
    anomaly_type: str
    delta: float
    driver_hints: str


def _build_driver_hints(history: pd.DataFrame, current_row: pd.Series) -> str:
    hints: list[str] = []
    candidate_features = ["income_to_debt_ratio", "avg_interest_rate", "repayment_rate"]

    for feature in candidate_features:
        if feature not in history.columns or feature not in current_row.index:
            continue

        baseline = history[feature].tail(min(6, len(history))).mean()
        current = current_row.get(feature)

        if pd.isna(baseline) or pd.isna(current) or baseline == 0:
            continue

        rel_change = (float(current) - float(baseline)) / abs(float(baseline))
        if abs(rel_change) >= 0.1:
            direction = "above" if rel_change > 0 else "below"
            hints.append(f"{feature} is {abs(rel_change) * 100:.1f}% {direction} recent average")

    return "; ".join(hints) if hints else "No strong feature jump vs recent average"


def detect_anomalies(
    segment_df: pd.DataFrame,
    test_periods: int = 6,
    margin: float = 0.0025,
) -> list[Anomaly]:
    """
    Label anomalous points in the recent test window using one-step forecast bands.

    High anomaly: actual > upper_band + margin
    Low anomaly: actual < lower_band - margin
    """
    ordered = segment_df.sort_values("date").reset_index(drop=True)
    if len(ordered) < test_periods + 6:
        return []

    start_index = len(ordered) - test_periods
    backtest = rolling_origin_forecasts(ordered, start_index=start_index)

    anomalies: list[Anomaly] = []

    for i, idx in enumerate(range(start_index, len(ordered))):
        row = backtest.iloc[i]
        current_row = ordered.iloc[idx]
        history = ordered.iloc[max(0, idx - 6) : idx]

        actual = float(row["actual"])
        lower = float(row["lower"])
        upper = float(row["upper"])
        expected = float(row["central"])

        if actual > upper + margin:
            anomaly_type = "high"
        elif actual < lower - margin:
            anomaly_type = "low"
        else:
            continue

        hints = _build_driver_hints(history, current_row)
        anomalies.append(
            Anomaly(
                date=pd.Timestamp(row["date"]),
                actual=actual,
                expected=expected,
                lower=lower,
                upper=upper,
                anomaly_type=anomaly_type,
                delta=actual - expected,
                driver_hints=hints,
            )
        )

    return anomalies


def anomalies_to_frame(anomalies: list[Anomaly]) -> pd.DataFrame:
    """Convert anomaly dataclasses to a streamlit-friendly dataframe."""
    if not anomalies:
        return pd.DataFrame(
            columns=["date", "anomaly_type", "actual", "expected", "lower", "upper", "delta", "driver_hints"]
        )

    rows = [asdict(item) for item in anomalies]
    return pd.DataFrame(rows).sort_values("date", ascending=False).reset_index(drop=True)
