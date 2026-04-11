from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd

REQUIRED_COLUMNS = [
    "date",
    "segment_id",
    "repayment_rate",
    "delinquency_rate",
    "income_to_debt_ratio",
    "avg_interest_rate",
]

OPTIONAL_MACRO_COLUMNS = ["unemployment_rate", "gdp_growth"]

NUMERIC_COLUMNS = [
    "repayment_rate",
    "delinquency_rate",
    "income_to_debt_ratio",
    "avg_interest_rate",
    *OPTIONAL_MACRO_COLUMNS,
]

_COLUMN_ALIASES = {
    "date": "date",
    "segment": "segment_id",
    "segment_id": "segment_id",
    "repayment_rate": "repayment_rate",
    "repayment": "repayment_rate",
    "delinquency_rate": "delinquency_rate",
    "delinquency": "delinquency_rate",
    "income_to_debt_ratio": "income_to_debt_ratio",
    "income_debt_ratio": "income_to_debt_ratio",
    "avg_interest_rate": "avg_interest_rate",
    "average_interest_rate": "avg_interest_rate",
    "unemployment_rate": "unemployment_rate",
    "gdp_growth": "gdp_growth",
}


@dataclass(frozen=True)
class DataConfig:
    min_segment_observations: int = 18
    test_periods: int = 6


def load_data(csv_path: Optional[str | Path] = None, dataframe: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Load and preprocess source data from CSV path or an in-memory dataframe."""
    if csv_path is None and dataframe is None:
        raise ValueError("Provide either csv_path or dataframe.")

    if dataframe is not None:
        raw_df = dataframe.copy()
    else:
        raw_df = pd.read_csv(csv_path)

    return preprocess_dataframe(raw_df)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize likely header variants from uploaded CSV files."""
    normalized = df.copy()
    rename_map: dict[str, str] = {}

    for column in normalized.columns:
        canonical = str(column).strip().lower().replace(" ", "_").replace("-", "_")
        canonical = _COLUMN_ALIASES.get(canonical, canonical)
        rename_map[column] = canonical

    normalized = normalized.rename(columns=rename_map)
    return normalized


def _coerce_rate_column(series: pd.Series) -> pd.Series:
    """Coerce a rate column to 0-1, handling percentage strings and 0-100 scales."""
    as_text = series.astype("string").str.strip().str.replace("%", "", regex=False)
    numeric = pd.to_numeric(as_text, errors="coerce")

    valid = numeric.dropna()
    if not valid.empty and (valid > 1.0).mean() > 0.5 and valid.max() <= 100.0:
        numeric = numeric / 100.0

    return numeric


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Validate schema and enforce datatypes used by the modeling pipeline."""
    clean = _normalize_columns(df)

    missing_cols = sorted(set(REQUIRED_COLUMNS) - set(clean.columns))
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    clean["date"] = pd.to_datetime(clean["date"], errors="coerce")
    clean["date"] = clean["date"].dt.to_period("M").dt.to_timestamp("M")

    clean["segment_id"] = clean["segment_id"].astype("string").str.strip()

    for rate_col in ["repayment_rate", "delinquency_rate"]:
        clean[rate_col] = _coerce_rate_column(clean[rate_col])

    for column in NUMERIC_COLUMNS:
        if column in clean.columns and column not in {"repayment_rate", "delinquency_rate"}:
            clean[column] = pd.to_numeric(clean[column], errors="coerce")

    clean = clean.dropna(subset=REQUIRED_COLUMNS)
    clean = clean.sort_values(["segment_id", "date"]).drop_duplicates(
        subset=["segment_id", "date"], keep="last"
    )

    clean["segment_id"] = clean["segment_id"].astype("category")
    return clean.reset_index(drop=True)


def filter_sparse_segments(df: pd.DataFrame, min_observations: int = 18) -> pd.DataFrame:
    """Drop segments that do not have enough time points for modeling."""
    counts = df.groupby("segment_id", observed=True).size()
    keep_segments = counts[counts >= min_observations].index
    filtered = df[df["segment_id"].isin(keep_segments)].copy()
    return filtered.reset_index(drop=True)


def list_segments(df: pd.DataFrame) -> list[str]:
    """Return sorted segment IDs as strings for UI selectors."""
    return sorted(df["segment_id"].astype(str).unique().tolist())


def split_train_test_by_time(segment_df: pd.DataFrame, test_periods: int = 6) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Time-ordered train/test split for a single segment."""
    ordered = segment_df.sort_values("date").reset_index(drop=True)
    if len(ordered) <= test_periods:
        raise ValueError(
            f"Segment has only {len(ordered)} rows; need more than test_periods={test_periods}."
        )

    train = ordered.iloc[:-test_periods].copy()
    test = ordered.iloc[-test_periods:].copy()
    return train, test


def build_segment_frames(df: pd.DataFrame, min_observations: int = 18) -> dict[str, pd.DataFrame]:
    """Build a dictionary of cleaned per-segment dataframes."""
    filtered = filter_sparse_segments(df, min_observations=min_observations)
    segment_frames: dict[str, pd.DataFrame] = {}
    for segment_id, segment_df in filtered.groupby("segment_id", observed=True):
        segment_frames[str(segment_id)] = segment_df.sort_values("date").reset_index(drop=True)
    return segment_frames
