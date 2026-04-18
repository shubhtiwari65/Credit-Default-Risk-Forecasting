from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

try:
    from src.anomalies import anomalies_to_frame, detect_anomalies
    from src.data_loader import filter_sparse_segments, list_segments, load_data
    from src.evaluation import evaluate_models
    from src.forecasting import ForecastResult, forecast_segment_delinquency
    from src.scenarios import fit_interest_rate_model, stress_test_interest_rate
except ImportError:
    from anomalies import anomalies_to_frame, detect_anomalies
    from data_loader import filter_sparse_segments, list_segments, load_data
    from evaluation import evaluate_models
    from forecasting import ForecastResult, forecast_segment_delinquency
    from scenarios import fit_interest_rate_model, stress_test_interest_rate


SAMPLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "sample_dataset.csv"
EXTENDED_SAMPLE_PATH = Path(__file__).resolve().parents[1] / "assets" / "demo_dataset_extended.csv"

DATA_SOURCE_STARTER = "Starter sample dataset"
DATA_SOURCE_EXTENDED = "Extended sample dataset"
DATA_SOURCE_UPLOAD = "Upload your own CSV"


# ─────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────
def _apply_custom_style(theme: str = "dark") -> None:
    # Theme variables injected into :root via a tiny separate style block.
    # The big CSS block is a plain string (not f-string) to avoid brace-escaping issues.
    if theme == "light":
        theme_vars = (
            "--bg-main:#FFFFFF;"
            "--bg-panel:#F8FAFC;"
            "--bg-card:#F1F5F9;"
            "--bg-card-hover:#E8EEF6;"
            "--border:rgba(0,0,0,0.08);"
            "--text-primary:#1E293B;"
            "--text-secondary:#475569;"
            "--text-muted:#94A3B8;"
        )
    else:
        theme_vars = (
            "--bg-main:#121212;"
            "--bg-panel:#1A1A1A;"
            "--bg-card:#1E1E1E;"
            "--bg-card-hover:#252525;"
            "--border:#2A2A2A;"
            "--text-primary:#FFFFFF;"
            "--text-secondary:#B0B0B0;"
            "--text-muted:#707070;"
        )

    # Step 1: inject :root theme variables
    st.markdown(
        "<style>:root{" + theme_vars + "}</style>",
        unsafe_allow_html=True,
    )

    # Step 2: inject the full static CSS (plain string, no f-string)
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

        /* Shared accent tokens */
        :root {
            --teal: #00E5CC;
            --teal-dim: rgba(0,229,204,0.10);
            --teal-glow: rgba(0,229,204,0.22);
            --red: #FF2D55;
            --red-dim: rgba(255,45,85,0.12);
            --green: #00D68F;
            --green-dim: rgba(0,214,143,0.12);
            --amber: #FFB800;
            --amber-dim: rgba(255,184,0,0.12);
            --border-accent: rgba(0,229,204,0.28);
            --font: 'DM Sans', sans-serif;
            --mono: 'DM Mono', monospace;
        }

        /* Global resets */
        html, body, [data-testid="stAppViewContainer"], .stApp, .main {
            background-color: var(--bg-main) !important;
            font-family: var(--font) !important;
            transition: background-color 0.25s ease, color 0.25s ease;
        }

        /* Remove Streamlit chrome */
        header[data-testid="stHeader"] { display: none !important; }
        [data-testid="stToolbar"] { display: none !important; }
        [data-testid="stDecoration"] { display: none !important; }
        [data-testid="stStatusWidget"] { display: none !important; }
        #MainMenu { display: none !important; }
        .stDeployButton { display: none !important; }
        footer { display: none !important; }
        .block-container { padding: 0 2rem 3rem 2rem !important; max-width: 1400px !important; }
        [data-testid="stSidebarNav"] { display: none !important; }
        [class*="keyboard"] { display: none !important; }
        button[aria-label="keyboard shortcuts"] { display: none !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"] { display: none !important; }
        [data-testid="stSidebar"] [data-testid="stSidebarNavItems"] { display: none !important; }
        .st-emotion-cache-1cypcdb, .st-emotion-cache-uf99v8 { display: none !important; }

        /* Typography */
        h1, h2, h3, h4, h5, h6 {
            font-family: var(--font) !important;
            color: var(--text-primary) !important;
            letter-spacing: -0.025em;
        }
        p, div, label, span, li, button, input, select, textarea {
            font-family: var(--font) !important;
        }

        /* Sidebar */
        [data-testid="stSidebar"] {
            background: var(--bg-panel) !important;
            border-right: 1px solid var(--border) !important;
            padding-top: 0 !important;
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 0 !important; }

        .sidebar-logo {
            background: linear-gradient(135deg, #1A2A3A 0%, #0D1520 100%);
            padding: 20px 20px 16px;
            margin: -1rem -1rem 4px -1rem;
        }
        .sidebar-logo .logo-title {
            font-size: 15px; font-weight: 700; color: #FFFFFF;
            letter-spacing: -0.01em; line-height: 1.2;
        }
        .sidebar-logo .logo-sub {
            font-size: 11px; font-weight: 500;
            color: rgba(255,255,255,0.55); margin-top: 3px;
        }

        .section-label {
            font-size: 10px; font-weight: 700; letter-spacing: 0.12em;
            text-transform: uppercase; color: var(--text-muted);
            margin: 20px 0 10px; padding: 0 4px;
        }

        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stFileUploader label {
            font-size: 12px !important; font-weight: 600 !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase; letter-spacing: 0.06em;
        }

        [data-testid="stFileUploader"] section {
            background-color: var(--bg-card);
            border: 1px dashed rgba(0,212,180,0.4);
            border-radius: 8px; padding: 16px;
        }
        [data-testid="stFileUploader"] section div { color: var(--text-secondary); }
        [data-testid="stFileUploader"] section small { font-size: 12px; color: var(--text-muted); }
        [data-testid="stFileUploader"] section button {
            background-color: rgba(0,212,180,0.1) !important;
            border: 1px solid rgba(0,212,180,0.3) !important;
            border-radius: 6px !important; color: #00D4B4 !important;
            padding: 4px 12px !important; font-weight: 600 !important;
        }
        [data-testid="stFileUploader"] section button:hover {
            background-color: rgba(0,212,180,0.2) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
            color: var(--text-muted) !important; font-size: 11px !important;
        }

        [data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] [data-testid="stSelectbox"] [data-baseweb="select"] > div {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [data-testid="stThumbValue"] {
            background: var(--teal) !important; color: #080E1A !important;
            font-weight: 600; font-size: 11px;
        }
        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:nth-child(3) {
            background: var(--teal) !important;
        }

        .sidebar-tip {
            background: var(--teal-dim); border: 1px solid var(--border-accent);
            border-radius: 8px; padding: 10px 12px; font-size: 12px;
            color: var(--teal); line-height: 1.5; margin-top: 16px;
        }
        .sidebar-tip b { color: var(--teal); }

        /* Top bar */
        .top-bar {
            display: flex; align-items: center; justify-content: space-between;
            padding: 20px 0 8px; border-bottom: 1px solid var(--border); margin-bottom: 24px;
        }
        .top-bar .app-title {
            font-size: 22px; font-weight: 700; color: var(--text-primary); letter-spacing: -0.03em;
        }
        .top-bar .app-title span { color: var(--teal); }
        .top-bar .segment-badge {
            background: var(--teal-dim); border: 1px solid var(--border-accent);
            color: var(--teal); font-size: 12px; font-weight: 600;
            padding: 5px 14px; border-radius: 999px; letter-spacing: 0.03em;
        }

        /* Insight Strip */
        .insight-strip {
            background: var(--bg-card);
            border: 1px solid var(--border-accent); border-radius: 14px;
            padding: 18px 24px; display: flex; align-items: center;
            gap: 16px; margin-bottom: 20px; position: relative; overflow: hidden;
        }
        .insight-strip::before {
            content: ''; position: absolute; top: 0; left: 0; width: 3px; height: 100%;
            background: linear-gradient(180deg, var(--teal) 0%, #0099DD 100%);
        }
        .insight-strip::after {
            content: ''; position: absolute; top: -40px; right: -40px;
            width: 180px; height: 180px;
            background: radial-gradient(circle, rgba(0,212,180,0.08) 0%, transparent 70%);
            pointer-events: none;
        }
        .insight-icon { font-size: 28px; flex-shrink: 0; filter: drop-shadow(0 0 8px rgba(0,212,180,0.4)); }
        .insight-text .insight-label {
            font-size: 11px; font-weight: 600; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--teal); margin-bottom: 4px;
        }
        .insight-text .insight-main {
            font-size: 15px; font-weight: 500; color: var(--text-primary); line-height: 1.5;
        }
        .insight-text .insight-main b { color: var(--text-primary); font-weight: 700; }

        /* KPI Cards */
        .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 12px; margin-bottom: 28px; }
        .kpi-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 14px; padding: 18px 16px 16px; position: relative;
            overflow: hidden; transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease; cursor: default;
        }
        .kpi-card:hover {
            transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.60);
            border-color: rgba(255,255,255,0.12);
        }
        .kpi-card .kpi-icon { font-size: 20px; margin-bottom: 10px; display: block; }
        .kpi-card .kpi-label {
            font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--text-muted); margin-bottom: 6px;
        }
        .kpi-card .kpi-value {
            font-size: 26px; font-weight: 700; letter-spacing: -0.03em; line-height: 1; margin-bottom: 8px;
        }
        .kpi-card .kpi-delta {
            font-size: 12px; font-weight: 600; display: inline-flex;
            align-items: center; gap: 4px; padding: 3px 8px; border-radius: 999px;
        }
        .kpi-card .kpi-delta.up { background: var(--red-dim); color: var(--red); }
        .kpi-card .kpi-delta.down { background: var(--green-dim); color: var(--green); }
        .kpi-card .kpi-delta.neutral { background: var(--teal-dim); color: var(--teal); }
        .kpi-card .kpi-bar { position: absolute; bottom: 0; left: 0; height: 3px; width: 100%; }
        .kpi-card.color-teal .kpi-value { color: var(--teal); }
        .kpi-card.color-teal .kpi-bar { background: linear-gradient(90deg, var(--teal), transparent); }
        .kpi-card.color-red .kpi-value { color: var(--red); }
        .kpi-card.color-red .kpi-bar { background: linear-gradient(90deg, var(--red), transparent); }
        .kpi-card.color-green .kpi-value { color: var(--green); }
        .kpi-card.color-green .kpi-bar { background: linear-gradient(90deg, var(--green), transparent); }
        .kpi-card.color-amber .kpi-value { color: var(--amber); }
        .kpi-card.color-amber .kpi-bar { background: linear-gradient(90deg, var(--amber), transparent); }

        /* Tabs */
        div[data-baseweb="tab-list"] {
            gap: 6px !important; background: transparent !important;
            border-bottom: 1px solid var(--border) !important; padding-bottom: 0 !important;
        }
        button[role="tab"] {
            background: transparent !important; border: none !important;
            border-radius: 0 !important; border-bottom: 2px solid transparent !important;
            color: var(--text-muted) !important; font-size: 13px !important;
            font-weight: 600 !important; padding: 10px 18px !important;
            transition: all 0.2s !important; margin-bottom: -1px !important;
        }
        button[role="tab"]:hover { color: var(--text-secondary) !important; }
        button[role="tab"][aria-selected="true"] {
            color: var(--teal) !important; border-bottom-color: var(--teal) !important;
            background: transparent !important;
        }
        div[data-baseweb="tab-panel"] { padding-top: 24px !important; }

        /* Anomaly Table */
        .anomaly-badge {
            display: inline-block; padding: 2px 10px; border-radius: 999px;
            font-size: 11px; font-weight: 700; letter-spacing: 0.06em;
        }
        .anomaly-badge.high { background: var(--red-dim); color: var(--red); border: 1px solid rgba(255,77,106,0.3); }
        .anomaly-badge.low { background: var(--green-dim); color: var(--green); border: 1px solid rgba(0,196,140,0.3); }
        .anomaly-badge.spike { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,181,71,0.3); }

        .anomaly-summary {
            display: flex; align-items: center; gap: 10px;
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 14px 20px; margin-bottom: 16px;
        }
        .anomaly-summary .count-bubble {
            background: var(--red-dim); border: 1px solid rgba(255,77,106,0.3);
            color: var(--red); font-size: 22px; font-weight: 700;
            width: 48px; height: 48px; border-radius: 50%;
            display: flex; align-items: center; justify-content: center; flex-shrink: 0;
        }
        .anomaly-summary .count-bubble.zero {
            background: var(--green-dim); border-color: rgba(0,196,140,0.3); color: var(--green);
        }
        .anomaly-summary .count-text .main { font-size: 15px; font-weight: 600; color: var(--text-primary); }
        .anomaly-summary .count-text .sub { font-size: 12px; color: var(--text-muted); margin-top: 2px; }

        /* Scenario cards */
        .scenario-compare {
            display: grid; grid-template-columns: 1fr auto 1fr; gap: 16px;
            align-items: center; margin-bottom: 20px;
        }
        .scenario-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 14px; padding: 20px;
        }
        .scenario-card.stressed { border-color: rgba(255,77,106,0.3); }
        .scenario-card .sc-label {
            font-size: 10px; font-weight: 700; letter-spacing: 0.1em;
            text-transform: uppercase; color: var(--text-muted); margin-bottom: 10px;
        }
        .scenario-card .sc-value { font-size: 30px; font-weight: 700; letter-spacing: -0.04em; margin-bottom: 4px; }
        .scenario-card.baseline .sc-value { color: var(--teal); }
        .scenario-card.stressed .sc-value { color: var(--red); }
        .scenario-card .sc-sub { font-size: 12px; color: var(--text-muted); }
        .scenario-arrow { font-size: 24px; text-align: center; color: var(--amber); filter: drop-shadow(0 0 6px rgba(255,181,71,0.4)); }
        .scenario-delta-badge {
            display: inline-block; background: var(--red-dim);
            border: 1px solid rgba(255,77,106,0.3); color: var(--red);
            font-size: 14px; font-weight: 700; padding: 4px 14px;
            border-radius: 999px; margin-top: 8px;
        }

        /* Executive card */
        .exec-card {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 16px; padding: 28px; position: relative;
            overflow: hidden; margin-bottom: 24px;
        }
        .exec-card::before {
            content: ''; position: absolute; top: 0; left: 0; right: 0; height: 3px;
            background: linear-gradient(90deg, var(--teal) 0%, #0099DD 50%, var(--teal) 100%);
        }
        .exec-card .exec-header { display: flex; align-items: center; gap: 10px; margin-bottom: 16px; }
        .exec-card .exec-header .exec-title { font-size: 16px; font-weight: 700; color: var(--text-primary); }
        .exec-card .exec-header .exec-badge {
            background: var(--teal-dim); border: 1px solid var(--border-accent);
            color: var(--teal); font-size: 10px; font-weight: 700;
            letter-spacing: 0.1em; text-transform: uppercase; padding: 3px 10px; border-radius: 999px;
        }
        .exec-body { font-size: 14px; color: var(--text-secondary); line-height: 1.8; }

        .meta-pill {
            display: inline-flex; align-items: center; gap: 6px;
            background: var(--bg-main); border: 1px solid var(--border);
            border-radius: 8px; padding: 6px 12px; font-size: 12px;
            color: var(--text-secondary); margin-right: 8px; margin-bottom: 8px;
        }
        .meta-pill span { color: var(--teal); font-weight: 600; }

        /* Chart container */
        .chart-wrap {
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 14px; padding: 4px; overflow: hidden;
        }

        /* Status indicators */
        .trend-indicator {
            display: inline-flex; align-items: center; gap: 6px;
            padding: 6px 14px; border-radius: 999px; font-size: 12px; font-weight: 700;
        }
        .trend-indicator.warning { background: var(--amber-dim); color: var(--amber); border: 1px solid rgba(255,181,71,0.3); }
        .trend-indicator.stable  { background: var(--green-dim);  color: var(--green);  border: 1px solid rgba(0,196,140,0.3); }
        .trend-indicator.danger  { background: var(--red-dim);    color: var(--red);    border: 1px solid rgba(255,77,106,0.3); }

        /* Dataframe overrides */
        .stDataFrame { border-radius: 10px !important; overflow: hidden !important; }
        .stDataFrame table { background: var(--bg-card) !important; }
        .stDataFrame thead tr th {
            background: var(--bg-main) !important; color: var(--text-muted) !important;
            font-size: 11px !important; font-weight: 700 !important;
            text-transform: uppercase; letter-spacing: 0.08em;
            border-bottom: 1px solid var(--border) !important;
        }
        .stDataFrame tbody tr td {
            color: var(--text-secondary) !important; font-size: 13px !important;
            font-family: var(--mono) !important; border-bottom: 1px solid var(--border) !important;
        }
        .stDataFrame tbody tr:hover td { background: var(--bg-card-hover) !important; }

        hr { border-color: var(--border) !important; }
        .stCaption { color: var(--text-muted) !important; font-size: 11px !important; }
        .stAlert { border-radius: 10px !important; }

        /* Animations */
        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(16px); }
            to   { opacity: 1; transform: translateY(0); }
        }
        .anim { animation: fadeUp 0.5s ease both; }
        .anim-delay-1 { animation-delay: 0.05s; }
        .anim-delay-2 { animation-delay: 0.10s; }
        .anim-delay-3 { animation-delay: 0.15s; }

        /* Selectbox dropdown popup */
        [data-baseweb="popover"] ul {
            background: var(--bg-panel) !important;
            border: 1px solid var(--border) !important;
            border-radius: 8px !important;
        }
        [data-baseweb="popover"] ul li {
            color: var(--text-primary) !important;
            font-size: 13px !important;
        }
        [data-baseweb="popover"] ul li:hover {
            background: var(--bg-card-hover) !important;
        }

        /* Sidebar section spacing */
        [data-testid="stSidebar"] .stFileUploader {
            margin-top: 8px;
        }

        /* Theme toggle */
        .theme-toggle-wrap {
            display: flex; align-items: center; justify-content: space-between;
            background: var(--bg-card); border: 1px solid var(--border);
            border-radius: 12px; padding: 10px 14px; margin: 10px 0 6px;
        }
        .theme-toggle-wrap .theme-label {
            font-size: 12px; font-weight: 700; letter-spacing: 0.06em;
            text-transform: uppercase; color: var(--text-secondary);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



# ─────────────────────────────────────────────
#  DATA LOADING  (unchanged logic)
# ─────────────────────────────────────────────
import io

@st.cache_data(show_spinner="Loading dataset…")
def _load_selected_dataset(data_source: str, uploaded_file_bytes: bytes | None) -> pd.DataFrame:
    if data_source == DATA_SOURCE_STARTER:
        return load_data(csv_path=SAMPLE_PATH)
    if data_source == DATA_SOURCE_EXTENDED:
        if EXTENDED_SAMPLE_PATH.exists():
            return load_data(csv_path=EXTENDED_SAMPLE_PATH)
        return load_data(csv_path=SAMPLE_PATH)
    if data_source == DATA_SOURCE_UPLOAD and uploaded_file_bytes is None:
        return pd.DataFrame()
    if uploaded_file_bytes is not None:
        uploaded_df = pd.read_csv(io.BytesIO(uploaded_file_bytes))
        return load_data(dataframe=uploaded_df)
    return pd.DataFrame()


@st.cache_data(show_spinner="Forecasting…")
def get_cached_forecast(_segment_df: pd.DataFrame, horizon: int) -> ForecastResult:
    return forecast_segment_delinquency(_segment_df.copy(), horizon_weeks=horizon)


@st.cache_data(show_spinner="Scanning anomalies…")
def get_cached_anomalies(_segment_df: pd.DataFrame, test_periods: int, margin: float):
    return detect_anomalies(_segment_df.copy(), test_periods=test_periods, margin=margin)


@st.cache_data(show_spinner="Running stress test…")
def get_cached_scenario(_segment_df: pd.DataFrame, _forecast_df: pd.DataFrame, delta: float) -> tuple[pd.DataFrame, str]:
    try:
        model, feats = fit_interest_rate_model(_segment_df.copy())
        df = stress_test_interest_rate(
            segment_features_df=_segment_df.copy(),
            base_forecast_df=_forecast_df.copy(),
            delta_rate=delta,
            fitted_model=model,
            feature_columns=feats,
        )
        return df, "regression"
    except ValueError:
        df = stress_test_interest_rate(
            segment_features_df=_segment_df.copy(),
            base_forecast_df=_forecast_df.copy(),
            delta_rate=delta,
            fitted_model=None,
            elasticity_of_default_wrt_rate=0.04,
        )
        return df, "elasticity"


# ─────────────────────────────────────────────
#  CHARTS  (unchanged logic, updated styling)
# ─────────────────────────────────────────────
def _build_forecast_figure(segment_df: pd.DataFrame, result: ForecastResult, theme: str = "dark") -> go.Figure:
    fig = go.Figure()

    # Theme-aware colors
    is_light = theme == "light"
    grid_color   = "rgba(0,0,0,0.07)"       if is_light else "rgba(255,255,255,0.07)"
    line_color   = "rgba(0,0,0,0.10)"       if is_light else "rgba(255,255,255,0.12)"
    font_color   = "#475569"                 if is_light else "#B0B0B0"
    hover_bg     = "#F1F5F9"                 if is_light else "#1E1E1E"
    hover_border = "rgba(0,0,0,0.10)"        if is_light else "rgba(255,255,255,0.1)"
    marker_ring  = "#FFFFFF"                 if is_light else "#121212"

    forecast_start = pd.Timestamp(result.forecast_df["date"].iloc[0])
    forecast_end   = pd.Timestamp(result.forecast_df["date"].iloc[-1])

    fig.add_vrect(
        x0=forecast_start, x1=forecast_end,
        fillcolor="rgba(255,181,71,0.07)", line_width=0, layer="below",
    )

    # Historical
    fig.add_trace(go.Scatter(
        x=segment_df["date"], y=segment_df["delinquency_rate"],
        mode="lines+markers", name="Historical",
        line=dict(color="#38BDF8", width=3.0),
        marker=dict(size=5, color="#38BDF8"),
    ))

    # Bands
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["lower"],
        mode="lines", line=dict(width=0), name="80% confidence band",
        fill="tonexty", fillcolor="rgba(255,45,85,0.13)",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["central"],
        mode="lines+markers", name="Model forecast",
        line=dict(color="#FF2D55", width=3.0),
        marker=dict(size=6, color="#FF2D55", line=dict(color=marker_ring, width=1.5)),
    ))

    # Baseline
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["baseline"],
        mode="lines", name="Naïve baseline",
        line=dict(color="#00C48C", dash="dot", width=1.8),
    ))

    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=font_color),
        xaxis=dict(
            gridcolor=grid_color,
            linecolor=line_color,
            tickfont=dict(size=11, color=font_color),
        ),
        yaxis=dict(
            gridcolor=grid_color,
            linecolor=line_color,
            tickformat=".1%",
            tickfont=dict(size=11, color=font_color),
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=font_color),
        ),
        margin=dict(l=12, r=12, t=16, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=hover_border, font=dict(size=12, color=font_color)),
    )
    return fig


def _build_scenario_figure(scenario_df: pd.DataFrame, theme: str = "dark") -> go.Figure:
    fig = go.Figure()

    # Theme-aware colors
    is_light = theme == "light"
    grid_color   = "rgba(0,0,0,0.07)"    if is_light else "rgba(255,255,255,0.07)"
    line_color   = "rgba(0,0,0,0.10)"    if is_light else "rgba(255,255,255,0.12)"
    font_color   = "#475569"              if is_light else "#B0B0B0"
    hover_bg     = "#F1F5F9"              if is_light else "#1E1E1E"
    hover_border = "rgba(0,0,0,0.10)"    if is_light else "rgba(255,255,255,0.1)"
    marker_ring  = "#FFFFFF"              if is_light else "#121212"

    fig.add_trace(go.Scatter(
        x=scenario_df["date"], y=scenario_df["baseline_delinquency"],
        mode="lines+markers", name="Baseline",
        line=dict(color="#00D4B4", width=2.5),
        marker=dict(size=6, color="#00D4B4", line=dict(color=marker_ring, width=1.5)),
    ))

    fig.add_trace(go.Scatter(
        x=scenario_df["date"], y=scenario_df["stressed_delinquency"],
        mode="lines+markers", name="Rate shock applied",
        line=dict(color="#FF2D55", width=3.0),
        marker=dict(size=6, color="#FF2D55", line=dict(color=marker_ring, width=1.5)),
        fill="tonexty", fillcolor="rgba(255,45,85,0.11)",
    ))

    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans", color=font_color),
        xaxis=dict(gridcolor=grid_color, linecolor=line_color, tickfont=dict(size=11, color=font_color)),
        yaxis=dict(gridcolor=grid_color, linecolor=line_color, tickformat=".1%", tickfont=dict(size=11, color=font_color)),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=12, color=font_color)),
        margin=dict(l=12, r=12, t=16, b=12),
        hovermode="x unified",
        hoverlabel=dict(bgcolor=hover_bg, bordercolor=hover_border, font=dict(size=12, color=font_color)),
    )
    return fig


# ─────────────────────────────────────────────
#  HELPERS  (unchanged logic)
# ─────────────────────────────────────────────
def _prepare_anomaly_display(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    if anomalies_df.empty:
        return anomalies_df
    display = anomalies_df.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["anomaly_type"] = display["anomaly_type"].str.upper()
    for col in ["actual", "expected", "lower", "upper", "delta"]:
        display[col] = display[col].map(lambda v: f"{float(v):.2%}")
    return display[["date", "anomaly_type", "actual", "expected", "lower", "upper", "delta", "driver_hints"]]


def _default_text_summary(
    segment_id: str,
    forecast_result: ForecastResult,
    anomalies_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    delta_rate: float,
) -> str:
    next_point = forecast_result.forecast_df.iloc[0]
    anomaly_count = int(len(anomalies_df))
    avg_stress_delta = float(scenario_df["delta"].mean()) if not scenario_df.empty else 0.0
    return (
        f"Segment {segment_id}: next forecasted delinquency is {next_point['central']:.2%} "
        f"with an 80% band of {next_point['lower']:.2%} to {next_point['upper']:.2%}. "
        f"Recent anomaly count in the selected window is {anomaly_count}. "
        f"Under a +{delta_rate:.2f}% interest-rate shock, expected delinquency changes by "
        f"about {avg_stress_delta:.2%} on average over the forecast horizon."
    )


@st.cache_data(show_spinner="Generating AI summary…", ttl="1h")
def _maybe_generate_llm_summary(fallback_summary: str, context_payload: dict) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return fallback_summary
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")
        prompt = (
            "Summarize this delinquency risk dashboard for a non-technical banking manager in 4 sentences. "
            "Use plain language and include one recommendation. Data: "
            f"{context_payload}"
        )
        response = model.generate_content(prompt)
        if hasattr(response, "text") and response.text:
            return response.text.strip()
    except Exception:
        pass
    return fallback_summary


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Credit-Default-Risk-Forecasting",
        page_icon="📊",
        layout="wide",
    )
    _apply_custom_style(theme=st.session_state.get("theme", "light"))

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="logo-title">📊 Credit-Default-Risk-Forecasting</div>
                <div class="logo-sub">Risk Forecasting Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme Toggle ──
        st.markdown("<div class='section-label'>Appearance</div>", unsafe_allow_html=True)
        current_theme = st.session_state.get("theme", "light")
        toggle_label = "🌙 Dark Mode" if current_theme == "dark" else "☀️ Light Mode"
        is_light_on = st.toggle(
            toggle_label,
            value=(current_theme == "light"),
            key="theme_toggle",
            help="Switch between ☀️ Light and 🌙 Dark mode",
        )
        new_theme = "light" if is_light_on else "dark"
        if new_theme != current_theme:
            st.session_state["theme"] = new_theme
            st.rerun()

        st.markdown("<div class='section-label'>Data Source</div>", unsafe_allow_html=True)
        data_source = st.selectbox(
            "Dataset",
            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],
            index=0,
            label_visibility="collapsed",
            help="Extended dataset gives richer segment behavior for demo purposes.",
        )
        if data_source == DATA_SOURCE_UPLOAD:
            uploaded_file = st.file_uploader(
                "Upload CSV",
                type=["csv"],
                help="Required columns: date, segment_id, repayment_rate, delinquency_rate, income_to_debt_ratio, avg_interest_rate",
                label_visibility="collapsed",
            )
        else:
            uploaded_file = None

        st.markdown("<div class='section-label'>Forecast Settings</div>", unsafe_allow_html=True)
        horizon_weeks = st.slider(
            "Forecast horizon (weeks)",
            min_value=4, max_value=8, value=6,
            help="How many weeks ahead to project delinquency rates.",
        )
        backtest_periods = st.slider(
            "Backtest periods",
            min_value=3, max_value=12, value=6,
            help="Number of historical periods used to evaluate model accuracy.",
        )
        min_observations = st.slider(
            "Min. observations per segment",
            min_value=8, max_value=24, value=12,
            help="Segments with fewer data points are excluded from modeling.",
        )

        st.markdown("<div class='section-label'>Risk Settings</div>", unsafe_allow_html=True)
        delta_rate = st.slider(
            "Interest-rate shock (+%)",
            min_value=0.0, max_value=2.0, value=0.5, step=0.1,
            help="Simulates an immediate rate increase of this many percentage points.",
        )
        anomaly_margin = st.slider(
            "Anomaly detection margin",
            min_value=0.0, max_value=0.02, value=0.0025, step=0.0005, format="%.4f",
            help="Extra buffer above/below the forecast band before flagging an anomaly.",
        )
        risk_threshold = st.slider(
            "High-risk delinquency threshold",
            min_value=0.03, max_value=0.20, value=0.08, step=0.005,
            help="Delinquency rate above this level is considered high risk.",
        )

        st.markdown(
            """
            <div class="sidebar-tip">
                <b>💡 Tip</b> Use the <em>Extended</em> dataset for demo. Select <em>Upload your own CSV</em> to analyze real portfolio data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── DATA LOADING ─────────────────────────
    if data_source == DATA_SOURCE_UPLOAD and uploaded_file is None:
        st.markdown(
            """
            <div style="text-align:center; padding: 80px xpx;">
                <div style="font-size:48px; margin-bottom:16px;">📂</div>
                <div style="font-size:20px; font-weight:700; color:var(--text-primary); margin-bottom:8px;">No file selected</div>
                <div style="font-size:14px; color:var(--text-secondary);">Upload a CSV from the sidebar to get started.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        data = _load_selected_dataset(data_source=data_source, uploaded_file_bytes=file_bytes)
    except Exception as exc:
        st.error(f"⚠️ Could not load CSV: {exc}")
        st.caption("Expected columns: date, segment_id, repayment_rate, delinquency_rate, income_to_debt_ratio, avg_interest_rate")
        st.stop()

    if data.empty:
        st.info("Select a dataset or upload a CSV to continue.")
        st.stop()

    modeled = filter_sparse_segments(data, min_observations=min_observations)
    segment_options = list_segments(modeled)

    if not segment_options:
        st.error("No segment has enough observations after filtering. Try lowering 'Min. observations'.")
        st.stop()

    selected_segment = st.sidebar.selectbox(
        "Customer segment",
        options=segment_options,
        help="Switch between portfolio segments to compare risk profiles.",
    )

    segment_df = modeled[modeled["segment_id"].astype(str) == selected_segment].copy()
    segment_df = segment_df.sort_values("date").reset_index(drop=True)

    # ── COMPUTE ──────────────────────────────
    forecast_result = get_cached_forecast(segment_df, horizon_weeks)
    anomalies = get_cached_anomalies(
        segment_df,
        test_periods=min(backtest_periods, max(3, len(segment_df) // 3)),
        margin=anomaly_margin,
    )
    anomalies_df = anomalies_to_frame(anomalies)
    scenario_df, scenario_used = get_cached_scenario(segment_df, forecast_result.forecast_df, delta_rate)

    next_point = forecast_result.forecast_df.iloc[0]
    last_actual = float(segment_df["delinquency_rate"].iloc[-1])
    avg_stress_delta = float(scenario_df["delta"].mean()) if not scenario_df.empty else 0.0
    interval_width = float(next_point["upper"] - next_point["lower"])
    forecast_change = float(next_point["central"]) - last_actual

    # ── TOP BAR ──────────────────────────────
    st.markdown(
        f"""
        <div class="top-bar">
            <div class="app-title">Credit-Default-Risk-<span>Forecasting</span></div>
            <div class="segment-badge">📍 {selected_segment}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── INSIGHT STRIP ────────────────────────
    trend_direction = "rising ⚠️" if forecast_change > 0.005 else ("falling ✅" if forecast_change < -0.005 else "stable 📊")
    anomaly_txt = f"{len(anomalies_df)} anomaly{'s' if len(anomalies_df) != 1 else ''} detected" if anomalies_df.empty is False else "No anomalies detected"
    icon = "⚠️" if len(anomalies_df) > 0 or forecast_change > 0.005 else "✅"

    st.markdown(
        f"""
        <div class="insight-strip anim">
            <div class="insight-icon">{icon}</div>
            <div class="insight-text">
                <div class="insight-label">Key Insight — Segment {selected_segment}</div>
                <div class="insight-main">
                    Next week forecast is <b>{float(next_point['central']):.2%}</b>
                    (trend is <b>{trend_direction}</b>).
                    {anomaly_txt} in backtest window.
                    A +{delta_rate:.1f}% rate shock would lift delinquency by ~<b>{avg_stress_delta:.2%}</b> on average.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI CARDS ────────────────────────────
    delta_sign = "▲" if forecast_change > 0 else "▼"
    delta_class = "up" if forecast_change > 0 else "down"
    anomaly_color = "color-red" if len(anomalies_df) > 0 else "color-green"
    stress_color = "color-red" if avg_stress_delta > 0.01 else "color-amber"
    delinquency_color = "color-red" if last_actual > risk_threshold else "color-green"
    forecast_color = "color-red" if float(next_point["central"]) > risk_threshold else "color-teal"

    def _kpi_card(col, css_color: str, icon: str, label: str, value: str, delta_text: str, delta_cls: str) -> None:
        col.markdown(
            f'<div class="kpi-card {css_color}">'
            f'<span class="kpi-icon">{icon}</span>'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value">{value}</div>'
            f'<div class="kpi-delta {delta_cls}">{delta_text}</div>'
            '<div class="kpi-bar"></div></div>',
            unsafe_allow_html=True,
        )

    _kc1, _kc2, _kc3, _kc4, _kc5 = st.columns(5)
    _kpi_card(_kc1, delinquency_color, "📈", "Latest Delinquency",
              f"{last_actual:.2%}", "Current rate", "neutral")
    _kpi_card(_kc2, forecast_color, "🎯", "Next Forecast",
              f"{float(next_point['central']):.2%}",
              f"{delta_sign} {abs(forecast_change):.2%} vs now", delta_class)
    _kpi_card(_kc3, "color-amber", "↔️", "80% Band Width",
              f"{interval_width:.2%}", "Uncertainty range", "neutral")
    _kpi_card(_kc4, stress_color, "⚡", "Rate Shock Impact",
              f"{avg_stress_delta:.2%}", f"+{delta_rate:.1f}% shock avg lift", "up")
    _kpi_card(_kc5, anomaly_color, "🔍", "Anomalies Found",
              str(len(anomalies_df)),
              "Needs review" if len(anomalies_df) > 0 else "All clear",
              "up" if len(anomalies_df) > 0 else "down")

    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────
    tab_forecast, tab_anomalies, tab_scenario, tab_summary = st.tabs([
        "📈  Forecast View",
        "🔍  Anomaly Monitor",
        "⚡  Scenario Lab",
        "📋  Executive Summary",
    ])

    active_theme = st.session_state.get("theme", "dark")

    # ── FORECAST TAB ─────────────────────────
    with tab_forecast:
        # Trend status
        if forecast_change > 0.01:
            indicator_html = '<span class="trend-indicator danger">⚠️ Risk Increasing</span>'
            guidance = "Delinquency is projected to rise meaningfully. Monitor closely and consider early interventions."
        elif forecast_change > 0.003:
            indicator_html = '<span class="trend-indicator warning">📊 Slight Upward Drift</span>'
            guidance = "Modest increase projected. Maintain standard monitoring cadence."
        else:
            indicator_html = '<span class="trend-indicator stable">✅ Stable Trend</span>'
            guidance = "Delinquency appears stable within normal variation. No immediate action required."

        st.markdown(
            f"""
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:16px;">
                {indicator_html}
                <span style="font-size:13px; color:var(--text-secondary);">{guidance}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(_build_forecast_figure(segment_df, forecast_result, theme=active_theme), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown(
            """
            <div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:16px;">
                <div style="background:rgba(91,156,246,0.08); border:1px solid rgba(91,156,246,0.2); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#38BDF8; text-transform:uppercase; margin-bottom:4px;">🔵 Blue line</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Observed historical delinquency rate.</div>
                </div>
                <div style="background:rgba(255,77,106,0.08); border:1px solid rgba(255,77,106,0.2); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#FF2D55; text-transform:uppercase; margin-bottom:4px;">🔴 Red line + band</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Model forecast with 80% confidence interval. Actuals above band = elevated risk.</div>
                </div>
                <div style="background:rgba(0,196,140,0.08); border:1px solid rgba(0,196,140,0.2); border-radius:10px; padding:12px 14px;">
                    <div style="font-size:11px; font-weight:700; letter-spacing:0.1em; color:#00C48C; text-transform:uppercase; margin-bottom:4px;">🟢 Dotted line</div>
                    <div style="font-size:12px; color:var(--text-secondary);">Naïve baseline — simple reference to benchmark model outperformance.</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ── ANOMALY TAB ──────────────────────────
    with tab_anomalies:
        count = len(anomalies_df)
        zero_class = "zero" if count == 0 else ""
        summary_text = "No anomalies detected in the backtest window — delinquency stayed within expected bands." if count == 0 else f"{count} period{'s' if count != 1 else ''} where delinquency broke outside the forecast band plus margin."

        st.markdown(
            f"""
            <div class="anomaly-summary">
                <div class="count-bubble {zero_class}">{count}</div>
                <div class="count-text">
                    <div class="main">{'✅ Clean window' if count == 0 else '⚠️ Anomalies detected'}</div>
                    <div class="sub">{summary_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if anomalies_df.empty:
            st.markdown(
                """
                <div style="text-align:center; padding:48px 20px;">
                    <div style="font-size:40px; margin-bottom:12px;">🟢</div>
                    <div style="font-size:16px; font-weight:600; color:var(--green); margin-bottom:6px;">All Clear</div>
                    <div style="font-size:13px; color:var(--text-secondary);">Delinquency remained within expected forecast bands during the backtest window.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.dataframe(_prepare_anomaly_display(anomalies_df), use_container_width=True)

        st.caption("Anomaly = actual delinquency rate exceeded the upper forecast band + anomaly margin, or fell below the lower band.")

    # ── SCENARIO TAB ─────────────────────────
    with tab_scenario:
        if not scenario_df.empty:
            baseline_avg = float(scenario_df["baseline_delinquency"].mean())
            stressed_avg = float(scenario_df["stressed_delinquency"].mean())
            avg_delta_pct = stressed_avg - baseline_avg

            st.markdown(
                f"""
                <div class="scenario-compare">
                    <div class="scenario-card baseline">
                        <div class="sc-label">📊 Baseline Forecast (avg)</div>
                        <div class="sc-value">{baseline_avg:.2%}</div>
                        <div class="sc-sub">No rate change applied</div>
                    </div>
                    <div class="scenario-arrow">
                        →<br>
                        <span style="font-size:13px; color:#FFB547; font-weight:700;">+{delta_rate:.1f}%<br>rate shock</span>
                    </div>
                    <div class="scenario-card stressed">
                        <div class="sc-label">⚡ Stressed Forecast (avg)</div>
                        <div class="sc-value">{stressed_avg:.2%}</div>
                        <div class="sc-sub">After interest rate shock</div>
                        <div><span class="scenario-delta-badge">▲ +{avg_delta_pct:.2%} avg uplift</span></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown('<div class="chart-wrap">', unsafe_allow_html=True)
        st.plotly_chart(_build_scenario_figure(scenario_df, theme=active_theme), use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

        if not scenario_df.empty:
            st.markdown("<br>", unsafe_allow_html=True)
            scenario_table = scenario_df.copy()
            scenario_table["date"] = pd.to_datetime(scenario_table["date"]).dt.strftime("%Y-%m-%d")
            for col in ["baseline_delinquency", "stressed_delinquency", "delta"]:
                scenario_table[col] = scenario_table[col].map(lambda v: f"{float(v):.2%}")
            st.dataframe(scenario_table, use_container_width=True)

        st.caption(f"Stress model in use: {scenario_used}. The shaded area represents the additional delinquency risk introduced by the rate shock.")

    # ── SUMMARY TAB ──────────────────────────
    with tab_summary:
        fallback_summary = _default_text_summary(
            segment_id=selected_segment,
            forecast_result=forecast_result,
            anomalies_df=anomalies_df,
            scenario_df=scenario_df,
            delta_rate=delta_rate,
        )
        summary_payload = {
            "segment": selected_segment,
            "next_forecast": forecast_result.forecast_df.iloc[0].to_dict(),
            "anomalies": len(anomalies_df),
            "avg_stress_delta": avg_stress_delta,
        }
        final_summary = _maybe_generate_llm_summary(fallback_summary, summary_payload)

        st.markdown(
            f"""
            <div class="exec-card">
                <div class="exec-header">
                    <span style="font-size:20px;">📋</span>
                    <div class="exec-title">Executive Summary</div>
                    <div class="exec-badge">{'AI Generated' if os.getenv('GEMINI_API_KEY') else 'Auto Generated'}</div>
                </div>
                <div class="exec-body">{final_summary}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            f"""
            <div style="margin-bottom: 20px;">
                <span class="meta-pill">🏷️ Segment: <span>{selected_segment}</span></span>
                <span class="meta-pill">🤖 Forecast model: <span>{forecast_result.model_name}</span></span>
                <span class="meta-pill">⚡ Stress model: <span>{scenario_used}</span></span>
                <span class="meta-pill">📅 Horizon: <span>{horizon_weeks} weeks</span></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:13px; font-weight:700; color:var(--text-primary); margin-bottom:12px; letter-spacing:-0.01em;'>📊 Backtest Metrics — All Segments</div>",
            unsafe_allow_html=True,
        )

        metrics = evaluate_models(modeled, test_periods=backtest_periods, risk_threshold=risk_threshold)
        if metrics.empty:
            st.markdown(
                """
                <div style="text-align:center; padding:32px; background:rgba(255,255,255,0.02); border:1px dashed rgba(255,255,255,0.07); border-radius:12px;">
                    <div style="font-size:13px; color:#4A5A78;">Not enough observations to compute backtest metrics yet.<br>Try reducing 'Min. observations' or 'Backtest periods'.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            display_metrics = metrics.copy()
            for col in ["model_mae", "model_rmse", "naive_mae", "naive_rmse", "rolling_mae", "rolling_rmse"]:
                display_metrics[col] = display_metrics[col].map(lambda v: f"{float(v):.4f}")
            for col in ["model_mape", "naive_mape", "rolling_mape"]:
                display_metrics[col] = display_metrics[col].map(lambda v: f"{float(v):.2f}%")
            display_metrics["roc_auc"] = display_metrics["roc_auc"].map(
                lambda v: "n/a" if pd.isna(v) else f"{float(v):.3f}"
            )
            st.dataframe(display_metrics, use_container_width=True)


if __name__ == "__main__":
    main()