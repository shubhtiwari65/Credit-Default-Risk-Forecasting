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


def _apply_custom_style() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        :root {
            --bg-main: #0B1120;
            --bg-sidebar: #111827;
            --text-heading: #F8FAFC;
            --text-body: #CBD5E1;
            --accent-primary: #14B8A6; /* Teal accent */
            --accent-secondary: #0D9488;
            --card-bg: #1E293B;
            --card-border: #334155;
            --success: #10B981;
            --warning: #F59E0B;
            --danger: #EF4444;
            --chart-bg: #1E293B;
        }

        /* Streamlit global resets */
        .stApp {
            background-color: var(--bg-main) !important;
        }

        h1, h2, h3, h4, h5, h6 {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-heading) !important;
            font-weight: 700 !important;
            letter-spacing: -0.02em;
        }
        
        .sidebar-caption {
            font-family: 'Inter', sans-serif !important;
            font-size: 14px;
            color: var(--text-body);
            margin-top: -12px;
            margin-bottom: 24px;
            font-weight: 500;
        }

        p, div, label, span, li, button {
            font-family: 'Inter', sans-serif !important;
            color: var(--text-body);
            font-weight: 500;
        }

        /* Override slider and radio label colors explicitly */
        .stSlider label, .stRadio label, .stFileUploader label, .stSelectbox label {
            color: var(--text-heading) !important;
            font-weight: 600 !important;
        }
        
        /* Subheaders inside sidebar */
        [data-testid="stSidebar"] h3 {
            color: var(--text-heading) !important;
            font-size: 16px !important;
            font-weight: 700 !important;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 16px;
        }

        [data-testid="stSidebar"] {
            background-color: var(--bg-sidebar) !important;
            border-right: 1px solid var(--card-border);
            box-shadow: 2px 0 15px rgba(0, 0, 0, 0.2);
            width: 320px !important;
        }

        div[data-testid="metric-container"] {
            background-color: var(--card-bg);
            border: 1px solid var(--card-border);
            border-radius: 12px;
            padding: 16px 20px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }
        
        div[data-testid="metric-container"]::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background-color: var(--accent-primary);
            opacity: 0.8;
        }
        
        div[data-testid="metric-container"]:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2), 0 4px 6px -4px rgba(0, 0, 0, 0.1);
        }

        div[data-testid="stMetricValue"] {
            color: var(--text-heading) !important;
            font-weight: 800;
            font-size: 2.2rem !important;
            letter-spacing: -0.03em;
        }
        
        div[data-testid="stMetricLabel"] {
            color: var(--text-body) !important;
            font-weight: 600;
        }

        div[data-testid="stMetricDelta"] svg {
            color: var(--accent-primary) !important;
        }
        
        div[data-testid="stMetricDelta"] {
            color: var(--accent-primary) !important;
        }

        .hero-card {
            background: linear-gradient(135deg, #1E293B, #111827);
            border: 1px solid var(--card-border);
            border-radius: 16px;
            padding: 32px;
            margin: 0px 0 24px 0;
            color: #FFFFFF;
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.3);
            animation: riseIn 600ms ease-out;
            position: relative;
            overflow: hidden;
        }

        .hero-card::after {
            content: '';
            position: absolute;
            top: 0px;
            right: 0px;
            width: 40%;
            height: 100%;
            background: radial-gradient(circle at 80% 20%, rgba(20, 184, 166, 0.15) 0%, transparent 60%);
            pointer-events: none;
        }

        .hero-title {
            font-family: 'Inter', sans-serif !important;
            font-size: 28px;
            font-weight: 800;
            margin-bottom: 8px;
            color: var(--text-heading);
            letter-spacing: -0.03em;
        }

        .hero-sub {
            font-size: 16px;
            font-weight: 400;
            opacity: 0.85;
            margin-bottom: 24px;
            color: var(--text-body);
            max-width: 650px;
            line-height: 1.5;
        }

        .hero-tags {
            display: flex;
            gap: 10px;
            flex-wrap: wrap;
        }

        .hero-tag {
            background: rgba(255, 255, 255, 0.05);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 999px; /* Pill shape */
            padding: 6px 16px;
            font-size: 13px;
            font-weight: 600;
            color: var(--text-heading);
            backdrop-filter: blur(4px);
            transition: background 0.2s;
        }
        
        .hero-tag:hover {
            background: rgba(20, 184, 166, 0.15);
            border: 1px solid rgba(20, 184, 166, 0.3);
        }

        .note-card {
            background-color: var(--card-bg);
            border-left: 4px solid var(--accent-primary);
            border-radius: 8px;
            padding: 16px 20px;
            margin-top: 16px;
            box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1);
            font-size: 14px;
            color: var(--text-body);
            border-right: 1px solid var(--card-border);
            border-top: 1px solid var(--card-border);
            border-bottom: 1px solid var(--card-border);
        }

        .summary-card {
            background-color: var(--card-bg);
            border-top: 4px solid var(--accent-primary);
            border-radius: 12px;
            padding: 24px;
            margin-top: 12px;
            box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.2);
            border-left: 1px solid var(--card-border);
            border-right: 1px solid var(--card-border);
            border-bottom: 1px solid var(--card-border);
            animation: riseIn 600ms ease-out;
            font-size: 15px;
            line-height: 1.6;
            color: var(--text-heading);
        }

        .side-tip {
            background-color: rgba(255, 255, 255, 0.03);
            border: 1px dashed var(--card-border);
            border-radius: 8px;
            padding: 12px;
            margin-top: 24px;
            font-size: 13px;
            color: var(--text-body);
        }

        /* Tabs styling */
        div[data-baseweb="tab-list"] {
            gap: 8px;
            padding-bottom: 16px;
            margin-top: 16px;
        }

        button[role="tab"] {
            border-radius: 8px !important;
            border: 1px solid var(--card-border) !important;
            background: var(--card-bg) !important;
            padding: 10px 20px !important;
            font-weight: 600 !important;
            color: var(--text-body) !important;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
            transition: all 0.2s ease !important;
        }
        
        button[role="tab"]:hover {
            border-color: var(--accent-primary) !important;
            color: var(--text-heading) !important;
        }

        button[role="tab"][aria-selected="true"] {
            color: #FFFFFF !important;
            background: var(--accent-primary) !important;
            border-color: var(--accent-primary) !important;
            box-shadow: 0 4px 6px -1px rgba(20, 184, 166, 0.3) !important;
        }

        /* Plotly background override */
        .js-plotly-plot .plotly .bg {
            fill: var(--card-bg) !important;
        }
        .js-plotly-plot .plotly .scrollbar {
            display: none !important;
        }

        @keyframes riseIn {
            from {opacity: 0; transform: translateY(10px);}
            to {opacity: 1; transform: translateY(0);}
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_hero() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-title">Segment Delinquency Intelligence</div>
            <div class="hero-sub">Forecast short-horizon delinquency, catch anomalies early, and stress test rate shocks with explainable outputs.</div>
            <div class="hero-tags">
                <span class="hero-tag">Forecast Bands</span>
                <span class="hero-tag">Anomaly Alerts</span>
                <span class="hero-tag">Rate Stress Test</span>
                <span class="hero-tag">Manager Summary</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


import io

@st.cache_data(show_spinner="Loading dataset...")
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


@st.cache_data(show_spinner="Forecasting...")
def get_cached_forecast(_segment_df: pd.DataFrame, horizon: int) -> ForecastResult:
    return forecast_segment_delinquency(_segment_df.copy(), horizon_weeks=horizon)

@st.cache_data(show_spinner="Scanning anomalies...")
def get_cached_anomalies(_segment_df: pd.DataFrame, test_periods: int, margin: float):
    return detect_anomalies(_segment_df.copy(), test_periods=test_periods, margin=margin)

@st.cache_data(show_spinner="Running stress test...")
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


def _build_forecast_figure(segment_df: pd.DataFrame, result: ForecastResult) -> go.Figure:
    fig = go.Figure()

    forecast_start = pd.Timestamp(result.forecast_df["date"].iloc[0])
    forecast_end = pd.Timestamp(result.forecast_df["date"].iloc[-1])

    fig.add_vrect(
        x0=forecast_start,
        x1=forecast_end,
        fillcolor="rgba(255, 177, 0, 0.17)",
        line_width=0,
        layer="below",
    )

    fig.add_trace(
        go.Scatter(
            x=segment_df["date"],
            y=segment_df["delinquency_rate"],
            mode="lines+markers",
            name="Historical delinquency",
            line=dict(color="#0f4c81", width=3),
            marker=dict(size=6),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result.forecast_df["date"],
            y=result.forecast_df["central"],
            mode="lines+markers",
            name="Model forecast",
            line=dict(color="#c23b49", width=3),
            marker=dict(size=7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result.forecast_df["date"],
            y=result.forecast_df["baseline"],
            mode="lines",
            name="Baseline forecast",
            line=dict(color="#177245", dash="dot", width=2),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result.forecast_df["date"],
            y=result.forecast_df["upper"],
            mode="lines",
            line=dict(width=0),
            name="Upper band",
            showlegend=False,
        )
    )

    fig.add_trace(
        go.Scatter(
            x=result.forecast_df["date"],
            y=result.forecast_df["lower"],
            mode="lines",
            line=dict(width=0),
            fill="tonexty",
            fillcolor="rgba(194, 59, 73, 0.18)",
            name="80% interval",
        )
    )

    fig.update_layout(
        title="Delinquency forecast with uncertainty band",
        xaxis_title="Date",
        yaxis_title="Delinquency rate",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".1%")
    return fig


def _build_scenario_figure(scenario_df: pd.DataFrame) -> go.Figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=scenario_df["date"],
            y=scenario_df["baseline_delinquency"],
            mode="lines+markers",
            name="Baseline",
            line=dict(color="#0f4c81", width=3),
            marker=dict(size=7),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=scenario_df["date"],
            y=scenario_df["stressed_delinquency"],
            mode="lines+markers",
            name="Stress (+rate shock)",
            line=dict(color="#c23b49", width=3),
            marker=dict(size=7),
            fill="tonexty",
            fillcolor="rgba(194, 59, 73, 0.15)",
        )
    )

    fig.update_layout(
        title="Scenario analysis: baseline vs interest-rate shock",
        xaxis_title="Date",
        yaxis_title="Forecast delinquency",
        template="plotly_white",
        margin=dict(l=20, r=20, t=60, b=20),
        hovermode="x unified",
    )
    fig.update_yaxes(tickformat=".1%")
    return fig


def _prepare_anomaly_display(anomalies_df: pd.DataFrame) -> pd.DataFrame:
    if anomalies_df.empty:
        return anomalies_df

    display = anomalies_df.copy()
    display["date"] = pd.to_datetime(display["date"]).dt.strftime("%Y-%m-%d")
    display["anomaly_type"] = display["anomaly_type"].str.upper()

    pct_cols = ["actual", "expected", "lower", "upper", "delta"]
    for col in pct_cols:
        display[col] = display[col].map(lambda v: f"{float(v):.2%}")

    return display[
        ["date", "anomaly_type", "actual", "expected", "lower", "upper", "delta", "driver_hints"]
    ]


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

    summary = (
        f"Segment {segment_id}: next forecasted delinquency is {next_point['central']:.2%} "
        f"with an 80% band of {next_point['lower']:.2%} to {next_point['upper']:.2%}. "
        f"Recent anomaly count in the selected window is {anomaly_count}. "
        f"Under a +{delta_rate:.2f}% interest-rate shock, expected delinquency changes by "
        f"about {avg_stress_delta:.2%} on average over the forecast horizon."
    )
    return summary


@st.cache_data(show_spinner="Generating AI summary...", ttl="1h")
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


def main() -> None:
    st.set_page_config(
        page_title="Segment Delinquency Forecaster",
        page_icon="N",
        layout="wide",
    )
    _apply_custom_style()

    _render_hero()

    with st.sidebar:
        st.header("Control Center")
        st.markdown("<div class='sidebar-caption'>Configure data and risk settings</div>", unsafe_allow_html=True)
        st.markdown("### Data source")
        data_source = st.radio(
            "Select dataset",
            label_visibility="collapsed",
            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Upload segment-month CSV",
            type=["csv"],
            disabled=data_source != DATA_SOURCE_UPLOAD,
        )

        st.divider()
        st.markdown("### Model controls")
        horizon_weeks = st.slider("Forecast horizon (weeks)", min_value=4, max_value=8, value=6)
        delta_rate = st.slider(
            "Interest-rate shock (+%)", min_value=0.0, max_value=2.0, value=0.5, step=0.1
        )
        anomaly_margin = st.slider(
            "Anomaly margin above/below band",
            min_value=0.0,
            max_value=0.02,
            value=0.0025,
            step=0.0005,
            format="%.4f",
        )
        min_observations = st.slider(
            "Minimum observations per segment",
            min_value=8,
            max_value=24,
            value=12,
        )
        backtest_periods = st.slider(
            "Backtest periods",
            min_value=3,
            max_value=12,
            value=6,
        )
        risk_threshold = st.slider(
            "High-risk threshold",
            min_value=0.03,
            max_value=0.20,
            value=0.08,
            step=0.005,
        )

        st.markdown(
            """
            <div class="side-tip">
                Tip: Start with Extended sample dataset for richer behavior, then switch to Upload your own CSV for real data.
            </div>
            """,
            unsafe_allow_html=True,
        )

    if data_source == DATA_SOURCE_UPLOAD and uploaded_file is None:
        st.info("Upload mode is selected. Please choose a CSV file in the sidebar to continue.")
        st.stop()

    try:
        file_bytes = uploaded_file.getvalue() if uploaded_file is not None else None
        data = _load_selected_dataset(data_source=data_source, uploaded_file_bytes=file_bytes)
    except Exception as exc:
        st.error(f"Could not load CSV: {exc}")
        st.caption(
            "Expected columns: date, segment_id, repayment_rate, delinquency_rate, income_to_debt_ratio, avg_interest_rate"
        )
        st.stop()

    if data.empty:
        st.info("Upload a CSV or enable sample dataset to continue.")
        st.stop()

    modeled = filter_sparse_segments(data, min_observations=min_observations)
    segment_options = list_segments(modeled)

    if not segment_options:
        st.error("No segment has enough observations after filtering.")
        st.stop()

    selected_segment = st.sidebar.selectbox("Customer segment", options=segment_options)
    segment_df = modeled[modeled["segment_id"].astype(str) == selected_segment].copy()
    segment_df = segment_df.sort_values("date").reset_index(drop=True)

    forecast_result = get_cached_forecast(segment_df, horizon_weeks)
    anomalies = get_cached_anomalies(
        segment_df,
        test_periods=min(backtest_periods, max(3, len(segment_df) // 3)),
        margin=anomaly_margin,
    )
    anomalies_df = anomalies_to_frame(anomalies)

    scenario_df, scenario_used = get_cached_scenario(
        segment_df, forecast_result.forecast_df, delta_rate
    )

    next_point = forecast_result.forecast_df.iloc[0]
    last_actual = float(segment_df["delinquency_rate"].iloc[-1])
    avg_stress_delta = float(scenario_df["delta"].mean()) if not scenario_df.empty else 0.0
    interval_width = float(next_point["upper"] - next_point["lower"])

    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Latest delinquency", f"{last_actual:.2%}")
    kpi2.metric("Next forecast", f"{float(next_point['central']):.2%}", f"{float(next_point['central'] - last_actual):+.2%}")
    kpi3.metric("80% interval width", f"{interval_width:.2%}")
    kpi4.metric("Avg stress uplift", f"{avg_stress_delta:.2%}")
    kpi5.metric("Anomalies in window", str(len(anomalies_df)))

    tab_forecast, tab_anomalies, tab_scenario, tab_summary = st.tabs(
        ["Forecast view", "Anomaly monitor", "Scenario lab", "Summary & backtest"]
    )

    with tab_forecast:
        st.plotly_chart(_build_forecast_figure(segment_df, forecast_result), use_container_width=True)
        st.markdown(
            """
            <div class="note-card">
                How to read this chart: the blue line is observed history, red is model forecast, green dotted is baseline,
                and the shaded red band is the uncertainty range. If future actuals break above the band,
                risk is rising faster than expected.
            </div>
            """,
            unsafe_allow_html=True,
        )

    with tab_anomalies:
        st.subheader("Detected anomalies")
        if anomalies_df.empty:
            st.success("No anomalies in the selected backtest window.")
        else:
            st.dataframe(_prepare_anomaly_display(anomalies_df), use_container_width=True)
        st.caption("High anomaly means actual delinquency is above the upper forecast band plus margin.")

    with tab_scenario:
        st.plotly_chart(_build_scenario_figure(scenario_df), use_container_width=True)
        scenario_table = scenario_df.copy()
        scenario_table["date"] = pd.to_datetime(scenario_table["date"]).dt.strftime("%Y-%m-%d")
        for col in ["baseline_delinquency", "stressed_delinquency", "delta"]:
            scenario_table[col] = scenario_table[col].map(lambda v: f"{float(v):.2%}")
        st.dataframe(scenario_table, use_container_width=True)
        st.caption(f"Stress model in use: {scenario_used}.")

    with tab_summary:
        st.subheader("Manager summary")
        st.markdown(
            f"""
            <div class="note-card">
                Segment selected: <b>{selected_segment}</b><br>
                Forecast model: <b>{forecast_result.model_name}</b><br>
                Stress model: <b>{scenario_used}</b>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
            "avg_stress_delta": float(scenario_df["delta"].mean()),
        }
        final_summary = _maybe_generate_llm_summary(fallback_summary, summary_payload)

        st.markdown(f"<div class='summary-card'>{final_summary}</div>", unsafe_allow_html=True)

        st.subheader("Backtest metrics (all segments)")
        metrics = evaluate_models(modeled, test_periods=backtest_periods, risk_threshold=risk_threshold)
        if metrics.empty:
            st.write("Not enough observations to compute backtests yet.")
        else:
            display_metrics = metrics.copy()
            for col in [
                "model_mae",
                "model_rmse",
                "naive_mae",
                "naive_rmse",
                "rolling_mae",
                "rolling_rmse",
            ]:
                display_metrics[col] = display_metrics[col].map(lambda v: f"{float(v):.4f}")

            for col in ["model_mape", "naive_mape", "rolling_mape"]:
                display_metrics[col] = display_metrics[col].map(lambda v: f"{float(v):.2f}%")

            display_metrics["roc_auc"] = display_metrics["roc_auc"].map(
                lambda v: "n/a" if pd.isna(v) else f"{float(v):.3f}"
            )

            st.dataframe(display_metrics, use_container_width=True)


if __name__ == "__main__":
    main()
