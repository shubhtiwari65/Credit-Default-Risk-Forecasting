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
    if theme == "light":
        theme_vars = (
            "--bg-main:#F0F4FA;"
            "--bg-panel:#FFFFFF;"
            "--bg-card:#FFFFFF;"
            "--bg-card-hover:#F8FAFC;"
            "--bg-soft:#EBF2FF;"
            "--border:#CBD5E1;"
            "--text-primary:#0F172A;"
            "--text-secondary:#1E293B;"
            "--text-muted:#475569;"
            "--shadow-sm:0 4px 14px rgba(15,23,42,0.08);"
            "--shadow-md:0 16px 36px rgba(15,23,42,0.12);"
        )
    else:
        theme_vars = (
            "--bg-main:#0B1320;"
            "--bg-panel:#111D30;"
            "--bg-card:#152238;"
            "--bg-card-hover:#1A2A43;"
            "--bg-soft:#1A2A43;"
            "--border:rgba(148,163,184,0.22);"
            "--text-primary:#E2E8F0;"
            "--text-secondary:#C3D2E6;"
            "--text-muted:#8EA4C0;"
            "--shadow-sm:0 6px 18px rgba(2,6,23,0.28);"
            "--shadow-md:0 20px 42px rgba(2,6,23,0.36);"
        )

    st.markdown("<style>:root{" + theme_vars + "}</style>", unsafe_allow_html=True)

    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {
            --primary:#1D4ED8;
            --primary-strong:#1E40AF;
            --primary-soft:rgba(29,78,216,0.10);
            --primary-border:rgba(29,78,216,0.24);
            --success:#059669;
            --success-soft:rgba(5,150,105,0.12);
            --warning:#D97706;
            --warning-soft:rgba(217,119,6,0.14);
            --danger:#DC2626;
            --danger-soft:rgba(220,38,38,0.12);
            --font:'Manrope', sans-serif;
            --mono:'IBM Plex Mono', monospace;
        }

        html, body, [data-testid="stAppViewContainer"], .stApp, .main {
            background-color: var(--bg-main) !important;
            font-family: var(--font) !important;
            transition: background-color 0.25s ease, color 0.25s ease;
        }

        header[data-testid="stHeader"],
        [data-testid="stToolbar"],
        [data-testid="stDecoration"],
        [data-testid="stStatusWidget"],
        #MainMenu,
        .stDeployButton,
        footer,
        [data-testid="stSidebarNav"],
        [class*="keyboard"],
        button[aria-label="keyboard shortcuts"],
        [data-testid="stSidebar"] [data-testid="stSidebarHeader"],
        [data-testid="stSidebar"] [data-testid="stSidebarNavItems"] {
            display: none !important;
        }

        .block-container {
            max-width: 1420px !important;
            padding: 0 1.9rem 3.1rem 1.9rem !important;
        }

        h1, h2, h3, h4, h5, h6,
        p, div, label, span, li, button, input, select, textarea {
            font-family: var(--font) !important;
        }

        [data-testid="stSidebar"] {
            background: var(--bg-panel) !important;
            border-right: 1px solid var(--border) !important;
            padding-top: 0 !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            padding-top: 0 !important;
        }

        .sidebar-logo {
            margin: -1rem -1rem 8px -1rem;
            padding: 20px 20px 16px;
            border-bottom: 1px solid var(--border);
            background: var(--bg-card);
        }

        .sidebar-logo .logo-title {
            color: var(--text-primary);
            font-size: 15px;
            font-weight: 800;
            letter-spacing: -0.01em;
            line-height: 1.25;
        }

        .sidebar-logo .logo-sub {
            color: var(--text-muted);
            font-size: 11px;
            font-weight: 500;
            margin-top: 3px;
        }

        /* Radio button styling for both themes */
        [data-testid="stSidebar"] .stRadio > div {
            background: var(--bg-card) !important;
            border-radius: 8px !important;
            padding: 4px !important;
            border: 1px solid var(--border) !important;
        }

        /* Radio button labels */
        [data-testid="stSidebar"] .stRadio label {
            color: var(--text-primary) !important;
            font-weight: 600 !important;
            background: transparent !important;
        }

        [data-testid="stSidebar"] .stRadio label:hover {
            background: var(--bg-soft) !important;
        }

        [data-testid="stSidebar"] .stRadio label[data-checked="true"],
        [data-testid="stSidebar"] .stRadio div[data-checked="true"] label {
            color: var(--primary) !important;
            background: var(--primary-soft) !important;
        }

        /* Radio button text - force visibility */
        [data-testid="stSidebar"] .stRadio label span,
        [data-testid="stSidebar"] .stRadio label p,
        [data-testid="stSidebar"] .stRadio label div,
        [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] .stRadio [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
        }

        /* Horizontal radio buttons */
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] {
            background: var(--bg-card) !important;
        }

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
            color: var(--text-primary) !important;
            padding: 6px 12px !important;
            border-radius: 6px !important;
        }

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label span,
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] label p,
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] [data-testid="stMarkdownContainer"],
        [data-testid="stSidebar"] .stRadio [role="radiogroup"] [data-testid="stMarkdownContainer"] p {
            color: var(--text-primary) !important;
        }

        /* Divider styling */
        [data-testid="stSidebar"] hr {
            border-color: var(--border) !important;
            margin: 16px 0 !important;
        }

        .section-label {
            margin: 18px 0 10px;
            padding: 0 4px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            color: var(--text-muted);
        }

        [data-testid="stSidebar"] .stSlider label,
        [data-testid="stSidebar"] .stSelectbox label,
        [data-testid="stSidebar"] .stFileUploader label {
            font-size: 12px !important;
            font-weight: 700 !important;
            color: var(--text-secondary) !important;
            text-transform: uppercase;
            letter-spacing: 0.06em;
        }

        [data-testid="stSidebar"] [data-baseweb="select"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
            box-shadow: var(--shadow-sm);
        }

        [data-testid="stSidebar"] [data-baseweb="select"] > div {
            color: var(--text-primary) !important;
            background: var(--bg-card) !important;
        }

        /* Dropdown menu popup styling */
        [data-baseweb="popover"] {
            background: var(--bg-card) !important;
            border: 1px solid var(--border) !important;
            border-radius: 10px !important;
        }

        [data-baseweb="menu"] {
            background: var(--bg-card) !important;
        }

        [data-baseweb="menu"] li {
            background: var(--bg-card) !important;
            color: var(--text-primary) !important;
        }

        [data-baseweb="menu"] li:hover {
            background: var(--bg-soft) !important;
        }

        [data-baseweb="menu"] li[aria-selected="true"] {
            background: var(--primary-soft) !important;
            color: var(--primary) !important;
        }

        /* Select input text */
        [data-baseweb="select"] input {
            color: var(--text-primary) !important;
        }

        [data-baseweb="select"] [data-baseweb="icon"] {
            color: var(--text-muted) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-baseweb="slider"] > div > div:nth-child(3) {
            background: var(--primary) !important;
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
            background: var(--primary) !important;
            color: #FFFFFF !important;
            font-weight: 700;
        }

        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
        [data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
            color: var(--text-muted) !important;
            font-size: 11px !important;
        }

        [data-testid="stFileUploader"] section {
            background: var(--bg-card) !important;
            border: 1px dashed var(--primary-border) !important;
            border-radius: 10px !important;
            padding: 14px !important;
        }

        [data-testid="stFileUploader"] section button {
            background: var(--primary-soft) !important;
            border: 1px solid var(--primary-border) !important;
            color: var(--primary) !important;
            border-radius: 8px !important;
            font-weight: 700 !important;
            padding: 8px 16px !important;
            position: relative !important;
            overflow: hidden !important;
        }

        /* Fix duplicate text in file uploader button */
        [data-testid="stFileUploader"] section button span[data-testid="stMarkdownContainer"] {
            display: none !important;
        }

        [data-testid="stFileUploader"] section button > div {
            display: none !important;
        }

        [data-testid="stFileUploader"] section button::after {
            content: "Browse files" !important;
            display: block !important;
            color: var(--primary) !important;
            font-weight: 700 !important;
            font-size: 13px !important;
        }

        [data-testid="stFileUploader"] section small,
        [data-testid="stFileUploader"] section > div:not(:has(button)) {
            color: var(--text-muted) !important;
            font-size: 12px !important;
        }

        /* Hide the native file input text */
        [data-testid="stFileUploader"] input[type="file"] {
            color: transparent !important;
        }

        .sidebar-tip {
            margin-top: 16px;
            padding: 12px 14px;
            border-radius: 10px;
            border: 1px solid var(--primary-border);
            background: var(--primary-soft);
            color: var(--text-secondary);
            font-size: 12px;
            line-height: 1.55;
        }

        .sidebar-tip b {
            color: var(--primary);
            font-weight: 800;
        }

        .top-bar {
            display: flex;
            align-items: flex-end;
            justify-content: space-between;
            padding: 22px 0 12px;
            border-bottom: 1px solid var(--border);
            margin-bottom: 20px;
        }

        .app-kicker {
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.12em;
            color: var(--primary);
            margin-bottom: 4px;
        }

        .app-title {
            font-size: 28px;
            font-weight: 800;
            letter-spacing: -0.04em;
            color: var(--text-primary);
            line-height: 1.05;
        }

        .app-subtitle {
            margin-top: 6px;
            font-size: 13px;
            color: var(--text-muted);
            font-weight: 500;
        }

        .segment-badge {
            display: inline-flex;
            align-items: center;
            gap: 8px;
            padding: 7px 14px;
            border-radius: 999px;
            border: 1px solid var(--primary-border);
            background: var(--primary-soft);
            color: var(--primary-strong);
            font-size: 12px;
            font-weight: 700;
            letter-spacing: 0.03em;
        }

        .segment-badge::before {
            content: "";
            width: 7px;
            height: 7px;
            border-radius: 50%;
            background: var(--primary);
            box-shadow: 0 0 0 4px rgba(29,78,216,0.14);
        }

        .workflow-strip {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin-bottom: 18px;
        }

        .workflow-step {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 11px 13px;
            box-shadow: var(--shadow-sm);
        }

        .workflow-step.active {
            border-color: var(--primary-border);
            background: linear-gradient(180deg, var(--bg-card) 0%, var(--bg-soft) 100%);
        }

        .workflow-index {
            display: inline-flex;
            align-items: center;
            justify-content: center;
            width: 22px;
            height: 22px;
            border-radius: 50%;
            background: var(--primary-soft);
            color: var(--primary-strong);
            font-size: 11px;
            font-weight: 800;
            margin-bottom: 8px;
        }

        .workflow-title {
            font-size: 12px;
            font-weight: 700;
            color: var(--text-primary);
            letter-spacing: -0.01em;
            margin-bottom: 2px;
        }

        .workflow-note {
            font-size: 11px;
            color: var(--text-muted);
            line-height: 1.35;
        }

        .risk-panel {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow-sm);
            padding: 18px 20px;
            margin-bottom: 14px;
        }

        .risk-head {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 14px;
            gap: 10px;
        }

        .risk-title-wrap .risk-kicker {
            font-size: 10px;
            font-weight: 800;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.11em;
            margin-bottom: 4px;
        }

        .risk-title-wrap .risk-title {
            font-size: 17px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.02em;
        }

        .risk-chip {
            border-radius: 999px;
            padding: 6px 12px;
            font-size: 12px;
            font-weight: 800;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .risk-chip.low {
            background: var(--success-soft);
            color: var(--success);
            border: 1px solid rgba(5,150,105,0.30);
        }

        .risk-chip.medium {
            background: var(--warning-soft);
            color: var(--warning);
            border: 1px solid rgba(217,119,6,0.28);
        }

        .risk-chip.high {
            background: var(--danger-soft);
            color: var(--danger);
            border: 1px solid rgba(220,38,38,0.30);
        }

        .risk-score-row {
            display: grid;
            grid-template-columns: 1.4fr 1fr;
            gap: 12px;
            align-items: end;
            margin-bottom: 12px;
        }

        .risk-score {
            font-size: 34px;
            font-weight: 800;
            color: var(--text-primary);
            letter-spacing: -0.05em;
            line-height: 1;
        }

        .risk-context {
            font-size: 12px;
            color: var(--text-muted);
            margin-top: 4px;
        }

        .risk-label {
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .risk-track {
            width: 100%;
            height: 10px;
            border-radius: 999px;
            overflow: hidden;
            background: rgba(148,163,184,0.25);
            border: 1px solid rgba(148,163,184,0.20);
        }

        .risk-fill {
            height: 100%;
            border-radius: 999px;
        }

        .risk-fill.low {
            background: linear-gradient(90deg, #10B981, #34D399);
        }

        .risk-fill.medium {
            background: linear-gradient(90deg, #F59E0B, #FBBF24);
        }

        .risk-fill.high {
            background: linear-gradient(90deg, #EF4444, #DC2626);
        }

        .risk-foot {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-top: 8px;
            gap: 10px;
            font-size: 11px;
            color: var(--text-muted);
        }

        .insight-strip {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-left: 4px solid var(--primary);
            border-radius: 14px;
            box-shadow: var(--shadow-sm);
            padding: 16px 18px;
            display: flex;
            align-items: center;
            gap: 14px;
            margin-bottom: 18px;
        }

        .insight-status-dot {
            width: 12px;
            height: 12px;
            border-radius: 50%;
            flex-shrink: 0;
        }

        .insight-status-dot.safe {
            background: var(--success);
            box-shadow: 0 0 0 6px rgba(5,150,105,0.18);
        }

        .insight-status-dot.alert {
            background: var(--danger);
            box-shadow: 0 0 0 6px rgba(220,38,38,0.16);
        }

        .insight-label {
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.12em;
            text-transform: uppercase;
            color: var(--primary);
            margin-bottom: 3px;
        }

        .insight-main {
            font-size: 14px;
            line-height: 1.6;
            color: var(--text-secondary);
            font-weight: 500;
        }

        .insight-main b {
            color: var(--text-primary);
            font-weight: 700;
        }

        .kpi-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 16px 14px 14px;
            position: relative;
            overflow: hidden;
            box-shadow: var(--shadow-sm);
            transition: transform 0.2s ease, box-shadow 0.2s ease, border-color 0.2s ease;
        }

        .kpi-card:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            border-color: var(--primary-border);
        }

        .kpi-icon {
            width: 34px;
            height: 34px;
            border-radius: 8px;
            background: var(--bg-soft);
            border: 1px solid var(--border);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            color: var(--text-secondary);
            margin-bottom: 10px;
            font-family: var(--mono) !important;
        }

        .kpi-label {
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.10em;
            text-transform: uppercase;
            color: var(--text-muted);
            margin-bottom: 6px;
        }

        .kpi-value {
            font-size: 32px;
            font-weight: 800;
            line-height: 1;
            letter-spacing: -0.05em;
            margin-bottom: 9px;
        }

        .kpi-delta {
            display: inline-flex;
            align-items: center;
            gap: 4px;
            border-radius: 999px;
            padding: 4px 9px;
            font-size: 11px;
            font-weight: 700;
        }

        .kpi-delta.up {
            background: var(--danger-soft);
            color: var(--danger);
        }

        .kpi-delta.down {
            background: var(--success-soft);
            color: var(--success);
        }

        .kpi-delta.neutral {
            background: var(--primary-soft);
            color: var(--primary-strong);
        }

        .kpi-bar {
            position: absolute;
            bottom: 0;
            left: 0;
            width: 100%;
            height: 3px;
        }

        .kpi-card.color-primary .kpi-value { color: var(--primary); }
        .kpi-card.color-primary .kpi-bar { background: linear-gradient(90deg, var(--primary), transparent); }
        .kpi-card.color-danger .kpi-value { color: var(--danger); }
        .kpi-card.color-danger .kpi-bar { background: linear-gradient(90deg, var(--danger), transparent); }
        .kpi-card.color-success .kpi-value { color: var(--success); }
        .kpi-card.color-success .kpi-bar { background: linear-gradient(90deg, var(--success), transparent); }
        .kpi-card.color-warning .kpi-value { color: var(--warning); }
        .kpi-card.color-warning .kpi-bar { background: linear-gradient(90deg, var(--warning), transparent); }

        div[data-baseweb="tab-list"] {
            gap: 8px !important;
            border-bottom: 1px solid var(--border) !important;
            padding-bottom: 1px !important;
        }

        button[role="tab"] {
            background: transparent !important;
            border: none !important;
            border-bottom: 2px solid transparent !important;
            border-radius: 0 !important;
            color: var(--text-muted) !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            padding: 10px 14px !important;
            margin-bottom: -1px !important;
            transition: color 0.2s ease !important;
        }

        button[role="tab"]:hover {
            color: var(--text-secondary) !important;
        }

        button[role="tab"][aria-selected="true"] {
            color: var(--primary) !important;
            border-bottom-color: var(--primary) !important;
        }

        div[data-baseweb="tab-panel"] {
            padding-top: 22px !important;
        }

        .trend-indicator {
            display: inline-flex;
            align-items: center;
            gap: 7px;
            padding: 7px 12px;
            border-radius: 999px;
            font-size: 12px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .trend-indicator::before {
            content: "";
            width: 8px;
            height: 8px;
            border-radius: 50%;
            display: inline-block;
        }

        .trend-indicator.warning {
            background: var(--warning-soft);
            color: var(--warning);
            border: 1px solid rgba(217,119,6,0.30);
        }

        .trend-indicator.warning::before {
            background: var(--warning);
        }

        .trend-indicator.stable {
            background: var(--success-soft);
            color: var(--success);
            border: 1px solid rgba(5,150,105,0.30);
        }

        .trend-indicator.stable::before {
            background: var(--success);
        }

        .trend-indicator.danger {
            background: var(--danger-soft);
            color: var(--danger);
            border: 1px solid rgba(220,38,38,0.30);
        }

        .trend-indicator.danger::before {
            background: var(--danger);
        }

        .chart-wrap {
            border: 1px solid var(--border);
            border-radius: 14px;
            background: var(--bg-card);
            box-shadow: var(--shadow-sm);
            padding: 4px;
            overflow: hidden;
        }

        .legend-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 12px;
            margin-top: 16px;
        }

        .legend-card {
            border-radius: 10px;
            border: 1px solid var(--border);
            background: var(--bg-card);
            padding: 12px 14px;
        }

        .legend-head {
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 5px;
            font-size: 10px;
            font-weight: 800;
            letter-spacing: 0.11em;
            text-transform: uppercase;
        }

        .legend-dot {
            width: 9px;
            height: 9px;
            border-radius: 50%;
        }

        .legend-blue { color: #2563EB; }
        .legend-blue .legend-dot { background: #2563EB; }
        .legend-red { color: #DC2626; }
        .legend-red .legend-dot { background: #DC2626; }
        .legend-green { color: #059669; }
        .legend-green .legend-dot { background: #059669; }

        .legend-body {
            font-size: 12px;
            color: var(--text-secondary);
            line-height: 1.55;
        }

        .anomaly-summary {
            display: flex;
            align-items: center;
            gap: 12px;
            background: var(--bg-card);
            border: 1px solid var(--border);
            box-shadow: var(--shadow-sm);
            border-radius: 12px;
            padding: 14px 18px;
            margin-bottom: 15px;
        }

        .anomaly-summary .count-bubble {
            width: 50px;
            height: 50px;
            border-radius: 50%;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            font-weight: 800;
            background: var(--danger-soft);
            border: 1px solid rgba(220,38,38,0.34);
            color: var(--danger);
            flex-shrink: 0;
        }

        .anomaly-summary .count-bubble.zero {
            background: var(--success-soft);
            border-color: rgba(5,150,105,0.30);
            color: var(--success);
        }

        .anomaly-summary .count-text .main {
            color: var(--text-primary);
            font-size: 15px;
            font-weight: 700;
        }

        .anomaly-summary .count-text .sub {
            color: var(--text-muted);
            font-size: 12px;
            margin-top: 3px;
            line-height: 1.45;
        }

        .scenario-compare {
            display: grid;
            grid-template-columns: 1fr auto 1fr;
            gap: 14px;
            margin-bottom: 18px;
            align-items: center;
        }

        .scenario-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 12px;
            box-shadow: var(--shadow-sm);
            padding: 18px;
        }

        .scenario-card.stressed {
            border-color: rgba(220,38,38,0.28);
        }

        .sc-label {
            font-size: 10px;
            letter-spacing: 0.11em;
            text-transform: uppercase;
            color: var(--text-muted);
            font-weight: 800;
            margin-bottom: 10px;
        }

        .sc-value {
            font-size: 34px;
            font-weight: 800;
            letter-spacing: -0.05em;
            margin-bottom: 3px;
            line-height: 1;
        }

        .scenario-card.baseline .sc-value {
            color: var(--primary);
        }

        .scenario-card.stressed .sc-value {
            color: var(--danger);
        }

        .sc-sub {
            font-size: 12px;
            color: var(--text-muted);
        }

        .scenario-arrow {
            text-align: center;
            min-width: 90px;
        }

        .scenario-arrow .arrow-line {
            font-size: 26px;
            font-weight: 700;
            color: var(--warning);
            line-height: 1;
        }

        .scenario-arrow .arrow-note {
            margin-top: 4px;
            font-size: 11px;
            color: var(--text-muted);
            font-weight: 700;
            letter-spacing: 0.06em;
            text-transform: uppercase;
        }

        .scenario-delta-badge {
            display: inline-block;
            margin-top: 8px;
            border-radius: 999px;
            padding: 4px 12px;
            background: var(--danger-soft);
            border: 1px solid rgba(220,38,38,0.30);
            color: var(--danger);
            font-size: 12px;
            font-weight: 800;
        }

        .exec-card {
            background: var(--bg-card);
            border: 1px solid var(--border);
            border-radius: 14px;
            box-shadow: var(--shadow-sm);
            padding: 24px;
            margin-bottom: 18px;
            position: relative;
            overflow: hidden;
        }

        .exec-card::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: linear-gradient(90deg, var(--primary) 0%, #3B82F6 100%);
        }

        .exec-header {
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 14px;
        }

        .exec-icon {
            width: 28px;
            height: 28px;
            border-radius: 8px;
            border: 1px solid var(--primary-border);
            background: var(--primary-soft);
            color: var(--primary-strong);
            display: inline-flex;
            align-items: center;
            justify-content: center;
            font-size: 11px;
            font-weight: 800;
            font-family: var(--mono) !important;
        }

        .exec-title {
            font-size: 17px;
            font-weight: 800;
            color: var(--text-primary);
        }

        .exec-badge {
            margin-left: auto;
            border-radius: 999px;
            padding: 4px 10px;
            font-size: 10px;
            font-weight: 800;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            background: var(--primary-soft);
            border: 1px solid var(--primary-border);
            color: var(--primary-strong);
        }

        .exec-body {
            font-size: 14px;
            color: var(--text-secondary);
            line-height: 1.75;
        }

        .meta-pill {
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 6px 11px;
            margin-right: 8px;
            margin-bottom: 8px;
            border-radius: 8px;
            border: 1px solid var(--border);
            background: var(--bg-soft);
            color: var(--text-secondary);
            font-size: 12px;
        }

        .meta-pill b {
            color: var(--text-primary);
            font-weight: 700;
        }

        .stDataFrame {
            border-radius: 10px !important;
            overflow: hidden !important;
            border: 1px solid var(--border) !important;
        }

        .stDataFrame table {
            background: var(--bg-card) !important;
        }

        .stDataFrame thead tr th {
            background: var(--bg-soft) !important;
            color: var(--text-muted) !important;
            font-size: 10px !important;
            font-weight: 800 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.09em !important;
            border-bottom: 1px solid var(--border) !important;
        }

        .stDataFrame tbody tr td {
            color: var(--text-secondary) !important;
            font-size: 12px !important;
            font-family: var(--mono) !important;
            border-bottom: 1px solid var(--border) !important;
        }

        .stDataFrame tbody tr:hover td {
            background: var(--bg-card-hover) !important;
        }

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

        hr {
            border-color: var(--border) !important;
        }

        .stCaption {
            color: var(--text-muted) !important;
            font-size: 11px !important;
        }

        .stAlert {
            border-radius: 10px !important;
        }

        @keyframes fadeUp {
            from { opacity: 0; transform: translateY(12px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .anim { animation: fadeUp 0.45s ease both; }
        .anim-delay-1 { animation-delay: 0.06s; }
        .anim-delay-2 { animation-delay: 0.12s; }
        .anim-delay-3 { animation-delay: 0.18s; }

        @media (max-width: 1200px) {
            .workflow-strip {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }

            .scenario-compare {
                grid-template-columns: 1fr;
            }

            .scenario-arrow {
                min-width: 0;
            }

            .legend-grid {
                grid-template-columns: 1fr;
            }
        }

        @media (max-width: 768px) {
            .block-container {
                padding: 0 1rem 2.2rem 1rem !important;
            }

            .top-bar {
                flex-direction: column;
                align-items: flex-start;
                gap: 10px;
            }

            .risk-score-row {
                grid-template-columns: 1fr;
                gap: 10px;
            }

            .workflow-strip {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )



# ─────────────────────────────────────────────
#  DATA LOADING  (unchanged logic)
# ─────────────────────────────────────────────
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
    grid_color   = "rgba(15,23,42,0.09)"     if is_light else "rgba(148,163,184,0.18)"
    line_color   = "rgba(15,23,42,0.14)"     if is_light else "rgba(148,163,184,0.24)"
    font_color   = "#334155"                 if is_light else "#C3D2E6"
    hover_bg     = "#EFF6FF"                 if is_light else "#1A2A43"
    hover_border = "rgba(29,78,216,0.25)"    if is_light else "rgba(148,163,184,0.28)"
    marker_ring  = "#FFFFFF"                 if is_light else "#121212"

    forecast_start = pd.Timestamp(result.forecast_df["date"].iloc[0])
    forecast_end   = pd.Timestamp(result.forecast_df["date"].iloc[-1])

    fig.add_vrect(
        x0=forecast_start, x1=forecast_end,
        fillcolor="rgba(37,99,235,0.08)", line_width=0, layer="below",
    )

    # Historical
    fig.add_trace(go.Scatter(
        x=segment_df["date"], y=segment_df["delinquency_rate"],
        mode="lines+markers", name="Historical",
        line=dict(color="#2563EB", width=3.0),
        marker=dict(size=5, color="#2563EB"),
    ))

    # Bands
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["upper"],
        mode="lines", line=dict(width=0), showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["lower"],
        mode="lines", line=dict(width=0), name="80% confidence band",
        fill="tonexty", fillcolor="rgba(37,99,235,0.14)",
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["central"],
        mode="lines+markers", name="Model forecast",
        line=dict(color="#1E40AF", width=3.0),
        marker=dict(size=6, color="#1E40AF", line=dict(color=marker_ring, width=1.5)),
    ))

    # Baseline
    fig.add_trace(go.Scatter(
        x=result.forecast_df["date"], y=result.forecast_df["baseline"],
        mode="lines", name="Naive baseline",
        line=dict(color="#059669", dash="dot", width=1.8),
    ))

    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", color=font_color),
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
    grid_color   = "rgba(15,23,42,0.09)"  if is_light else "rgba(148,163,184,0.18)"
    line_color   = "rgba(15,23,42,0.14)"  if is_light else "rgba(148,163,184,0.24)"
    font_color   = "#334155"              if is_light else "#C3D2E6"
    hover_bg     = "#EFF6FF"              if is_light else "#1A2A43"
    hover_border = "rgba(29,78,216,0.25)" if is_light else "rgba(148,163,184,0.28)"
    marker_ring  = "#FFFFFF"              if is_light else "#121212"

    fig.add_trace(go.Scatter(
        x=scenario_df["date"], y=scenario_df["baseline_delinquency"],
        mode="lines+markers", name="Baseline",
        line=dict(color="#2563EB", width=2.5),
        marker=dict(size=6, color="#2563EB", line=dict(color=marker_ring, width=1.5)),
    ))

    fig.add_trace(go.Scatter(
        x=scenario_df["date"], y=scenario_df["stressed_delinquency"],
        mode="lines+markers", name="Rate shock applied",
        line=dict(color="#DC2626", width=3.0),
        marker=dict(size=6, color="#DC2626", line=dict(color=marker_ring, width=1.5)),
        fill="tonexty", fillcolor="rgba(220,38,38,0.11)",
    ))

    fig.update_layout(
        title=dict(text=""),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Manrope", color=font_color),
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


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Credit-Default-Risk-Forecasting",
        layout="wide",
    )
    _apply_custom_style(theme=st.session_state.get("theme", "light"))

    # ── SIDEBAR ──────────────────────────────
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
                <div class="logo-title">Credit-Default-Risk-Forecasting</div>
                <div class="logo-sub">Risk Forecasting Platform</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

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
                <b>Note:</b> Use the <em>Extended</em> dataset for demo depth. Select <em>Upload your own CSV</em> to analyze portfolio data.
            </div>
            """,
            unsafe_allow_html=True,
        )

        # ── Theme Toggle (at bottom) ──
        st.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)
        st.divider()
        st.markdown("<div class='section-label'>Appearance</div>", unsafe_allow_html=True)
        current_theme = st.session_state.get("theme", "light")
        theme_options = ["Light", "Dark"]
        selected_theme_label = st.radio(
            "Theme",
            options=theme_options,
            index=0 if current_theme == "light" else 1,
            horizontal=True,
            label_visibility="collapsed",
        )
        new_theme = "light" if selected_theme_label == "Light" else "dark"
        if new_theme != current_theme:
            st.session_state["theme"] = new_theme
            st.rerun()

    # ── DATA LOADING ─────────────────────────
    if data_source == DATA_SOURCE_UPLOAD and uploaded_file is None:
        st.markdown(
            """
            <div style="text-align:center; padding:72px 24px; background:var(--bg-card); border:1px dashed var(--border); border-radius:14px;">
                <div style="font-size:20px; font-weight:800; color:var(--text-primary); margin-bottom:8px;">No file selected</div>
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
        st.error(f"Could not load CSV: {exc}")
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
    forecast_probability = float(next_point["central"])
    risk_ratio = (forecast_probability / risk_threshold) if risk_threshold > 0 else 0.0
    risk_probability_pct = max(0.0, min(forecast_probability * 100.0, 100.0))

    if risk_ratio >= 1.0:
        risk_level = "High"
        risk_class = "high"
        risk_takeaway = "Forecast exceeds the configured high-risk threshold."
    elif risk_ratio >= 0.65:
        risk_level = "Medium"
        risk_class = "medium"
        risk_takeaway = "Forecast is approaching the high-risk threshold."
    else:
        risk_level = "Low"
        risk_class = "low"
        risk_takeaway = "Forecast remains comfortably below the high-risk threshold."

    insight_status = "alert" if (len(anomalies_df) > 0 or forecast_change > 0.005 or risk_level == "High") else "safe"
    risk_fill_width = min(max(risk_probability_pct, 1.0), 100.0)

    # ── TOP BAR ──────────────────────────────
    st.markdown(
        f"""
        <div class="top-bar">
            <div>
                <div class="app-title">Credit Default Risk Forecasting</div>
                <div class="app-subtitle">Portfolio-level delinquency outlook, anomaly monitoring, and stress testing</div>
            </div>
            <div class="segment-badge">Segment {selected_segment}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="workflow-strip anim">
            <div class="workflow-step active">
                <div class="workflow-index">1</div>
                <div class="workflow-title">Input</div>
                <div class="workflow-note">Select dataset, horizon, and risk assumptions.</div>
            </div>
            <div class="workflow-step active">
                <div class="workflow-index">2</div>
                <div class="workflow-title">Prediction</div>
                <div class="workflow-note">Generate segment-specific delinquency forecast.</div>
            </div>
            <div class="workflow-step active">
                <div class="workflow-index">3</div>
                <div class="workflow-title">Result</div>
                <div class="workflow-note">Review risk probability and confidence interval.</div>
            </div>
            <div class="workflow-step active">
                <div class="workflow-index">4</div>
                <div class="workflow-title">Insights</div>
                <div class="workflow-note">Inspect anomalies, scenario shifts, and summary.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="risk-panel anim anim-delay-1">
            <div class="risk-head">
                <div class="risk-title-wrap">
                    <div class="risk-kicker">Predicted Default Risk</div>
                    <div class="risk-title">Next Week Outlook</div>
                </div>
                <div class="risk-chip {risk_class}">{risk_level}</div>
            </div>
            <div class="risk-score-row">
                <div>
                    <div class="risk-score">{forecast_probability:.2%}</div>
                    <div class="risk-context">Forecasted delinquency probability for the selected segment.</div>
                </div>
                <div>
                    <div class="risk-label">Probability Scale</div>
                    <div class="risk-track">
                        <div class="risk-fill {risk_class}" style="width:{risk_fill_width:.1f}%"></div>
                    </div>
                    <div class="risk-foot">
                        <span>Threshold: {risk_threshold:.2%}</span>
                        <span>{risk_probability_pct:.1f}% / 100%</span>
                    </div>
                </div>
            </div>
            <div class="risk-context">{risk_takeaway}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── INSIGHT STRIP ────────────────────────
    trend_direction = "rising" if forecast_change > 0.005 else ("falling" if forecast_change < -0.005 else "stable")
    anomaly_txt = f"{len(anomalies_df)} anomaly{'s' if len(anomalies_df) != 1 else ''} detected" if anomalies_df.empty is False else "No anomalies detected"

    st.markdown(
        f"""
        <div class="insight-strip anim">
            <div class="insight-status-dot {insight_status}"></div>
            <div>
                <div class="insight-label">Key Insight | Segment {selected_segment}</div>
                <div class="insight-main">Next week forecast is <b>{float(next_point['central']):.2%}</b> with a <b>{trend_direction}</b> trend. {anomaly_txt} in the selected backtest window. Under a +{delta_rate:.1f}% rate shock, expected delinquency shifts by approximately <b>{avg_stress_delta:.2%}</b> on average.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ── KPI CARDS ────────────────────────────
    delta_sign = "UP" if forecast_change > 0 else "DOWN"
    delta_class = "up" if forecast_change > 0 else "down"
    anomaly_color = "color-danger" if len(anomalies_df) > 0 else "color-success"
    stress_color = "color-danger" if avg_stress_delta > 0.01 else "color-warning"
    delinquency_color = "color-danger" if last_actual > risk_threshold else "color-success"
    forecast_color = "color-danger" if float(next_point["central"]) > risk_threshold else "color-primary"

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
    _kpi_card(_kc1, delinquency_color, "DLQ", "Latest Delinquency",
              f"{last_actual:.2%}", "Current rate", "neutral")
    _kpi_card(_kc2, forecast_color, "NXT", "Next Forecast",
              f"{float(next_point['central']):.2%}",
              f"{delta_sign} {abs(forecast_change):.2%} vs now", delta_class)
    _kpi_card(_kc3, "color-warning", "BND", "80% Band Width",
              f"{interval_width:.2%}", "Uncertainty range", "neutral")
    _kpi_card(_kc4, stress_color, "SHK", "Rate Shock Impact",
              f"{avg_stress_delta:.2%}", f"+{delta_rate:.1f}% shock avg lift", "up")
    _kpi_card(_kc5, anomaly_color, "ANM", "Anomalies Found",
              str(len(anomalies_df)),
              "Needs review" if len(anomalies_df) > 0 else "All clear",
              "up" if len(anomalies_df) > 0 else "down")

    st.markdown("<div style='margin-bottom:12px'></div>", unsafe_allow_html=True)

    # ── TABS ─────────────────────────────────
    tab_forecast, tab_anomalies, tab_scenario, tab_summary = st.tabs([
        "Forecast View",
        "Anomaly Monitor",
        "Scenario Lab",
        "Executive Summary",
    ])

    active_theme = st.session_state.get("theme", "light")

    # ── FORECAST TAB ─────────────────────────
    with tab_forecast:
        # Trend status
        if forecast_change > 0.01:
            indicator_html = '<span class="trend-indicator danger">Risk Increasing</span>'
            guidance = "Delinquency is projected to rise meaningfully. Monitor closely and consider early interventions."
        elif forecast_change > 0.003:
            indicator_html = '<span class="trend-indicator warning">Slight Upward Drift</span>'
            guidance = "Modest increase projected. Maintain standard monitoring cadence."
        else:
            indicator_html = '<span class="trend-indicator stable">Stable Trend</span>'
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
            <div class="legend-grid">
                <div class="legend-card">
                    <div class="legend-head legend-blue"><span class="legend-dot"></span>Historical series</div>
                    <div class="legend-body">Observed delinquency trajectory for the selected segment.</div>
                </div>
                <div class="legend-card">
                    <div class="legend-head legend-red"><span class="legend-dot"></span>Forecast and interval</div>
                    <div class="legend-body">Model projection with the 80% confidence band for near-term risk estimation.</div>
                </div>
                <div class="legend-card">
                    <div class="legend-head legend-green"><span class="legend-dot"></span>Baseline benchmark</div>
                    <div class="legend-body">Reference path used to compare model lift and directional confidence.</div>
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
                    <div class="main">{'Window Within Band' if count == 0 else 'Anomalies Detected'}</div>
                    <div class="sub">{summary_text}</div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if anomalies_df.empty:
            st.markdown(
                """
                <div style="text-align:center; padding:44px 20px; background:var(--bg-card); border:1px solid var(--border); border-radius:12px;">
                    <div style="font-size:16px; font-weight:600; color:var(--success); margin-bottom:6px;">All Clear</div>
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
                        <div class="sc-label">Baseline Forecast (avg)</div>
                        <div class="sc-value">{baseline_avg:.2%}</div>
                        <div class="sc-sub">No rate change applied</div>
                    </div>
                    <div class="scenario-arrow">
                        <div class="arrow-line">-&gt;</div>
                        <div class="arrow-note">+{delta_rate:.1f}% rate shock</div>
                    </div>
                    <div class="scenario-card stressed">
                        <div class="sc-label">Stressed Forecast (avg)</div>
                        <div class="sc-value">{stressed_avg:.2%}</div>
                        <div class="sc-sub">After interest rate shock</div>
                        <div><span class="scenario-delta-badge">Uplift +{avg_delta_pct:.2%} average</span></div>
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
                    <span class="exec-icon">SUM</span>
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
                <span class="meta-pill">Segment: <b>{selected_segment}</b></span>
                <span class="meta-pill">Forecast model: <b>{forecast_result.model_name}</b></span>
                <span class="meta-pill">Stress model: <b>{scenario_used}</b></span>
                <span class="meta-pill">Horizon: <b>{horizon_weeks} weeks</b></span>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.markdown(
            "<div style='font-size:13px; font-weight:700; color:var(--text-primary); margin-bottom:12px; letter-spacing:-0.01em;'>Backtest Metrics - All Segments</div>",
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