# Credit Default Risk Forecasting

Segment-level credit delinquency forecasting, anomaly monitoring, and interest-rate shock scenario analysis in an interactive Streamlit dashboard.

## Overview

This project helps risk teams detect early signs of portfolio stress through:

- short-horizon delinquency forecasting
- anomaly detection against forecast bands
- interest-rate stress simulation
- segment-level backtesting metrics

The focus is explainability and decision support, not black-box modeling.

## Key Features

- Forecasts delinquency rates using SARIMAX with baseline fallbacks
- Produces forecast intervals for uncertainty-aware monitoring
- Flags anomalies in recent periods with configurable tolerance
- Simulates delinquency shifts under interest-rate shocks
- Reports MAE, RMSE, MAPE, ROC-AUC, and confusion-matrix counts
- Provides a four-tab Streamlit experience:
  - Forecast View
  - Anomaly Monitor
  - Scenario Lab
  - Executive Summary

## Repository Structure

```text
assets/
  architecture.png
  sample_dataset.csv
  demo_dataset_extended.csv
  forecast_panel.svg
  anomalies_panel.svg
  scenario_panel.svg
scripts/
src/
  streamlit_app.py
  data_loader.py
  forecasting.py
  anomalies.py
  scenarios.py
  evaluation.py
tests/
  test_pipeline.py
  test_evaluation.py
  test_upload_resilience.py
.env.example
requirements.txt
README.md
LICENSE
```

## Architecture

![Project Architecture](assets/architecture.png)

### Data Flow

1. Load data from starter sample, extended sample, or uploaded CSV.
2. Normalize headers, coerce datatypes, and validate required fields.
3. Filter sparse segments and select a segment in the UI.
4. Forecast delinquency with SARIMAX (or fallback baseline).
5. Detect anomalies via rolling-origin checks.
6. Run interest-rate stress scenarios.
7. Render dashboard views and executive summary.

## Installation

### Prerequisites

- Python 3.9+
- pip

### Setup

```bash
git clone https://github.com/shubhtiwari65/Credit-Default-Risk-Forecasting.git
cd Credit-Default-Risk-Forecasting

python -m venv venv
# Windows PowerShell
venv\Scripts\Activate.ps1
# macOS/Linux
# source venv/bin/activate

pip install -r requirements.txt
```

## Run the Dashboard

```bash
streamlit run src/streamlit_app.py
```

Open http://localhost:8501 in your browser.

## Evaluate from CLI

```bash
python -m src.evaluation --input assets/sample_dataset.csv --test-periods 6 --risk-threshold 0.08
```

## Run Tests

```bash
pytest -q
```

## Input Schema

Required columns:

- date
- segment_id
- repayment_rate
- delinquency_rate
- income_to_debt_ratio
- avg_interest_rate

Optional macro columns:

- unemployment_rate
- gdp_growth

Notes:

- Rate fields can be provided as percentages (for example, 5.2%) or 0-1 decimals.
- Dates are normalized to month-end timestamps.
- Common uploaded-header variants are normalized (for example, segment to segment_id).

## Tech Stack

- Streamlit
- Pandas, NumPy
- Statsmodels
- Scikit-learn
- Plotly
- Pytest
- Google Generative AI (summary generation with fallback)

## Current Limitations

- Forecast horizon is short term (configured via week-to-month mapping).
- Sparse segments are excluded by minimum-observation filtering.
- SARIMAX order is fixed to a simple configuration.
- Scenario engine is sensitivity-based and not causal.

## AI Summary Behavior

- The executive summary attempts LLM-based generation.
- If the LLM call fails, the app falls back to deterministic summary text.

## License

Released under Apache License 2.0. See LICENSE.

## Support

- Issues: https://github.com/shubhtiwari65/Credit-Default-Risk-Forecasting/issues
- Email: shubhtiwari651@gmail.com
