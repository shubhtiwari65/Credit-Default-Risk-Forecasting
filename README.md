# Segment Delinquency Forecasting Dashboard

## 1. Overview

This project is a **credit-risk analytics application** designed for segment-level delinquency monitoring in banking and lending contexts. It forecasts short-horizon delinquency rates using SARIMAX models, detects anomalous repayment stress patterns, and simulates how default risk changes under interest-rate shocks. The dashboard provides explainable, actionable insights for portfolio managers and risk committees, enabling data-driven decision-making on credit exposure and risk mitigation strategies.

**Problem Solved:** Banks need early warning signals for deteriorating credit quality at the segment level. This tool provides interpretable forecasts with uncertainty bands and anomaly detection to catch emerging risks before they escalate.

**Intended Users:** Risk managers, credit portfolio analysts, compliance teams, and executive stakeholders in financial institutions.

## 2. Features

### ✅ Implemented & Working
- **Short-horizon delinquency forecasting** (4-8 weeks → 1-2 months) using SARIMAX(1,0,0) models
- **Explainable baseline forecasts** with Simple Exponential Smoothing fallback
- **80% prediction intervals** for uncertainty quantification
- **Anomaly detection** via rolling-origin evaluation with configurable margins
- **Interest-rate scenario stress testing** with sensitivity-based delinquency projections
- **Streamlit interactive dashboard** with real-time segment selection and parameter control
- **CSV upload support** for custom datasets with schema validation
- **Backtesting & evaluation** with MAE, RMSE, MAPE, and ROC-AUC metrics
- **Driver hints** identifying feature anomalies (income, interest rates) contributing to alerts
- **Optional AI summaries** via Google Gemini for non-technical stakeholders

### Dashboard Components
- Data source selector (sample datasets or custom CSV)
- Segment-level filtering and selection
- Forecast view with historical + predicted delinquency bands
- Anomaly monitor showing detected outliers and driver explanations
- Scenario laboratory for rate shock simulations
- Summary tab with backtesting results

## 3. Install and Run Instructions

### Prerequisites
- Python 3.9+
- pip package manager
- Virtual environment (recommended)

### Step 1: Clone the Repository
```bash
git clone https://github.com/shubhtiwari65/Credit-Default-Risk-Forecasting.git
cd Credit-Default-Risk-Forecasting
```

### Step 2: Create Virtual Environment
```bash
python -m venv venv
# On Windows:
venv\Scripts\Activate.ps1
# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 4: Run the Application
```bash
# Streamlit version (recommended for local use)
python -m streamlit run src/streamlit_app.py

# Opens at http://localhost:8501
```

### Step 5 (Optional): Configure Environment Variables
Copy `.env.example` to `.env` and set your API key for AI summaries:
```
GOOGLE_GENERATIVEAI_API_KEY=your_api_key_here
```

## 4. Tech Stack

### Programming & Frameworks
- **Language:** Python 3.11+
- **Frontend:** Streamlit (interactive dashboard)
- **Web Framework:** Flask (for Vercel deployment alternative)

### Data & ML Libraries
- **Data Processing:** Pandas, NumPy
- **Time Series Forecasting:** Statsmodels (SARIMAX, SimpleExpSmoothing)
- **Machine Learning:** Scikit-learn (regression, metrics)
- **Visualization:** Plotly (interactive charts)

### AI/LLM
- **Google Generative AI:** Gemini API (optional natural-language summaries)

### Cloud & Deployment
- **Deployment Platforms:** Streamlit Cloud, Vercel (Flask version)
- **Version Control:** Git/GitHub

### Testing
- **Testing Framework:** Pytest

## 5. Usage Examples

### Example 1: Run the Dashboard Locally
```bash
python -m streamlit run src/streamlit_app.py
```
- Open browser at `http://localhost:8501`
- Select "Extended sample dataset" from sidebar
- Choose a segment (e.g., `age_25_34_personal_loan`)
- Adjust "Forecast horizon (weeks)" slider from 4 to 8
- Click tabs to view Forecast, Anomalies, and Scenario results

### Example 2: Load Custom Data
1. Prepare CSV with columns:
   - `date` (month-end date, e.g., 2025-01-31)
   - `segment_id` (string identifier)
   - `delinquency_rate` (0-1 scale)
   - `repayment_rate` (0-1 scale)
   - `income_to_debt_ratio` (numeric)
   - `avg_interest_rate` (numeric)
2. Select "Upload your own CSV" in sidebar
3. Upload file and select segment

### Example 3: Run Backtesting from CLI
```bash
python -m src.evaluation --input assets/sample_dataset.csv --test-periods 6 --risk-threshold 0.08
```
Output: MAE, RMSE, MAPE, ROC-AUC, confusion matrix per segment

### Sample Output (JSON from Forecast)
```json
{
  "segment_id": "age_25_34_personal_loan",
  "model_name": "SARIMAX(1,0,0)",
  "forecast": [
    {
      "date": "2025-11-30",
      "central": 0.0415,
      "lower": 0.0385,
      "upper": 0.0445
    }
  ],
  "diagnostics": {"aic": 125.43}
}
```

## 6. Architecture

### System Design
```
Streamlit Frontend (src/streamlit_app.py)
    ↓
Core Business Logic
├─ Data Loader (src/data_loader.py)        → CSV parsing, schema validation
├─ Forecasting (src/forecasting.py)        → SARIMAX & baseline models
├─ Anomalies (src/anomalies.py)           → Rolling-origin detection
├─ Scenarios (src/scenarios.py)           → Stress testing & sensitivity
└─ Evaluation (src/evaluation.py)         → Backtesting & metrics

Data Sources
├─ assets/sample_dataset.csv               → Starter sample (~8 segments)
└─ assets/demo_dataset_extended.csv        → Extended sample (~15 segments)

Optional: Google Gemini API → LLM summaries
```

### Data Flow
1. **Load:** User selects dataset → CSV loaded and validated
2. **Filter:** Segments with <12 months dropped automatically
3. **Select:** User picks segment from dropdown
4. **Forecast:** SARIMAX fits on history → generates 1-2 month predictions
5. **Detect:** Rolling-origin evaluation flags anomalies
6. **Stress:** Linear regression applies interest-rate scenario
7. **Display:** Charts, tables, and summaries rendered

<<<<<<< HEAD
## 7 Sample Outputs & Screenshots

### Dashboard Screenshots
- **Forecast Panel** - Historical + predicted delinquency with 80% confidence bands
  <img width="941" height="439" alt="output_Forecast" src="https://github.com/user-attachments/assets/300eda1f-0518-478a-bbdf-51e78daf7593" />

- **Anomaly Monitor** - Detected outliers with driver explanations
- <img width="948" height="438" alt="anomaly" src="https://github.com/user-attachments/assets/c42b9702-7934-48ea-9e74-3aee4585e2d2" />

- **Scenario Lab** - Interest rate shock simulations
- <img width="952" height="440" alt="output_scenario" src="https://github.com/user-attachments/assets/c1b39b02-5df7-4122-b460-20582b06dbc3" />

- **Summary Tab** - Backtesting metrics (MAE, RMSE, ROC-AUC)
- <img width="950" height="442" alt="output_Executive_summary" src="https://github.com/user-attachments/assets/584dfcc3-813a-4054-924d-1509fc7b0d5b" />


### Data Tables

## 8. Limitations

### Current Limitations
- **Forecast horizon:** Fixed to 4-8 weeks (4-6 weeks = 1 month, 7+ weeks = 2 months) due to weekly-to-monthly mapping
- **Segment granularity:** Monthly data only; sub-monthly (weekly) data not supported
- **Model complexity:** SARIMAX(1,0,0) is simple; seasonal patterns not captured
- **Data requirements:** Minimum 12 observations per segment; sparse segments excluded
- **Exogenous variables:** Limited to income, interest rate, unemployment, GDP growth; custom features not yet supported
- **Scalability:** Single-threaded; large datasets (100k+ rows) may be slow
- **Deployment:** Vercel version has 100-second timeout limit; long forecasts may fail
- **No real-time updates:** Static CSV inputs only; live data feeds not integrated
- **AI summaries:** Optional only; requires Gemini API key; not all users have access

## 9. Future Improvements

### Short-term (Next Sprint)
- [ ] Multi-step forecasting with auto-ARIMA for model selection
- [ ] Custom exogenous variable support for user-provided features
- [ ] Export functionality (PDF reports, CSV downloads)
- [ ] Dark/light theme toggle
- [ ] Performance optimization for 100k+ row datasets

### Medium-term (Q2 2026)
- [ ] Real-time data integration (SQL database, cloud data warehouse)
- [ ] Ensemble forecasting (combine SARIMAX + Prophet + LSTM)
- [ ] Causal inference for driver attribution (DoWhy library)
- [ ] Multi-segment portfolio risk aggregation
- [ ] Historical scenario backtesting (past rate shocks)

### Long-term (Q3-Q4 2026)
- [ ] Deep learning models (LSTM, Transformer) for longer-horizon forecasts
- [ ] Hierarchical forecasting (portfolio → segment → product)
- [ ] Automatic model retraining pipeline
- [ ] Mobile app (React Native)
- [ ] REST API with authentication for enterprise integration
- [ ] Real-time alerting system (Slack/Email integration)

### Quality & DevOps
- [ ] Increase test coverage (currently ~70%)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Docker containerization for consistent deployment
- [ ] Monitoring & logging (DataDog, Sentry)
- [ ] Documentation improvements (API docs, architecture diagrams)

## Contact & Support

For questions or issues:
- **GitHub Issues:** [Project Issues](https://github.com/shubhtiwari65/Credit-Default-Risk-Forecasting/issues)
- **Email:** shubhtiwari651@gmail.com

---


=======
## 7. Limitations

### Current Limitations
- **Forecast horizon:** Fixed to 4-8 weeks (4-6 weeks = 1 month, 7+ weeks = 2 months) due to weekly-to-monthly mapping
- **Segment granularity:** Monthly data only; sub-monthly (weekly) data not supported
- **Model complexity:** SARIMAX(1,0,0) is simple; seasonal patterns not captured
- **Data requirements:** Minimum 12 observations per segment; sparse segments excluded
- **Exogenous variables:** Limited to income, interest rate, unemployment, GDP growth; custom features not yet supported
- **Scalability:** Single-threaded; large datasets (100k+ rows) may be slow
- **Deployment:** Vercel version has 100-second timeout limit; long forecasts may fail
- **No real-time updates:** Static CSV inputs only; live data feeds not integrated
- **AI summaries:** Optional only; requires Gemini API key; not all users have access

## 8. Future Improvements

### Short-term (Next Sprint)
- [ ] Multi-step forecasting with auto-ARIMA for model selection
- [ ] Custom exogenous variable support for user-provided features
- [ ] Export functionality (PDF reports, CSV downloads)
- [ ] Dark/light theme toggle
- [ ] Performance optimization for 100k+ row datasets

### Medium-term (Q2 2026)
- [ ] Real-time data integration (SQL database, cloud data warehouse)
- [ ] Ensemble forecasting (combine SARIMAX + Prophet + LSTM)
- [ ] Causal inference for driver attribution (DoWhy library)
- [ ] Multi-segment portfolio risk aggregation
- [ ] Historical scenario backtesting (past rate shocks)

### Long-term (Q3-Q4 2026)
- [ ] Deep learning models (LSTM, Transformer) for longer-horizon forecasts
- [ ] Hierarchical forecasting (portfolio → segment → product)
- [ ] Automatic model retraining pipeline
- [ ] Mobile app (React Native)
- [ ] REST API with authentication for enterprise integration
- [ ] Real-time alerting system (Slack/Email integration)

### Quality & DevOps
- [ ] Increase test coverage (currently ~70%)
- [ ] CI/CD pipeline with GitHub Actions
- [ ] Docker containerization for consistent deployment
- [ ] Monitoring & logging (DataDog, Sentry)
- [ ] Documentation improvements (API docs, architecture diagrams)

## Contact & Support

For questions or issues:
- **GitHub Issues:** [Project Issues](https://github.com/shubhtiwari65/Credit-Default-Risk-Forecasting/issues)
- **Email:** shubhtiwari65@example.com

---

**Last Updated:** April 12, 2026
>>>>>>> 8cf147f (Updated project files)
