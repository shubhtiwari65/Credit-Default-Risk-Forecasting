import sys
with open("src/streamlit_app.py", "r", encoding="utf-8") as f:
    c = f.read()

# 1. Update the sidebar
sidebar_old = """    with st.sidebar:
        st.header("Control Center")
        data_source = st.radio(
            "Data source",
            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Upload segment-month CSV",
            type=["csv"],
            disabled=data_source != DATA_SOURCE_UPLOAD,
        )

        with st.expander("Model controls", expanded=True):
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
        )"""

sidebar_new = """    with st.sidebar:
        st.header("Control Center")
        st.markdown("<div class='sidebar-caption'>Configure data and risk settings</div>", unsafe_allow_html=True)
        
        st.markdown("### Data source")
        data_source = st.radio(
            "Data source",
            label_visibility="collapsed",
            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],
            index=0,
        )

        uploaded_file = st.file_uploader(
            "Upload segment-month CSV",
            type=["csv"],
            help="Hint: Must contain segment, date, and delinquency columns",
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
            \"\"\"
            <div class="side-tip">
                Tip: Start with Extended sample dataset for richer behavior, then switch to Upload your own CSV for real data.
            </div>
            \"\"\",
            unsafe_allow_html=True,
        )"""

if sidebar_old in c:
    c = c.replace(sidebar_old, sidebar_new)
    print("Sidebar replaced successfully.")
else:
    print("Failed to find sidebar_old. It might have trailing spaces or formatting issues.")

# 2. Update the KPIs
kpi_old = """    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Latest delinquency", f"{last_actual:.2%}")
    kpi2.metric("Next forecast", f"{float(next_point['central']):.2%}", f"{float(next_point['central'] - last_actual):+.2%}")
    kpi3.metric("80% interval width", f"{interval_width:.2%}")
    kpi4.metric("Avg stress uplift", f"{avg_stress_delta:.2%}")
    kpi5.metric("Anomalies in window", str(len(anomalies_df)))"""

kpi_new = """    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Latest delinquency", f"{last_actual:.2%}", "")
    kpi2.metric("Next forecast", f"{float(next_point['central']):.2%}", f"{float(next_point['central'] - last_actual):+.2%}")
    kpi3.metric("80% interval width", f"{interval_width:.2%}", "")
    kpi4.metric("Avg stress uplift", f"{avg_stress_delta:+.2%}", "")
    kpi5.metric("Anomalies in window", str(len(anomalies_df)), "")"""

if kpi_old in c:
    c = c.replace(kpi_old, kpi_new)
    print("KPI replaced successfully.")

# 3. Update tabs
c = c.replace("[\"Forecast view\", \"Anomaly monitor\", \"Scenario lab\", \"Summary and backtest\"]", "[\"Forecast view\", \"Anomaly monitor\", \"Scenario lab\", \"Summary & backtest\"]")

with open("src/streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(c)

print("Done with ultimate fix.")
