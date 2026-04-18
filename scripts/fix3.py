with open("src/streamlit_app.py", "r", encoding="utf-8") as f:
    c = f.read()

old_kpi = """    kpi1, kpi2, kpi3 = st.columns(3)
    kpi1.metric("Latest delinquency", f"{last_actual:.2%}")
    kpi2.metric("Next forecast", f"{float(next_point['central']):.2%}", f"{float(next_point['central'] - last_actual):+.2%}")
    kpi3.metric("80% interval width", f"{interval_width:.2%}")"""

new_kpi = """    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    kpi1.metric("Latest delinquency", f"{last_actual:.2%}", "")
    kpi2.metric("Next forecast", f"{float(next_point['central']):.2%}", f"{float(next_point['central'] - last_actual):+.2%}")
    kpi3.metric("80% interval width", f"{interval_width:.2%}", "")
    kpi4.metric("Avg stress uplift", f"{avg_stress_delta:+.2%}", "")
    kpi5.metric("Anomalies in window", str(len(anomalies_df)), "")"""

if old_kpi in c:
    c = c.replace(old_kpi, new_kpi)
else:
    print("KPIs not matched. Already replaced?")

c = c.replace("\"Summary and backtest\"", "\"Summary & backtest\"")

with open("src/streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(c)
print("Done!")
