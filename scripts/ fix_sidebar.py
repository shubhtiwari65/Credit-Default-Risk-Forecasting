with open("src/streamlit_app.py", "r", encoding="utf-8") as f:
    c = f.read()

import re

# Replace Sidebar Expander with dividing header
c = re.sub(
    r'with st\.expander\("Model controls", expanded=True\):',
    'st.divider()\n        st.markdown("### Model controls")',
    c
)

# Unindent all the sliders inside the sidebar section
def unindent_sliders(text):
    lines = text.split("\n")
    new_lines = []
    in_model_controls = False
    for line in lines:
        if "### Model controls" in line:
            in_model_controls = True
            new_lines.append(line)
            continue
        if in_model_controls and line.strip() == "st.markdown(":
            in_model_controls = False
            new_lines.append(line)
            continue

        if in_model_controls and line.startswith("            "):
            new_lines.append(line[4:])
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

c = unindent_sliders(c)

# Clean up Data source section
c = re.sub(
    r'data_source = st\.radio\(\s*"Data source",\s*options=\[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD\],\s*index=0,\s*\)',
    '''st.markdown("### Data source")
        data_source = st.radio(
            "Select dataset",
            label_visibility="collapsed",
            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],
            index=0,
        )''',
    c
)

# Add Control Center caption
c = c.replace('st.header("Control Center")', 'st.header("Control Center")\n        st.markdown("<div class=\'sidebar-caption\'>Configure data and risk settings</div>", unsafe_allow_html=True)')

# Change tab names
c = c.replace('["Forecast view", "Anomaly monitor", "Scenario lab", "Summary and backtest"]', '["Forecast view", "Anomaly monitor", "Scenario lab", "Summary & backtest"]')

with open("src/streamlit_app.py", "w", encoding="utf-8") as f:
    f.write(c)

print("Fix completed cleanly!")
