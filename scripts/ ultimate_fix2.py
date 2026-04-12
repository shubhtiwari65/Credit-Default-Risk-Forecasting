import re 
with open('src/streamlit_app.py', 'r', encoding='utf-8') as f: c = f.read() 
c = re.sub(r'with st\.expander\(\" Model "controls\, expanded=True\):', 'st.divider()\n        st.markdown(\###" Model "controls\)', c)
import re 
c = re.sub(r'data_source = st\.radio\(\n\s*\" Data "source\,\n\s*options=\[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD\],\n\s*index=0,\n\s*\)', 'st.markdown(\###" Data "source\)\n        data_source = st.radio(\n            \Data" "source\,\n            label_visibility=\collapsed\,\n            options=[DATA_SOURCE_EXTENDED, DATA_SOURCE_STARTER, DATA_SOURCE_UPLOAD],\n            index=0,\n        )', c)
import re 
