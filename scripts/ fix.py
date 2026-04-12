import sys 
with open('src/streamlit_app.py', 'r', encoding='utf-8') as f: c = f.read()
c = c.replace('with st.expander(\" Model "controls\, expanded=True):', 'st.divider()\n        st.markdown(\###" Model "controls\)') 
with open('src/streamlit_app.py', 'w', encoding='utf-8') as f: f.write(c) 
