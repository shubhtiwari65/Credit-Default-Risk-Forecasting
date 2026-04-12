import re  
with open('src/streamlit_app.py', 'r', encoding='utf-8') as f: c = f.read() 
lines = c.split('\n') 
out = [] 
flag = False 
for line in lines: 
    if '### Model controls' in line: 
        flag = True 
    if flag and line.startswith('            '): 
        out.append(line[4:]) 
    else: 
        out.append(line) 
with open('src/streamlit_app.py', 'w', encoding='utf-8') as f: f.write('\n'.join(out)) 
