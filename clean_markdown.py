import re

with open('streamlit_app.py', 'r', encoding='utf-8') as f:
    content = f.read()

# Make sure render_html is defined
if 'def render_html(' not in content:
    func_code = '''def render_html(html_str):
    cleaned = '\\n'.join(line.strip() for line in html_str.strip().split('\\n'))
    st.markdown(cleaned, unsafe_allow_html=True)

'''
    content = content.replace('st.set_page_config(', func_code + 'st.set_page_config(')

# Pattern 1: st.markdown(f"""...""", unsafe_allow_html=True)
pattern_triple = r'st\.markdown\(\s*(f?"""[\s\S]*?""")\s*,\s*unsafe_allow_html=True\s*\)'
content = re.sub(pattern_triple, r'render_html(\1)', content)

# Pattern 2: st.markdown(f'<div...', unsafe_allow_html=True)
pattern_single = r'st\.markdown\(\s*(f?["\']<div[\s\S]*?["\'])\s*,\s*unsafe_allow_html=True\s*\)'
content = re.sub(pattern_single, r'render_html(\1)', content)

with open('streamlit_app.py', 'w', encoding='utf-8') as f:
    f.write(content)

print('Updated streamlit_app.py successfully')
