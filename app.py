import streamlit as st
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

st.set_page_config(
    page_title="Bất động sản Việt Nam 2024",
    layout="wide",
    initial_sidebar_state="expanded",
    initial_sidebar_state="expanded",
)

pg = st.navigation(
    [
        st.Page("pages/Toan.py", title="Cấu trúc & Quy mô"),
        st.Page("pages/Dia_ly.py", title="Phân tích Địa lý"),
    ]
)
pg.run()