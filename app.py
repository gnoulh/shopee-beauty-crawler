import streamlit as st
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

st.set_page_config(
    page_title="Dashboard Phân Tích Bất Động Sản Việt Nam 2024",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/Home.py", title="Tổng quan"),
    st.Page("pages/EDA.py", title="Mô tả dữ liệu"),
    st.Page("pages/Thu.py", title="Thị trường & Giá trị vô hình"),
    st.Page("pages/draft.py", title="NHÁP"),
    ])
pg.run()
