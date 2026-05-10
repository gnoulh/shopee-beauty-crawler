import streamlit as st
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))

st.set_page_config(
<<<<<<< HEAD
    page_title="Bất động sản Việt Nam 2024",
=======
    page_title="Dashboard Phân Tích Bất Động Sản Việt Nam 2024",
>>>>>>> MinhThu
    layout="wide",
    initial_sidebar_state="expanded",
    initial_sidebar_state="expanded",
)

<<<<<<< HEAD
pg = st.navigation(
    [
        st.Page("pages/Toan.py", title="Cấu trúc & Quy mô"),
        st.Page("pages/Dia_ly.py", title="Phân tích Địa lý"),
    ]
)
pg.run()
=======
pg = st.navigation([
    st.Page("pages/Home.py", title="Tổng quan"),
    st.Page("pages/EDA.py", title="Mô tả dữ liệu"),
    st.Page("pages/Thu.py", title="Thị trường & Giá trị vô hình"),
    st.Page("pages/draft.py", title="NHÁP"),
    ])
pg.run()
>>>>>>> MinhThu
