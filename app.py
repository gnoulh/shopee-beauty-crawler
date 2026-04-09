import streamlit as st
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))

st.set_page_config(
    page_title="Shopee Mỹ phẩm VN – Lab 01",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/Home.py", title="Tổng quan"),
    st.Page("pages/EDA.py", title="Mô tả dữ liệu"),
    st.Page("pages/01_Pricing.py", title="Giá & Danh mục"),
    st.Page("pages/02_Reviews.py", title="Đánh giá & Cảm xúc khách hàng"),
    st.Page("pages/03_Market.py", title="Phân khúc Thị trường"),
    st.Page("pages/04_Shops.py", title="Hiệu quả Cửa hàng"),
    st.Page("pages/05_Geo_Mall.py", title="Địa lý & Shop Mall"),
    ])
pg.run()
