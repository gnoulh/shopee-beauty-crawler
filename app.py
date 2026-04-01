"""
app.py — Trang chủ Dashboard Shopee Mỹ phẩm VN
Chạy: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.helpers import load_data, setup_sidebar, CB_BLUE, CB_ORANGE

st.set_page_config(
    page_title="Shopee Mỹ phẩm VN – Lab 01",
    layout="wide",
    initial_sidebar_state="expanded",
)

setup_sidebar()          # Branding nhất quán

# ====================== Kiểm tra data ======================
try:
    products, shops, reviews = load_data()
    DATA_OK = True
except FileNotFoundError as e:
    DATA_OK = False
    missing = str(e)

# ====================== Header ========================================================================================
st.title("Phân tích Thị trường Mỹ phẩm & Chăm sóc Cá nhân")
st.caption("Shopee Vietnam, Snapshot crawl 18–19/3/2026, Lab 01 – Trực quan hóa Dữ liệu 23KHDL")

if not DATA_OK:
    st.error(
        f"Không tìm thấy file dữ liệu: `{missing}`\n\n"
    )
    st.stop()

st.markdown("---")

# ====================== Bài toán chung ============================================================
st.markdown("""
### Bài toán phân tích chung

> **Phân tích thị trường mỹ phẩm và chăm sóc cá nhân trên Shopee Việt Nam** dựa trên bộ dữ liệu
> crawl ngày 18–19/3/2026 (**{:,} sản phẩm; {:,} cửa hàng; {:,} đánh giá**) nhằm xác định
> các yếu tố then chốt ảnh hưởng đến **doanh thu ước tính** và **hành vi đánh giá** của người tiêu dùng,
> từ đó **đề xuất ít nhất 3 chiến lược kinh doanh** có cơ sở dữ liệu cho người bán hàng trên sàn.
""".format(len(products), len(shops), len(reviews)))

st.markdown("---")

# ====================== Dataset stats ======================
st.markdown("### Thống kê nhanh bộ dữ liệu")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sản phẩm", f"{len(products):,}")
c2.metric("Cửa hàng", f"{len(shops):,}")
c3.metric("Đánh giá", f"{len(reviews):,}")
c4.metric("Danh mục con", f"{products['sub_category'].nunique()}")
c5.metric("Tổng DT ước tính", f"{products['revenue_est'].sum()/1e12:.1f} nghìn tỷ đồng")

st.markdown("---")

# ====================== Navigation ============================================
st.markdown("### Điều hướng")

col1, col2 = st.columns(2)

with col1:
    st.page_link("pages/EDA.py", label="Tổng quan dữ liệu (EDA)")
    st.caption("KPIs, phân bố giá, rating, top danh mục, ma trận tương quan")

    st.page_link("pages/22127254.py", label="22127254")
    st.caption("Obj 1: Ngưỡng discount tối ưu theo price_tier, Obj 2: ...")

    st.page_link("pages/22127418.py", label="22127418")
    st.caption("...")

with col2:
    st.page_link("pages/23127361.py", label="23127361")
    st.caption("...")

    st.page_link("pages/23127488.py", label="23127488")
    st.caption("...")

st.markdown("---")

# ====================== Objectives map ========================================================
st.markdown("### Bản đồ Objectives -> Trang -> Thành viên")

obj_df = pd.DataFrame({
    "Obj": ["EDA","Obj 1","Obj 2","Obj 3 (ML)","Obj 4","Obj 5 (ML)","Obj 6","Obj 7","Obj 8"],
    "Nội dung": [
        "Tổng quan: KPIs, phân bố biến, correlation",
        "Ngưỡng discount tối ưu theo price_tier",
        "Top 5 danh mục theo revenue_est",
        "K-Means phân khúc thị trường (Silhouette)",
        "Mann-Whitney: Mall vs Non-Mall",
        "TF-IDF + Logistic Regression phân loại review",
        "Tương quan chỉ số cửa hàng -> revenue",
        "Phân tích địa lý: top tỉnh/thành",
        "Bubble chart BCG + Lollipop danh mục",
    ],
    "Trang": [
        "EDA","Trang 1","Trang 2",
        "...","...","...",
        "...","...","...",
    ],
    "Trạng thái": [
        "Hoàn thiện","","",
        "Hoàn thiện","","",
        "","","",
    ],
    "Thành viên": ["Cả nhóm","22127254","22127254","...","...","...","...","...","..."],
})
st.dataframe(obj_df, hide_index=True, use_container_width=True)