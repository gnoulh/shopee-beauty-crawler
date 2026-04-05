"""
app.py — Trang chủ Dashboard Shopee Mỹ phẩm VN
Chạy: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.helpers import load_data, setup_sidebar

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

# ====================== Header ========================================================================================
st.title("Phân tích Thị trường Mỹ phẩm & Chăm sóc Cá nhân")
st.caption("Shopee Vietnam, Snapshot crawl 18–19/3/2026, Lab 01 – Trực quan hóa Dữ liệu 23KHDL")

if not DATA_OK:
    st.error(
        f"Không tìm thấy file dữ liệu: `{missing}`\n\n"
    )
    st.stop()

st.markdown("---")
st.markdown("""
### Bài toán phân tích chung

> **Phân tích thị trường mỹ phẩm và chăm sóc cá nhân trên Shopee Việt Nam** dựa trên bộ dữ liệu
> crawl ngày 18–19/3/2026 (**{:,} sản phẩm; {:,} cửa hàng; {:,} đánh giá**) nhằm xác định
> các yếu tố then chốt ảnh hưởng đến **doanh thu ước tính** và **hành vi đánh giá** của người tiêu dùng,
> từ đó **đề xuất ít nhất 3 chiến lược kinh doanh** có cơ sở dữ liệu cho người bán hàng trên sàn.
""".format(len(products), len(shops), len(reviews)))

st.markdown("---")
st.markdown("### Điều hướng")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/EDA.py", label="Tổng quan Dữ liệu (EDA)")
    st.caption("Schema - KPIs - phân bố biến - correlation matrix")
    st.page_link("pages/01_Pricing.py", label="22127254 – Giá, Danh mục & BCG")
    st.caption("MT1 Discount tối ưu - MT2 Top danh mục - MT8 BCG + Lollipop")
    st.page_link("pages/02_Reviews.py", label="Đánh giá & Phân tích Cảm xúc")
    st.caption("MT9 (23127488) - J-curve - từ khóa 1–3 sao vs 5 sao - TF-IDF + LogReg ML")
with col2:
    st.page_link("pages/03_Market.py", label="Phân khúc Thị trường & SP Chiến lược")
    st.caption("MT3 K-Means (price x sold) - MT6 Volume/Margin driver (23127488)")
    st.page_link("pages/04_Shops.py", label="Hiệu quả Cửa hàng")
    st.caption("MT10 K-Means shop - MT5 Pearson correlation (23127361)")
    st.page_link("pages/05_Geo_Mall.py", label="Địa lý & Mall vs Non-Mall",              icon="🗺️")
    st.caption("MT7 Top tỉnh/thành · MT4 Box/Violin/Donut/Radar  (22127418)")

st.markdown("---")

# === Objectives table ===
st.markdown("### Danh sách 11 Objectives đã hoàn thành")
obj_df = pd.DataFrame({
    "Obj": ["EDA","MT1","MT2","MT3","MT4","MT5","MT6","MT7","MT8","MT9","MT10"],
    "Nội dung":   ["Tổng quan dataset — kích thước, schema, phân bố biến, correlation",
                   "Discount tối ưu theo price_tier — Heatmap + Box + Scatter+OLS",
                   "Top 5 danh mục theo revenue_est — Bar + Grouped bar + Bubble",
                   "K-Means market (price×sold, Silhouette) — Scatter cluster + Pie",
                   "Mall vs Non-Mall — Box x 3, Grouped bar, Donut x 2, Violin",
                   "Tương quan chỉ số shop -> revenue — Heatmap Pearson",
                   "Volume Driver vs Margin Driver theo cluster — Grouped bar",
                   "Địa lý top tỉnh/thành — Bar ngang, Bubble, Radar chart",
                   "BCG bubble 22 danh mục + Lollipop composite score",
                   "Review 1–3 sao vs 5 sao — từ khóa, 4 nhóm nguyên nhân, TF-IDF+LogReg",
                   "K-Means shop 4 tiêu chí lòng tin — Silhouette + Pie + Bar"],
    "Trang":      ["EDA","01 Pricing","01 Pricing","03 Market","05 Geo","04 Shops",
                   "03 Market","05 Geo","01 Pricing","02 Reviews","04 Shops"],
    "Thành viên": ["Cả nhóm","22127254","22127254","23127488","22127418","23127361",
                   "23127488","22127418","22127254","23127488","23127361"],
    "ML":         ["—","—","—","K-Means","—","—","—","—","—","TF-IDF+LogReg","K-Means"],
})
st.dataframe(obj_df, hide_index=True, use_container_width=True)

st.markdown("---")

# === Cross-objective insights & Overall conclusion ===
st.markdown("### Kết luận Tổng hợp & Chiến lược Kinh doanh")

st.markdown("""
#### Những phát hiện nổi bật liên kết nhiều mục tiêu

**1. Chuỗi Discount -> Phân khúc -> Chiến lược Sản phẩm (MT1 -> MT2 -> MT8)**
> MT1 cho thấy phân khúc **premium/luxury** bán tốt nhất khi discount thấp (0–10%). MT2 xác nhận **kem dưỡng ẩm và serum** — hai danh mục sinh lời (BCG, MT8) — đều thuộc phân khúc giá cao. Kết hợp: người bán danh mục premium không nên chạy đua giảm giá — sẽ giảm cả margin lẫn lượng bán.

**2. Mặt nạ là ngoại lệ của thị trường (MT2, 3, 8)**
> Mặt nạ dẫn đầu doanh thu (700B), lượng bán TB cao nhất (5,280), Composite Score = 1.000 (MT8) và thể hiện trong cluster "Phổ thông – Bán chạy" (MT3). Ba mục tiêu đều xác nhận: đây là category duy nhất đạt cả hai chiều (volume và revenue) nhưng cũng là sân chơi cạnh tranh nhất.

**3. Followers quyết định doanh thu — Không phải Rating (MT5, 10)**
> MT5 (Pearson correlation) xác nhận followers là yếu tố có |r| cao nhất với revenue. MT10 (K-Means shop) cho thấy "Shop Dẫn Đầu" — nhóm followers cao nhất — chiếm tỷ trọng doanh thu không cân xứng. Nhất quán với MT9 (reviews): rating 5 sao phổ biến trên toàn nền tảng nên không còn là yếu tố phân biệt.

**4. Non-Mall vượt Mall về doanh thu — Nhưng không đều (MT4, 5)**
> MT4: Non-Mall chiếm ~74% tổng doanh thu dù chỉ chiếm ~60% số SP. Nhưng MT5 cho thấy followers — yếu tố tương quan cao nhất — có phân phối lệch phải mạnh: một số ít shop Non-Mall quy mô lớn đang kéo trung bình lên. Median (từ box plot) cho thấy sản phẩm Non-Mall điển hình không vượt trội Mall.

**5. Tập trung địa lý song hành với tập trung thị trường (MT7)**
> TP.HCM và Hà Nội chiếm ~75% doanh thu (MT7) — tương tự như phân khúc "Cao cấp – Bán chạy" (MT3) tập trung phần lớn doanh thu. Cả địa lý và phân khúc sản phẩm đều cho thấy thị trường winner-takes-most.
""")

st.markdown("---")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.info("""
    **Chiến lược 1:**
    **Discount theo phân khúc, không đồng đều**

    Dựa trên MT1: Budget (>50%), mid-low (10–20%), mid (30–40%), premium/luxury (<=10%).
    Áp voucher đúng tier — tránh giảm giá tràn lan làm loãng brand premium.
    """)
with col_s2:
    st.info("""
    **Chiến lược 2:**
    **Phễu -> Sinh lời (danh mục ladder)**

    Dựa trên MT2 & 8: Bắt đầu từ sữa rửa mặt/kem chống nắng (Phễu) để xây 50+ reviews và followers.
    Pivot sang serum/kem dưỡng ẩm (Sinh lời) khi uy tín đủ.
    """)
with col_s3:
    st.info("""
    **Chiến lược 3:**
    **Đầu tư Followers thay vì chạy đua giá**

    Dựa trên MT5 & 10: Followers có tương quan cao nhất với revenue.
    Chiến dịch tăng Followers (follow-to-get-voucher) ROI cao hơn chạy Flash Sale.
    Maintain response_rate >90% để vào nhóm "Shop Uy Tín".
    """)
