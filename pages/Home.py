"""
Home.py — Trang chủ Dashboard Shopee Mỹ phẩm VN
"""

import streamlit as st
import pandas as pd
import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from utils.helpers import load_data, inject_css, setup_sidebar

inject_css()
setup_sidebar()

try:
    products, shops, reviews = load_data()
    DATA_OK = True
except FileNotFoundError:
    DATA_OK = False

# ====================== Header ========================================================================================
st.title("Phân tích Thị trường Mỹ phẩm & Chăm sóc Cá nhân")
st.caption(
    "Shopee Vietnam, Snapshot crawl 18–19/3/2026, Lab 01 – Trực quan hóa Dữ liệu 23KHDL"
)

if not DATA_OK:
    st.error(f"Không tìm thấy file dữ liệu \n")
    st.stop()

st.markdown("---")
st.markdown(
    """
### Bài toán phân tích chung

> **Phân tích thị trường mỹ phẩm và chăm sóc cá nhân trên Shopee Việt Nam** dựa trên bộ dữ liệu
> crawl ngày 18–19/3/2026 (**{:,} sản phẩm; {:,} cửa hàng; {:,} đánh giá**) nhằm xác định
> các yếu tố then chốt ảnh hưởng đến **doanh thu ước tính** và **hành vi đánh giá** của người tiêu dùng,
> từ đó **đề xuất ít nhất 3 chiến lược kinh doanh** có cơ sở dữ liệu cho người bán hàng trên sàn.
""".format(
        len(products), len(shops), len(reviews)
    )
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sản phẩm", f"{len(products):,}")
c2.metric("Cửa hàng", f"{len(shops):,}")
c3.metric("Đánh giá", f"{len(reviews):,}")
c4.metric("Danh mục con", f"{products['sub_category'].nunique()}")
rev_tong = products["revenue_est"].sum() / 1e9  # convert to "tỷ"
c5.metric("Tổng doanh thu ước tính", f"{rev_tong:,.0f} tỷ đồng")

st.markdown("---")
st.markdown("### Điều hướng")
col1, col2 = st.columns(2)
with col1:
    st.page_link("pages/EDA.py", label="Mô tả dữ liệu")
    st.caption("Schema - KPIs - phân bố biến - correlation matrix")
    st.page_link("pages/01_Pricing.py", label="Giá & Danh mục")
    st.caption("MT1 Discount tối ưu - MT2 Top danh mục - MT7 BCG + Lollipop")
    st.page_link("pages/02_Reviews.py", label="Đánh giá & Cảm xúc khách hàng")
    st.caption("MT8 (23127488) - J-curve - từ khóa 1–3 sao vs 5 sao")
with col2:
    st.page_link("pages/03_Market.py", label="Phân khúc Thị trường")
    st.caption("MT3 K-Means (price x sold)")
    st.page_link("pages/04_Shops.py", label="Hiệu quả Cửa hàng")
    st.caption("MT9 K-Means shop - MT5 Pearson correlation (23127361)")
    st.page_link("pages/05_Geo_Mall.py", label="Địa lý & Shop Mall")
    st.caption("MT6 Top tỉnh/thành - MT4 Box/Violin/Donut/Radar (22127418)")

st.markdown("---")

# === Objectives table ===
st.markdown("### Danh sách 10 Objectives đã hoàn thành")
obj_df = pd.DataFrame(
    {
        "Mục tiêu": [
            "EDA",
            "MT1",
            "MT2",
            "MT3",
            "MT4",
            "MT5",
            "MT6",
            "MT7",
            "MT8",
            "MT9",
        ],
        "Nội dung": [
            "Tổng quan dataset - kích thước, schema, phân bố biến, correlation",
            "Sử dụng dữ liệu discount_pct, sold và price_tier của 20,658 sản phẩm crawl ngày 18/3/2026 để xác định khoảng discount_pct có trung vị sold cao nhất trong từng phân khúc giá (budget -> luxury), từ đó đề xuất ngưỡng chiết khấu tối ưu cho mỗi phân khúc, hoàn thành trong phạm vi phân tích snapshot này.",
            "Tính tổng revenue_est và trung bình price, discount_pct, sold của từng sub_category trong bộ dữ liệu 18/3/2026 để xác định top 5 danh mục có doanh thu ước tính cao nhất và mô tả đặc trưng định giá của từng nhóm bằng ít nhất 2 loại biểu đồ khác nhau.",
            "Phân chia thị trường mỹ phẩm Shopee thành 5 phân khúc chiến lược dựa trên mối tương quan giữa giá bán và sản lượng. Từ đó, đánh giá tỷ trọng đóng góp doanh thu của từng phân khúc.",
            "So sánh sold, rating, revenue_est giữa Shopee Mall (8,298 sp) và Non-mall (12,360 sp) từ dữ liệu 18/3/2026.",
            "Xác định chỉ số shop (followers, response_rate, shop_rating) có tương quan Pearson mạnh nhất với revenue_est",
            "Phân tích phân bố doanh thu theo tỉnh/thành để xác định 3 khu vực dẫn đầu và đặc trưng riêng",
            "So sánh đồng thời avg_sold và total_revenue_est trên 22 sub_category trong snapshot 18/3/2026, phân loại từng danh mục vào một trong 4 chiến lược (Ngôi sao/Sinh lời/Phễu/Câu hỏi) — để người bán mới lựa chọn đúng danh mục phù hợp với chiến lược của mình ngay từ đầu: muốn volume cao -> phễu; muốn margin cao -> sinh lời; muốn cả hai -> ngôi sao.",
            "Trích xuất và phân tích 10 từ khóa phổ biến trong các đánh giá từ 1 - 3 sao mà nhóm thu thập được, tính tới thời điểm tháng 3 năm 2026, phân loại theo 4 nhóm nguyên nhân (sự cố sản phẩm, trải nghiệm khách hàng, chất lượng đóng gói và vận chuyển) để từ đó để xuất các giải pháp tương ứng giúp ngăn chặn hoặc thuyết phục khách hàng chỉnh sửa đánh giá.",
            "Phân các shop trên thị trường Shopee thành 4 nhóm dựa trên 4 tiêu chí lòng tin và dịch vụ (lượt người theo dõi, tổng lượt đánh giá, tổng điểm đánh giá và tốc độ phản hồi tin nhắn) để từ đó xác định 2 nhóm mang lại doanh thu ước tính và lượt bán cao nhất.",
        ],
        "Biểu đồ": [
            "-",
            "Heatmap, Box, Scatter + OLS",
            "Bar, Grouped bar, Bubble",
            "Scatter cluster, Pie",
            "Box x 3, Grouped bar, Donut x 2, Violin",
            "Heatmap Pearson",
            "Bar ngang, Bubble, Radar chart",
            "BCG bubble 22 danh mục + Lollipop composite score",
            "Từ khóa phổ biến",
            "Silhouette, Pie, Bar",
        ],
        "Trang": [
            "EDA",
            "Giá & Danh mục",
            "Giá & Danh mục",
            "03 Market",
            "05 Geo",
            "04 Shops",
            "05 Geo",
            "Giá & Danh mục",
            "02 Reviews",
            "04 Shops",
        ],
        "Thành viên": [
            "Cả nhóm",
            "22127254",
            "22127254",
            "23127488",
            "22127418",
            "23127361",
            "22127418",
            "22127254",
            "23127488",
            "23127361",
        ],
        "ML": [
            "—",
            "Random Forest",
            "—",
            "K-Means",
            "—",
            "—",
            "—",
            "—",
            "—",
            "K-Means",
        ],
    }
)
st.dataframe(obj_df, hide_index=True, use_container_width=True)

st.markdown("---")

# === Cross-objective insights & Overall conclusion ===
st.markdown("### Kết luận Tổng hợp & Chiến lược Kinh doanh")

st.markdown(
    """
#### Những phát hiện nổi bật liên kết nhiều mục tiêu

**1. Chuỗi Discount -> Phân khúc -> Chiến lược Sản phẩm (MT1 -> MT2 -> MT7)**
> MT1 cho thấy phân khúc **premium/luxury** bán tốt nhất khi discount thấp (0–10%). MT2 xác nhận **kem dưỡng ẩm và serum** — hai danh mục sinh lời (BCG, MT7) — đều thuộc phân khúc giá cao. Kết hợp: người bán danh mục premium không nên chạy đua giảm giá — sẽ giảm cả margin lẫn lượng bán.

**2. Mặt nạ là ngoại lệ của thị trường (MT2, 3, 8)**
> Mặt nạ dẫn đầu doanh thu (700B), lượng bán TB cao nhất (5,280), Composite Score = 1.000 (MT7) và thể hiện trong cluster "Phổ thông – Bán chạy" (MT3). Ba mục tiêu đều xác nhận: đây là category duy nhất đạt cả hai chiều (volume và revenue) nhưng cũng là sân chơi cạnh tranh nhất.

**3. Followers quyết định doanh thu — Không phải Rating (MT5, 9)**
> MT5 (Pearson correlation) xác nhận followers là yếu tố có |r| cao nhất với revenue. MT9 (K-Means shop) cho thấy "Shop Dẫn Đầu" — nhóm followers cao nhất — chiếm tỷ trọng doanh thu không cân xứng. Nhất quán với MT8 (reviews): rating 5 sao phổ biến trên toàn nền tảng nên không còn là yếu tố phân biệt.

**4. Non-Mall vượt Mall về doanh thu — Nhưng không đều (MT4, 5)**
> MT4: Non-Mall chiếm ~74% tổng doanh thu dù chỉ chiếm ~60% số SP. Nhưng MT5 cho thấy followers — yếu tố tương quan cao nhất — có phân phối lệch phải mạnh: một số ít shop Non-Mall quy mô lớn đang kéo trung bình lên. Median (từ box plot) cho thấy sản phẩm Non-Mall điển hình không vượt trội Mall.

**5. Tập trung địa lý song hành với tập trung thị trường (MT6)**
> TP.HCM và Hà Nội chiếm ~75% doanh thu (MT6) — tương tự như phân khúc "Cao cấp – Bán chạy" (MT3) tập trung phần lớn doanh thu. Cả địa lý và phân khúc sản phẩm đều cho thấy thị trường winner-takes-most.
"""
)

st.markdown("---")

col_s1, col_s2, col_s3 = st.columns(3)
with col_s1:
    st.info(
        """
    **Chiến lược 1:**
    **Discount theo phân khúc, không đồng đều**

    Dựa trên MT1: Budget (>50%), mid-low (10–20%), mid (30–40%), premium/luxury (<=10%).
    Áp voucher đúng tier — tránh giảm giá tràn lan làm loãng brand premium.
    """
    )
with col_s2:
    st.info(
        """
    **Chiến lược 2:**
    **Phễu -> Sinh lời (danh mục ladder)**

    Dựa trên MT2 & 8: Bắt đầu từ sữa rửa mặt/kem chống nắng (Phễu) để xây 50+ reviews và followers.
    Pivot sang serum/kem dưỡng ẩm (Sinh lời) khi uy tín đủ.
    """
    )
with col_s3:
    st.info(
        """
    **Chiến lược 3:**
    **Đầu tư Followers thay vì chạy đua giá**

    Dựa trên MT5 & 9: Followers có tương quan cao nhất với revenue.
    Chiến dịch tăng Followers (follow-to-get-voucher) ROI cao hơn chạy Flash Sale.
    Maintain response_rate >90% để vào nhóm "Shop Uy Tín".
    """
    )
