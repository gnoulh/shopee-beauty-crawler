"""
pages/EDA.py — Tổng quan dữ liệu (EDA)
Yêu cầu: "Phân tích tổng quan: kích thước mẫu, cấu trúc dữ liệu, phân bố các biến."
"""

import pandas as pd
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from utils.helpers import (
    load_data, get_active, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_BLUE, CB_VERMIL, CB_GRAY,
    SEQ_BLUES, DIV_RDBU,
)

st.set_page_config(page_title="EDA – Tổng quan", layout="wide")
setup_sidebar()

# ====== Load ======
products, shops, reviews = load_data()
active = get_active(products)

# ====== Header ======
st.title("Tổng quan dữ liệu – Shopee Mỹ phẩm VN")
st.caption("Snapshot crawl 18–19/3/2026 | Yêu cầu §2.3.3: kích thước mẫu, cấu trúc, phân bố biến")

# ====== KPIs ======
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Sản phẩm", f"{len(products):,}")
c2.metric("Cửa hàng", f"{len(shops):,}")
c3.metric("Đánh giá", f"{len(reviews):,}")
c4.metric("Danh mục con", f"{products['sub_category'].nunique()}")
c5.metric("Tổng DT ước tính", f"{products['revenue_est'].sum()/1e12:.1f} nghìn tỷ đồng")

st.info("""
**Vị trí code thu thập dữ liệu:**
File **`crawling/shopee_crawler.py`** — chứa toàn bộ code crawl sản phẩm, shop và reviews từ Shopee.
""")

st.markdown("---")

# ========================
# 1. Cấu trúc dữ liệu (schema)
# ========================
st.subheader("1. Cấu trúc dữ liệu")
tab_p, tab_s, tab_r = st.tabs(["products.csv", "shops.csv", "reviews.csv"])

with tab_p:
    schema = pd.DataFrame({
        "Trường": ["item_id","shop_id","sub_category","sort_source","name",
                   "brand / brand_normalized","price","original_price","discount_pct",
                   "price_tier","sold","monthly_sold","rating","rating_count",
                   "r5…r1","is_mall","has_flash_sale","is_free_ship","revenue_est",
                   "shop_followers","shop_rating","shop_response_rate","shop_location"],
        "Kiểu": ["int","int","str","str","str","str","float","float","float (%)","str",
                 "int","int","float","int","int","bool","bool","bool","float",
                 "int","float","float","str"],
        "Mô tả": ["ID sản phẩm","ID cửa hàng","Danh mục con (22 loại)","Keyword crawl","Tên SP",
                  "Thương hiệu","Giá hiện tại (VND)","Giá gốc","% chiết khấu",
                  "budget/mid-low/mid/premium/luxury","Tổng đã bán","Bán trong tháng",
                  "Điểm TB (0–5)","Số lượt đánh giá","Số lượt 5 sao -> 1 sao",
                  "1=Shopee Mall","1=Flash sale","1=Miễn ship","price x sold (ước tính)",
                  "Followers cửa hàng","Rating cửa hàng","Tỷ lệ phản hồi","Tỉnh/thành"],
    })
    st.dataframe(schema, hide_index=True, use_container_width=True)
    st.caption(f"{len(products):,} dòng x {len(products.columns)} cột."
               f"Missing: {products.isnull().sum().sum():,} ô")

with tab_s:
    schema_s = pd.DataFrame({
        "Trường": ["shop_id","shop_name","is_mall","is_verified","follower_count",
                   "rating_star","rating_count","response_rate","location","total_products","total_sold"],
        "Mô tả": ["Khóa chính","Tên cửa hàng","Shopee Mall","Xác thực","Số followers",
                  "Rating TB (0–5)","Số lượt đánh giá","Tỷ lệ phản hồi (0–1)","Tỉnh/thành",
                  "Tổng SP đang bán","Tổng đã bán"],
    })
    st.dataframe(schema_s, hide_index=True, use_container_width=True)
    st.caption(f"{len(shops):,} dòng x {len(shops.columns)} cột")

with tab_r:
    schema_r = pd.DataFrame({
        "Trường": ["item_id","shop_id","reviewer_id","rating","review_text",
                   "review_length","has_image","has_video","helpful_count","reviewed_ts"],
        "Mô tả": ["FK->products","FK->shops","ID ẩn danh","Số sao (1–5)","Nội dung tự do",
                  "Độ dài (ký tự)","Kèm ảnh","Kèm video","Lượt thấy hữu ích","Timestamp"],
    })
    st.dataframe(schema_r, hide_index=True, use_container_width=True)
    st.caption(f"{len(reviews):,} dòng x {len(reviews.columns)} cột")

st.markdown("---")

# ========================
# 2. Phân bố các biến chính
# ========================
st.subheader("2. Phân bố các biến chính")

r1c1, r1c2 = st.columns(2)

with r1c1:
    st.markdown("**Phân bố phân khúc giá (price_tier)**")
    tier_order = ["budget", "mid-low", "mid", "premium", "luxury"]
    tc = products["price_tier"].value_counts().reindex(tier_order).dropna()
    fig = px.bar(
        x=tc.index, y=tc.values,
        color=tc.values, color_continuous_scale=SEQ_BLUES,
        text=tc.values,
        labels={"x": "Phân khúc", "y": "Số sản phẩm"},
        title="Phân bố price_tier",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white", coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Nhận xét:** ~90% sản phẩm ở budget + mid-low. Luxury chỉ ~28 sản phẩm.")

with r1c2:
    st.markdown("**Top 10 danh mục theo doanh thu ước tính**")
    top10 = (
        active.groupby("sub_category")["revenue_est"]
        .sum()
        .sort_values(ascending=False)
        .head(10) / 1e9
    )
    fig = px.bar(
        x=top10.values, y=top10.index, orientation="h",
        color=top10.values, color_continuous_scale=SEQ_BLUES,
        text=top10.values.round(0),
        labels={"x": "Tỷ VND", "y": ""},
        title="Top 10 danh mục – Doanh thu ước tính (tỷ VND)",
    )
    fig.update_traces(texttemplate="%{text:.0f}B", textposition="outside")
    fig.update_layout(
        showlegend=False, plot_bgcolor="white",
        yaxis={"categoryorder": "total ascending"},
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Nhận xét:** Mặt nạ dẫn đầu (~700B), gấp 1.6x sữa rửa mặt.")

r2c1, r2c2 = st.columns(2)

with r2c1:
    st.markdown("**Phân bố rating đánh giá (J-curve)**")
    rc = reviews["rating"].value_counts().sort_index()
    bar_colors = [CB_VERMIL, CB_VERMIL, CB_ORANGE, CB_SKYBLUE, CB_BLUE]
    fig = go.Figure()
    for i, (star, cnt) in enumerate(rc.items()):
        fig.add_bar(
            x=[star], y=[cnt],
            marker_color=bar_colors[i],
            text=[f"{cnt:,}<br>({cnt/len(reviews)*100:.1f}%)"],
            textposition="outside",
            name=f"{star} sao",
        )
    fig.update_layout(
        showlegend=True, plot_bgcolor="white",
        xaxis_title="Số sao", yaxis_title="Số đánh giá",
        title=f"Phân bố rating – {len(reviews):,} đánh giá",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("**Nhận xét:** 99.58% là 5 sao — J-curve điển hình TMĐT.")

with r2c2:
    st.markdown("**Ma trận tương quan các biến số**")
    num_cols = ["price", "discount_pct", "sold", "rating", "revenue_est",
                "shop_followers", "shop_rating"]
    available = [c for c in num_cols if c in active.columns]
    corr = active[available].corr()
    fig = px.imshow(
        corr, text_auto=".2f",
        color_continuous_scale=DIV_RDBU,
        zmin=-1, zmax=1, aspect="auto",
        title="Ma trận tương quan Pearson",
    )
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
    st.caption(
        "**Nhận xét:** sold<->revenue r=0.79 (mạnh nhất). "
        "shop_rating↔revenue r=−0.15 (nghịch chiều — counterintuitive)."
    )

st.markdown("---")

# ========================
# 3. Thống kê mô tả
# ========================
st.subheader("3. Thống kê mô tả tổng hợp")
desc_cols = ["price", "original_price", "discount_pct", "sold",
             "monthly_sold", "rating", "rating_count", "revenue_est"]
avail = [c for c in desc_cols if c in products.columns]
st.dataframe(products[avail].describe().round(2), use_container_width=True)

st.markdown("---")
st.markdown("""
### 4. Nhận xét tổng quan

- **Kích thước mẫu:** 20,658 sản phẩm (38 cột, 4,448 ô missing); 5,746 cửa hàng; 23,989 đánh giá — đủ lớn để phân tích phân phối và so sánh nhóm. Dữ liệu bao phủ 28 danh mục con.

- **Phân bố price_tier (budget/mid-low chiếm ưu thế):** Budget có 10,051 SP và mid-low có 8,633 SP — cộng lại chiếm ~90% tổng số sản phẩm. Phân khúc mid chỉ có 1,787 SP, premium 159 SP, luxury 28 SP. Phản ánh đúng định vị Shopee là nền tảng đại chúng giá phải chăng.

- **Phân bố giá và sold (lệch phải mạnh):** Giá trung bình 144,452 đồng nhưng trung vị chỉ 103,206 đồng — chênh lệch lớn do outlier giá cao kéo trung bình lên. Sold trung bình 1,776 nhưng trung vị chỉ 95 — phân phối lũy thừa điển hình TMĐT: phần lớn SP bán ít, một thiểu số nhỏ bán rất nhiều.

- **Phân bố rating (J-curve):** 23,888/23,989 đánh giá (99.6%) là 5 sao — J-curve điển hình TMĐT Việt Nam. Shopee khuyến khích rating bằng xu/voucher -> cần cân bằng khi dùng ML phân loại cảm xúc (tham khảo thêm tại trang Đánh giá).

- **Tương quan sold <-> revenue (r=0.79):** Liên kết mạnh nhất trong toàn bộ ma trận — doanh thu ước tính được quyết định chủ yếu bởi lượng bán, không phải giá. Hàm ý: chiến lược volume-first hiệu quả hơn premium-only trên Shopee.

- **Counterintuitive — shop_rating <-> revenue (r=–0.15):** Tương quan âm nhẹ giữa rating cửa hàng và doanh thu sản phẩm. Giải thích: các mega-shop bán hàng chục nghìn SP có thể có vài review tiêu cực làm giảm nhẹ điểm rating, trong khi shop nhỏ mới mở có ít review toàn 5 sao nhưng doanh thu thấp. Đây là trường hợp Simpson's Paradox — xem chi tiết tại trang Shops.
""")
