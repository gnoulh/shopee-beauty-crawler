import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings

warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide", page_title="Phân tích Shopee Mỹ Phẩm")

# ─── CUSTOM CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Be+Vietnam+Pro:wght@300;400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Be Vietnam Pro', sans-serif;
}
/* Hero banner */
.hero-banner {
    background: linear-gradient(135deg, #1a0a2e 0%, #16213e 40%, #0f3460 100%);
    border: 1px solid rgba(230, 90, 150, 0.3);
    border-radius: 16px;
    padding: 36px 40px;
    margin-bottom: 32px;
    position: relative;
    overflow: hidden;
}

.hero-banner::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(230,90,150,0.15) 0%, transparent 70%);
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    background: linear-gradient(90deg, #ff6b9d, #c44dff, #4d79ff);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.hero-sub {
    color: #9090b0;
    font-size: 0.95rem;
    font-weight: 300;
}

/* Objective cards */
.obj-card {
    background: linear-gradient(135deg, rgba(255,107,157,0.08), rgba(77,121,255,0.06));
    border: 1px solid rgba(255,107,157,0.25);
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
}
.obj-num {
    font-family: 'JetBrains Mono', monospace;
    color: #ff6b9d;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 6px;
}
.obj-text {
    color: #dde;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* Section header */
.section-header {
    display: flex;
    align-items: center;
    gap: 14px;
    margin: 40px 0 20px 0;
    padding-bottom: 12px;
    border-bottom: 1px solid rgba(255,107,157,0.2);
}
.section-badge {
    background: linear-gradient(135deg, #ff6b9d, #c44dff);
    color: white;
    font-size: 0.75rem;
    font-weight: 700;
    padding: 4px 12px;
    border-radius: 20px;
    letter-spacing: 1px;
    white-space: nowrap;
}
.section-title {
    font-size: 1.4rem;
    font-weight: 700;
    color: #f0f0ff;
}

/* Insight box */
.insight-box {
    background: rgba(77,121,255,0.08);
    border-left: 3px solid #4d79ff;
    border-radius: 0 8px 8px 0;
    padding: 14px 18px;
    margin: 16px 0;
    color: #c0c8e8;
    font-size: 0.9rem;
    line-height: 1.7;
}
.insight-box strong {
    color: #7fa8ff;
}

/* Conclusion box */
.conclusion-box {
    background: linear-gradient(135deg, rgba(255,107,157,0.1), rgba(196,77,255,0.08));
    border: 1px solid rgba(255,107,157,0.3);
    border-radius: 12px;
    padding: 20px 24px;
    margin-top: 24px;
    color: #dde;
    font-size: 0.92rem;
    line-height: 1.75;
}
.conclusion-title {
    font-size: 1rem;
    font-weight: 700;
    color: #ff6b9d;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* Metric cards */
.metric-row {
    display: flex;
    gap: 12px;
    margin-bottom: 20px;
    flex-wrap: wrap;
}
.metric-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 14px 18px;
    flex: 1;
    min-width: 140px;
    text-align: center;
}
.metric-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.5rem;
    font-weight: 600;
    color: #ff6b9d;
}
.metric-label {
    font-size: 0.78rem;
    color: #8888aa;
    margin-top: 4px;
}

/* Chart label */
.chart-label {
    font-size: 0.85rem;
    font-weight: 600;
    color: #9090c0;
    letter-spacing: 1px;
    text-transform: uppercase;
    margin-bottom: 6px;
    font-family: 'JetBrains Mono', monospace;
}
</style>
""",
    unsafe_allow_html=True,
)


# ─── LOAD DATA ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    products = pd.read_csv("data/products.csv")
    shops = pd.read_csv("data/shops.csv")
    reviews = pd.read_csv("data/reviews.csv")
    return products, shops, reviews


products, shops, reviews = load_data()

# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
products["revenue_est"] = products["revenue_est"].fillna(0).astype(float)
products["sold"] = products["sold"].fillna(0).astype(float)
products["rating"] = products["rating"].fillna(0).astype(float)


# Chuẩn hóa tên tỉnh/thành
def normalize_location(loc):
    if pd.isna(loc):
        return "Không xác định"
    loc = str(loc).strip()
    mappings = {
        "TP. Hồ Chí Minh": "Hồ Chí Minh",
        "TP.HCM": "Hồ Chí Minh",
        "Ho Chi Minh": "Hồ Chí Minh",
        "HCM": "Hồ Chí Minh",
        "Hà Nội": "Hà Nội",
        "Ha Noi": "Hà Nội",
        "Đà Nẵng": "Đà Nẵng",
        "Da Nang": "Đà Nẵng",
        "Bình Dương": "Bình Dương",
        "Hải Phòng": "Hải Phòng",
        "Cần Thơ": "Cần Thơ",
    }
    for k, v in mappings.items():
        if k.lower() in loc.lower():
            return v
    return loc


products["location_norm"] = products["shop_location"].apply(normalize_location)


# ═══════════════════════════════════════════════════════════════════════════════
# HERO BANNER
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="hero-banner">
    <div class="hero-title">Phân tích Thị trường Mỹ phẩm Shopee</div>
    <div class="hero-sub">Snapshot 18–19/3/2026 · 20,658 sản phẩm · 5,746 cửa hàng · 23,989 đánh giá</div>
</div>
""",
    unsafe_allow_html=True,
)

# ─── DANH SÁCH MỤC TIÊU ─────────────────────────────────────────────────────────
st.markdown("### 🎯 Mục tiêu phân tích")

st.markdown(
    """
<div class="obj-card">
    <div class="obj-num">Mục tiêu 01</div>
    <div class="obj-text">
        <strong>Phân tích phân bố doanh thu theo tỉnh/thành</strong> — Xác định 3 khu vực địa lý
        dẫn đầu về doanh thu ước tính và số lượng bán, đồng thời mô tả đặc trưng riêng của
        từng khu vực (mức giá trung bình, tỷ lệ Shopee Mall, danh mục nổi bật) dựa trên
        snapshot 18/3/2026.
    </div>
</div>
<div class="obj-card">
    <div class="obj-num">Mục tiêu 02</div>
    <div class="obj-text">
        <strong>So sánh hiệu quả bán hàng giữa Shopee Mall và Non-Mall</strong> — So sánh các chỉ số
        sold, rating và revenue_est giữa 8,298 sản phẩm Mall và 12,360 sản phẩm Non-Mall
        từ dữ liệu 18/3/2026 để đánh giá lợi thế thực sự của nhãn hàng chính hãng.
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MỤC TIÊU 1 — PHÂN BỐ ĐỊA LÝ
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="section-header">
    <span class="section-badge">MỤC TIÊU 01</span>
    <span class="section-title">Phân bố doanh thu theo tỉnh/thành</span>
</div>
""",
    unsafe_allow_html=True,
)

# Aggregate by location
geo_df = (
    products.groupby("location_norm")
    .agg(
        revenue_est=("revenue_est", "sum"),
        sold=("sold", "sum"),
        avg_price=("price", "mean"),
        product_count=("id", "count"),
        mall_count=("is_mall", "sum"),
        avg_rating=("rating", "mean"),
    )
    .reset_index()
)
geo_df["mall_ratio"] = geo_df["mall_count"] / geo_df["product_count"] * 100
geo_df["revenue_B"] = geo_df["revenue_est"] / 1e9
geo_df["revenue_M"] = geo_df["revenue_est"] / 1e6

# Lọc top 15
geo_top15 = geo_df.nlargest(15, "revenue_est").copy()
geo_top3 = geo_df.nlargest(3, "revenue_est").copy()

PINK = "#ff6b9d"
PURPLE = "#c44dff"
BLUE = "#4d79ff"
TEAL = "#00d4aa"
AMBER = "#ffb347"

colors_bar = [PINK if i < 3 else "#3a3a5c" for i in range(len(geo_top15))]

# ── Biểu đồ 1.1: Top 15 tỉnh/thành theo doanh thu ──────────────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 1.1 — Top 15 tỉnh/thành theo doanh thu ước tính</div>',
    unsafe_allow_html=True,
)

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=geo_top15["revenue_B"],
        y=geo_top15["location_norm"],
        orientation="h",
        marker=dict(
            color=colors_bar,
            line=dict(color="rgba(255,255,255,0.05)", width=1),
        ),
        text=[f"{v:.1f} tỷ" for v in geo_top15["revenue_B"]],
        textposition="outside",
        textfont=dict(color="#dde", size=12),
        hovertemplate="<b>%{y}</b><br>Doanh thu: %{x:.2f} tỷ VNĐ<extra></extra>",
    )
)
fig1.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=480,
    xaxis=dict(
        color="#8888aa",
        gridcolor="rgba(255,255,255,0.06)",
        title=dict(
            text="Doanh thu ước tính (tỷ VNĐ)",
            font=dict(size=12),
        ),
    ),
    yaxis=dict(
        autorange="reversed",
        color="#dde",
        tickfont=dict(size=12),
    ),
    font=dict(family="Be Vietnam Pro", color="#dde"),
    margin=dict(l=20, r=80, t=20, b=40),
    showlegend=False,
    annotations=[
        dict(
            x=geo_top15["revenue_B"].max() * 0.6,
            y=0,
            text="⭐ Top 3",
            showarrow=False,
            font=dict(color=PINK, size=11),
            xanchor="left",
        )
    ],
)
st.plotly_chart(fig1, use_container_width=True)

st.markdown(
    f"""
<div class="insight-box">
    <strong>Nhận xét:</strong> Ba khu vực dẫn đầu là
    <strong>{geo_top3.iloc[0]['location_norm']}</strong>,
    <strong>{geo_top3.iloc[1]['location_norm']}</strong> và
    <strong>{geo_top3.iloc[2]['location_norm']}</strong> —
    chiếm tỷ trọng áp đảo trong tổng doanh thu toàn thị trường.
    Sự tập trung doanh thu theo địa lý phản ánh mật độ người dùng và hạ tầng logistics
    tại các đô thị lớn.
</div>
""",
    unsafe_allow_html=True,
)

# ── Biểu đồ 1.2: Bubble chart — Số sản phẩm vs Doanh thu ───────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 1.2 — Mối quan hệ giữa số sản phẩm, doanh thu và tỷ lệ Mall</div>',
    unsafe_allow_html=True,
)

geo_plot = geo_df[geo_df["product_count"] >= 50].copy()

fig2 = px.scatter(
    geo_plot,
    x="product_count",
    y="revenue_B",
    size="sold",
    color="mall_ratio",
    hover_name="location_norm",
    color_continuous_scale=[[0, "#4d79ff"], [0.5, "#c44dff"], [1, "#ff6b9d"]],
    size_max=60,
    labels={
        "product_count": "Số lượng sản phẩm",
        "revenue_B": "Doanh thu (tỷ VNĐ)",
        "sold": "Tổng lượt bán",
        "mall_ratio": "Tỷ lệ Mall (%)",
    },
    hover_data={"avg_price": ":.0f", "avg_rating": ":.2f"},
)
fig2.update_traces(
    marker=dict(opacity=0.85, line=dict(color="white", width=0.5)),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Sản phẩm: %{x:,}<br>"
        "Doanh thu: %{y:.2f} tỷ<br>"
        "Lượt bán: %{marker.size:,}<br>"
        "Tỷ lệ Mall: %{marker.color:.1f}%<extra></extra>"
    ),
)
fig2.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=420,
    font=dict(family="Be Vietnam Pro", color="#dde"),
    xaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        color="#8888aa",
        title=dict(font=dict(size=12)),
    ),
    yaxis=dict(
        gridcolor="rgba(255,255,255,0.06)",
        color="#8888aa",
        title=dict(font=dict(size=12)),
    ),
    coloraxis_colorbar=dict(
        tickfont=dict(color="#dde"),
        title=dict(text="Tỷ lệ Mall (%)", font=dict(color="#dde")),
    ),
    margin=dict(l=20, r=20, t=20, b=40),
)
st.plotly_chart(fig2, use_container_width=True)

st.markdown(
    """
<div class="insight-box">
    <strong>Nhận xét:</strong> Các tỉnh/thành có nhiều sản phẩm nhất không nhất thiết có
    doanh thu cao nhất — <strong>chất lượng danh mục và tỷ lệ Mall cao</strong> (màu hồng)
    tương quan rõ hơn với doanh thu. Bong bóng lớn hơn (lượt bán nhiều) tập trung
    ở nhóm dẫn đầu, cho thấy hiệu ứng "người thắng lấy tất" trong phân phối địa lý.
</div>
""",
    unsafe_allow_html=True,
)

# ── Biểu đồ 1.3: Radar chart cho Top 3 ─────────────────────────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 1.3 — Hồ sơ đặc trưng 3 khu vực dẫn đầu (Radar chart)</div>',
    unsafe_allow_html=True,
)

top3_names = geo_top3["location_norm"].tolist()


def minmax_norm(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return series * 0
    return (series - mn) / (mx - mn)


radar_cols = [
    "revenue_B",
    "sold",
    "avg_price",
    "mall_ratio",
    "avg_rating",
    "product_count",
]
radar_labels = ["Doanh thu", "Lượt bán", "Giá TB", "Tỷ lệ Mall", "Đánh giá TB", "Số SP"]

radar_data = geo_df[geo_df["location_norm"].isin(top3_names)].copy()
for c in radar_cols:
    radar_data[c + "_norm"] = minmax_norm(geo_df[c])[
        geo_df["location_norm"].isin(top3_names)
    ].values

fig3 = go.Figure()
palette = [PINK, BLUE, TEAL]
for i, (_, row) in enumerate(radar_data.iterrows()):
    vals = [row[c + "_norm"] for c in radar_cols]
    vals += [vals[0]]
    fig3.add_trace(
        go.Scatterpolar(
            r=vals,
            theta=radar_labels + [radar_labels[0]],
            fill="toself",
            name=row["location_norm"],
            line=dict(color=palette[i], width=2),
            fillcolor=palette[i],
            opacity=0.9,
        )
    )

fig3.update_layout(
    polar=dict(
        bgcolor="rgba(255,255,255,0.03)",
        radialaxis=dict(
            visible=True,
            range=[0, 1],
            color="#666688",
            gridcolor="rgba(255,255,255,0.08)",
        ),
        angularaxis=dict(
            color="#aaaacc",
            gridcolor="rgba(255,255,255,0.08)",
        ),
    ),
    paper_bgcolor="rgba(0,0,0,0)",
    height=400,
    font=dict(family="Be Vietnam Pro", color="#dde"),
    legend=dict(
        font=dict(color="#dde"),
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    margin=dict(l=60, r=60, t=30, b=30),
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown(
    f"""
<div class="insight-box">
    <strong>Nhận xét:</strong>
    <strong>{top3_names[0]}</strong> nổi bật với hồ sơ cân bằng — dẫn đầu cả doanh thu, lượt bán
    và số sản phẩm. <strong>{top3_names[1] if len(top3_names) > 1 else ''}</strong> có tỷ lệ Mall cao
    hơn tương đối, gợi ý thị trường cao cấp hơn. Mỗi khu vực thể hiện một chiến lược
    khác nhau: volume cao, mall-heavy, hoặc giá trung bình thấp kèm lượt bán lớn.
</div>
""",
    unsafe_allow_html=True,
)

# ── Conclusion 1 ────────────────────────────────────────────────────────────────
revenue_total = geo_df["revenue_est"].sum()
top3_revenue = geo_top3["revenue_est"].sum()
top3_pct = top3_revenue / revenue_total * 100 if revenue_total > 0 else 0

st.markdown(
    f"""
<div class="conclusion-box">
    <div class="conclusion-title">📌 Kết luận Mục tiêu 01</div>
    Ba khu vực dẫn đầu là <strong>{top3_names[0]}</strong>,
    <strong>{top3_names[1] if len(top3_names) > 1 else 'N/A'}</strong> và
    <strong>{top3_names[2] if len(top3_names) > 2 else 'N/A'}</strong>,
    chiếm khoảng <strong>{top3_pct:.1f}%</strong> tổng doanh thu ước tính toàn thị trường.
    Doanh thu mỹ phẩm trên Shopee tập trung cao độ tại hai đầu tàu kinh tế Bắc–Nam,
    phản ánh mật độ người dùng và hạ tầng giao nhận vượt trội. Các tỉnh thành còn lại
    đóng góp phần nhỏ nhưng có tiềm năng tăng trưởng nếu được đầu tư logistics và
    chính sách giá phù hợp. Người bán mới nên ưu tiên đặt kho/vận chuyển tại
    <strong>{top3_names[0]}</strong> để tối ưu tốc độ giao hàng và phủ sóng khách hàng.
</div>
""",
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MỤC TIÊU 2 — MALL vs NON-MALL
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown(
    """
<div class="section-header">
    <span class="section-badge">MỤC TIÊU 02</span>
    <span class="section-title">So sánh Shopee Mall vs Non-Mall</span>
</div>
""",
    unsafe_allow_html=True,
)

mall = products[products["is_mall"] == 1].copy()
nonmall = products[products["is_mall"] == 0].copy()

n_mall = len(mall)
n_nonmall = len(nonmall)

# KPI row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-val">{n_mall:,}</div>
        <div class="metric-label">Sản phẩm Mall</div>
    </div>""",
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-val">{n_nonmall:,}</div>
        <div class="metric-label">Sản phẩm Non-Mall</div>
    </div>""",
        unsafe_allow_html=True,
    )
with col3:
    mall_rev = mall["revenue_est"].sum() / 1e9
    non_rev = nonmall["revenue_est"].sum() / 1e9
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-val">{mall_rev:.0f}B</div>
        <div class="metric-label">Doanh thu Mall (tỷ VNĐ)</div>
    </div>""",
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""<div class="metric-card">
        <div class="metric-val">{non_rev:.0f}B</div>
        <div class="metric-label">Doanh thu Non-Mall (tỷ VNĐ)</div>
    </div>""",
        unsafe_allow_html=True,
    )

# ── Biểu đồ 2.1: Box plot so sánh 3 chỉ số ──────────────────────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 2.1 — Phân phối Sold / Rating / Revenue theo loại cửa hàng</div>',
    unsafe_allow_html=True,
)

fig4 = make_subplots(
    rows=1,
    cols=3,
    subplot_titles=[
        "Lượt bán (sold)",
        "Đánh giá (rating)",
        "Doanh thu ước tính (triệu VNĐ)",
    ],
    horizontal_spacing=0.08,
)

for col_idx, (col_name, label, scale) in enumerate(
    [
        ("sold", "Lượt bán", 1),
        ("rating", "Điểm đánh giá", 1),
        ("revenue_est", "Triệu VNĐ", 1e6),
    ],
    start=1,
):
    for grp, color, name in [(mall, PINK, "Mall"), (nonmall, BLUE, "Non-Mall")]:
        data_vals = grp[col_name].dropna()
        if scale != 1:
            data_vals = data_vals / scale
        # Cap at 99th percentile for readability
        p99 = data_vals.quantile(0.99)
        data_vals = data_vals[data_vals <= p99]
        fig4.add_trace(
            go.Box(
                y=data_vals,
                name=name,
                marker_color=color,
                boxmean="sd",
                showlegend=(col_idx == 1),
                line=dict(width=1.5),
            ),
            row=1,
            col=col_idx,
        )

fig4.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=400,
    font=dict(family="Be Vietnam Pro", color="#dde"),
    legend=dict(
        font=dict(color="#dde"),
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
        orientation="h",
        x=0.5,
        xanchor="center",
        y=-0.12,
    ),
    margin=dict(l=20, r=20, t=50, b=60),
)
for i in range(1, 4):
    fig4.update_xaxes(showgrid=False, row=1, col=i)
    fig4.update_yaxes(gridcolor="rgba(255,255,255,0.06)", color="#8888aa", row=1, col=i)
    fig4.update_annotations(font=dict(color="#aaaacc", size=12))

st.plotly_chart(fig4, use_container_width=True)

st.markdown(
    """
<div class="insight-box">
    <strong>Nhận xét:</strong> Sản phẩm <strong>Mall</strong> có trung vị sold và revenue_est
    cao hơn rõ rệt, nhưng phân phối cũng rải rộng hơn — một số Mall có doanh thu cực lớn
    kéo phân phối lên. <strong>Rating</strong> giữa hai nhóm gần nhau, cho thấy
    chất lượng cảm nhận của khách hàng không chênh lệch nhiều chỉ vì nhãn Mall.
</div>
""",
    unsafe_allow_html=True,
)

# ── Biểu đồ 2.2: Grouped bar — Trung bình 3 chỉ số ──────────────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 2.2 — Trung bình các chỉ số: Mall vs Non-Mall</div>',
    unsafe_allow_html=True,
)

compare_data = {
    "Chỉ số": [
        "Sold TB",
        "Rating TB",
        "Revenue TB (triệu)",
        "Giá TB (nghìn)",
        "Discount TB (%)",
    ],
    "Mall": [
        mall["sold"].mean(),
        mall["rating"].mean(),
        mall["revenue_est"].mean() / 1e6,
        mall["price"].mean() / 1e3,
        mall["discount_pct"].mean(),
    ],
    "Non-Mall": [
        nonmall["sold"].mean(),
        nonmall["rating"].mean(),
        nonmall["revenue_est"].mean() / 1e6,
        nonmall["price"].mean() / 1e3,
        nonmall["discount_pct"].mean(),
    ],
}
cmp_df = pd.DataFrame(compare_data)

fig5 = go.Figure()
fig5.add_trace(
    go.Bar(
        name="Mall",
        x=cmp_df["Chỉ số"],
        y=cmp_df["Mall"],
        marker_color=PINK,
        text=[f"{v:.1f}" for v in cmp_df["Mall"]],
        textposition="outside",
        textfont=dict(color=PINK, size=11),
    )
)
fig5.add_trace(
    go.Bar(
        name="Non-Mall",
        x=cmp_df["Chỉ số"],
        y=cmp_df["Non-Mall"],
        marker_color=BLUE,
        text=[f"{v:.1f}" for v in cmp_df["Non-Mall"]],
        textposition="outside",
        textfont=dict(color=BLUE, size=11),
    )
)
fig5.update_layout(
    barmode="group",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=380,
    font=dict(family="Be Vietnam Pro", color="#dde"),
    legend=dict(
        font=dict(color="#dde"),
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    xaxis=dict(color="#aaaacc", tickfont=dict(size=12)),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", color="#8888aa"),
    margin=dict(l=20, r=20, t=30, b=40),
    bargap=0.25,
    bargroupgap=0.05,
)
st.plotly_chart(fig5, use_container_width=True)

st.markdown(
    """
<div class="insight-box">
    <strong>Nhận xét:</strong> Mall có lượt bán trung bình và doanh thu ước tính cao hơn
    hẳn — nhưng cũng có <strong>giá trung bình cao hơn</strong> và thường áp dụng
    <strong>discount cao hơn</strong> (chiến lược giảm giá sâu để tạo khối lượng).
    Non-Mall cạnh tranh ở phân khúc giá thấp hơn với biên lợi nhuận mỏng hơn.
</div>
""",
    unsafe_allow_html=True,
)

# ── Biểu đồ 2.3: Pie chart — Thị phần doanh thu + Lượt bán ─────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 2.3 — Thị phần doanh thu & lượt bán theo loại cửa hàng</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)
with col_a:
    labels = ["Shopee Mall", "Non-Mall"]
    rev_vals = [mall["revenue_est"].sum(), nonmall["revenue_est"].sum()]
    fig6a = go.Figure(
        go.Pie(
            labels=labels,
            values=rev_vals,
            hole=0.55,
            marker=dict(colors=[PINK, BLUE]),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value:,.0f} VNĐ<extra></extra>",
        )
    )
    fig6a.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        font=dict(family="Be Vietnam Pro", color="#dde"),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(
            text="Doanh thu ước tính", font=dict(color="#aaaacc", size=13), x=0.5
        ),
        annotations=[
            dict(
                text=f"{mall['revenue_est'].sum()/1e9:.0f}B<br>Mall",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=PINK),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig6a, use_container_width=True)

with col_b:
    sold_vals = [mall["sold"].sum(), nonmall["sold"].sum()]
    fig6b = go.Figure(
        go.Pie(
            labels=labels,
            values=sold_vals,
            hole=0.55,
            marker=dict(colors=[PINK, BLUE]),
            textinfo="label+percent",
            textfont=dict(size=13, color="white"),
            hovertemplate="%{label}: %{value:,} lượt<extra></extra>",
        )
    )
    fig6b.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        height=300,
        font=dict(family="Be Vietnam Pro", color="#dde"),
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Tổng lượt bán", font=dict(color="#aaaacc", size=13), x=0.5),
        annotations=[
            dict(
                text=f"{mall['sold'].sum()/1e6:.1f}M<br>Mall",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=PINK),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig6b, use_container_width=True)

mall_rev_pct = mall["revenue_est"].sum() / products["revenue_est"].sum() * 100
mall_sold_pct = mall["sold"].sum() / products["sold"].sum() * 100
mall_prod_pct = n_mall / (n_mall + n_nonmall) * 100

st.markdown(
    f"""
<div class="insight-box">
    <strong>Nhận xét:</strong>
    Mặc dù Mall chỉ chiếm <strong>{mall_prod_pct:.1f}%</strong> số lượng sản phẩm,
    nhưng đóng góp đến <strong>{mall_rev_pct:.1f}%</strong> doanh thu và
    <strong>{mall_sold_pct:.1f}%</strong> tổng lượt bán toàn thị trường.
    Điều này cho thấy nhãn Shopee Mall mang lại hiệu ứng uy tín (<em>trust signal</em>)
    rõ rệt, giúp chuyển đổi cao hơn đáng kể so với cửa hàng thông thường.
</div>
""",
    unsafe_allow_html=True,
)

# ── Biểu đồ 2.4: Violin plot phân phối rating ────────────────────────────────────
st.markdown(
    '<div class="chart-label">Biểu đồ 2.4 — Phân phối điểm đánh giá (Violin plot)</div>',
    unsafe_allow_html=True,
)

products_plot = products[products["rating"] > 0].copy()
products_plot["Loại"] = products_plot["is_mall"].map({1: "Shopee Mall", 0: "Non-Mall"})

fig7 = go.Figure()
for grp, color in [("Shopee Mall", PINK), ("Non-Mall", BLUE)]:
    data_r = products_plot[products_plot["Loại"] == grp]["rating"]
    fig7.add_trace(
        go.Violin(
            y=data_r,
            name=grp,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.9,
            points=False,
        )
    )

fig7.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.02)",
    height=360,
    font=dict(family="Be Vietnam Pro", color="#dde"),
    yaxis=dict(
        range=[3, 5.1],
        gridcolor="rgba(255,255,255,0.06)",
        color="#8888aa",
        title=dict(text="Điểm đánh giá", font=dict(size=12)),
    ),
    legend=dict(
        font=dict(color="#dde"),
        bgcolor="rgba(0,0,0,0.3)",
        bordercolor="rgba(255,255,255,0.1)",
        borderwidth=1,
    ),
    margin=dict(l=20, r=20, t=20, b=30),
    violingap=0.3,
    violinmode="group",
)
st.plotly_chart(fig7, use_container_width=True)

mall_rating = mall[mall["rating"] > 0]["rating"].mean()
nonmall_rating = nonmall[nonmall["rating"] > 0]["rating"].mean()

st.markdown(
    f"""
<div class="insight-box">
    <strong>Nhận xét:</strong>
    Phân phối rating của Mall ({mall_rating:.2f}★ trung bình) và Non-Mall ({nonmall_rating:.2f}★)
    rất gần nhau và tập trung cao ở vùng 4.5–5.0★.
    Điều này cho thấy đánh giá sản phẩm trên Shopee có xu hướng <strong>tích cực</strong>
    ở cả hai nhóm — khách hàng có xu hướng chỉ để lại đánh giá khi hài lòng,
    hoặc hệ thống khuyến khích đánh giá 5 sao. Mall không có lợi thế rõ ràng về rating.
</div>
""",
    unsafe_allow_html=True,
)

# ── Conclusion 2 ────────────────────────────────────────────────────────────────
st.markdown(
    f"""
<div class="conclusion-box">
    <div class="conclusion-title">📌 Kết luận Mục tiêu 02</div>
    Shopee Mall thể hiện ưu thế rõ ràng về <strong>doanh thu và lượt bán</strong>:
    chỉ với {mall_prod_pct:.1f}% số sản phẩm nhưng tạo ra {mall_rev_pct:.1f}% tổng doanh thu —
    cho thấy nhãn Mall là một <em>trust signal</em> mạnh trong hành vi mua sắm mỹ phẩm.
    Tuy nhiên, về <strong>điểm đánh giá</strong>, hai nhóm gần như ngang nhau ({mall_rating:.2f}★
    vs {nonmall_rating:.2f}★), cho thấy Non-Mall vẫn có thể cạnh tranh về chất lượng dịch vụ.
    <br><br>
    <strong>Khuyến nghị:</strong> Người bán Non-Mall nên tập trung vào việc tích lũy
    đánh giá tích cực, cải thiện tỷ lệ phản hồi và cân nhắc đăng ký chương trình
    Shopee Mall để được hưởng lợi từ hiệu ứng uy tín thương hiệu. Trong khi đó,
    chiến lược giảm giá sâu kết hợp free shipping có thể là đòn bẩy hiệu quả
    cho nhóm Non-Mall muốn tăng volume bán hàng.
</div>
""",
    unsafe_allow_html=True,
)

# ─── FOOTER ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center; padding: 40px 0 20px 0; color: #444466; font-size: 0.82rem; font-family: 'JetBrains Mono', monospace;">
    Trực quan hóa dữ liệu · 23KHDL · Snapshot 18–19/3/2026
</div>
""",
    unsafe_allow_html=True,
)
