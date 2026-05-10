import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ── Load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/vietnam_housing_dataset_cleaned.csv")
    return df


df = load_data()

# ── Header ───────────────────────────────────────────────
st.title("🏗️ Cấu trúc & Quy mô Bất động sản")
st.markdown(
    """
### ❓ Câu hỏi phân tích
> **Diện tích và cấu trúc vật lý của bất động sản (số tầng, phòng ngủ, phòng tắm, mặt tiền)
> ảnh hưởng như thế nào đến giá bán? Liệu "nhà to hơn" có luôn đồng nghĩa với "đắt hơn"?**

Phân tích tập trung vào các biến: `Area`, `Floors`, `Bedrooms`, `Bathrooms`, `Frontage`
so với `Price` và `Price_per_m2`.
"""
)
st.markdown("---")

# ── Filters sidebar ──────────────────────────────────────
with st.sidebar:
    st.header("🔧 Bộ lọc")
    price_range = st.slider(
        "Khoảng giá (tỷ đồng)",
        float(df["Price"].min()),
        float(df["Price"].quantile(0.99)),
        (float(df["Price"].min()), float(df["Price"].quantile(0.95))),
    )
    selected_provinces = st.multiselect(
        "Tỉnh/Thành phố", options=sorted(df["Province"].dropna().unique()), default=[]
    )

# Apply filter
mask = df["Price"].between(*price_range)
if selected_provinces:
    mask &= df["Province"].isin(selected_provinces)
dff = df[mask].copy()

st.caption(f"Đang hiển thị **{len(dff):,}** / {len(df):,} bất động sản sau lọc")

# ═══════════════════════════════════════════════════════
# 1. SCATTER: Area vs Price (colored by Area_group)
# ═══════════════════════════════════════════════════════
st.subheader("1. Diện tích và Giá bán")
st.markdown(
    """
**Biểu đồ tán xạ** thể hiện mối quan hệ giữa diện tích (`Area`, m²) và giá bán (`Price`, tỷ đồng).
Màu sắc phân biệt nhóm diện tích (`Area_group`). Đường xu hướng (OLS) cho thấy chiều hướng chung.
"""
)

fig1 = px.scatter(
    dff.sample(min(3000, len(dff)), random_state=42),
    x="Area",
    y="Price",
    color="Area_group",
    trendline="ols",
    trendline_scope="overall",
    opacity=0.5,
    labels={
        "Area": "Diện tích (m²)",
        "Price": "Giá (tỷ đồng)",
        "Area_group": "Nhóm diện tích",
    },
    title="Diện tích vs Giá bán",
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=450,
)
fig1.update_traces(marker=dict(size=5))
fig1.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    yaxis=dict(gridcolor="#E8E8E8"),
)
st.plotly_chart(fig1, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 2. BOX: Price_per_m2 theo Area_group
# ═══════════════════════════════════════════════════════
st.subheader("2. Đơn giá/m² theo Nhóm diện tích")
st.markdown(
    """
**Box plot** so sánh phân phối đơn giá (`Price_per_m2`, tỷ/m²) giữa các nhóm diện tích.
Giúp kiểm tra nghịch lý: nhà nhỏ nhưng đắt hơn/m² so với nhà lớn?
Mỗi hộp thể hiện Q1, Median, Q3; các điểm ngoài là outlier.
"""
)

area_order = ["<50m²", "50-80m²", "80-120m²", "120-200m²", ">200m²"]
area_order_present = [a for a in area_order if a in dff["Area_group"].values]

fig2 = px.box(
    dff,
    x="Area_group",
    y="Price_per_m2",
    category_orders={"Area_group": area_order_present},
    color="Area_group",
    points=False,
    labels={"Area_group": "Nhóm diện tích", "Price_per_m2": "Đơn giá (tỷ/m²)"},
    title="Phân phối đơn giá/m² theo nhóm diện tích",
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=400,
)
fig2.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    showlegend=False,
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    yaxis=dict(gridcolor="#E8E8E8"),
)
st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 3. BAR: Giá trung bình theo số tầng & số phòng ngủ (subplots)
# ═══════════════════════════════════════════════════════
st.subheader("3. Số tầng và Số phòng ngủ ảnh hưởng đến giá")
st.markdown(
    """
**Biểu đồ cột** thể hiện giá bán trung bình (`Price`) theo số tầng (`Floors`) và số phòng ngủ (`Bedrooms`).
Thanh màu cam là Median giá; thanh màu xanh là Mean giá — so sánh 2 chỉ số giúp nhận biết mức độ lệch phân phối.
"""
)

col_left, col_right = st.columns(2)

for col_widget, col_var, label in [
    (col_left, "Floors", "Số tầng"),
    (col_right, "Bedrooms", "Số phòng ngủ"),
]:
    grp = (
        dff.groupby(col_var)["Price"]
        .agg(Mean="mean", Median="median", Count="count")
        .reset_index()
        .query("Count >= 10")
        .sort_values(col_var)
    )
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=grp[col_var].astype(str),
            y=grp["Mean"],
            name="Mean",
            marker_color="#2E86AB",
            opacity=0.85,
        )
    )
    fig.add_trace(
        go.Bar(
            x=grp[col_var].astype(str),
            y=grp["Median"],
            name="Median",
            marker_color="#F28E2B",
            opacity=0.85,
        )
    )
    fig.update_layout(
        title=f"Giá TB theo {label}",
        xaxis_title=label,
        yaxis_title="Giá (tỷ đồng)",
        barmode="group",
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#2C3E50"),
        xaxis=dict(gridcolor="#E8E8E8"),
        yaxis=dict(gridcolor="#E8E8E8"),
        legend=dict(orientation="h", y=1.1),
        height=380,
    )
    col_widget.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 4. SCATTER: Frontage vs Price_per_m2
# ═══════════════════════════════════════════════════════
st.subheader("4. Mặt tiền và Đơn giá")
st.markdown(
    """
**Biểu đồ tán xạ** thể hiện mối quan hệ giữa chiều rộng mặt tiền (`Frontage`, mét)
và đơn giá/m² (`Price_per_m2`). Kích thước điểm tỷ lệ với diện tích (`Area`).
Mặt tiền rộng hơn có thực sự đẩy đơn giá lên không?
"""
)

fig4 = px.scatter(
    dff[dff["Frontage"].notna()].sample(min(3000, len(dff)), random_state=1),
    x="Frontage",
    y="Price_per_m2",
    size="Area",
    size_max=18,
    color="Area_group",
    trendline="ols",
    trendline_scope="overall",
    opacity=0.55,
    labels={
        "Frontage": "Mặt tiền (m)",
        "Price_per_m2": "Đơn giá (tỷ/m²)",
        "Area_group": "Nhóm diện tích",
        "Area": "Diện tích (m²)",
    },
    title="Mặt tiền vs Đơn giá/m²",
    color_discrete_sequence=px.colors.qualitative.Set2,
    height=430,
)
fig4.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    yaxis=dict(gridcolor="#E8E8E8"),
)
st.plotly_chart(fig4, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 5. HEATMAP: Tương quan giữa các biến số
# ═══════════════════════════════════════════════════════
st.subheader("5. Ma trận tương quan")
st.markdown(
    """
**Heatmap tương quan Pearson** giữa các biến số liên tục.
Giá trị từ **-1** (tương quan nghịch hoàn toàn) đến **+1** (tương quan thuận hoàn toàn).
Ô màu đậm = tương quan mạnh; ô trắng = gần như không tương quan.
"""
)

corr_cols = [
    "Area",
    "Frontage",
    "Access Road",
    "Floors",
    "Bedrooms",
    "Bathrooms",
    "Price",
    "Price_per_m2",
]
corr_cols_present = [c for c in corr_cols if c in dff.columns]
corr_matrix = dff[corr_cols_present].corr().round(2)

fig5 = px.imshow(
    corr_matrix,
    text_auto=True,
    color_continuous_scale="RdBu_r",
    zmin=-1,
    zmax=1,
    aspect="auto",
    title="Ma trận tương quan — Các biến số liên tục",
    labels={"color": "Pearson r"},
    height=450,
)
fig5.update_layout(
    font=dict(color="#2C3E50"),
    paper_bgcolor="#FFFFFF",
)
st.plotly_chart(fig5, use_container_width=True)

# ═══════════════════════════════════════════════════════
# 6. VIOLIN: Price theo Bathrooms
# ═══════════════════════════════════════════════════════
st.subheader("6. Phân phối giá theo số phòng tắm")
st.markdown(
    """
**Violin plot** thể hiện phân phối đầy đủ của giá bán (`Price`) theo số phòng tắm (`Bathrooms`).
Chiều rộng violin thể hiện mật độ dữ liệu; điểm trắng bên trong là Median.
Chỉ hiển thị các nhóm có ≥ 30 quan sát.
"""
)

bath_counts = dff["Bathrooms"].value_counts()
valid_baths = bath_counts[bath_counts >= 30].index
dff_bath = dff[dff["Bathrooms"].isin(valid_baths)].copy()
dff_bath["Bathrooms"] = dff_bath["Bathrooms"].astype(str)

fig6 = px.violin(
    dff_bath,
    x="Bathrooms",
    y="Price",
    color="Bathrooms",
    box=True,
    points=False,
    category_orders={
        "Bathrooms": sorted(dff_bath["Bathrooms"].unique(), key=lambda x: float(x))
    },
    labels={"Bathrooms": "Số phòng tắm", "Price": "Giá (tỷ đồng)"},
    title="Phân phối giá theo số phòng tắm",
    color_discrete_sequence=px.colors.qualitative.Pastel,
    height=430,
)
fig6.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    showlegend=False,
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    yaxis=dict(gridcolor="#E8E8E8"),
)
st.plotly_chart(fig6, use_container_width=True)
