"""
pages/22127418.py
==========================================================
PHÂN CÔNG: 22127418

Obj 1: Phân tích phân bố doanh thu theo tỉnh/thành
        -> Xác định 3 khu vực địa lý dẫn đầu và đặc trưng riêng
Obj 2: So sánh hiệu quả bán hàng giữa Shopee Mall và Non-Mall
        -> So sánh sold, rating, revenue_est giữa 2 nhóm

TEST ĐỘC LẬP:
    streamlit run app.py   # Mở trang này từ sidebar
"""

import sys, pathlib

sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.helpers import (
    load_data,
    get_active,
    setup_sidebar,
    CB_ORANGE,
    CB_SKYBLUE,
    CB_GREEN,
    CB_BLUE,
    CB_VERMIL,
    CB_GRAY,
    SEQ_BLUES,
)

st.set_page_config(page_title="22127418", page_icon="", layout="wide")
setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

# ====== Header ======
st.title("22127418")
st.caption("products.csv · 20,658 sản phẩm · crawl 18/3/2026")

# ====== Mục tiêu SMART ======
with st.expander("Mục tiêu SMART", expanded=True):
    st.markdown(
        """
    **Obj 1 — Phân bố doanh thu theo địa lý:**
    > Phân tích phân bố doanh thu ước tính theo tỉnh/thành từ snapshot 18/3/2026
    > để xác định **3 khu vực địa lý dẫn đầu** và mô tả đặc trưng riêng của từng khu vực
    > (mức giá trung bình, tỷ lệ Shopee Mall, số sản phẩm) —
    > giúp người bán biết nên đặt kho/vận chuyển ở đâu để tối ưu phủ sóng và tốc độ giao hàng.

    **Obj 2 — Shopee Mall vs Non-Mall:**
    > So sánh các chỉ số `sold`, `rating` và `revenue_est` giữa 8,298 sản phẩm Mall
    > và 12,360 sản phẩm Non-Mall từ dữ liệu 18/3/2026 —
    > để đánh giá lợi thế thực sự của nhãn hàng chính hãng và đề xuất chiến lược
    > phù hợp cho từng nhóm người bán.
    """
    )

st.markdown("---")

# ─── PREPROCESSING ──────────────────────────────────────────────────────────────
products["revenue_est"] = products["revenue_est"].fillna(0).astype(float)
products["sold"] = products["sold"].fillna(0).astype(float)
products["rating"] = products["rating"].fillna(0).astype(float)


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

# Màu sắc (lấy từ helpers; fallback nếu helpers chưa định nghĩa đủ)
_ACCENT = CB_VERMIL  # màu nhấn chính  (tương đương PINK trước đây)
_SECOND = CB_BLUE  # màu phụ         (tương đương BLUE trước đây)
_THIRD = CB_GREEN  # màu thứ ba      (tương đương TEAL trước đây)
_MUTED = CB_GRAY  # màu mờ cho cột không nổi bật


# ======================
# OBJ 1 — PHÂN BỐ ĐỊA LÝ
# ======================
st.markdown("## Obj 1 — Phân bố doanh thu theo tỉnh/thành")

# ── Chuẩn bị dữ liệu ──
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

geo_top15 = geo_df.nlargest(15, "revenue_est").copy()
geo_top3 = geo_df.nlargest(3, "revenue_est").copy()
top3_names = geo_top3["location_norm"].tolist()

colors_bar = [_ACCENT if i < 3 else _MUTED for i in range(len(geo_top15))]

# ====== Biểu đồ 1a: Bar chart ngang — Top 15 tỉnh/thành ======
st.subheader("Biểu đồ 1a: Bar chart — Top 15 tỉnh/thành theo doanh thu ước tính")

fig1 = go.Figure()
fig1.add_trace(
    go.Bar(
        x=geo_top15["revenue_B"],
        y=geo_top15["location_norm"],
        orientation="h",
        marker=dict(color=colors_bar, line=dict(color="rgba(0,0,0,0.1)", width=1)),
        text=[f"{v:.1f} tỷ" for v in geo_top15["revenue_B"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>Doanh thu: %{x:.2f} tỷ VNĐ<extra></extra>",
    )
)
fig1.update_layout(
    plot_bgcolor="white",
    height=480,
    xaxis=dict(
        title="Doanh thu ước tính (tỷ VNĐ)",
        gridcolor="#eeeeee",
    ),
    yaxis=dict(autorange="reversed"),
    margin=dict(l=20, r=80, t=20, b=40),
    showlegend=False,
    annotations=[
        dict(
            x=geo_top15["revenue_B"].max() * 0.6,
            y=0,
            text="⭐ Top 3",
            showarrow=False,
            font=dict(color=_ACCENT, size=11),
            xanchor="left",
        )
    ],
)
st.plotly_chart(fig1, use_container_width=True)
st.caption(
    "Cột màu đỏ cam = 3 khu vực dẫn đầu; cột xám = các tỉnh còn lại. "
    "Trục x là tổng doanh thu ước tính (tỷ VNĐ) của tất cả sản phẩm có shop đặt tại tỉnh đó."
)

st.markdown(
    f"""
**Nhận xét Biểu đồ 1a:**
- Ba khu vực dẫn đầu là **{top3_names[0]}**, **{top3_names[1] if len(top3_names) > 1 else 'N/A'}**
  và **{top3_names[2] if len(top3_names) > 2 else 'N/A'}** — chiếm tỷ trọng áp đảo so với
  phần còn lại của thị trường.
- Khoảng cách giữa vị trí #1 và #2 rất lớn, cho thấy sự tập trung mạnh tại một đô thị trung tâm.
- Sự phân phối doanh thu theo địa lý phản ánh mật độ người dùng và hạ tầng logistics tại các
  đô thị lớn phía Nam và phía Bắc.
"""
)

# ====== Biểu đồ 1b: Bubble chart — Số sản phẩm vs Doanh thu ======
st.subheader("Biểu đồ 1b: Bubble chart — Số sản phẩm × Doanh thu × Tỷ lệ Mall")

geo_plot = geo_df[geo_df["product_count"] >= 50].copy()

fig2 = px.scatter(
    geo_plot,
    x="product_count",
    y="revenue_B",
    size="sold",
    color="mall_ratio",
    hover_name="location_norm",
    color_continuous_scale="RdYlGn",
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
    marker=dict(opacity=0.8, line=dict(color="white", width=0.5)),
    hovertemplate=(
        "<b>%{hovertext}</b><br>"
        "Sản phẩm: %{x:,}<br>"
        "Doanh thu: %{y:.2f} tỷ<br>"
        "Lượt bán: %{marker.size:,}<br>"
        "Tỷ lệ Mall: %{marker.color:.1f}%<extra></extra>"
    ),
)
fig2.update_layout(
    plot_bgcolor="white",
    height=420,
    xaxis=dict(gridcolor="#eeeeee", title="Số lượng sản phẩm"),
    yaxis=dict(gridcolor="#eeeeee", title="Doanh thu (tỷ VNĐ)"),
    margin=dict(l=20, r=20, t=20, b=40),
)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    "Kích thước bong bóng = tổng lượt bán. "
    "Màu sắc = tỷ lệ sản phẩm Mall (xanh lá = Mall nhiều, đỏ = Mall ít). "
    "Chỉ hiển thị tỉnh/thành có ≥ 50 sản phẩm."
)

st.markdown(
    """
**Nhận xét Biểu đồ 1b:**
- Các tỉnh/thành có **nhiều sản phẩm nhất không nhất thiết** có doanh thu cao nhất —
  **tỷ lệ Mall cao** tương quan rõ hơn với doanh thu.
- Bong bóng lớn (lượt bán nhiều) tập trung ở nhóm dẫn đầu, cho thấy hiệu ứng
  "người thắng lấy tất" trong phân phối địa lý.
- Một số tỉnh có số sản phẩm thấp nhưng tỷ lệ Mall cao → doanh thu trên mỗi sản phẩm
  cao hơn trung bình, gợi ý phân khúc cao cấp hơn.
"""
)

# ====== Biểu đồ 1c: Radar chart — Hồ sơ 3 khu vực dẫn đầu ======
st.subheader("Biểu đồ 1c: Radar chart — Hồ sơ đặc trưng 3 khu vực dẫn đầu")


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
palette = [_ACCENT, _SECOND, _THIRD]
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
            opacity=0.75,
        )
    )

fig3.update_layout(
    polar=dict(
        bgcolor="#f8f8f8",
        radialaxis=dict(visible=True, range=[0, 1], color="#999"),
        angularaxis=dict(color="#555"),
    ),
    plot_bgcolor="white",
    height=420,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.1),
    margin=dict(l=60, r=60, t=30, b=60),
)
st.plotly_chart(fig3, use_container_width=True)
st.caption(
    "Giá trị mỗi trục đã được chuẩn hóa min-max (0–1) so với toàn bộ tỉnh/thành. "
    "Diện tích polygon lớn hơn = hồ sơ mạnh hơn toàn diện."
)

st.markdown(
    f"""
**Nhận xét Biểu đồ 1c:**
- **{top3_names[0]}** nổi bật với hồ sơ cân bằng — dẫn đầu cả doanh thu, lượt bán và số sản phẩm.
- **{top3_names[1] if len(top3_names) > 1 else ''}** có tỷ lệ Mall tương đối cao hơn,
  gợi ý thị trường cao cấp và giá bán trung bình lớn hơn.
- Mỗi khu vực thể hiện một chiến lược khác nhau: volume cao, mall-heavy, hoặc giá thấp
  kèm lượt bán lớn.
"""
)

# ====== Kết luận Obj 1 ======
revenue_total = geo_df["revenue_est"].sum()
top3_revenue = geo_top3["revenue_est"].sum()
top3_pct = top3_revenue / revenue_total * 100 if revenue_total > 0 else 0

st.markdown("### Kết luận — Obj 1")
st.success(
    f"""
Ba khu vực dẫn đầu là **{top3_names[0]}**, **{top3_names[1] if len(top3_names) > 1 else 'N/A'}**
và **{top3_names[2] if len(top3_names) > 2 else 'N/A'}**,
chiếm khoảng **{top3_pct:.1f}%** tổng doanh thu ước tính toàn thị trường.

Doanh thu mỹ phẩm trên Shopee tập trung cao độ tại hai đầu tàu kinh tế Bắc–Nam, phản ánh
mật độ người dùng và hạ tầng giao nhận vượt trội. Các tỉnh thành còn lại đóng góp phần nhỏ
nhưng có tiềm năng tăng trưởng nếu được đầu tư logistics và chính sách giá phù hợp.
**Người bán mới nên ưu tiên đặt kho/vận chuyển tại {top3_names[0]}** để tối ưu tốc độ giao hàng
và phủ sóng khách hàng.
"""
)

st.markdown("---")

# ======================
# OBJ 2 — MALL vs NON-MALL
# ======================
st.markdown("## Obj 2 — So sánh Shopee Mall vs Non-Mall")

mall = products[products["is_mall"] == 1].copy()
nonmall = products[products["is_mall"] == 0].copy()

n_mall = len(mall)
n_nonmall = len(nonmall)
mall_rev = mall["revenue_est"].sum() / 1e9
non_rev = nonmall["revenue_est"].sum() / 1e9

# KPI metrics (dùng st.metric thay custom HTML)
c1, c2, c3, c4 = st.columns(4)
c1.metric("Sản phẩm Mall", f"{n_mall:,}")
c2.metric("Sản phẩm Non-Mall", f"{n_nonmall:,}")
c3.metric("Doanh thu Mall", f"{mall_rev:.0f} tỷ VNĐ")
c4.metric("Doanh thu Non-Mall", f"{non_rev:.0f} tỷ VNĐ")

# ====== Biểu đồ 2a: Box plot — Phân phối 3 chỉ số ======
st.subheader(
    "Biểu đồ 2a: Box plot — Phân phối Sold / Rating / Revenue theo loại cửa hàng"
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

for col_idx, (col_name, scale) in enumerate(
    [("sold", 1), ("rating", 1), ("revenue_est", 1e6)], start=1
):
    for grp, color, name in [(mall, _ACCENT, "Mall"), (nonmall, _SECOND, "Non-Mall")]:
        data_vals = grp[col_name].dropna()
        if scale != 1:
            data_vals = data_vals / scale
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
    plot_bgcolor="white",
    height=400,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.15),
    margin=dict(l=20, r=20, t=50, b=60),
)
for i in range(1, 4):
    fig4.update_xaxes(showgrid=False, row=1, col=i)
    fig4.update_yaxes(gridcolor="#eeeeee", row=1, col=i)

st.plotly_chart(fig4, use_container_width=True)
st.caption(
    "Hộp = IQR (25th–75th percentile); đường ngang = median; dấu X = mean ± SD. "
    "Đã cắt tại percentile 99 để tránh outlier cực đoan làm méo biểu đồ."
)

st.markdown(
    """
**Nhận xét Biểu đồ 2a:**
- **Sold:** Sản phẩm Mall có trung vị sold cao hơn rõ rệt; phân phối rải rộng hơn do một số
  sản phẩm Mall có lượt bán cực lớn.
- **Rating:** Hai nhóm gần như ngang nhau — chất lượng cảm nhận của khách hàng không chênh
  lệch đáng kể chỉ vì nhãn Mall.
- **Revenue:** Mall dẫn đầu rõ ràng; median của Mall cao hơn trung bình toàn bộ Non-Mall.
"""
)

# ====== Biểu đồ 2b: Grouped bar — Trung bình 5 chỉ số ======
st.subheader("Biểu đồ 2b: Grouped bar — Trung bình các chỉ số Mall vs Non-Mall")

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
        marker_color=_ACCENT,
        text=[f"{v:.1f}" for v in cmp_df["Mall"]],
        textposition="outside",
    )
)
fig5.add_trace(
    go.Bar(
        name="Non-Mall",
        x=cmp_df["Chỉ số"],
        y=cmp_df["Non-Mall"],
        marker_color=_SECOND,
        text=[f"{v:.1f}" for v in cmp_df["Non-Mall"]],
        textposition="outside",
    )
)
fig5.update_layout(
    barmode="group",
    plot_bgcolor="white",
    height=380,
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
    yaxis=dict(gridcolor="#eeeeee"),
    margin=dict(l=20, r=20, t=30, b=50),
    bargap=0.25,
    bargroupgap=0.05,
)
st.plotly_chart(fig5, use_container_width=True)
st.caption(
    "Mỗi cặp cột thể hiện trung bình một chỉ số cho Mall (đỏ cam) và Non-Mall (xanh dương)."
)

st.markdown(
    """
**Nhận xét Biểu đồ 2b:**
- Mall có **lượt bán trung bình và doanh thu ước tính cao hơn** hẳn.
- Mall cũng có **giá trung bình cao hơn** và thường áp dụng **discount cao hơn**
  (chiến lược giảm giá sâu để tạo khối lượng bán).
- Non-Mall cạnh tranh ở phân khúc giá thấp hơn với biên lợi nhuận mỏng hơn,
  nhưng bù lại bằng số lượng sản phẩm đa dạng.
"""
)

# ====== Biểu đồ 2c: Donut chart — Thị phần doanh thu & lượt bán ======
st.subheader("Biểu đồ 2c: Donut chart — Thị phần doanh thu và lượt bán")

col_a, col_b = st.columns(2)

with col_a:
    labels = ["Shopee Mall", "Non-Mall"]
    rev_vals = [mall["revenue_est"].sum(), nonmall["revenue_est"].sum()]
    fig6a = go.Figure(
        go.Pie(
            labels=labels,
            values=rev_vals,
            hole=0.55,
            marker=dict(colors=[_ACCENT, _SECOND]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,.0f} VNĐ<extra></extra>",
        )
    )
    fig6a.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Doanh thu ước tính", x=0.5),
        annotations=[
            dict(
                text=f"{mall['revenue_est'].sum()/1e9:.0f}B<br>Mall",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=_ACCENT),
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
            marker=dict(colors=[_ACCENT, _SECOND]),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value:,} lượt<extra></extra>",
        )
    )
    fig6b.update_layout(
        height=300,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        title=dict(text="Tổng lượt bán", x=0.5),
        annotations=[
            dict(
                text=f"{mall['sold'].sum()/1e6:.1f}M<br>Mall",
                x=0.5,
                y=0.5,
                font=dict(size=14, color=_ACCENT),
                showarrow=False,
            )
        ],
    )
    st.plotly_chart(fig6b, use_container_width=True)

mall_rev_pct = mall["revenue_est"].sum() / products["revenue_est"].sum() * 100
mall_sold_pct = mall["sold"].sum() / products["sold"].sum() * 100
mall_prod_pct = n_mall / (n_mall + n_nonmall) * 100

st.caption(
    f"Mall chiếm {mall_prod_pct:.1f}% số sản phẩm nhưng đóng góp {mall_rev_pct:.1f}% "
    f"doanh thu và {mall_sold_pct:.1f}% tổng lượt bán."
)

st.markdown(
    f"""
**Nhận xét Biểu đồ 2c:**
- Mặc dù Mall chỉ chiếm **{mall_prod_pct:.1f}%** số lượng sản phẩm,
  nhưng đóng góp đến **{mall_rev_pct:.1f}%** doanh thu và **{mall_sold_pct:.1f}%** tổng lượt bán.
- Điều này cho thấy nhãn Shopee Mall mang lại hiệu ứng uy tín (*trust signal*) rõ rệt,
  giúp tỷ lệ chuyển đổi cao hơn đáng kể so với cửa hàng thông thường.
"""
)

# ====== Biểu đồ 2d: Violin plot — Phân phối rating ======
st.subheader("Biểu đồ 2d: Violin plot — Phân phối điểm đánh giá Mall vs Non-Mall")

products_plot = products[products["rating"] > 0].copy()
products_plot["Loại"] = products_plot["is_mall"].map({1: "Shopee Mall", 0: "Non-Mall"})

fig7 = go.Figure()
for grp, color in [("Shopee Mall", _ACCENT), ("Non-Mall", _SECOND)]:
    data_r = products_plot[products_plot["Loại"] == grp]["rating"]
    fig7.add_trace(
        go.Violin(
            y=data_r,
            name=grp,
            box_visible=True,
            meanline_visible=True,
            fillcolor=color,
            line_color=color,
            opacity=0.7,
            points=False,
        )
    )

fig7.update_layout(
    plot_bgcolor="white",
    height=360,
    yaxis=dict(
        title="Điểm đánh giá",
        range=[3, 5.1],
        gridcolor="#eeeeee",
    ),
    legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.12),
    margin=dict(l=20, r=20, t=20, b=50),
    violingap=0.3,
    violinmode="group",
)
st.plotly_chart(fig7, use_container_width=True)
st.caption(
    "Chiều rộng violin = mật độ phân phối; hộp bên trong = IQR; đường ngang = median; "
    "đường đứt = mean. Trục Y giới hạn từ 3.0 để tập trung vào vùng có dữ liệu chính."
)

mall_rating = mall[mall["rating"] > 0]["rating"].mean()
nonmall_rating = nonmall[nonmall["rating"] > 0]["rating"].mean()

st.markdown(
    f"""
**Nhận xét Biểu đồ 2d:**
- Phân phối rating của Mall ({mall_rating:.2f}★) và Non-Mall ({nonmall_rating:.2f}★) rất gần nhau
  và tập trung cao ở vùng **4.5–5.0★**.
- Mall không có lợi thế rõ ràng về rating — khách hàng đánh giá dựa trên trải nghiệm thực tế,
  không phải nhãn hiệu cửa hàng.
- Xu hướng rating tích cực ở cả hai nhóm có thể do: khách hàng chỉ để lại đánh giá khi
  hài lòng, hoặc hệ thống khuyến khích đánh giá 5 sao bằng xu/voucher.
"""
)

# ====== Kết luận Obj 2 ======
st.markdown("### Kết luận — Obj 2")
st.success(
    f"""
Shopee Mall thể hiện ưu thế rõ ràng về **doanh thu và lượt bán**: chỉ với {mall_prod_pct:.1f}%
số sản phẩm nhưng tạo ra {mall_rev_pct:.1f}% tổng doanh thu — cho thấy nhãn Mall là một
*trust signal* mạnh trong hành vi mua sắm mỹ phẩm. Tuy nhiên, về **điểm đánh giá**, hai nhóm
gần như ngang nhau ({mall_rating:.2f}★ vs {nonmall_rating:.2f}★), cho thấy Non-Mall vẫn có thể
cạnh tranh về chất lượng dịch vụ.

**Chiến lược đề xuất:**
- **Non-Mall:** Tập trung tích lũy đánh giá tích cực, cải thiện tỷ lệ phản hồi, cân nhắc
  đăng ký Shopee Mall khi đủ điều kiện để hưởng lợi từ hiệu ứng uy tín thương hiệu.
- **Mọi nhóm:** Chiến lược giảm giá sâu kết hợp free shipping là đòn bẩy hiệu quả để
  tăng volume bán hàng, đặc biệt trong các đợt sale lớn.
"""
)
