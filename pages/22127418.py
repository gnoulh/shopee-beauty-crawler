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
    > Phân tích phân bố doanh thu ước tính theo tỉnh/thành
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
_ACCENT = CB_VERMIL  # màu nhấn chính
_SECOND = CB_BLUE  # màu phụ
_THIRD = CB_GREEN  # màu thứ ba
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
    """
**Nhận xét Biểu đồ 1a:**
- **Hồ Chí Minh** áp đảo hoàn toàn với doanh thu ước tính vượt trội so với tất cả các tỉnh thành còn lại,
  phản ánh vai trò trung tâm thương mại điện tử lớn nhất cả nước — nơi tập trung phần lớn các shop mỹ phẩm
  chuyên nghiệp, Shopee Mall và các thương hiệu lớn.
- **Hà Nội** xếp thứ hai nhưng khoảng cách với vị trí #1 rất lớn, cho thấy thị trường mỹ phẩm online
  vẫn đang trong giai đoạn tập trung hóa mạnh tại đầu tàu phía Nam.
- Từ vị trí thứ 3 trở đi, doanh thu giảm rất nhanh và gần như đồng đều — đường cong phân phối dạng
  long-tail điển hình, phần lớn tỉnh thành đóng góp rất nhỏ vào tổng doanh thu chung.
- Các tỉnh miền Trung như Đà Nẵng, Thừa Thiên Huế xuất hiện trong top nhưng ở mức rất thấp,
  gợi ý tiềm năng thị trường chưa được khai thác tại khu vực này.
- Các tỉnh công nghiệp phía Nam như Bình Dương, Đồng Nai có mặt trong top 15 do mật độ dân số
  và thu nhập khả dụng tương đối cao, dù không phải trung tâm thương mại lớn.
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
- Biểu đồ cho thấy mối quan hệ **không tuyến tính** giữa số lượng sản phẩm và doanh thu:
  một số tỉnh có ít sản phẩm nhưng doanh thu cao bất ngờ — điều này thường gặp ở những nơi
  có tỷ lệ Mall cao, nơi mỗi sản phẩm mang giá trị và lượt bán lớn hơn nhiều.
- **Hồ Chí Minh** nằm hoàn toàn tách biệt ở góc trên phải — vừa nhiều sản phẩm nhất, vừa
  doanh thu cao nhất, với bong bóng lớn cho thấy lượt bán cũng dẫn đầu. Đây là điểm ngoại lệ
  duy nhất kết hợp cả ba chiều mạnh cùng lúc.
- **Hà Nội** có số sản phẩm đứng thứ hai nhưng doanh thu trên mỗi sản phẩm thấp hơn Hồ Chí Minh,
  gợi ý cơ cấu sản phẩm nghiêng về phân khúc giá vừa và thấp hơn.
- Các tỉnh có tỷ lệ Mall cao (màu xanh lá đậm) thường xuất hiện ở phần trên của trục Y dù
  số sản phẩm không nhiều — xác nhận rằng **chất lượng danh mục quan trọng hơn số lượng**
  trong việc tạo ra doanh thu.
- Nhóm tỉnh có bong bóng nhỏ và màu đỏ (nhiều Non-Mall, ít lượt bán) tập trung ở góc dưới trái —
  đây là thị trường phân tán, cạnh tranh chủ yếu bằng giá thấp và khó tạo ra doanh thu đột phá.
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
- **{top3_names[0]}** có diện tích polygon lớn nhất và cân bằng nhất trên cả 6 chiều —
  dẫn đầu đồng thời về doanh thu, lượt bán, số sản phẩm và tỷ lệ Mall, đây là thị trường
  trưởng thành nhất với hệ sinh thái người bán đa dạng và cạnh tranh cao.
- **{top3_names[1] if len(top3_names) > 1 else ''}** nổi bật ở chiều **Tỷ lệ Mall** và
  **Giá trung bình** — cho thấy thị trường này thiên về phân khúc cao cấp hơn, ít sản phẩm
  budget, người mua sẵn sàng chi nhiều hơn cho mỗi đơn hàng.
- **{top3_names[2] if len(top3_names) > 2 else ''}** có hồ sơ khiêm tốn hơn trên hầu hết
  các chiều nhưng vẫn lọt top 3 nhờ **khối lượng lượt bán** đáng kể — chiến lược cạnh tranh
  bằng giá thấp và số lượng giao dịch lớn thay vì doanh thu trên mỗi đơn.
- Sự khác biệt rõ ràng giữa 3 hồ sơ cho thấy mỗi khu vực có đặc thù riêng và người bán
  cần điều chỉnh chiến lược sản phẩm, định giá và khuyến mãi cho từng thị trường địa phương
  thay vì áp dụng một công thức chung.
- Chiều **Đánh giá trung bình** gần như bằng nhau giữa 3 khu vực — xác nhận rằng chất lượng
  dịch vụ và sản phẩm không phụ thuộc vào địa lý mà phụ thuộc vào từng người bán cụ thể.
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
và **{top3_names[2] if len(top3_names) > 2 else 'N/A'}**, chiếm khoảng **{top3_pct:.1f}%**
tổng doanh thu ước tính toàn thị trường mỹ phẩm Shopee — một mức độ tập trung cực kỳ cao
cho thấy thị trường đang ở giai đoạn winner-takes-most rõ rệt theo địa lý.

Sự tập trung này không phải ngẫu nhiên: Hồ Chí Minh và Hà Nội là nơi đặt trụ sở của phần lớn
các thương hiệu mỹ phẩm lớn, các nhà phân phối chính thức và Shopee Mall — những đơn vị tạo
ra doanh thu lớn nhất. Hạ tầng logistics phát triển ở hai đầu tàu này cũng giúp rút ngắn
thời gian giao hàng, một yếu tố tác động trực tiếp đến tỷ lệ chuyển đổi trong mua sắm online.

Các tỉnh thành bên ngoài top 3 đóng góp rất nhỏ và phân tán — nhưng đây không hẳn là thị trường
yếu, mà có thể là **thị trường chưa được khai thác**: người mua tại các tỉnh này vẫn mua mỹ phẩm
trên Shopee nhưng từ các shop đặt tại Hồ Chí Minh hoặc Hà Nội, nghĩa là doanh thu thực tế
của người tiêu dùng địa phương cao hơn nhiều so với những gì dữ liệu shop_location thể hiện.

**Khuyến nghị chiến lược:**
- Người bán mới nên **đặt kho và địa chỉ shop tại Hồ Chí Minh hoặc Hà Nội** để tối ưu
  thời gian giao hàng và tăng khả năng hiển thị trong kết quả tìm kiếm Shopee (thuật toán
  ưu tiên shop gần người mua).
- Người bán đang hoạt động tại tỉnh nhỏ nên cân nhắc hợp tác với đơn vị fulfillment tại
  hai đầu tàu để cạnh tranh về tốc độ giao hàng.
- Tỷ lệ Mall cao tại top 3 cho thấy đây là thị trường đã trưởng thành — người bán Non-Mall
  cần có USP rõ ràng (giá tốt hơn, sản phẩm ngách, dịch vụ tư vấn) để tồn tại bên cạnh
  các thương hiệu lớn.
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
    "Hộp = IQR (25th–75th percentile); đường ngang = median; hình thoi = mean ± SD. "
    "Đã cắt tại percentile 99 để tránh outlier cực đoan làm méo biểu đồ."
)

st.markdown(
    """
**Nhận xét Biểu đồ 2a:**
- **Sold:** Hộp IQR của Mall và Non-Mall đều nằm rất thấp, gần 0 — đại đa số sản phẩm ở cả
  hai nhóm có lượt bán khá thấp. Tuy nhiên Mall có phần đuôi trên (râu và outlier) kéo dài
  lên đến ~10k, trong khi Non-Mall có một outlier đơn lẻ lên đến ~20k. Mean của Mall
  (đường đứt nét) nằm cao hơn median rõ rệt, cho thấy một nhóm nhỏ sản phẩm Mall bán rất
  chạy đang kéo trung bình lên — phân phối lệch phải mạnh ở cả hai nhóm.
- **Rating:** Đây là chỉ số thú vị nhất trong ba chiều. Hộp IQR của cả Mall và Non-Mall
  đều nằm trong khoảng 4.5–5.0★, median gần sát 5★. Hình thoi độ lệch chuẩn của hai nhóm
  khá rộng và gần như bằng nhau, kéo xuống tận vùng 3.5–4.0★ — cho thấy dù phần lớn sản
  phẩm được đánh giá cao, vẫn có một bộ phận đáng kể ở cả hai nhóm nhận rating thấp.
  Quan trọng hơn, không có sự khác biệt rõ ràng nào giữa Mall và Non-Mall — nhãn Mall
  không đảm bảo rating cao hơn.
- **Revenue:** Hộp IQR của cả hai nhóm đều nằm rất thấp gần 0, nhưng Mall có râu trên
  kéo đến ~2,500 triệu trong khi Non-Mall kéo đến ~5,000 triệu — tức là outlier doanh thu
  cực lớn lại xuất hiện ở Non-Mall nhiều hơn, không phải Mall. Điều này cho thấy một số
  shop Non-Mall quy mô lớn đang tạo ra doanh thu khổng lồ, cạnh tranh ngang ngửa thậm chí
  vượt trội so với Mall về tổng doanh thu tuyệt đối.
- Nhìn chung cả ba biểu đồ đều thể hiện phân phối **lệch phải rất mạnh** (right-skewed) —
  phần lớn sản phẩm ở cả hai nhóm hoạt động ở mức khiêm tốn, trong khi một thiểu số nhỏ
  tạo ra phần lớn giá trị thị trường. Đây là đặc trưng điển hình của thị trường thương mại
  điện tử theo quy luật phân phối lũy thừa (power-law distribution).
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
- **Sold trung bình:** Ngược với kỳ vọng, **Non-Mall (1960.6) cao hơn Mall (1500.3)** —
  chênh lệch khoảng 30%. Điều này nhất quán với quan sát outlier từ biểu đồ 2a: một số
  shop Non-Mall quy mô lớn có lượt bán rất cao đang kéo trung bình của nhóm này lên.
  Mean bị ảnh hưởng mạnh bởi outlier, nên con số này không đại diện cho sản phẩm
  Non-Mall điển hình — median ở biểu đồ 2a cho thấy bức tranh chính xác hơn.
- **Rating trung bình:** Hai nhóm gần như bằng nhau — Mall 4.7★ và Non-Mall 4.6★,
  chênh lệch chỉ 0.1★. Xác nhận lại kết luận từ biểu đồ 2a: nhãn Mall không tạo ra
  lợi thế về chất lượng cảm nhận của khách hàng. Non-Mall hoàn toàn có thể duy trì
  rating ngang bằng Mall nếu đảm bảo chất lượng sản phẩm và dịch vụ.
- **Revenue trung bình (triệu VNĐ):** Tương tự sold, **Non-Mall (326.3 triệu) cao hơn
  Mall (167.7 triệu)** — gần gấp đôi. Kết hợp với sold trung bình cao hơn, điều này
  cho thấy nhóm Non-Mall đang bị kéo lên bởi một số shop có doanh thu rất lớn.Tuy nhiên trung bình không phản ánh đúng sản phẩm Non-Mall thông
  thường vì phân phối lệch phải rất mạnh.
- **Giá trung bình (nghìn VNĐ):** **Non-Mall (166.0 nghìn) cao hơn Mall (112.4 nghìn)**
  — đây là điểm bất ngờ. Có thể giải thích bởi: các shop Non-Mall lớn bán sản phẩm
  cao cấp nhập khẩu không qua kênh chính hãng, hoặc cơ cấu sản phẩm của Mall nghiêng
  về hàng tiêu dùng nhanh giá thấp (sữa rửa mặt, kem dưỡng phổ thông) nhiều hơn.
- **Discount trung bình (%):** Gần như bằng nhau — Mall 18.8% và Non-Mall 18.5%,
  chênh lệch không đáng kể. Cả hai nhóm áp dụng mức chiết khấu tương đương nhau
  như một công cụ kích cầu tiêu chuẩn trên nền tảng Shopee, không có sự khác biệt
  chiến lược rõ ràng giữa hai nhóm về mức giảm giá.
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
- **Về doanh thu:** Mall chỉ chiếm **{mall_rev_pct:.1f}%** (tương đương 1,391 tỷ VNĐ) —
  thấp hơn đáng kể so với tỷ lệ số sản phẩm ({mall_prod_pct:.1f}%). Điều này cho thấy
  doanh thu trung bình trên mỗi sản phẩm của Mall **thấp hơn** Non-Mall, nhất quán với
  kết quả từ biểu đồ 2b. Non-Mall chiếm tới **74.4%** tổng doanh thu — phần lớn đến từ
  một nhóm nhỏ shop Non-Mall quy mô lớn có doanh thu rất cao.
- **Về lượt bán:** Mall chiếm **{mall_sold_pct:.1f}%** (12.4 triệu lượt), thấp hơn tỷ lệ
  số sản phẩm ({mall_prod_pct:.1f}%). Non-Mall chiếm tới **66.1%** lượt bán — xác nhận
  rằng phần lớn giao dịch trên thị trường mỹ phẩm Shopee đang diễn ra ở kênh Non-Mall,
  không phải Mall.
- **So sánh hai vòng tròn:** Tỷ lệ Mall trong lượt bán (33.9%) cao hơn trong doanh thu
  (25.6%) — nghĩa là giá bán trung bình mỗi giao dịch của Mall **thấp hơn** Non-Mall.
  Điều này phản ánh cơ cấu sản phẩm Mall nghiêng về hàng tiêu dùng nhanh giá thấp
  (sữa rửa mặt, kem dưỡng phổ thông), trong khi một số shop Non-Mall lớn đang kinh doanh
  sản phẩm cao cấp với giá trị đơn hàng cao hơn.
- Kết quả này đảo ngược hoàn toàn nhận định ban đầu: **Non-Mall đang chiếm ưu thế áp đảo**
  trên thị trường mỹ phẩm Shopee cả về doanh thu lẫn lượt bán, bất chấp việc Mall được
  hưởng nhiều lợi thế về hiển thị và uy tín nền tảng.
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
- **Hình dạng violin:** Cả hai nhóm đều có hình dạng **"giọt nước ngược"** — phần thân
  phình rất rộng ở vùng gần 5★ và thu hẹp nhanh chóng thành đuôi mảnh kéo xuống 3★.
  Điều này cho thấy tuyệt đại đa số sản phẩm có rating tập trung rất cao ở mức 4.8–5.0★,
  trong khi rating thấp hơn rất hiếm gặp.
- **Mall (cam):** Phần thân violin rộng và tập trung gần sát 5★, đuôi kéo xuống đến 3★
  nhưng rất mảnh — cho thấy có tồn tại sản phẩm bị đánh giá thấp nhưng số lượng không
  đáng kể so với tổng thể. Mật độ tập trung ở đỉnh 5★ rất cao, phản ánh phần lớn sản
  phẩm Mall đang duy trì chất lượng dịch vụ tốt.
- **Non-Mall (xanh):** Hình dạng tương tự Mall nhưng phần thân hơi nhỏ hơn và đuôi
  kéo xuống đến 3★ cũng mảnh tương đương. Điểm khác biệt là vùng 4.5★ của Non-Mall
  có phần thắt rõ hơn — mật độ sản phẩm ở khoảng 4.5–4.8★ thưa hơn so với Mall,
  nghĩa là Non-Mall có xu hướng phân cực hơn: hoặc rất gần 5★ hoặc thấp hơn 4.5★.
- **So sánh tổng thể:** Hai violin gần như đối xứng và có hình dạng rất giống nhau —
  xác nhận rằng phân phối rating giữa Mall và Non-Mall **không có sự khác biệt có ý
  nghĩa**. Với mean {mall_rating:.2f}★ (Mall) và {nonmall_rating:.2f}★ (Non-Mall), chênh
  lệch 0.1★ hoàn toàn không đủ để kết luận Mall có chất lượng cảm nhận vượt trội hơn.
- Hiện tượng rating dồn về 5★ ở cả hai nhóm là đặc trưng điển hình của thương mại điện
  tử Việt Nam: Shopee tích cực khuyến khích người mua để lại đánh giá thông qua xu
  Shopee và voucher, dẫn đến hành vi đánh giá mang tính xã giao nhiều hơn là phản ánh
  trải nghiệm thực tế.
"""
)

# ====== Kết luận Obj 2 ======
st.markdown("### Kết luận — Obj 2")
st.success(
    f"""
Phân tích so sánh giữa Shopee Mall và Non-Mall trên ba chiều chính — sold, rating và revenue_est —
cho thấy một bức tranh **đảo ngược hoàn toàn so với kỳ vọng ban đầu**: Non-Mall đang chiếm
ưu thế áp đảo trên thị trường mỹ phẩm Shopee, không phải Mall.

**Về doanh thu và lượt bán:** Non-Mall chiếm tới {100-mall_rev_pct:.1f}% tổng doanh thu và
{100-mall_sold_pct:.1f}% tổng lượt bán, dù Mall chiếm {mall_prod_pct:.1f}% số sản phẩm.
Doanh thu trung bình trên mỗi sản phẩm của Non-Mall (326.3 triệu) cao gần gấp đôi Mall
(167.7 triệu), và giá bán trung bình của Non-Mall (166.0 nghìn) cũng cao hơn Mall (112.4 nghìn).
Điều này cho thấy thị trường mỹ phẩm Shopee không vận hành theo mô hình "Mall thống trị"
mà là một thị trường đa dạng, nơi các shop Non-Mall quy mô lớn đang tạo ra phần lớn giá trị.
Cần lưu ý rằng kết quả này bị ảnh hưởng bởi một nhóm nhỏ shop Non-Mall có doanh thu
cực lớn kéo trung bình lên — phân phối lệch phải mạnh khiến mean không đại diện cho
sản phẩm Non-Mall điển hình.

**Về đánh giá:** Không có sự khác biệt có ý nghĩa giữa hai nhóm ({mall_rating:.2f}★ vs
{nonmall_rating:.2f}★), chênh lệch chỉ 0.1★. Cả hai nhóm đều có violin tập trung dày đặc
ở vùng 4.8–5.0★ với đuôi mảnh kéo xuống 3★. Nhãn Mall không mang lại lợi thế về chất lượng
cảm nhận — người tiêu dùng đánh giá dựa trên trải nghiệm thực tế với sản phẩm và giao hàng,
không phải dựa trên loại cửa hàng.

**Chiến lược đề xuất:**
- **Người bán Non-Mall** không cần vội đăng ký Mall — dữ liệu cho thấy Non-Mall hoàn toàn
  có thể đạt doanh thu và rating ngang bằng hoặc vượt trội hơn Mall nếu xây dựng được
  quy mô và uy tín đủ lớn. Ưu tiên đầu tư vào chất lượng sản phẩm, tốc độ giao hàng
  và chăm sóc khách hàng sau bán để tích lũy đánh giá tích cực bền vững.
- **Người bán Mall** cần nhìn nhận lại lợi thế cạnh tranh thực sự: nhãn Mall không tự
  động đảm bảo doanh thu hay rating cao hơn. Lợi thế của Mall nằm ở uy tín thương hiệu
  và khả năng tiếp cận phân khúc khách hàng ưu tiên hàng chính hãng — cần khai thác
  đúng phân khúc này thay vì cạnh tranh giá với Non-Mall.
- **Về chiến lược giảm giá:** Cả hai nhóm áp dụng discount gần như bằng nhau (~18.5–18.8%),
  cho thấy đây đã là mức chuẩn thị trường. Người bán không nên chạy đua giảm giá thêm
  mà nên tập trung vào giá trị gia tăng: tặng kèm mẫu thử, hướng dẫn sử dụng, tư vấn
  chọn sản phẩm — đặc biệt quan trọng trong ngành mỹ phẩm nơi người mua thường không
  chắc chắn về sản phẩm phù hợp với da mình.
"""
)
