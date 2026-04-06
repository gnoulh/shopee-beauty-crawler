"""
pages/05_Geo_Mall.py — Địa lý & So sánh Shopee Mall vs Non-Mall
================================================================
PHÂN CÔNG: 22127418
  MT6 — Phân tích phân bố doanh thu theo tỉnh/thành (top 3 khu vực)
  MT4 — So sánh Mall vs Non-Mall: box plot, grouped bar, donut, violin
================================================================
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    inject_css, member_badge, conclusion_box,
    load_data, get_active, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_GRAY, SEQ_BLUES,
)

st.set_page_config(page_title="05 – Địa lý & Mall/NonMall", layout="wide")
inject_css(); setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

st.title("Địa lý & So sánh Shopee Mall vs Non-Mall")
st.caption("products.csv - 20,658 sản phẩm - crawl 18/3/2026 - MSSV: 22127418")
member_badge("22127418", "MT4 & 6")

with st.expander("Mục tiêu SMART (MT6 & 4)", expanded=False):
    st.markdown("""
    **MT6:** Phân tích phân bố `revenue_est` theo `shop_location` để xác định
    **3 khu vực dẫn đầu** và mô tả đặc trưng riêng (mức giá TB, tỷ lệ Mall, số SP) —
    giúp người bán biết nên đặt kho/vận chuyển ở đâu để tối ưu phủ sóng.

    **MT4:** So sánh định lượng `sold`, `rating`, `revenue_est` giữa **8,298 SP Mall**
    và **12,360 SP Non-Mall** — đánh giá lợi thế thực sự của nhãn chính hãng.
    """)

st.markdown("---")

# === Preprocessing ===
for col in ["revenue_est","sold","rating"]:
    products[col] = pd.to_numeric(products.get(col), errors="coerce").fillna(0)

def normalize_location(loc):
    if pd.isna(loc): return "Không xác định"
    loc = str(loc).strip()
    for k, v in [("Hồ Chí Minh","Hồ Chí Minh"),("HCM","Hồ Chí Minh"),("Ho Chi Minh","Hồ Chí Minh"),
                 ("Hà Nội","Hà Nội"),("Ha Noi","Hà Nội"),("Đà Nẵng","Đà Nẵng"),
                 ("Da Nang","Đà Nẵng"),("Bình Dương","Bình Dương"),("Hải Phòng","Hải Phòng")]:
        if k.lower() in loc.lower(): return v
    return loc

products["location_norm"] = products["shop_location"].apply(normalize_location)

_ACCENT = CB_VERMIL; _SECOND = CB_BLUE; _MUTED = CB_GRAY

# ======
# MT6 — Địa lý
# ======
st.markdown("## Mục tiêu 6 — Phân bố Doanh thu theo Tỉnh/Thành")

geo_df = (
    products.groupby("location_norm")
    .agg(revenue_est=("revenue_est","sum"), sold=("sold","sum"),
         avg_price=("price","mean"),
         product_count=("item_id","count"),  
         mall_count=("is_mall","sum"), avg_rating=("rating","mean"))
    .reset_index()
)
geo_df["mall_ratio"] = geo_df["mall_count"] / geo_df["product_count"] * 100
geo_df["revenue_B"]  = geo_df["revenue_est"] / 1e9
geo_top15 = geo_df.nlargest(15, "revenue_est").copy()
geo_top3 = geo_df.nlargest(3, "revenue_est").copy()
top3_names = geo_top3["location_norm"].tolist()

# === Biểu đồ 1a: Bar chart ngang Top 15 ============================================================
st.subheader("Biểu đồ 1a: Bar chart — Top 15 tỉnh/thành theo doanh thu ước tính")

colors_bar = [_ACCENT if i < 3 else _MUTED for i in range(len(geo_top15))]
fig1 = go.Figure()
fig1.add_trace(go.Bar(
    x=geo_top15["revenue_B"], y=geo_top15["location_norm"], orientation="h",
    marker=dict(color=colors_bar, line=dict(color="rgba(0,0,0,0.1)",width=1)),
    text=[f"{v:.1f} tỷ" for v in geo_top15["revenue_B"]], textposition="outside",
    hovertemplate="<b>%{y}</b><br>Doanh thu: %{x:.2f} tỷ VNĐ<extra></extra>",
))
fig1.update_layout(plot_bgcolor="white", height=480,
                   xaxis=dict(title="Doanh thu ước tính (tỷ VNĐ)", gridcolor="#eeeeee"),
                   yaxis=dict(autorange="reversed"),
                   margin=dict(l=20,r=80,t=20,b=40), showlegend=False)
st.plotly_chart(fig1, use_container_width=True)
st.caption("Cột đỏ cam = 3 khu vực dẫn đầu. Trục X = tổng doanh thu ước tính (tỷ VNĐ) của "
           "tất cả sản phẩm có shop đặt tại tỉnh đó.")
st.markdown("""
**Nhận xét 1a:**
- Hồ Chí Minh áp đảo hoàn toàn — trung tâm thương mại điện tử lớn nhất, tập trung phần lớn
  shop mỹ phẩm chuyên nghiệp, Shopee Mall và thương hiệu lớn.
- Hà Nội xếp thứ hai nhưng khoảng cách với #1 rất lớn -> tập trung hóa thị trường mạnh ở phía Nam.
- Từ vị trí thứ 3 trở đi doanh thu giảm rất nhanh — long-tail distribution điển hình TMĐT.
""")

# === Biểu đồ 1b: Bubble chart ===========================================================================
st.subheader("Biểu đồ 1b: Bubble chart — Số sản phẩm x Doanh thu x Tỷ lệ Mall")

geo_plot = geo_df[geo_df["product_count"] >= 50].copy()
fig2 = px.scatter(geo_plot, x="product_count", y="revenue_B", size="sold",
                  color="mall_ratio", hover_name="location_norm",
                  color_continuous_scale="RdYlGn", size_max=60,
                  labels={"product_count":"Số lượng sản phẩm","revenue_B":"Doanh thu (tỷ VNĐ)",
                          "sold":"Tổng lượt bán","mall_ratio":"Tỷ lệ Mall (%)"},
                  hover_data={"avg_price":":.0f","avg_rating":":.2f"})
fig2.update_traces(marker=dict(opacity=0.8, line=dict(color="white",width=0.5)))
fig2.update_layout(plot_bgcolor="white", height=420,
                   xaxis=dict(gridcolor="#eeeeee",title="Số lượng sản phẩm"),
                   yaxis=dict(gridcolor="#eeeeee",title="Doanh thu (tỷ VNĐ)"),
                   margin=dict(l=20,r=20,t=20,b=40))
st.plotly_chart(fig2, use_container_width=True)
st.caption("Kích thước bong bóng = tổng lượt bán. Màu xanh lá = tỷ lệ Mall cao. "
           "Chỉ hiển thị tỉnh/thành >= 50 sản phẩm.")
st.markdown("""
**Nhận xét 1b:**
- Quan hệ số lượng SP -> doanh thu **không tuyến tính**: tỷ lệ Mall cao (xanh lá) mang lại
  doanh thu/SP cao hơn, dù tổng số SP ít hơn.
- Hồ Chí Minh nằm tách biệt hoàn toàn ở góc trên phải — duy nhất kết hợp cả 3 chiều mạnh.
""")

# === Biểu đồ 1c: Radar chart ===========================================================================
st.subheader("Biểu đồ 1c: Radar chart — Hồ sơ đặc trưng 3 khu vực dẫn đầu")

radar_cols = ["revenue_B","sold","avg_price","mall_ratio","avg_rating","product_count"]
radar_labels = ["Doanh thu","Lượt bán","Giá TB","Tỷ lệ Mall","Đánh giá TB","Số SP"]
radar_data = geo_df[geo_df["location_norm"].isin(top3_names)].copy().reset_index(drop=True)

# Thêm chuẩn hóa min-max trên toàn bộ geo_df, sau đó lọc top3
for c in radar_cols:
    mn, mx = geo_df[c].min(), geo_df[c].max()
    geo_df[c+"_norm"] = (geo_df[c]-mn)/(mx-mn) if mx > mn else 0.0
radar_data = geo_df[geo_df["location_norm"].isin(top3_names)].copy().reset_index(drop=True)

fig3 = go.Figure()
palette3 = [_ACCENT, _SECOND, CB_GREEN]
for i, row in radar_data.iterrows():
    vals = [row[c+"_norm"] for c in radar_cols] + [row[radar_cols[0]+"_norm"]]
    fig3.add_trace(go.Scatterpolar(
        r=vals, theta=radar_labels+[radar_labels[0]],
        fill="toself", name=row["location_norm"],
        line=dict(color=palette3[i % len(palette3)], width=2), opacity=0.75,
    ))
fig3.update_layout(
    polar=dict(bgcolor="#f8f8f8",
               radialaxis=dict(visible=True,range=[0,1],color="#999"),
               angularaxis=dict(color="#555")),
    plot_bgcolor="white", height=420,
    legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.1),
    margin=dict(l=60,r=60,t=30,b=60),
)
st.plotly_chart(fig3, use_container_width=True)
st.caption("Giá trị mỗi trục chuẩn hóa min-max (0–1) so với toàn bộ tỉnh/thành. "
           "Diện tích polygon lớn = hồ sơ mạnh hơn.")

top3_rev = geo_top3["revenue_est"].sum()
total_rev = geo_df["revenue_est"].sum()
top3_pct = top3_rev / total_rev * 100 if total_rev > 0 else 0

st.success(f"""
**Kết luận MT6:** Ba khu vực dẫn đầu — **{', '.join(top3_names)}** — chiếm ~**{top3_pct:.1f}%**
tổng doanh thu: mức tập trung địa lý cực cao (winner-takes-most).
Người bán mới nên đặt kho tại Hồ Chí Minh/Hà Nội để tối ưu thời gian giao hàng và hiển thị Shopee.
""")

st.markdown("---")

# ======
# MT4 — Mall vs Non-Mall
# ======
st.markdown("## Mục tiêu 4 — So sánh Shopee Mall vs Non-Mall")

mall = products[products["is_mall"] == 1].copy()
nonmall = products[products["is_mall"] == 0].copy()
n_mall, n_nonmall = len(mall), len(nonmall)
mall_rev, non_rev = mall["revenue_est"].sum()/1e9, nonmall["revenue_est"].sum()/1e9

c1,c2,c3,c4 = st.columns(4)
c1.metric("Sản phẩm Mall", f"{n_mall:,}")
c2.metric("Sản phẩm Non-Mall", f"{n_nonmall:,}")
c3.metric("DT Mall", f"{mall_rev:.0f} tỷ VNĐ")
c4.metric("DT Non-Mall", f"{non_rev:.0f} tỷ VNĐ")

# === Biểu đồ 2a: Box plot (3 chỉ số) ===============================================================
st.subheader("Biểu đồ 2a: Box plot — Phân phối Sold / Rating / Revenue")

fig4 = make_subplots(rows=1, cols=3,
                     subplot_titles=["Lượt bán (sold)","Đánh giá (rating)","DT ước tính (triệu VNĐ)"],
                     horizontal_spacing=0.08)

for col_i, (col_name, scale) in enumerate([("sold",1),("rating",1),("revenue_est",1e6)], 1):
    for grp, color, name in [(mall,_ACCENT,"Mall"),(nonmall,_SECOND,"Non-Mall")]:
        d = grp[col_name].dropna()
        if scale != 1: d = d / scale
        d = d[d <= d.quantile(0.99)]
        fig4.add_trace(go.Box(y=d, name=name, marker_color=color, boxmean="sd",
                              showlegend=(col_i==1), line=dict(width=1.5)), row=1, col=col_i)
fig4.update_layout(plot_bgcolor="white", height=400,
                   legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.15),
                   margin=dict(l=20,r=20,t=50,b=60))
for i in range(1,4):
    fig4.update_xaxes(showgrid=False, row=1, col=i)
    fig4.update_yaxes(gridcolor="#eeeeee", row=1, col=i)
st.plotly_chart(fig4, use_container_width=True)
st.caption("Hộp = IQR (25th–75th); đường ngang = median; hình thoi = mean ± SD. "
           "Đã cắt tại percentile 99 để tránh outlier làm méo biểu đồ.")
st.markdown("""
**Nhận xét 2a:**
- **Sold & Revenue:** Cả hai nhóm phân phối lệch phải mạnh — đại đa số sản phẩm bán ít,
  một thiểu số nhỏ tạo ra phần lớn giá trị. Non-Mall có outlier doanh thu cao hơn Mall —
  một số shop Non-Mall quy mô lớn đang cạnh tranh ngang ngửa Mall.
- **Rating:** Hai nhóm gần như không có sự khác biệt — median ~4.8–5.0 sao. Nhãn Mall không
  đảm bảo rating cao hơn.
""")

# === Biểu đồ 2b: Grouped bar — trung bình 5 chỉ số ==========================================
st.subheader("Biểu đồ 2b: Grouped bar — Trung bình các chỉ số Mall vs Non-Mall")

cmp_df = pd.DataFrame({
    "Chỉ số": ["Sold TB","Rating TB","Revenue TB (triệu)","Giá TB (nghìn)","Discount TB (%)"],
    "Mall":    [mall["sold"].mean(), mall["rating"].mean(),
                mall["revenue_est"].mean()/1e6, mall["price"].mean()/1e3,
                mall["discount_pct"].mean()],
    "Non-Mall":[nonmall["sold"].mean(), nonmall["rating"].mean(),
                nonmall["revenue_est"].mean()/1e6, nonmall["price"].mean()/1e3,
                nonmall["discount_pct"].mean()],
})
fig5 = go.Figure()
fig5.add_trace(go.Bar(name="Mall", x=cmp_df["Chỉ số"], y=cmp_df["Mall"],
                       marker_color=_ACCENT,
                       text=[f"{v:.1f}" for v in cmp_df["Mall"]], textposition="outside"))
fig5.add_trace(go.Bar(name="Non-Mall", x=cmp_df["Chỉ số"], y=cmp_df["Non-Mall"],
                       marker_color=_SECOND,
                       text=[f"{v:.1f}" for v in cmp_df["Non-Mall"]], textposition="outside"))
fig5.update_layout(barmode="group", plot_bgcolor="white", height=380,
                   legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.12),
                   yaxis=dict(gridcolor="#eeeeee"), margin=dict(l=20,r=20,t=30,b=50),
                   bargap=0.25, bargroupgap=0.05)
st.plotly_chart(fig5, use_container_width=True)
st.caption("Mỗi cặp cột = trung bình một chỉ số cho Mall (đỏ cam) và Non-Mall (xanh).")
st.markdown("""
**Nhận xét 2b (counterintuitive):**
- Non-Mall có sold TB, revenue TB, và giá TB cao hơn Mall — bị kéo lên bởi một nhóm nhỏ
  shop Non-Mall quy mô lớn (phân phối lệch phải -> mean không đại diện tốt).
- Discount gần như bằng nhau (~18.5%) -> cả hai nhóm cùng chiến lược kích cầu tiêu chuẩn.
""")

# === Biểu đồ 2c: Donut chart ===========================================================================
st.subheader("Biểu đồ 2c: Donut chart — Thị phần Doanh thu & Lượt bán")

col_a, col_b = st.columns(2)
with col_a:
    rev_vals = [mall["revenue_est"].sum(), nonmall["revenue_est"].sum()]
    fig6a = go.Figure(go.Pie(labels=["Shopee Mall","Non-Mall"], values=rev_vals, hole=0.55,
                             marker=dict(colors=[_ACCENT,_SECOND]),
                             textinfo="label+percent"))
    fig6a.update_layout(height=300, showlegend=False, margin=dict(l=10,r=10,t=30,b=10),
                        title=dict(text="Doanh thu ước tính",x=0.5))
    st.plotly_chart(fig6a, use_container_width=True)

with col_b:
    sold_vals = [mall["sold"].sum(), nonmall["sold"].sum()]
    fig6b = go.Figure(go.Pie(labels=["Shopee Mall","Non-Mall"], values=sold_vals, hole=0.55,
                             marker=dict(colors=[_ACCENT,_SECOND]),
                             textinfo="label+percent"))
    fig6b.update_layout(height=300, showlegend=False, margin=dict(l=10,r=10,t=30,b=10),
                        title=dict(text="Tổng lượt bán",x=0.5))
    st.plotly_chart(fig6b, use_container_width=True)

mall_rev_pct = mall["revenue_est"].sum() / products["revenue_est"].sum() * 100
mall_sold_pct = mall["sold"].sum() / products["sold"].sum() * 100
mall_prod_pct = n_mall / (n_mall+n_nonmall) * 100
st.caption(f"Mall chiếm {mall_prod_pct:.1f}% số sản phẩm nhưng chỉ {mall_rev_pct:.1f}% "
           f"doanh thu và {mall_sold_pct:.1f}% lượt bán.")

# === Biểu đồ 2d: Violin plot — Rating ============================================================
st.subheader("Biểu đồ 2d: Violin plot — Phân phối Rating Mall vs Non-Mall")

products_r = products[products["rating"] > 0].copy()
products_r["Loại"] = products_r["is_mall"].map({1:"Shopee Mall",0:"Non-Mall"})
fig7 = go.Figure()
for grp, color in [("Shopee Mall",_ACCENT),("Non-Mall",_SECOND)]:
    d = products_r[products_r["Loại"]==grp]["rating"]
    fig7.add_trace(go.Violin(y=d, name=grp, box_visible=True, meanline_visible=True,
                              fillcolor=color, line_color=color, opacity=0.7, points=False))
fig7.update_layout(plot_bgcolor="white", height=360,
                   yaxis=dict(title="Điểm đánh giá", range=[3,5.1], gridcolor="#eeeeee"),
                   legend=dict(orientation="h",x=0.5,xanchor="center",y=-0.12),
                   margin=dict(l=20,r=20,t=20,b=50), violingap=0.3, violinmode="group")
st.plotly_chart(fig7, use_container_width=True)
st.caption("Chiều rộng violin = mật độ phân phối. Hộp = IQR. Đường ngang = median. "
           "Đường đứt = mean. Trục Y giới hạn 3.0–5.1.")

mall_r = mall[mall["rating"]>0]["rating"].mean()
non_r  = nonmall[nonmall["rating"]>0]["rating"].mean()
st.success(f"""
**Kết luận MT4:**
Non-Mall chiếm {100-mall_rev_pct:.1f}% doanh thu và {100-mall_sold_pct:.1f}% lượt bán,
dù Mall chiếm {mall_prod_pct:.1f}% số sản phẩm — **đảo ngược hoàn toàn kỳ vọng ban đầu**.
Rating không khác biệt có ý nghĩa ({mall_r:.2f} sao vs {non_r:.2f}★, chênh 0.1 sao).

**Chiến lược:** Non-Mall không cần vội đăng ký Mall — ưu tiên chất lượng SP, giao hàng nhanh,
chăm sóc KH để tích lũy rating bền vững. Mức discount ~18.5% đã là chuẩn thị trường — không
nên chạy đua giảm giá thêm mà tập trung vào giá trị gia tăng (tặng kèm mẫu thử, tư vấn).
""")
