"""
pages/03_Market.py — Phân khúc Thị trường & Chiến lược Sản phẩm
================================================================
PHÂN CÔNG:
  MT3 — Phân khúc K-Means (price x sold, log-normalized, Silhouette)
        Đánh giá tỷ trọng doanh thu từng phân khúc
  MT6 — 23127488: Volume Driver vs Margin Driver theo phân khúc
================================================================
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    load_data, get_active, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_PURPLE, CB_GRAY,
)

st.set_page_config(page_title="03 – Phân khúc Thị trường", layout="wide")
setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

st.title("Phân khúc Thị trường & Chiến lược Sản phẩm")
st.caption("products.csv - 20,658 sản phẩm - crawl 18/3/2026 - MSSV: 23127488")

with st.expander("Mục tiêu SMART (MT3 & 6)", expanded=False):
    st.markdown("""
    **MT3:** Phân chia thị trường mỹ phẩm Shopee thành k phân khúc chiến lược dựa trên
    mối tương quan giữa `price` và `sold` (log-normalized, Silhouette score), đánh giá tỷ trọng
    `revenue_est` từng phân khúc — để người bán biết mình đang ở phân khúc nào và phân khúc nào
    chiếm phần lớn giá trị thị trường.

    **MT6:** Với từng phân khúc đã phân chia, xác định **top 3 Volume Driver** (sold cao nhất —
    chiến lược phễu) và **top 3 Margin Driver** (revenue_est cao, sold < trung vị phân khúc —
    chiến lược sinh lời), trực quan hóa bằng grouped bar chart.
    """)

st.markdown("---")

# === K-Means ===
@st.cache_data
def run_kmeans(active_df):
    df = active_df[["price","sold","revenue_est","sub_category","price_tier","is_mall",
                     "name"]].dropna().copy()
    X  = np.log1p(df[["price","sold"]])
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    sil, iner = {}, {}
    for k in range(2, 8):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(Xs)
        sil[k] = silhouette_score(Xs, lb); iner[k] = km.inertia_
    bk  = max(sil, key=sil.get)
    km2 = KMeans(n_clusters=bk, random_state=42, n_init=10)
    df["cluster"] = km2.fit_predict(Xs)
    return df, sil, iner, bk

df_km, sil_s, iner_s, bk = run_kmeans(active)

cs = (df_km.groupby("cluster")
      .agg(avg_price=("price","mean"), avg_sold=("sold","mean"),
           total_rev=("revenue_est","sum"), n=("price","count")).round(0))
cs["rev_B"] = cs["total_rev"] / 1e9
cs["rev_pct"] = (cs["total_rev"] / cs["total_rev"].sum() * 100).round(1)
pm, sm = cs["avg_price"].median(), cs["avg_sold"].median()

def name_cluster(row):
    hp, hs = row["avg_price"] > pm, row["avg_sold"] > sm
    if hp and hs: return "Cao cấp – Bán chạy"
    if hp and not hs: return "Cao cấp – Bán chậm"
    if not hp and hs: return "Phổ thông – Bán chạy"
    return "Phổ thông – Bán chậm"

cs["tên"] = cs.apply(name_cluster, axis=1)
df_km["cluster_name"] = df_km["cluster"].map(cs["tên"])
pal  = [CB_BLUE, CB_ORANGE, CB_GREEN, CB_PURPLE, CB_SKYBLUE, CB_VERMIL]
cmap = {n: pal[i] for i, n in enumerate(cs["tên"])}

# ==================
# Biểu đồ 1: Silhouette + Elbow
# ==================
st.subheader(f"Biểu đồ 1: Xác định số phân khúc tối ưu (k={bk})")
c1, c2 = st.columns(2)

with c1:
    fig = px.line(x=list(sil_s.keys()), y=list(sil_s.values()), markers=True,
                  title="Silhouette Score theo k",
                  labels={"x":"k","y":"Silhouette Score"},
                  color_discrete_sequence=[CB_BLUE])
    fig.add_vline(x=bk, line_dash="dash", line_color=CB_VERMIL,
                  annotation_text=f"k={bk} tối ưu", annotation_position="top right")
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = px.line(x=list(iner_s.keys()), y=list(iner_s.values()), markers=True,
                  title="Elbow Method – Inertia theo k",
                  labels={"x":"k","y":"Inertia"},
                  color_discrete_sequence=[CB_BLUE])
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Silhouette cao nhất tại k={bk}: phân cụm tốt nhất theo cả 2 chiều (price x sold, log-scale). "
           "Elbow xác nhận điểm gãy tương ứng. Silhouette > 0.4 = phân cụm có ý nghĩa.")

# ==================
# Biểu đồ 2: Scatter cluster
# ==================
st.markdown("---")
st.subheader("Biểu đồ 2: Bản đồ phân khúc – Giá vs Lượng bán (log scale)")

samp = df_km.sample(min(3000, len(df_km)), random_state=42)
fig = px.scatter(samp, x="price", y="sold", color="cluster_name",
                 color_discrete_map=cmap, log_x=True, log_y=True, opacity=0.45,
                 title="Phân bố sản phẩm theo phân khúc K-Means (price x sold, log scale)",
                 labels={"price":"Giá bán (VND, log)","sold":"Lượng bán (log)",
                         "cluster_name":"Phân khúc"},
                 hover_name="name" if "name" in samp.columns else None)
fig.update_layout(plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=1))
st.plotly_chart(fig, use_container_width=True)
st.caption("Sample 3,000 sản phẩm. K-Means trên log1p(price) x log1p(sold) sau StandardScaler — "
           "tránh bias do phân phối lệch phải mạnh và đơn vị khác nhau.")

# ==================
# Biểu đồ 3: Revenue share
# ==================
st.markdown("---")
st.subheader("Biểu đồ 3: Tỷ trọng Doanh thu & Đặc trưng từng Phân khúc")

c3, c4 = st.columns(2)
with c3:
    fig = px.pie(cs.reset_index(), names="tên", values="rev_B", color="tên",
                 color_discrete_map=cmap, hole=0.35,
                 title="% Doanh thu ước tính theo phân khúc")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with c4:
    disp = cs[["tên","n","avg_price","avg_sold","rev_B","rev_pct"]].copy()
    disp.columns = ["Tên phân khúc","Số SP","Giá TB (VND)","Sold TB","DT (Tỷ đồng)","%DT"]
    disp["Giá TB (VND)"] = disp["Giá TB (VND)"].apply(lambda x: f"{x:,.0f}")
    disp["Sold TB"] = disp["Sold TB"].apply(lambda x: f"{x:,.0f}")
    disp["DT (Tỷ đồng)"] = disp["DT (Tỷ đồng)"].apply(lambda x: f"{x:.0f}B")
    st.markdown("**Đặc trưng từng phân khúc:**")
    st.dataframe(disp.reset_index(drop=True), use_container_width=True, hide_index=True)
    st.caption("Tên phân khúc dựa trên so sánh avg_price và avg_sold mỗi cluster với median tổng thể.")

st.markdown("---")

# ==================
# Biểu đồ 4: Volume Driver vs Margin Driver (MT6 — 23127488)
# ==================
st.subheader("Biểu đồ 4: Volume Driver vs Margin Driver theo Phân khúc (MT6)")
st.caption("Volume Driver = top 3 sản phẩm sold cao nhất (chiến lược phễu). "
           "Margin Driver = top 3 sản phẩm revenue cao nhất trong sold < trung vị phân khúc.")

exclude_kw = r"quà tặng|gift|không bán|tặng kèm|phần quà"
tier_order = list(cs["tên"].values)
tabs = st.tabs([t.upper() for t in tier_order])

for i, tier in enumerate(tier_order):
    with tabs[i]:
        tier_data = df_km[
            (df_km["cluster_name"] == tier) &
            (~df_km["name"].str.contains(exclude_kw, case=False, na=False))
        ].copy()
        if tier_data.empty:
            st.info("Không có dữ liệu cho phân khúc này."); continue

        sold_median = tier_data["sold"].median()

        top3_vol_df  = tier_data.nlargest(3, "sold")
        top3_marg_df = tier_data[tier_data["sold"] < sold_median].nlargest(3, "revenue_est")

        # Grouped bar chart so sánh 2 nhóm
        bar_data = pd.DataFrame({
            "Rank": ["#1","#2","#3"],
            "Volume Driver (Sold)": top3_vol_df["sold"].values,
            "Margin Driver (Revenue est, M đồng)": (top3_marg_df["revenue_est"].values / 1e6
                                                 if len(top3_marg_df) == 3 else [0,0,0]),
        })
        fig = go.Figure()
        fig.add_bar(name="Volume Driver (Sold)", x=bar_data["Rank"],
                    y=bar_data["Volume Driver (Sold)"], marker_color=CB_BLUE,
                    text=bar_data["Volume Driver (Sold)"].apply(lambda x: f"{x:,.0f}"),
                    textposition="outside")
        fig.add_bar(name="Margin Driver (Rev M đồng)", x=bar_data["Rank"],
                    y=bar_data["Margin Driver (Revenue est, M đồng)"],
                    marker_color=CB_ORANGE,
                    text=bar_data["Margin Driver (Revenue est, M đồng)"].apply(lambda x: f"{x:,.0f}M"),
                    textposition="outside")
        fig.update_layout(barmode="group", plot_bgcolor="white",
                          title=f"Top 3 Volume Driver vs Margin Driver — {tier}",
                          xaxis_title="Hạng", yaxis_title="Giá trị",
                          legend=dict(orientation="h", y=1))
        st.plotly_chart(fig, use_container_width=True, key=f"vol_margin_{i}")
        # st.caption("Lưu ý: hai trục có đơn vị khác nhau (lượt bán vs triệu VND) — "
        #           "so sánh theo hạng trong cùng nhóm, không so sánh giữa 2 cột.")

        # Chi tiết sản phẩm
        col_l, col_r = st.columns(2)
        with col_l:
            st.markdown(f"**Volume Driver — {tier}** *(sold cao -> chiến lược phễu)*")
            show_v = top3_vol_df[["name","sold","price"]].copy()
            show_v.columns = ["Sản phẩm","Đã bán","Giá (VND)"]
            show_v["Đã bán"] = show_v["Đã bán"].apply(lambda x: f"{x:,.0f}")
            show_v["Giá (VND)"] = show_v["Giá (VND)"].apply(lambda x: f"{x:,.0f}")
            st.dataframe(show_v, hide_index=True, use_container_width=True)

        with col_r:
            st.markdown(f"**Margin Driver — {tier}** *(revenue cao, sold thấp -> sinh lời)*")
            if len(top3_marg_df) == 0:
                st.info("Không đủ dữ liệu.")
            else:
                show_m = top3_marg_df[["name","revenue_est","sold"]].copy()
                show_m.columns = ["Sản phẩm","Revenue ước tính","Đã bán"]
                show_m["Revenue ước tính"] = show_m["Revenue ước tính"].apply(lambda x: f"{x:,.0f} đồng")
                show_m["Đã bán"] = show_m["Đã bán"].apply(lambda x: f"{x:,.0f}")
                st.dataframe(show_m, hide_index=True, use_container_width=True)

st.markdown("---")
st.markdown("""
### Kết luận — MT3 & 6
*(Điền sau khi xem kết quả phân cụm thực tế từ data)*
- Phân khúc chiếm tỷ trọng doanh thu lớn nhất: **...**
- Chiến lược phễu (Volume Driver): **...** — bán nhiều, lôi kéo khách mới
- Chiến lược sinh lời (Margin Driver): **...** — biên lợi nhuận cao, ít cạnh tranh volume
- Đề xuất cho người bán mới: bắt đầu từ phân khúc **...** để xây reviews, sau đó pivot sang **...**
""")
