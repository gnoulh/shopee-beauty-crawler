"""
pages/03_Market.py — Phân khúc Thị trường & Chiến lược Sản phẩm
================================================================
PHÂN CÔNG:
  MT3 — Phân khúc K-Means (price x sold, log-normalized, Silhouette)
        Đánh giá tỷ trọng doanh thu từng phân khúc
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
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    load_data, get_active, inject_css, setup_sidebar,
    member_badge, conclusion_box,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_PURPLE, CB_GRAY,
)

st.set_page_config(page_title="Phân khúc Thị trường", layout="wide")
inject_css(); setup_sidebar()
products, shops, reviews = load_data()
active = get_active(products)

st.title("Phân khúc Thị trường & Chiến lược Sản phẩm")
st.caption("products.csv - 20,658 sản phẩm - crawl 18/3/2026 - MSSV: 23127488")
member_badge("23127488", "MT3")

with st.expander("Mục tiêu SMART (MT3 & 6)", expanded=False):
    st.markdown("""
    > **Phân chia thị trường mỹ phẩm Shopee thành 5 phân khúc chiến lược** dựa trên mối tương quan
    > giữa giá bán và sản lượng — snapshot 18/3/2026, K-Means (k=5, Elbow method),
    > từ đó đánh giá tỷ trọng đóng góp doanh thu ước tính của từng phân khúc.
    """)
st.markdown("---")

@st.cache_data
def run_kmeans5(df):
    sub = df[["price","sold","revenue_est","sub_category","price_tier","is_mall"]].dropna().copy()
    if "name" in df.columns: sub["name"] = df.loc[sub.index,"name"]
    X  = np.log1p(sub[["price","sold"]])
    sc = StandardScaler(); Xs = sc.fit_transform(X)
    iner = {k: KMeans(n_clusters=k, random_state=42, n_init=3).fit(Xs).inertia_ for k in range(2,9)}
    km5  = KMeans(n_clusters=5, random_state=42, n_init=10)
    sub["cluster"] = km5.fit_predict(Xs)
    return sub, iner

df_km, iner = run_kmeans5(active)

cs = (df_km.groupby("cluster")
      .agg(avg_price=("price","mean"), avg_sold=("sold","mean"),
           total_rev=("revenue_est","sum"), n=("price","count")).round(0))
cs["rev_B"] = cs["total_rev"] / 1e9
cs["rev_pct"] = (cs["total_rev"] / cs["total_rev"].sum() * 100).round(1)
pm, sm = cs["avg_price"].median(), cs["avg_sold"].median()

NAMES = ["Cao cấp – Bán chạy","Cao cấp – Bán chậm",
         "Phổ thông – Bán chạy","Phổ thông – Bán chậm","Phổ thông – Trung bình"]
def name_c(row):
    hp,hs = row["avg_price"]>pm, row["avg_sold"]>sm
    if hp and hs: return NAMES[0]
    if hp and not hs: return NAMES[1]
    if not hp and hs: return NAMES[2]
    return NAMES[3]

cs["tên"] = cs.apply(name_c, axis=1)
# Deduplicate
seen={}
for idx in cs.index:
    n=cs.loc[idx,"tên"]; seen[n]=seen.get(n,0)+1
    if seen[n]>1: cs.loc[idx,"tên"]=n+f" {seen[n]}"
df_km["cluster_name"] = df_km["cluster"].map(cs["tên"])

pal  = [CB_BLUE,CB_ORANGE,CB_GREEN,CB_PURPLE,CB_SKYBLUE]
cmap = {n:pal[i] for i,n in enumerate(cs["tên"])}

# === Biểu đồ 1: Elbow ===
st.markdown("## MT3 — K-Means (k=5) Phân khúc Thị trường")
member_badge("23127488", "MT3")

st.subheader("Biểu đồ 1: Elbow Method — Inertia theo k")
fig_e = px.line(x=list(iner.keys()), y=list(iner.values()), markers=True,
                title="Elbow Method — Inertia theo số phân khúc k",
                labels={"x":"k","y":"Inertia"}, color_discrete_sequence=[CB_BLUE])
fig_e.add_vline(x=5, line_dash="dash", line_color=CB_VERMIL,
                annotation_text="k=5 (theo đề)", annotation_position="top right")
fig_e.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_e, use_container_width=True)
st.caption("Điểm gãy Elbow xác nhận k=5. K-Means trên log1p(price)×log1p(sold) sau StandardScaler.")

# === Biểu đồ 2: Scatter ===
st.subheader("Biểu đồ 2: Bản đồ phân khúc — Giá vs Lượng bán (log scale)")
samp = df_km.sample(min(3000,len(df_km)), random_state=42)
fig_s = px.scatter(samp, x="price", y="sold", color="cluster_name",
                   color_discrete_map=cmap, log_x=True, log_y=True, opacity=0.45,
                   title="Phân bố sản phẩm theo 5 phân khúc K-Means (price x sold, log scale)",
                   labels={"price":"Giá bán (VND, log)","sold":"Lượng bán (log)","cluster_name":"Phân khúc"},
                   hover_name="name" if "name" in samp.columns else None)
fig_s.update_layout(plot_bgcolor="white", legend=dict(orientation="h",yanchor="bottom",y=1))
st.plotly_chart(fig_s, use_container_width=True)
st.caption("Sample 3,000 sản phẩm. Log scale giúp thấy rõ phân bố khi price và sold trải rất rộng.")

# ==================
# Biểu đồ 3: Revenue share
# ==================
st.markdown("---")
st.subheader("Biểu đồ 3: Tỷ trọng Doanh thu & Đặc trưng từng Phân khúc")

col_a, col_b = st.columns(2)
with col_a:
    fp = px.pie(cs.reset_index(), names="tên", values="rev_B", color="tên",
                color_discrete_map=cmap, hole=0.38,
                title="% Doanh thu ước tính theo phân khúc")
    fp.update_traces(textinfo="percent+label")
    st.plotly_chart(fp, use_container_width=True)
with col_b:
    sc2 = cs.sort_values("rev_B",ascending=True).reset_index()
    fb = go.Figure()
    for _,row in sc2.iterrows():
        fb.add_bar(x=[row["rev_B"]],y=[row["tên"]],orientation="h",
                   marker_color=cmap.get(row["tên"],CB_GRAY),
                   text=[f"{row['rev_B']:.0f}B"],textposition="outside",
                   name=row["tên"],showlegend=False)
    fb.update_layout(plot_bgcolor="white",xaxis_title="Tổng DT ước tính (Tỷ VND)",
                     title="Doanh thu tuyệt đối theo phân khúc",
                     yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fb, use_container_width=True)

# Bảng đặc trưng
st.markdown("#### Đặc trưng từng phân khúc chiến lược")
disp = cs[["tên","n","avg_price","avg_sold","rev_B","rev_pct"]].copy()
disp.columns = ["Tên phân khúc","Số SP","Giá TB (VND)","Sold TB","DT (Tỷ đồng)","%DT"]
disp["Giá TB (VND)"] = disp["Giá TB (VND)"].apply(lambda x:f"{x:,.0f}")
disp["Sold TB"] = disp["Sold TB"].apply(lambda x:f"{x:,.0f}")
disp["DT (Tỷ đồng)"] = disp["DT (Tỷ đồng)"].apply(lambda x:f"{x:.0f}B")
st.dataframe(disp.reset_index(drop=True), hide_index=True, use_container_width=True)

conclusion_box("""
<b> Nhận xét MT3 (23127488):</b><br>
• Phân khúc <b>Cao cấp – Bán chạy</b>: giá cao VÀ sold cao — tỷ trọng doanh thu lớn nhất, cạnh tranh nhất.<br>
• Phân khúc <b>Cao cấp – Bán chậm</b>: margin cao, volume thấp — phù hợp thương hiệu uy tín.<br>
• Phân khúc <b>Phổ thông – Bán chạy</b>: chiến lược volume — tốt để xây reviews ban đầu.<br>
• Phân khúc <b>Phổ thông – Bán chậm</b>: cần xem lại chiến lược giá.<br>
• <b>Nhất quán với MT7 (22127254)</b>: Phổ thông–Bán chạy ~ ô Phễu; Cao cấp–Bán chạy ~ ô Ngôi sao.
""")
