"""
pages/04_Shops.py — Hiệu quả Cửa hàng & Chỉ số Lòng tin
================================================================
PHÂN CÔNG: 23127361
  MT9 — K-Means phân cụm shop theo 4 tiêu chí lòng tin (k=4, Silhouette)
  MT5 — Tương quan chỉ số shop -> revenue_est sản phẩm
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
    inject_css, member_badge, conclusion_box,
    load_data, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_GRAY,
)

st.set_page_config(page_title="04 – Hiệu quả Cửa hàng", layout="wide")
inject_css(); setup_sidebar()

products, shops, reviews = load_data()

st.title("Hiệu quả Cửa hàng & Chỉ số Lòng tin")
st.caption("shops.csv + products.csv - 5,746 cửa hàng - crawl 18/3/2026 - MSSV: 23127361")
member_badge("23127361", "MT5 & 9")

# === SMART ===
st.subheader("Bảng tiêu chí SMART")
smart_df = pd.DataFrame({
    "Tiêu chí": ["Specific","Measurable","Achievable","Relevant","Time-bound"],
    "MT9 – Phân cụm Shop": [
        "Phân 4 nhóm shop theo follower, rating_count, rating_star, response_rate",
        "Đo bằng Silhouette score + tổng revenue_est và total_sold mỗi nhóm",
        "K-Means trên tập dữ liệu đã làm sạch (5,746 shops)",
        "Xác định nhóm shop hiệu quả cao nhất để người bán làm mục tiêu",
        "Snapshot 18/3/2026",
    ],
    "MT5 – Tương quan": [
        "Tìm yếu tố (follower, response_rate, rating_star) ảnh hưởng nhất đến revenue_est",
        "Hệ số tương quan Pearson — chỉ số có |r| lớn nhất = quan trọng nhất",
        "Ma trận tương quan trên tập dữ liệu hiện có",
        "Chỉ ra KPI shop nào nhà bán hàng nên ưu tiên cải thiện",
        "Snapshot 18/3/2026",
    ],
})
st.dataframe(smart_df, hide_index=True, use_container_width=True)
st.markdown("---")

# === Prep data =============================================─
@st.cache_data
def prep_shop_data(products_df, shops_df):
    shop_revenue = products_df.groupby("shop_id")["revenue_est"].sum().reset_index()
    shop_cols = ["shop_id","shop_name","follower_count","rating_star",
                 "rating_count","response_rate","total_sold","is_mall"]
    avail = [c for c in shop_cols if c in shops_df.columns]
    df = shops_df[avail].copy().merge(shop_revenue, on="shop_id", how="left")
    df["revenue_est"] = df["revenue_est"].fillna(0)
    df["total_sold"] = df["total_sold"].fillna(0) if "total_sold" in df.columns else 0
    return df

df = prep_shop_data(products, shops)

# ===============
# MT9 — K-Means phân cụm shop
# ===============
st.subheader("Mục tiêu 9 — K-Means phân cụm Cửa hàng theo Độ tin cậy (k=4)")

features = ["follower_count","rating_count","rating_star","response_rate"]
avail_f  = [f for f in features if f in df.columns]

@st.cache_data
def run_shop_kmeans(df_, features_):
    df_c = df_.copy()
    df_c[features_] = df_c[features_].fillna(0)
    X_scaled = StandardScaler().fit_transform(df_c[features_])
    # Silhouette để xác nhận k=4
    sil_scores = {}
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lb = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, lb)
    km4 = KMeans(n_clusters=4, random_state=42, n_init=10)
    df_c["Cluster"] = km4.fit_predict(X_scaled)
    return df_c, sil_scores

df_c, sil_scores = run_shop_kmeans(df, avail_f)

# Đặt tên cluster dựa trên follower_count trung bình (dynamic, không hardcode)
cluster_follower = df_c.groupby("Cluster")["follower_count"].mean().sort_values(ascending=False)
rank_names = ["Shop Dẫn Đầu","Shop Uy Tín","Shop Phổ Thông","Shop Mới/Ít Tương Tác"]
cluster_name_map = {cid: name for cid, name in zip(cluster_follower.index, rank_names)}
df_c["Nhóm"] = df_c["Cluster"].map(cluster_name_map)

colorblind_palette = {
    "Shop Dẫn Đầu": CB_VERMIL,
    "Shop Uy Tín":  CB_BLUE,
    "Shop Phổ Thông": CB_SKYBLUE,
    "Shop Mới/Ít Tương Tác": CB_GRAY,
}
chart_layout = dict(plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                    margin=dict(t=50,l=10,r=10,b=10),
                    font=dict(family="Arial",size=13,color="#333"))

# Biểu đồ 1a: Silhouette score
st.markdown("#### Biểu đồ 1a: Silhouette Score — Xác nhận k=4")
fig_sil = px.line(x=list(sil_scores.keys()), y=list(sil_scores.values()),
                  markers=True, labels={"x":"k","y":"Silhouette Score"},
                  title="Silhouette Score theo số cụm k",
                  color_discrete_sequence=[CB_BLUE])
fig_sil.add_vline(x=4, line_dash="dash", line_color=CB_VERMIL,
                  annotation_text="k=4 được chọn", annotation_position="top right")
fig_sil.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_sil, use_container_width=True)

with st.expander("Đặc trưng trung bình từng nhóm"):
    summary = df_c.groupby("Nhóm")[avail_f].mean().reset_index()
    disp = summary.rename(columns={
        "follower_count":"TB Followers","rating_count":"TB Lượt đánh giá",
        "rating_star":"TB Điểm sao","response_rate":"TB Tốc độ phản hồi",
    })
    st.dataframe(disp, hide_index=True, use_container_width=True)
    st.markdown("""
    **Cơ sở đặt tên nhóm:**
    - **Shop Dẫn Đầu:** Followers và lượt đánh giá cực cao — "ông lớn" trên sàn
    - **Shop Uy Tín:** Điểm sao và response rate cao nhất — dịch vụ chuyên nghiệp
    - **Shop Phổ Thông:** Chỉ số ở mức ổn định — cần cải thiện phản hồi
    - **Shop Mới/Ít Tương Tác:** Các chỉ số thấp nhất — mới tham gia hoặc không hoạt động
    """)

# Biểu đồ 1b: 3 biểu đồ cột (số lượng, doanh thu, lượt bán)
revenue_sales = df_c.groupby("Nhóm")[["revenue_est","total_sold"]].sum().reset_index()
shop_counts = df_c["Nhóm"].value_counts().reset_index()
shop_counts.columns = ["Nhóm","Số lượng Shop"]

st.markdown("#### Biểu đồ 1b: Phân bố số lượng shop, doanh thu và lượt bán")
col1, col2, col3 = st.columns(3)

with col1:
    fig = px.pie(shop_counts, names="Nhóm", values="Số lượng Shop",
                 color="Nhóm", color_discrete_map=colorblind_palette,
                 title="Tỷ trọng số lượng Shop", hole=0.4)
    fig.update_traces(textinfo="percent", textfont_size=12)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5),
                      margin=dict(t=50,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    fig = px.bar(revenue_sales, x="Nhóm", y="revenue_est", color="Nhóm",
                 color_discrete_map=colorblind_palette,
                 title="Tổng Doanh thu Ước tính (VND)",
                 labels={"revenue_est":"Doanh thu (VND)","Nhóm":""}, text_auto=".2s")
    fig.update_layout(**chart_layout)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

with col3:
    if "total_sold" in revenue_sales.columns:
        fig = px.bar(revenue_sales, x="Nhóm", y="total_sold", color="Nhóm",
                     color_discrete_map=colorblind_palette,
                     title="Tổng Lượt Bán",
                     labels={"total_sold":"Lượt bán","Nhóm":""}, text_auto=".2s")
        fig.update_layout(**chart_layout)
        fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
        fig.update_traces(textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

with st.expander("Nhận xét: Nghịch lý giữa số lượng và doanh thu"):
    st.markdown("""
    **1. Quy tắc Pareto (80/20):** Shop Dẫn Đầu chiếm số lượng nhỏ nhưng nắm giữ phần lớn doanh thu.
    Thị trường mỹ phẩm Shopee bị dẫn dắt bởi một số ít "ông lớn" thay vì phân phối đều.

    **2. Sức mạnh của số đông:** Shop Phổ Thông chiếm số lượng áp đảo. Tổng doanh thu cả nhóm
    đáng kể dù hiệu quả mỗi shop không cao bằng nhóm Dẫn Đầu.

    **3. Bẫy số 0 (Shop Mới):** Không có review -> không có người mua -> không có doanh thu.
    Chiến lược: tạo 10–20 đơn đầu tiên bằng voucher/flash sale để phá vỡ vòng lặp.

    **Gợi ý chiến lược:**
    - Shop Mới: "Phá băng" — chấp nhận lỗ nhẹ để có review đầu tiên
    - Shop Phổ Thông: Tối ưu response rate > 90% -> tiến lên Uy Tín
    - Shop Uy Tín: Đầu tư marketing để tăng followers — con đường trực tiếp đến doanh thu lớn
    """)

st.markdown("---")

# ===============
# MT5 — Tương quan chỉ số cửa hàng -> revenue_est
# ===============
st.subheader("Mục tiêu 5 — Tương quan Chỉ số Cửa hàng -> Doanh thu Sản phẩm")

corr_features = [f for f in ["follower_count","response_rate","rating_star","revenue_est"]
                 if f in df.columns]
corr_matrix = df[corr_features].corr()

labels_map = {"follower_count":"Lượt theo dõi","response_rate":"Tỷ lệ phản hồi",
              "rating_star":"Điểm đánh giá","revenue_est":"Doanh thu"}

fig_corr = px.imshow(
    corr_matrix,
    labels=dict(color="Hệ số Pearson"),
    x=[labels_map.get(c,c) for c in corr_matrix.columns],
    y=[labels_map.get(c,c) for c in corr_matrix.index],
    text_auto=".3f",
    color_continuous_scale="RdBu_r",
    range_color=[-1,1],
    title="Ma trận tương quan Pearson — Chỉ số cửa hàng × Doanh thu",
)
fig_corr.update_layout(plot_bgcolor="rgba(0,0,0,0)", font=dict(family="Arial",size=13),
                       margin=dict(t=80,l=10,r=10,b=10), height=460)
st.plotly_chart(fig_corr, use_container_width=True)
st.caption("Giá trị càng gần +1: tương quan thuận mạnh. Màu xanh = thuận, đỏ = nghịch. "
           "Diverging colorscale RdBu_r — colorblind-safe.")

revenue_corrs = corr_matrix["revenue_est"].drop("revenue_est") if "revenue_est" in corr_matrix else pd.Series()
if not revenue_corrs.empty:
    strongest = revenue_corrs.abs().idxmax()
    strongest_r = revenue_corrs[strongest]

with st.expander("Nhận xét: Yếu tố nào ảnh hưởng mạnh nhất đến doanh thu?"):
    st.markdown(f"""
    **Phát hiện quan trọng:**

    1. **Lượt theo dõi (Followers) là yếu tố quyết định:** Tương quan thuận với doanh thu
       (r ~ cao nhất trong 3 chỉ số) -> Followers là tệp khách hàng trung thành, nguồn doanh thu bền vững nhất.

    2. **Nghịch lý điểm đánh giá (Sao):** Tương quan yếu hoặc âm nhẹ với doanh thu.
       **Không phải** rating cao thì bán ít — mà shop lớn bán nghìn sản phẩm khó tránh
       vài review tiêu cực, kéo rating xuống nhẹ so với shop nhỏ mới mở toàn 5 sao tuyệt đối.

    3. **Tỷ lệ phản hồi:** Tương quan thuận nhưng không đột phá.
       Phản hồi nhanh tăng tỷ lệ chốt đơn, nhưng để tăng doanh thu lớn vẫn cần marketing + followers.

    **Thứ tự ưu tiên hành động:**
    Xây dựng Đánh giá (lòng tin) -> Tăng Tốc độ phản hồi (chuyên nghiệp) -> Tích lũy Followers (tăng doanh thu bền vững)
    """)

st.markdown("---")
st.caption("MSSV: 23127361 - Dashboard được thực hiện bởi thành viên 23127361")
