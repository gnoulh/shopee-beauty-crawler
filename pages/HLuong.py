import unicodedata

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


# === Load & chuẩn hóa ===
@st.cache_data
def load_data():
    df = pd.read_csv("data/vietnam_housing_dataset_cleaned.csv")
    # Fix NFD -> NFC (4 dòng 'Hà Nội' bị encode lệch trong CSV)
    df["Province"] = df["Province"].apply(
        lambda x: unicodedata.normalize("NFC", x) if isinstance(x, str) else x
    )
    # Thêm Price_segment (không có trong CSV)
    df["Price_segment"] = pd.cut(
        df["Price"],
        bins=[0, 3, 5, 7, 9, float("inf")],
        labels=["<3 tỷ", "3–5 tỷ", "5–7 tỷ", "7–9 tỷ", ">9 tỷ"],
    ).astype(str)
    # Phân vùng địa lý
    bac = [
        "Hà Nội","Hải Phòng","Hưng Yên","Bắc Ninh","Quảng Ninh","Hà Nam",
        "Hải Dương","Thái Bình","Nam Định","Ninh Bình","Vĩnh Phúc","Bắc Giang",
        "Phú Thọ","Thái Nguyên","Hà Giang","Cao Bằng","Lào Cai","Điện Biên",
        "Sơn La","Yên Bái","Tuyên Quang","Lạng Sơn","Bắc Kạn","Hòa Bình",
    ]
    trung = [
        "Đà Nẵng","Thừa Thiên Huế","Quảng Nam","Quảng Ngãi","Bình Định","Phú Yên",
        "Khánh Hòa","Ninh Thuận","Bình Thuận","Thanh Hóa","Nghệ An","Hà Tĩnh",
        "Quảng Bình","Quảng Trị","Kon Tum","Gia Lai","Đắk Lắk","Đắk Nông","Lâm Đồng",
    ]
    df["Region"] = df["Province"].apply(
        lambda p: "Miền Bắc" if p in bac else ("Miền Trung" if p in trung else "Miền Nam")
    )
    return df


df = load_data()

# === Header ==================
st.title("Phân tích Giá BĐS theo Địa lý")
st.markdown(
    """
### Câu hỏi phân tích
> **Thị trường bất động sản phân bố và định giá như thế nào giữa các tỉnh thành Việt Nam?
> Hà Nội hay TP.HCM đắt hơn? Quận/huyện nào là "đắt đỏ nhất"?
> Tại sao Hưng Yên xuất hiện dày đặc trong top giá cao?**

Phân tích tập trung vào: `Province`, `District`, `Price`, `Price_per_m2`, `Region`, `Is_Project`.
"""
)
st.markdown("---")

# === Sidebar filters ======
with st.sidebar:
    st.header("Bộ lọc")
    price_range = st.slider(
        "Khoảng giá (tỷ đồng)",
        float(df["Price"].min()),
        float(df["Price"].quantile(0.99)),
        (float(df["Price"].min()), float(df["Price"].quantile(0.95))),
    )
    selected_regions = st.multiselect(
        "Vùng miền", ["Miền Bắc", "Miền Trung", "Miền Nam"], default=[]
    )

mask = df["Price"].between(*price_range)
if selected_regions:
    mask &= df["Region"].isin(selected_regions)
dff = df[mask].copy()
st.caption(f"Đang hiển thị **{len(dff):,}** / {len(df):,} bất động sản sau lọc")

# === Thống kê tỉnh thành ===================================================
prov_stats = (
    dff.groupby("Province")
    .agg(
        Count=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Median_Price=("Price", "median"),
        Avg_Pm2=("Price_per_m2", lambda x: x.mean() * 1000),
        Proj_pct=("Is_Project", lambda x: x.mean() * 100),
    )
    .query("Count >= 30")
    .reset_index()
    .sort_values("Avg_Pm2", ascending=False)
)

# ======================================
# 1. BAR: Top 15 tỉnh thành theo giá/m^2
# ======================================
st.subheader("1. Giá/m^2 trung bình theo Tỉnh thành")
st.markdown(
    """
**Biểu đồ cột ngang** xếp hạng 15 tỉnh thành có đơn giá/m^2 trung bình cao nhất.
Đơn vị: **triệu VND/m^2** - Chỉ tính tỉnh có >= 30 bất động sản.

> **Lưu ý:** Hưng Yên đứng thứ 2 toàn quốc (113 tr/m^2), cao hơn TP.HCM (106 tr/m^2)
> — lý do: **98% BĐS tại Hưng Yên là nhà dự án** (chủ yếu Vinhomes Ocean Park),
> đẩy giá/m^2 lên cao bất thường so với quy mô kinh tế thực tế của tỉnh.
"""
)

top15 = prov_stats.head(15).sort_values("Avg_Pm2")

# Tô màu đặc biệt cho HN, HCM, Hưng Yên
color_map = []
for _, row in top15.iterrows():
    if row["Province"] == "Hà Nội":
        color_map.append("#1D3557")         # Xanh đậm — đắt nhất
    elif row["Province"] == "Hưng Yên":
        color_map.append("#E9C46A")         # Vàng — outlier dự án
    elif row["Province"] == "Hồ Chí Minh":
        color_map.append("#E63946")         # Magenta
    else:
        color_map.append("#A8DADC")         # Mặc định

fig1 = go.Figure(go.Bar(
    x=top15["Avg_Pm2"],
    y=top15["Province"],
    orientation="h",
    text=top15["Avg_Pm2"].apply(lambda x: f"{x:.0f}"),
    textposition="outside",
    marker_color=color_map,
    hovertemplate=(
        "<b>%{y}</b><br>"
        "Đơn giá TB: %{x:.0f} triệu/m^2<br>"
        "<extra></extra>"
    ),
))
fig1.update_layout(
    title="Top 15 tỉnh thành — Đơn giá/m^2 trung bình (triệu VND)",
    xaxis_title="Triệu VND/m^2",
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    height=520, margin=dict(l=10, r=80, t=50, b=20),
)

# Chú thích màu
st.plotly_chart(fig1, use_container_width=True)
col_leg1, col_leg2, col_leg3 = st.columns(3)
col_leg1.markdown("**Hà Nội** — đắt nhất toàn quốc (149 tr/m^2)")
col_leg2.markdown("**Hưng Yên** — outlier do 98% là nhà dự án")
col_leg3.markdown("**TP.HCM** — thị trường lớn nhất (106 tr/m^2)")

# ==========================================
# 2. BOX: Phân phối giá 6 thành phố lớn nhất
# ==========================================
st.subheader("2. Phân phối giá tại các thành phố lớn")
st.markdown(
    """
**Box plot** so sánh phân phối giá bán tại 6 tỉnh thành có nhiều BĐS nhất.
Hộp = Q1–Median–Q3 - Whisker = 1.5×IQR - Điểm bên ngoài = outlier.
"""
)

top6 = prov_stats.nlargest(6, "Count")["Province"].tolist()
order_box = (
    dff[dff["Province"].isin(top6)]
    .groupby("Province")["Price"].median()
    .sort_values(ascending=False).index.tolist()
)
fig2 = px.box(
    dff[dff["Province"].isin(top6)],
    x="Province", y="Price", color="Province",
    category_orders={"Province": order_box},
    points=False,
    labels={"Province": "Tỉnh/Thành phố", "Price": "Giá (tỷ đồng)"},
    title="Phân phối giá bán tại 6 tỉnh thành lớn nhất (sắp xếp theo median giảm dần)",
    color_discrete_sequence=px.colors.qualitative.Bold,
    height=420,
)
fig2.update_layout(
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    showlegend=False, font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"), yaxis=dict(gridcolor="#E8E8E8"),
)
st.plotly_chart(fig2, use_container_width=True)

# Insight từ data thực
hcm_med = dff[dff["Province"]=="Hồ Chí Minh"]["Price"].median()
hn_med  = dff[dff["Province"]=="Hà Nội"]["Price"].median()
bd_med  = dff[dff["Province"]=="Bình Dương"]["Price"].median() if "Bình Dương" in dff["Province"].values else None
st.info(
    f"**Insight:** TP.HCM (median {hcm_med:.1f} tỷ) nhỉnh hơn Hà Nội (median {hn_med:.1f} tỷ) "
    f"về giá bán tuyệt đối, nhưng **Hà Nội đắt hơn về giá/m^2** (149 vs 106 tr/m^2) do "
    f"diện tích trung bình nhà Hà Nội nhỏ hơn TP.HCM. "
    + (f"Bình Dương (median {bd_med:.1f} tỷ) — thị trường vệ tinh giá rẻ hơn 40%." if bd_med else "")
)

# ==============================================
# 3. STACKED BAR: Cơ cấu phân khúc giá theo tỉnh
# ==============================================
st.subheader("3. Cơ cấu phân khúc thị trường theo Tỉnh thành")
st.markdown(
    """
**Biểu đồ cột xếp chồng (%)** thể hiện tỷ trọng từng phân khúc giá trong cơ cấu
nguồn cung tại 8 tỉnh thành lớn nhất.
> *"Thị trường nào thiên về phân khúc bình dân? Thị trường nào có nhiều BĐS cao cấp?"*
"""
)

top8 = prov_stats.nlargest(8, "Count")["Province"].tolist()
seg_order = ["<3 tỷ", "3–5 tỷ", "5–7 tỷ", "7–9 tỷ", ">9 tỷ"]
seg_colors = ["#2A9D8F", "#A8DADC", "#E9C46A", "#F4A261", "#E63946"]

df_seg = (
    dff[dff["Province"].isin(top8)]
    .groupby(["Province", "Price_segment"])
    .size().reset_index(name="Count")
)
df_seg["Price_segment"] = pd.Categorical(df_seg["Price_segment"], categories=seg_order, ordered=True)
df_seg = df_seg.sort_values("Price_segment")
totals = df_seg.groupby("Province")["Count"].transform("sum")
df_seg["Pct"] = (df_seg["Count"] / totals * 100).round(1)

fig3 = px.bar(
    df_seg, x="Province", y="Pct", color="Price_segment",
    category_orders={"Province": top8, "Price_segment": seg_order},
    color_discrete_sequence=seg_colors,
    labels={"Province": "Tỉnh/Thành phố", "Pct": "Tỷ lệ (%)", "Price_segment": "Phân khúc"},
    title="Cơ cấu phân khúc giá tại 8 tỉnh thành lớn nhất (%)",
    text="Pct", height=430,
)
fig3.update_traces(texttemplate="%{text:.0f}%", textposition="inside", textfont_size=10)
fig3.update_layout(
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"), yaxis=dict(gridcolor="#E8E8E8", ticksuffix="%"),
    barmode="stack",
    legend=dict(orientation="h", y=-0.2, title="Phân khúc"),
)
st.plotly_chart(fig3, use_container_width=True)
st.info(
    "**Insight:** Bình Dương có 74% BĐS dưới 5 tỷ — thị trường bình dân điển hình. "
    "TP.HCM và Hà Nội tập trung nhiều nhất ở phân khúc 5–9 tỷ (~60%). "
    "Hà Nội có tỷ trọng >9 tỷ cao hơn TP.HCM, phản ánh giá/m^2 đắt hơn."
)

# ================================================
# 4. BAR: Top quận/huyện đắt nhất — chọn thành phố
# ================================================
st.subheader("4. Top Quận/Huyện đắt nhất theo Thành phố")
st.markdown(
    """
**Biểu đồ cột** xếp hạng quận/huyện theo đơn giá/m^2 trong tỉnh được chọn.
Chỉ tính quận/huyện có ≥ 20 bất động sản.
"""
)

city_choices = prov_stats.nlargest(8, "Count")["Province"].tolist()
selected_city = st.selectbox("Chọn tỉnh/thành phố:", city_choices, index=0)

city_df = dff[dff["Province"] == selected_city]
dist_stats = (
    city_df.groupby("District")
    .agg(
        Count=("Price", "count"),
        Avg_Pm2=("Price_per_m2", lambda x: x.mean() * 1000),
        Avg_Price=("Price", "mean"),
        Median_Price=("Price", "median"),
    )
    .query("Count >= 20")
    .sort_values("Avg_Pm2", ascending=False)
    .head(15).reset_index().sort_values("Avg_Pm2")
)

if dist_stats.empty:
    st.warning("Không đủ dữ liệu quận/huyện cho tỉnh này.")
else:
    fig4 = px.bar(
        dist_stats, x="Avg_Pm2", y="District", orientation="h",
        text=dist_stats["Avg_Pm2"].apply(lambda x: f"{x:.0f}"),
        color="Avg_Pm2",
        color_continuous_scale="Blues",
        custom_data=["Avg_Price", "Median_Price", "Count"],
        labels={"Avg_Pm2": "Triệu VND/m^2", "District": "Quận/Huyện"},
        title=f"Top quận/huyện đắt nhất — {selected_city} (triệu VND/m^2)",
        height=max(380, len(dist_stats) * 38),
    )
    fig4.update_traces(
        textposition="outside",
        hovertemplate=(
            "<b>%{y}</b><br>"
            "Đơn giá TB: %{x:.0f} triệu/m^2<br>"
            "Giá TB: %{customdata[0]:.1f} tỷ<br>"
            "Giá Median: %{customdata[1]:.1f} tỷ<br>"
            "Số BĐS: %{customdata[2]}<extra></extra>"
        ),
    )
    fig4.update_layout(
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        font=dict(color="#2C3E50"), coloraxis_showscale=False,
        xaxis=dict(gridcolor="#E8E8E8"), yaxis=dict(linecolor="#CCCCCC"),
        margin=dict(l=10, r=80, t=50, b=20),
    )
    st.plotly_chart(fig4, use_container_width=True)

    # Insight động theo thành phố được chọn
    top_dist = dist_stats.iloc[-1]
    bot_dist = dist_stats.iloc[0]
    ratio = top_dist["Avg_Pm2"] / bot_dist["Avg_Pm2"] if bot_dist["Avg_Pm2"] > 0 else 1
    st.info(
        f"**{selected_city}:** Quận/huyện đắt nhất là **{top_dist['District']}** "
        f"({top_dist['Avg_Pm2']:.0f} tr/m^2), đắt hơn quận rẻ nhất "
        f"(**{bot_dist['District']}** — {bot_dist['Avg_Pm2']:.0f} tr/m^2) "
        f"tới **{ratio:.1f} lần** — cho thấy chênh lệch địa lý nội tỉnh rất lớn."
    )

# ==============================================
# 5. BUBBLE CHART: Thị phần vs Đơn giá toàn quốc
# ==============================================
st.subheader("5. Thị phần và Đơn giá — Toàn quốc")
st.markdown(
    """
**Biểu đồ bong bóng** — mỗi bong bóng là 1 tỉnh thành.
Trục X = số lượng BĐS (log scale) - Trục Y = giá/m^2 (triệu) - Kích thước = giá TB - Màu = vùng miền.
> *"Tỉnh nào vừa có thị phần lớn, vừa có giá cao?"*
"""
)

bubble_df = (
    dff.groupby(["Province", "Region"])
    .agg(
        Count=("Price", "count"),
        Avg_Price=("Price", "mean"),
        Avg_Pm2=("Price_per_m2", lambda x: x.mean() * 1000),
    )
    .query("Count >= 30").reset_index()
)
region_colors = {"Miền Bắc": "#457B9D", "Miền Trung": "#E9C46A", "Miền Nam": "#E63946"}

fig5 = px.scatter(
    bubble_df, x="Count", y="Avg_Pm2", size="Avg_Price", color="Region",
    text="Province",
    color_discrete_map=region_colors,
    size_max=55, opacity=0.8,
    labels={
        "Count": "Số lượng BĐS", "Avg_Pm2": "Đơn giá TB (triệu VND/m^2)",
        "Avg_Price": "Giá TB (tỷ)", "Region": "Vùng miền",
    },
    title="Thị phần BĐS vs Đơn giá — Bong bóng = Giá TB (tỷ đồng)",
    height=530,
)
fig5.update_traces(textposition="top center", textfont=dict(size=9, color="#2C3E50"))
fig5.update_layout(
    plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8", type="log", title="Số lượng BĐS (log scale)"),
    yaxis=dict(gridcolor="#E8E8E8"),
    legend=dict(title="Vùng miền"),
)
st.plotly_chart(fig5, use_container_width=True)

# ======================
# 6. BẢNG TÓM TẮT
# ======================
st.subheader("6. Bảng tóm tắt theo Tỉnh thành")
st.markdown("Thống kê đầy đủ — nhấn vào tiêu đề cột để sắp xếp.")

summary = (
    dff.groupby(["Province", "Region"])
    .agg(
        Số_BĐS=("Price", "count"),
        Giá_TB=("Price", lambda x: round(x.mean(), 2)),
        Giá_trung_vị=("Price", lambda x: round(x.median(), 2)),
        Giá_m2_TB=("Price_per_m2", lambda x: round(x.mean() * 1000, 0)),
        Pct_có_sổ=("Has_certificate", lambda x: f"{x.mean()*100:.0f}%"),
        Pct_dự_án=("Is_Project", lambda x: f"{x.mean()*100:.0f}%"),
    )
    .query("Số_BĐS >= 30")
    .sort_values("Số_BĐS", ascending=False)
    .reset_index()
)
summary.columns = [
    "Tỉnh/TP", "Vùng miền", "Số BĐS",
    "Giá TB (tỷ)", "Giá trung vị (tỷ)", "Giá/m^2 TB (tr)",
    "% Có sổ", "% Dự án",
]
st.dataframe(summary, use_container_width=True, hide_index=True)

# === Key Insights =========─
st.markdown("---")

# Tính dynamic từ data thực
hcm_row = prov_stats[prov_stats["Province"] == "Hồ Chí Minh"]
hn_row = prov_stats[prov_stats["Province"] == "Hà Nội"]
hy_row = prov_stats[prov_stats["Province"] == "Hưng Yên"]
hcm_n = int(hcm_row["Count"].values[0]) if not hcm_row.empty else 0
hn_n = int(hn_row["Count"].values[0])  if not hn_row.empty  else 0
top2_pct = round((hcm_n + hn_n) / len(dff) * 100, 1)
hn_pm2 = round(float(hn_row["Avg_Pm2"].values[0]))  if not hn_row.empty  else 0
hcm_pm2 = round(float(hcm_row["Avg_Pm2"].values[0])) if not hcm_row.empty else 0
hy_pm2 = round(float(hy_row["Avg_Pm2"].values[0]))  if not hy_row.empty  else 0

c1, c2, c3 = st.columns(3)
with c1:
    st.info(
        f"**Hà Nội đắt nhất** về giá/m^2: **{hn_pm2} tr/m^2** "
        f"(cao hơn TP.HCM {hn_pm2-hcm_pm2:.0f} tr/m^2, tức +{(hn_pm2/hcm_pm2-1)*100:.0f}%) "
        f"— do diện tích nhà Hà Nội trung bình nhỏ hơn, nhưng vị trí trung tâm đắt hơn."
    )
with c2:
    st.info(
        f"**TP.HCM lớn nhất** về thị phần: **{hcm_n:,} BĐS** ({hcm_pm2} tr/m^2). "
        f"Cùng với Hà Nội ({hn_n:,} BĐS) chiếm **{top2_pct}%** tổng nguồn cung — "
        f"thị trường cực kỳ tập trung."
    )
with c3:
    st.warning(
        f"**Hưng Yên — Outlier đáng chú ý:** {hy_pm2} tr/m^2, cao hơn TP.HCM "
        f"dù quy mô kinh tế nhỏ hơn rất nhiều. Nguyên nhân: "
        f"**98% BĐS là nhà dự án** (Vinhomes Ocean Park), "
        f"đẩy giá/m^2 lên mức không đại diện cho mặt bằng chung."
    )