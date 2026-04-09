"""
pages/01_Pricing.py — 22127254: Chiến lược Giá, Danh mục & Ma trận BCG
======================================================================
PHÂN CÔNG: 22127254

Mục tiêu 1 — Ngưỡng discount tối ưu theo price_tier
         Heatmap + Box plot (faceted) + Scatter + OLS
Mục tiêu 2 — Top 5 danh mục theo revenue_est + đặc trưng định giá
         Bar ngang + Grouped bar + Bubble chart (22 danh mục)
Mục tiêu 7 — Ma trận chiến lược BCG (22 danh mục) + Lollipop composite score
         Bubble BCG + Lollipop

"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    load_data, get_active, inject_css, setup_sidebar,
    member_badge, conclusion_box, add_trendline,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_GRAY, SEQ_BLUES,
)

inject_css(); setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

st.title("Chiến lược Giá, Danh mục & Ma trận BCG")
st.caption("products.csv - 20,658 sản phẩm - crawl 18/3/2026 - 22127254")
member_badge("22127254", "MT1 - MT2 - MT7 - ML")

with st.expander("Tổng hợp Mục tiêu SMART (22127254)", expanded=False):
    st.markdown("""
| MT | Mục tiêu SMART | Biểu đồ |
|---|---|---|
| **1** | Sử dụng `discount_pct`, `sold`, `price_tier` của 20,658 SP crawl 18/3/2026 để xác định khoảng `discount_pct` có trung vị `sold` cao nhất trong từng `price_tier` (budget -> luxury), đề xuất ngưỡng chiết khấu tối ưu cho mỗi phân khúc, hoàn thành trong phạm vi phân tích snapshot này | Heatmap + Box plot + Scatter+OLS |
| **2** | Tính tổng `revenue_est` và trung bình `price`, `discount_pct`, `sold` của từng `sub_category` trong bộ dữ liệu 18/3/2026 để xác định top 5 danh mục doanh thu cao nhất và mô tả đặc trưng định giá bằng >= 2 loại biểu đồ khác nhau | Bar ngang + Grouped bar + Bubble |
| **7** | So sánh trung bình `sold`, `rating`, `revenue_est` của 22 `sub_category` trong snapshot 18/3/2026 để xác định top 5 danh mục tiềm năng nhất (doanh thu cao VÀ lượng bán cao), trực quan hóa bằng bubble chart kết hợp lollipop chart | Bubble BCG + Lollipop |
| **ML** | Random Forest phân loại sản phẩm bán chạy (`sold > median`) từ `price_tier`, `sub_category`, `discount_pct`, `is_mall` -> Feature importance | Confusion matrix + Feature importance bar |
""")
st.markdown("---")

# === Shared prep ===
bins = [0,10,20,30,40,50,100]
labels = ["0–10%","10–20%","20–30%","30–40%","40–50%",">50%"]
work = active[active["discount_pct"] >= 0].copy()
work["discount_bin"] = pd.cut(work["discount_pct"], bins=bins, labels=labels, right=False)
t_ord = [t for t in ["budget","mid-low","mid","premium","luxury"] if t in work["price_tier"].unique()]

cat_agg = (
    active.groupby("sub_category")
    .agg(revenue_total=("revenue_est","sum"), price_mean=("price","mean"),
         discount_mean=("discount_pct","mean"), sold_mean=("sold","mean"),
         n_products=("item_id","count"))
    .reset_index().sort_values("revenue_total", ascending=False)
)
cat_agg["rev_B"] = cat_agg["revenue_total"] / 1e9
top5 = cat_agg.head(5)

# ======
# MT1 — Ngưỡng discount tối ưu
# ======
st.markdown("## Mục tiêu 1 — Ngưỡng chiết khấu tối ưu theo phân khúc giá")
member_badge("22127254", "MT1")

selected_tiers = st.multiselect(
    "Lọc phân khúc giá (tất cả nếu để trống):",
    options=t_ord,
    default=[]
)
if selected_tiers:
    work_filtered = work[work["price_tier"].isin(selected_tiers)]
    t_ord_filtered = [t for t in t_ord if t in selected_tiers]
else:
    work_filtered = work
    t_ord_filtered = t_ord
tcmap = dict(zip(t_ord_filtered, [CB_BLUE,CB_SKYBLUE,CB_GREEN,CB_ORANGE,CB_VERMIL]))

pivot = (work_filtered.groupby(["price_tier","discount_bin"])["sold"]
         .median().unstack(fill_value=0).reindex(columns=labels).reindex(t_ord_filtered))

st.subheader("Ngữ cảnh: Phân bố giá theo phân khúc")
fig_ctx = px.histogram(
    work_filtered, x="price", color="price_tier",
    nbins=80, log_y=True,
    color_discrete_map=tcmap,
    category_orders={"price_tier": t_ord_filtered},
    template="plotly_white",
    labels={"price": "Giá (VND)", "price_tier": "Phân khúc"},
    title="Phân bố giá sản phẩm theo phân khúc (log Y)"
)
fig_ctx.update_layout(barmode="overlay")
fig_ctx.update_traces(opacity=0.6)
st.plotly_chart(fig_ctx, use_container_width=True)
st.caption("Cung cấp ngữ cảnh cho MT1: ranh giới giá thực tế giữa các tier.")

st.subheader("Biểu đồ 1a: Heatmap — Trung vị sold theo price_tier x discount_bin")
fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                title="Trung vị lượng bán theo phân khúc giá x khoảng chiết khấu",
                labels={"color":"Median sold","x":"Khoảng chiết khấu","y":"Phân khúc giá"})
for tier in t_ord_filtered:
    if tier not in pivot.index:
        continue
    best_col = pivot.loc[tier].idxmax()
    best_val = pivot.loc[tier].max()
    col_idx = list(pivot.columns).index(best_col)
    row_idx = list(pivot.index).index(tier)
    fig.add_annotation(
        x=col_idx, y=row_idx,
        text="O",
        showarrow=False,
        font=dict(size=50, color="#FFD700"),
        xref="x", yref="y"
    )
fig.update_layout(plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)
st.caption("Ô màu đậm hơn = trung vị sold cao hơn. Đọc từng hàng để tìm khoảng discount tối ưu mỗi tier.")

st.subheader("Biểu đồ 1b: Box plot — Phân bố sold theo discount_bin (từng tier)")
box_s = work_filtered[work_filtered["sold"] < work_filtered["sold"].quantile(0.97)].copy()
if not box_s.empty:
    DISCOUNT_ORDER = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", ">50%"]
    fig_box = px.box(box_s, x="discount_bin", y="sold", log_y=True, color="price_tier",
                     facet_col="price_tier", facet_col_wrap=3, points=False,
                     color_discrete_map=tcmap, template="plotly_white",
                     category_orders={"discount_bin": DISCOUNT_ORDER},
                     labels={"discount_bin":"Khoảng chiết khấu","sold":"Số đã bán"},
                     title="Phân bố lượng bán theo khoảng chiết khấu — theo price_tier (bỏ 3% outlier)")
    fig_box.update_layout(showlegend=False)
    fig_box.update_yaxes(range=[0, 4])  # log10 range: 1 to 10,000
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Hộp = IQR (Q1–Q3). Đường giữa = trung vị. Whisker = 1.5xIQR. Trục Y: log scale.")

st.subheader("Biểu đồ 1c: Scatter — Discount (%) vs Lượng bán (log Y)")
samp = work_filtered.dropna(subset=["discount_pct","sold","price_tier"]).sample(min(3000,len(work_filtered)), random_state=42)
fig_sc = px.scatter(samp, x="discount_pct", y="sold", color="price_tier",
                    log_y=True, opacity=0.45, color_discrete_map=tcmap, template="plotly_white",
                    labels={"discount_pct":"% Chiết khấu","sold":"Số đã bán (log)","price_tier":"Phân khúc"},
                    title="Mối quan hệ Chiết khấu vs Lượng bán — màu theo price_tier (log Y, sample 3,000)",
                    hover_name="name" if "name" in samp.columns else None)
for tier, grp in samp.groupby("price_tier"):
    if len(grp) < 10:
        continue
    xs = grp["discount_pct"].values.astype(float)
    ys = np.log10(grp["sold"].clip(1).values.astype(float))
    mask = np.isfinite(xs) & np.isfinite(ys)
    if mask.sum() < 5:
        continue
    z = np.polyfit(xs[mask], ys[mask], 1)
    xl = np.linspace(xs[mask].min(), xs[mask].max(), 80)
    yl = 10 ** np.polyval(z, xl)
    fig_sc.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines", name=f"Xu hướng {tier}",
        line=dict(color=tcmap.get(tier, CB_GRAY), width=1.5, dash="dash"),
        showlegend=True,
    ))
fig_sc.update_layout(plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=1))
st.plotly_chart(fig_sc, use_container_width=True)
st.caption(
    "Đường chấm = xu hướng OLS tổng thể. "
    "Đường đứt = xu hướng riêng từng tier. "
    "Premium/luxury: slope gần phẳng hoặc âm -> discount không tăng sold. "
    "Log scale trục Y."
)

st.markdown("#### Tổng hợp ngưỡng discount tối ưu từng phân khúc")
if not pivot.empty:
    opt = [{"Phân khúc":t,"Ngưỡng tối ưu":str(pivot.loc[t].idxmax()),
            "Trung vị sold cao nhất":int(pivot.loc[t].max())}
           for t in t_ord_filtered if t in pivot.index]
    st.dataframe(pd.DataFrame(opt), hide_index=True, use_container_width=True)

conclusion_box("""
<b>Nhận xét MT1:</b><br>
- <b>premium (306)</b> và <b>luxury (220)</b> có trung vị sold cao nhất khi discount <b>0–10%</b> — counterintuitive: giảm giá ít bán tốt hơn, vì giảm sâu làm mất hình ảnh cao cấp.<br>
- <b>mid (211)</b>: tối ưu tại 30–40% — cần mức giảm đủ lớn để khách cảm nhận giá trị.<br>
- <b>mid-low (173)</b>: tối ưu 10–20% — nhạy cảm giá nhưng không cần discount quá sâu.<br>
- <b>budget (133)</b>: tối ưu >50% nhưng trung vị vẫn thấp nhất — discount sâu giúp nhưng cạnh tranh quá cao.<br>
- <b>Scatter OLS</b>: Đường xu hướng gần phẳng — discount một mình không đủ, price_tier mới là biến phân biệt chính.<br>
- <b>Chiến lược:</b> Phân tầng discount theo tier thay vì áp đồng đều toàn kho.
""")
st.markdown("---")

# ======
# MT2 — Top 5 danh mục theo doanh thu
# ======
st.markdown("## MT2 — Top 5 danh mục theo doanh thu ước tính")
member_badge("22127254", "MT2")

# Biểu đồ 2a: Bar ngang
st.subheader("Biểu đồ 2a: Bar chart ngang — Top 5 danh mục x Doanh thu")

col_a, col_b = st.columns([3, 2])
with col_a:
    fig_rv = px.bar(top5.sort_values("rev_B"), x="rev_B", y="sub_category",
                    orientation="h", text="rev_B",
                    color="rev_B", color_continuous_scale=SEQ_BLUES,
                    labels={"rev_B":"Doanh thu ước tính (Tỷ VND)","sub_category":""},
                    title="Top 5 danh mục — Tổng doanh thu ước tính")
    fig_rv.update_traces(texttemplate="%{text:.0f}B", textposition="outside")
    fig_rv.update_layout(coloraxis_showscale=False,
                         yaxis={"categoryorder":"total ascending"}, plot_bgcolor="white")
    st.plotly_chart(fig_rv, use_container_width=True)
    st.caption("revenue_est = price x sold (ước tính). Chỉ dùng để so sánh tương đối.")
with col_b:
    disp = top5[["sub_category","rev_B","price_mean","discount_mean","sold_mean","n_products"]].copy()
    disp.columns = ["Danh mục","DT (Tỷ đồng)","Giá TB","Giảm TB%","Bán TB","Số SP"]
    disp["DT (Tỷ đồng)"]  = disp["DT (Tỷ đồng)"].apply(lambda x: f"{x:.0f}B")
    disp["Giá TB"]     = disp["Giá TB"].apply(lambda x: f"{x:,.0f}")
    disp["Giảm TB%"]   = disp["Giảm TB%"].apply(lambda x: f"{x:.1f}%")
    disp["Bán TB"]     = disp["Bán TB"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)

# Biểu đồ 2b: Grouped bar đặc trưng định giá
st.subheader("Biểu đồ 2b: Grouped bar — Giá TB và Lượng bán TB (Top 5 danh mục)")

top5_gb = top5[["sub_category","price_mean","sold_mean"]].melt(
    id_vars="sub_category", var_name="Chỉ số", value_name="Giá trị"
)
top5_gb["Chỉ số"] = top5_gb["Chỉ số"].map({
    "price_mean": "Giá TB (VND)",
    "sold_mean": "Lượng bán TB"
})
fig_grouped = px.bar(
    top5_gb, x="sub_category", y="Giá trị",
    color="Chỉ số", barmode="group",
    template="plotly_white",
    labels={"sub_category": "Danh mục", "Giá trị": "Giá trị"},
    title="Giá TB và Lượng bán TB — Top 5 danh mục"
)
fig_grouped.update_layout(xaxis_tickangle=-20)
st.plotly_chart(fig_grouped, use_container_width=True)
st.caption("Grouped bar: mỗi nhóm = 1 danh mục, 2 thanh = Giá TB và Lượng bán TB.")

st.subheader("Biểu đồ 2c: Hai bar chart — Đặc trưng định giá Top 5 danh mục")
col_c, col_d = st.columns(2)
with col_c:
    fp = px.bar(top5, x="sub_category", y="price_mean", color="sub_category",
                text="price_mean", template="plotly_white",
                labels={"price_mean":"Giá TB (VND)","sub_category":""},
                title="Giá trung bình — Top 5 danh mục")
    fp.update_traces(texttemplate="%{text:,.0f}", textposition="outside", showlegend=False)
    fp.update_layout(xaxis_tickangle=-20, showlegend=False)
    st.plotly_chart(fp, use_container_width=True)
with col_d:
    fd = px.bar(top5, x="sub_category", y="discount_mean", color="sub_category",
                text="discount_mean", template="plotly_white",
                labels={"discount_mean":"Giảm giá TB (%)","sub_category":""},
                title="Mức giảm giá TB — Top 5 danh mục")
    fd.update_traces(texttemplate="%{text:.1f}%", textposition="outside", showlegend=False)
    fd.update_layout(xaxis_tickangle=-20, showlegend=False)
    st.plotly_chart(fd, use_container_width=True)

# Biểu đồ 2d: Bubble chart 22 danh mục
st.subheader("Biểu đồ 2d: Bubble chart — Giá TB x Sold TB (tất cả 22 danh mục)")

fig_bub = px.scatter(
    cat_agg, x="price_mean", y="sold_mean", size="rev_B", color="rev_B",
    hover_name="sub_category", text="sub_category",
    color_continuous_scale="YlOrRd", log_x=True, log_y=True, size_max=60,
    template="plotly_white",
    labels={"price_mean":"Giá TB (VND, log)","sold_mean":"Lượng bán TB (log)","rev_B":"DT (Tỷ đồng)"},
    title="22 danh mục: Giá TB x Lượng bán TB - kích thước doanh thu ước tính",
)
fig_bub.update_traces(textposition="top center", textfont_size=8,
                      marker=dict(opacity=0.8, line=dict(width=1, color="white")))
fig_bub.update_layout(
    plot_bgcolor="white",
    xaxis=dict(
        title="Giá TB (VND)",
        tickvals=[10000, 50000, 100000, 500000, 1000000],
        ticktext=["10k", "50k", "100k", "500k", "1M"]
    ),
    yaxis=dict(
        title="Lượng bán TB",
        tickvals=[10, 50, 100, 500, 1000, 5000, 10000],
        ticktext=["10", "50", "100", "500", "1k", "5k", "10k"]
    )
)
st.plotly_chart(fig_bub, use_container_width=True)
st.caption("Bong bóng lớn = doanh thu cao (YlOrRd: vàng→đỏ). Log scale cả 2 trục.")

conclusion_box("""
<b> Nhận xét MT2:</b><br>
- <b>Top 5:</b> mặt nạ (700B) > sữa rửa mặt (438B) > kem dưỡng ẩm (374B) > kem chống nắng (336B) > dầu gội (325B).<br>
- <b>mặt nạ dẫn đầu nhờ volume:</b> giá TB thấp nhất (68,822đ) nhưng sold TB cao nhất (5,280). Tách biệt hoàn toàn trên bubble chart.<br>
- <b>kem dưỡng ẩm dẫn đầu nhờ giá cao:</b> 229,004đ TB (gấp 3.3x mặt nạ), dù sold TB chỉ 1,815.<br>
- <b>Khoảng cách #1 vs #5:</b> 700B/325B = 2.15x — top 5 tương đối đồng đều, không có danh mục thống trị tuyệt đối.
""")
st.markdown("---")

# ======
# MT7 — Ma trận chiến lược BCG + Lollipop
# ======
st.markdown("## MT7 — Ma trận Chiến lược Danh mục (BCG-style) & Lollipop Tiềm năng")
member_badge("22127254", "MT7")

st.subheader("Biểu đồ 3a: Bubble chart BCG — 22 danh mục - 4 ô chiến lược")
st.caption("Trục X = Lượng bán TB - Trục Y = Doanh thu tổng (Tỷ VND) - Kích thước = Số SP - Màu = Giá TB (Viridis)")

cat_plot = cat_agg[cat_agg["n_products"] >= 20].copy()
cat_plot["avp_k"] = cat_plot["price_mean"] / 1000
ms, mr = cat_plot["sold_mean"].median(), cat_plot["rev_B"].median()

fig_bcg = px.scatter(
    cat_plot, x="sold_mean", y="rev_B", size="n_products", color="avp_k",
    text="sub_category", hover_name="sub_category",
    color_continuous_scale="Viridis", size_max=55, template="plotly_white",
    labels={"sold_mean":"Lượng bán TB","rev_B":"Tổng DT ước tính (Tỷ VND)",
            "n_products":"Số SP","avp_k":"Giá TB (nghìn đồng)"},
    title="Bản đồ chiến lược 22 danh mục mỹ phẩm Shopee VN (18/3/2026)")
fig_bcg.update_traces(textposition="top center", textfont_size=8)
fig_bcg.add_hline(y=mr, line_dash="dot", line_color=CB_GRAY, opacity=0.7,
                  annotation_text="Median DT", annotation_position="right")
fig_bcg.add_vline(x=ms, line_dash="dot", line_color=CB_GRAY, opacity=0.7,
                  annotation_text="Median sold", annotation_position="top")
xm, ym = cat_plot["sold_mean"].max(), cat_plot["rev_B"].max()
quadrants = {"NGÔI SAO": 0, "SINH LỜI": 0, "PHỄU": 0, "CÂU HỎI": 0}
for _, row in cat_plot.iterrows():
    if row["sold_mean"] >= ms and row["rev_B"] >= mr:
        quadrants["NGÔI SAO"] += 1
    elif row["sold_mean"] < ms and row["rev_B"] >= mr:
        quadrants["SINH LỜI"] += 1
    elif row["sold_mean"] >= ms and row["rev_B"] < mr:
        quadrants["PHỄU"] += 1
    else:
        quadrants["CÂU HỎI"] += 1

for lbl, x, y, bg, bc in [
    (f"NGÔI SAO\n(nhiều, DT cao): {quadrants['NGÔI SAO']} danh mục", xm*.80, ym*.90, "rgba(255,215,0,.18)", CB_ORANGE),
    (f"SINH LỜI\n(ít, DT cao): {quadrants['SINH LỜI']} danh mục", ms*.18, ym*.90, "rgba(86,180,233,.18)", CB_SKYBLUE),
    (f"PHỄU\n(nhiều, DT thấp): {quadrants['PHỄU']} danh mục", xm*.80, mr*(-0.5), "rgba(0,158,115,.18)", CB_GREEN),
    (f"CÂU HỎI\n(ít, DT thấp): {quadrants['CÂU HỎI']} danh mục", ms*.18, mr*(-0.5), "rgba(153,153,153,.18)", CB_GRAY),
]:
    fig_bcg.add_annotation(x=x, y=y, text=lbl, showarrow=False,
                            bgcolor=bg, bordercolor=bc, borderwidth=1, font=dict(size=9))
fig_bcg.update_layout(plot_bgcolor="white", height=580)
st.plotly_chart(fig_bcg, use_container_width=True)
st.caption("4 ô chiến lược tương tự BCG Matrix. Viridis = perceptually uniform + colorblind-safe.")

st.subheader("Biểu đồ 3b: Lollipop chart — Điểm tiềm năng tổng hợp (22 danh mục)")
for c in ["rev_B","sold_mean"]:
    mn, mx = cat_agg[c].min(), cat_agg[c].max()
    cat_agg[c+"_n"] = (cat_agg[c]-mn)/(mx-mn) if mx>mn else 0.0
cat_agg["composite"] = (cat_agg["rev_B_n"] + cat_agg["sold_mean_n"]) / 2
top5nm = set(cat_agg.nlargest(5,"composite")["sub_category"])
lol = cat_agg.sort_values("composite").reset_index(drop=True)
clrs = [CB_VERMIL if n in top5nm else CB_GRAY for n in lol["sub_category"]]

fig_lol = go.Figure()
for i, row in lol.iterrows():
    fig_lol.add_shape(type="line", x0=0, x1=row["composite"], y0=i, y1=i,
                      line=dict(color=CB_GRAY, width=1.5))
fig_lol.add_trace(go.Scatter(
    x=lol["composite"], y=list(range(len(lol))), mode="markers+text",
    marker=dict(size=13, color=clrs),
    text=[f" {v:.3f}" for v in lol["composite"]],
    textposition="middle right", showlegend=False))
fig_lol.update_layout(
    title="Điểm tiềm năng tổng hợp (Composite = avg rank %DT + rank %Sold)",
    xaxis_title="Composite Score (0–1)", plot_bgcolor="white", height=580,
    margin=dict(l=160, r=120),
    yaxis=dict(tickvals=list(range(len(lol))), ticktext=lol["sub_category"].tolist()))
fig_lol.add_vline(x=0.5, line_dash="dash", line_color=CB_SKYBLUE, opacity=0.6,
                  annotation_text="Ngưỡng 0.5")
st.plotly_chart(fig_lol, use_container_width=True)
st.caption("Lollipop = biến thể bar chart giảm ink-to-data ratio (nguyên tắc Tufte). Cam = Top 5.")

conclusion_box("""
<b> Nhận xét MT7:</b><br>
- <b>Ngôi sao — mặt nạ</b> (Composite=1.000): Outlier tuyệt đối. Chỉ có 2 danh mục vượt ngưỡng 0.5.<br>
- <b>Ngôi sao — sữa rửa mặt</b> (0.614): Lựa chọn an toàn thứ 2 để gia nhập thị trường.<br>
- <b>Sinh lời:</b> kem dưỡng ẩm, serum, toner — margin tốt, phù hợp người bán có sản phẩm chất lượng cao.<br>
- <b>Phễu:</b> dầu gội, kem chống nắng, sữa tắm — volume cao, dùng để xây reviews và followers.<br>
- <b>Hair care:</b> tất cả composite < 0.1 -> tránh nếu mới gia nhập.
""")
st.markdown("---")

# === ML ===
st.markdown("## Machine Learning — Random Forest: Phân loại Sản phẩm Bán chạy")
member_badge("22127254", "ML")

with st.expander("Tại sao Random Forest?", expanded=False):
    st.markdown("""
    - **Bổ sung MT1 & 2:** RF học đồng thời tác động của `price_tier`, `sub_category`, `discount_pct`, `is_mall` thay vì xét từng biến riêng
    - **Feature importance:** Xác nhận hoặc bác bỏ các nhận định từ phân tích đơn biến
    - **Target:** `is_top_seller` = 1 nếu `sold > median sold` (sản phẩm bán chạy hơn mức trung bình)
    """)

@st.cache_data
def run_rf(df):
    sub = df.dropna(subset=["price_tier","sub_category","discount_pct","is_mall","sold"]).copy()
    med = sub["sold"].median()
    sub["target"] = (sub["sold"] > med).astype(int)
    le1 = LabelEncoder(); sub["pt_e"] = le1.fit_transform(sub["price_tier"])
    le2 = LabelEncoder(); sub["cat_e"] = le2.fit_transform(sub["sub_category"])
    X = sub[["pt_e","cat_e","discount_pct","is_mall"]].values
    y = sub["target"].values
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    cv_scores = cross_val_score(clf, X, y, cv=5, scoring="accuracy", n_jobs=-1)
    yp = clf.predict(Xte)
    cm = confusion_matrix(yte, yp)
    acc = (yte == yp).mean()
    fi = pd.DataFrame({
        "Feature": ["price_tier","sub_category","discount_pct","is_mall"],
        "Importance": clf.feature_importances_
    }).sort_values("Importance")
    report_dict = classification_report(yte, yp, 
                                        target_names=["Không bán chạy","Bán chạy"],
                                        output_dict=True)
    report_df = pd.DataFrame(report_dict).T.round(3)
    return cm, fi, acc, med, cv_scores, report_df

cm_rf, fi_rf, acc_rf, sold_med, cv_scores, report_df_rf = run_rf(active)

col_r1, col_r2 = st.columns(2)
with col_r1:
    fig_cm = px.imshow(cm_rf, text_auto=True, color_continuous_scale="Blues",
                       x=["Không bán chạy","Bán chạy"], y=["Không bán chạy","Bán chạy"],
                       title=f"Ma trận nhầm lẫn (Accuracy = {acc_rf*100:.1f}%)",
                       labels={"x":"Dự đoán","y":"Thực tế"})
    st.plotly_chart(fig_cm, use_container_width=True)
with col_r2:
    fi_colors = [CB_VERMIL if v==fi_rf["Importance"].max() else CB_BLUE for v in fi_rf["Importance"]]
    fig_fi = go.Figure()
    fig_fi.add_bar(x=fi_rf["Importance"], y=fi_rf["Feature"], orientation="h",
                   marker_color=fi_colors,
                   text=[f"{v:.3f}" for v in fi_rf["Importance"]], textposition="outside")
    fig_fi.update_layout(plot_bgcolor="white", title="Feature Importance — Random Forest",
                         xaxis_title="Importance (Gini)")
    st.plotly_chart(fig_fi, use_container_width=True)
st.caption(f"Target: sold > {sold_med:.0f} (median). 200 trees, max_depth=8. "
           f"Màu cam = feature quan trọng nhất.")

best_feat = fi_rf.iloc[-1]["Feature"]
best_imp = fi_rf.iloc[-1]["Importance"]
disc_imp = fi_rf[fi_rf["Feature"]=="discount_pct"]["Importance"].values[0]

st.markdown("#### Báo cáo phân loại chi tiết")
st.dataframe(report_df_rf.loc[["Không bán chạy","Bán chạy","macro avg","weighted avg"],
                               ["precision","recall","f1-score","support"]],
             use_container_width=True)
st.caption(f"Baseline (random): 50%. Model: {acc_rf*100:.1f}% -> cải thiện ~{(acc_rf-0.5)*100:.1f} điểm phần trăm so với baseline.")
st.markdown(f"**5-fold Cross-Validation Accuracy:** {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%")

conclusion_box(f"""
<b> Kết quả ML — Random Forest (Accuracy = {acc_rf*100:.1f}%):</b><br>
- <b>Feature quan trọng nhất: {best_feat}</b> (importance = {best_imp:.3f}) — nhất quán với MT1 & 2: danh mục sản phẩm và phân khúc giá là yếu tố quyết định chính.<br>
- <b>discount_pct</b> (importance = {disc_imp:.3f}) — vai trò vừa phải, xác nhận MT1: discount một mình không đủ.<br>
- <b>is_mall</b> có importance thấp nhất — nhất quán với MT4 (22127418): Mall không phải yếu tố phân biệt chính.<br>
- <b>Kết nối với phân tích đơn biến:</b> sub_category importance ~ 0.60 xác nhận MT2 (danh mục quyết định doanh thu), discount_pct ~ 0.29 xác nhận MT1 (discount có vai trò nhưng không phải yếu tố chính), is_mall thấp nhất (0.036) xác nhận MT4 (Mall không phải yếu tố phân biệt).
- Accuracy {acc_rf*100:.1f}% trên bài toán 2 lớp cân bằng là khả quan; giới hạn: không có đặc trưng thời gian, marketing, hay hình ảnh sản phẩm.
""")
