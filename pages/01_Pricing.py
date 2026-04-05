"""
pages/01_Pricing.py — 22127254: Chiến lược Giá, Danh mục & Ma trận BCG
======================================================================
PHÂN CÔNG: 22127254

Mục tiêu 1 — Ngưỡng discount tối ưu theo price_tier
         Heatmap + Box plot (faceted) + Scatter + OLS
Mục tiêu 2 — Top 5 danh mục theo revenue_est + đặc trưng định giá
         Bar ngang + Grouped bar + Bubble chart (22 danh mục)
Mục tiêu 8 — Ma trận chiến lược BCG (22 danh mục) + Lollipop composite score
         Bubble BCG + Lollipop

"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    load_data, get_active, setup_sidebar, add_trendline,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_GRAY, CB_PURPLE,
    SEQ_BLUES,
)

st.set_page_config(page_title="Giá, Danh mục & BCG", layout="wide")
setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

st.title("Chiến lược Giá, Danh mục & Ma trận BCG")
st.caption("products.csv - 20,658 sản phẩm - crawl 18/3/2026 - 22127254")

with st.expander("Tổng hợp Mục tiêu SMART (22127254)", expanded=False):
    st.markdown("""
    | MT | Mục tiêu SMART | Biểu đồ |
    |-----|----------------|---------|
    | **1** | Sử dụng `discount_pct`, `sold`, `price_tier` của 20,658 SP crawl 18/3/2026 để xác định khoảng `discount_pct` có **trung vị sold cao nhất** trong từng `price_tier` (budget->luxury), đề xuất ngưỡng chiết khấu tối ưu mỗi phân khúc | Heatmap + Box plot + Scatter+OLS |
    | **2** | Tính tổng `revenue_est`, trung bình `price`, `discount_pct`, `sold` của 22 `sub_category` từ snapshot 18/3/2026 để xác định top 5 danh mục doanh thu cao nhất và mô tả đặc trưng định giá bằng >= 2 loại biểu đồ | Bar ngang + Grouped bar + Bubble |
    | **8** | Phân loại 22 `sub_category` vào 4 ô chiến lược BCG dựa trên `avg_sold` x `total_revenue`, xếp hạng tiềm năng tổng hợp bằng lollipop chart | Bubble BCG + Lollipop |
    """)

st.markdown("---")

# === Shared prep ===
bins   = [0, 10, 20, 30, 40, 50, 100]
labels = ["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", ">50%"]
work   = active[active["discount_pct"] >= 0].copy()
work["discount_bin"] = pd.cut(work["discount_pct"], bins=bins, labels=labels, right=False)
tier_order  = [t for t in ["budget","mid-low","mid","premium","luxury"] if t in work["price_tier"].unique()]
tier_colors = [CB_BLUE, CB_SKYBLUE, CB_GREEN, CB_ORANGE, CB_VERMIL]
tier_cmap   = {t: tier_colors[i] for i, t in enumerate(tier_order)}

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

# Biểu đồ 1a: Heatmap
st.subheader("Biểu đồ 1a: Heatmap — Trung vị sold theo price_tier x discount_bin")

pivot = (
    work.groupby(["price_tier","discount_bin"])["sold"]
    .median().unstack(fill_value=0).reindex(tier_order)
)
fig = px.imshow(pivot, text_auto=".0f", color_continuous_scale="Blues", aspect="auto",
                title="Trung vị lượng bán theo phân khúc giá x khoảng chiết khấu",
                labels={"color":"Median sold","x":"Khoảng chiết khấu","y":"Phân khúc giá"})
fig.update_layout(plot_bgcolor="white")
st.plotly_chart(fig, use_container_width=True)
st.caption("Ô màu đậm hơn = trung vị sold cao hơn. Đọc từng hàng để tìm khoảng discount tối ưu mỗi tier.")

# Biểu đồ 1b: Box plot
st.subheader("Biểu đồ 1b: Box plot — Phân bố sold theo discount_bin (từng tier)")

sample_box = work[work["sold"] < work["sold"].quantile(0.97)].copy()
if not sample_box.empty:
    fig_box = px.box(
        sample_box, x="discount_bin", y="sold",
        color="price_tier", facet_col="price_tier", facet_col_wrap=3,
        points=False, color_discrete_map=tier_cmap, template="plotly_white",
        labels={"discount_bin":"Khoảng chiết khấu","sold":"Số đã bán","price_tier":"Phân khúc"},
        title="Phân bố lượng bán theo khoảng chiết khấu — theo price_tier (bỏ 3% outlier cao nhất)",
    )
    fig_box.update_layout(showlegend=False)
    st.plotly_chart(fig_box, use_container_width=True)
    st.caption("Hộp = IQR (Q1–Q3). Đường giữa = trung vị. Whisker = 1.5xIQR. "
               "Box plot bổ sung heatmap — thể hiện mức độ biến động trong mỗi ô.")

# Biểu đồ 1c: Scatter + OLS
st.subheader("Biểu đồ 1c: Scatter — Discount (%) vs Lượng bán (log Y)")

samp_sc = work.dropna(subset=["discount_pct","sold","price_tier"]).sample(
    min(3000, len(work)), random_state=42)
fig_sc = px.scatter(
    samp_sc, x="discount_pct", y="sold", color="price_tier",
    log_y=True, opacity=0.45, color_discrete_map=tier_cmap, template="plotly_white",
    labels={"discount_pct":"% Chiết khấu","sold":"Số đã bán (log)","price_tier":"Phân khúc"},
    title="Mối quan hệ Chiết khấu vs Lượng bán — màu theo price_tier (log Y, sample 3,000)",
    hover_name="name" if "name" in samp_sc.columns else None,
)
fig_sc = add_trendline(fig_sc, samp_sc["discount_pct"], samp_sc["sold"], log_y=True)
fig_sc.update_layout(plot_bgcolor="white", legend=dict(orientation="h", yanchor="bottom", y=1))
st.plotly_chart(fig_sc, use_container_width=True)
st.caption("Đường chấm đỏ = xu hướng OLS tổng thể. Mỗi điểm = 1 sản phẩm. "
           "Log scale trục Y giúp thấy rõ phân bố khi sold trải rộng từ 1 đến 100,000+.")

# Bảng tổng hợp
st.markdown("#### Tổng hợp ngưỡng discount tối ưu từng phân khúc")
if not pivot.empty:
    opt_rows = []
    for tier in tier_order:
        if tier in pivot.index:
            best_bin = pivot.loc[tier].idxmax()
            best_val = int(pivot.loc[tier, best_bin])
            opt_rows.append({"Phân khúc": tier,
                             "Ngưỡng discount tối ưu": str(best_bin),
                             "Trung vị sold cao nhất": best_val})
    st.dataframe(pd.DataFrame(opt_rows), hide_index=True, use_container_width=True)

st.markdown("""
**Nhận xét MT1:**

- **budget:** Khoảng **>50%** có trung vị sold cao nhất (133). Sản phẩm giá rẻ cần chiết khấu sâu để thu hút trong thị trường cạnh tranh cao. Tuy nhiên, trung vị vẫn thấp (133) so với các phân khúc khác -> discount giúp nhưng không đủ để bứt phá.

- **mid-low:** Khoảng **10–20%** có trung vị sold cao nhất (173). Người mua phân khúc này nhạy cảm với giảm giá vừa phải — không cần discount quá sâu, tiết kiệm được margin so với budget.

- **mid:** Khoảng **30–40%** có trung vị sold cao nhất (211). Phân khúc tầm trung cần mức giảm đáng kể để thuyết phục khách hàng so sánh — đây là "sweet spot" để kích cầu mà không làm loãng giá trị thương hiệu.

- **premium:** Khoảng **0–10%** có trung vị sold **cao nhất toàn bảng (306)** — counterintuitive. Premium bán tốt nhất khi KHÔNG giảm giá nhiều. Giảm sâu có thể làm mất hình ảnh cao cấp và gây nghi ngờ về chất lượng.

- **luxury:** Khoảng **0–10%** có trung vị sold 220 (chỉ 2 bin có dữ liệu do số lượng sản phẩm rất ít — 28 SP). Nhất quán với premium: luxury không cần và không nên dùng discount sâu.

- **Từ scatter (xu hướng OLS):** Đường xu hướng gần như phẳng với độ dốc nhỏ — discount một mình không phải yếu tố quyết định sold. Yếu tố phân khúc (price_tier) mới là biến phân loại quan trọng hơn.

- **Chiến lược đề xuất:** Người bán không nên áp dụng discount đồng đều — cần phân tầng theo price_tier: budget (>50%), mid-low (10–20%), mid (30–40%), premium/luxury (<=10%).
""")

st.markdown("---")

# ======
# MT2 — Top 5 danh mục theo doanh thu
# ======
st.markdown("## MT2 — Top 5 danh mục theo doanh thu ước tính")

# Biểu đồ 2a: Bar ngang
st.subheader("Biểu đồ 2a: Bar chart ngang — Top 5 danh mục x Doanh thu")

col_a, col_b = st.columns([3, 2])
with col_a:
    fig_rev = px.bar(
        top5.sort_values("rev_B"), x="rev_B", y="sub_category", orientation="h",
        text="rev_B", color="rev_B", color_continuous_scale=SEQ_BLUES,
        labels={"rev_B":"Doanh thu ước tính (Tỷ VND)","sub_category":""},
        title="Top 5 danh mục — Tổng doanh thu ước tính (revenue = price x sold)",
    )
    fig_rev.update_traces(texttemplate="%{text:.0f}B", textposition="outside")
    fig_rev.update_layout(coloraxis_showscale=False,
                          yaxis={"categoryorder":"total ascending"}, plot_bgcolor="white")
    st.plotly_chart(fig_rev, use_container_width=True)
    st.caption("revenue_est = price x sold (ước tính gộp). Chỉ dùng để so sánh tương đối.")

with col_b:
    st.markdown("**Bảng so sánh Top 5:**")
    disp = top5[["sub_category","rev_B","price_mean","discount_mean","sold_mean","n_products"]].copy()
    disp.columns = ["Danh mục","DT (Tỷ đồng)","Giá TB","Giảm TB%","Bán TB","Số SP"]
    disp["DT (Tỷ đồng)"]  = disp["DT (Tỷ đồng)"].apply(lambda x: f"{x:.0f}B")
    disp["Giá TB"]     = disp["Giá TB"].apply(lambda x: f"{x:,.0f}")
    disp["Giảm TB%"]   = disp["Giảm TB%"].apply(lambda x: f"{x:.1f}%")
    disp["Bán TB"]     = disp["Bán TB"].apply(lambda x: f"{x:,.0f}")
    st.dataframe(disp, hide_index=True, use_container_width=True)

# Biểu đồ 2b: Grouped bar đặc trưng định giá
st.subheader("Biểu đồ 2b: Grouped bar — Đặc trưng định giá Top 5 danh mục")

col_c, col_d = st.columns(2)
with col_c:
    fig_p = px.bar(top5, x="sub_category", y="price_mean", color="sub_category",
                   text="price_mean", template="plotly_white",
                   labels={"price_mean":"Giá trung bình (VND)","sub_category":""},
                   title="Giá trung bình — Top 5 danh mục")
    fig_p.update_traces(texttemplate="%{text:,.0f}", textposition="outside", showlegend=False)
    fig_p.update_layout(xaxis_tickangle=-25, showlegend=False)
    st.plotly_chart(fig_p, use_container_width=True)

with col_d:
    fig_d = px.bar(top5, x="sub_category", y="discount_mean", color="sub_category",
                   text="discount_mean", template="plotly_white",
                   labels={"discount_mean":"Mức giảm giá TB (%)","sub_category":""},
                   title="Mức giảm giá trung bình — Top 5 danh mục")
    fig_d.update_traces(texttemplate="%{text:.1f}%", textposition="outside", showlegend=False)
    fig_d.update_layout(xaxis_tickangle=-25, showlegend=False)
    st.plotly_chart(fig_d, use_container_width=True)

# Biểu đồ 2c: Bubble chart 22 danh mục
st.subheader("Biểu đồ 2c: Bubble chart — Giá TB x Sold TB (tất cả 22 danh mục)")

fig_bub = px.scatter(
    cat_agg, x="price_mean", y="sold_mean",
    size="rev_B", color="rev_B",
    hover_name="sub_category", text="sub_category",
    color_continuous_scale="YlOrRd", log_x=True, log_y=True, size_max=60,
    template="plotly_white",
    labels={"price_mean":"Giá TB (VND, log)","sold_mean":"Lượng bán TB (log)","rev_B":"DT (Tỷ đồng)"},
    title="22 danh mục: Giá TB x Lượng bán TB - kích thước ∝ doanh thu ước tính",
)
fig_bub.update_traces(textposition="top center", textfont_size=8,
                      marker=dict(opacity=0.8, line=dict(width=1, color="white")))
fig_bub.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_bub, use_container_width=True)
st.caption("Bong bóng lớn hơn = doanh thu cao hơn (YlOrRd: vàng -> đỏ). Log scale cả 2 trục. "
           "Bubble chart truyền 4 chiều thông tin đồng thời — nhược điểm: area encoding kém chính xác hơn bar, "
           "dùng để định vị tổng quan.")

st.markdown("""
**Nhận xét MT2:**

- **Top 5 danh mục:** mặt nạ (700B) > sữa rửa mặt (438B) > kem dưỡng ẩm (374B) > kem chống nắng (336B) > dầu gội (325B).

- **mặt nạ dẫn đầu nhờ volume:** Giá TB thấp nhất trong top 5 (68,822đ) nhưng lượng bán TB cao nhất (5,280) — chiến lược giá rẻ, bán cực nhiều. Bong bóng mặt nạ nằm tách biệt ở góc trên của bubble chart, xa hẳn các danh mục còn lại.

- **kem dưỡng ẩm dẫn đầu nhờ giá cao:** Avg price 229,004đ (gấp 3.3x mặt nạ), discount 19.7% — doanh thu cao dù sold TB thấp hơn mặt nạ nhiều (1,815 vs 5,280).

- **kem chống nắng** có mức giảm giá TB cao nhất trong top 5 (22.5%) — cạnh tranh bằng khuyến mãi mạnh để duy trì volume.

- **dầu gội** xuất hiện trong top 5 mặc dù là danh mục chăm sóc tóc — lượng nhu cầu thường xuyên (daily use) tạo ra volume ổn định (sold TB 2,486) bù cho giá thấp (160,139đ).

- **Khoảng cách #1 vs #5:** 700B vs 325B ~ 2.15x — top 5 tương đối đồng đều, không quá chênh lệch, cho thấy không có một danh mục nào "thống trị" tuyệt đối ngoài mặt nạ.

- **Từ bubble chart:** Phần lớn 22 danh mục tập trung ở vùng trung tâm (sold TB 1,000–3,000, giá 100k–300k) — thị trường khá đồng đều ngoài mặt nạ.
""")

st.markdown("---")

# ======
# MT8 — Ma trận chiến lược BCG + Lollipop
# ======
st.markdown("## MT8 — Ma trận Chiến lược Danh mục (BCG-style) & Lollipop Tiềm năng")

# Biểu đồ 3a: BCG bubble
st.subheader("Biểu đồ 3a: Bubble chart BCG — 22 danh mục - 4 ô chiến lược")
st.caption("Trục X = Lượng bán TB - Trục Y = Doanh thu tổng (Tỷ VND)"
           "Kích thước = Số SP - Màu = Giá TB (Viridis — colorblind safe)")

cat_plot = cat_agg[cat_agg["n_products"] >= 20].copy()
cat_plot["avp_k"] = cat_plot["price_mean"] / 1000

fig_bcg = px.scatter(
    cat_plot, x="sold_mean", y="rev_B", size="n_products", color="avp_k",
    text="sub_category", hover_name="sub_category",
    color_continuous_scale="Viridis", size_max=55,
    template="plotly_white",
    labels={"sold_mean":"Lượng bán TB / sản phẩm","rev_B":"Tổng DT ước tính (Tỷ VND)",
            "n_products":"Số SP","avp_k":"Giá TB (nghìn đồng)"},
    title="Bản đồ chiến lược 22 danh mục mỹ phẩm Shopee VN (18/3/2026)",
)
fig_bcg.update_traces(textposition="top center", textfont_size=8)

ms, mr = cat_plot["sold_mean"].median(), cat_plot["rev_B"].median()
fig_bcg.add_hline(y=mr, line_dash="dot", line_color=CB_GRAY, opacity=0.7,
                  annotation_text="Median DT", annotation_position="right")
fig_bcg.add_vline(x=ms, line_dash="dot", line_color=CB_GRAY, opacity=0.7,
                  annotation_text="Median sold", annotation_position="top")

xm, ym = cat_plot["sold_mean"].max(), cat_plot["rev_B"].max()
for label, x, y, bg, bc in [
    ("NGÔI SAO\n(nhiều + DT cao)", xm*0.80, ym*0.90, "rgba(255,215,0,0.2)", CB_ORANGE),
    ("SINH LỜI\n(ít + DT cao)", ms*0.20, ym*0.90, "rgba(86,180,233,0.2)", CB_SKYBLUE),
    ("PHỄU\n(nhiều + DT thấp)", xm*0.80, mr*0.25, "rgba(0,158,115,0.2)", CB_GREEN),
    ("CÂU HỎI\n(ít + DT thấp)", ms*0.20, mr*0.25, "rgba(153,153,153,0.2)", CB_GRAY),
]:
    fig_bcg.add_annotation(x=x, y=y, text=label, showarrow=False,
                            bgcolor=bg, bordercolor=bc, borderwidth=1, font=dict(size=9))

fig_bcg.update_layout(plot_bgcolor="white", height=600)
st.plotly_chart(fig_bcg, use_container_width=True)
st.caption("4 ô chiến lược tương tự BCG Matrix. Viridis = perceptually uniform + colorblind-safe. "
           "Kích thước bong bóng = area encoding (ít chính xác hơn bar) -> dùng để định vị, "
           "không so sánh số liệu chính xác.")

# Biểu đồ 3b: Lollipop
st.subheader("Biểu đồ 3b: Lollipop chart — Điểm tiềm năng tổng hợp (22 danh mục)")

for c in ["rev_B","sold_mean"]:
    mn, mx = cat_agg[c].min(), cat_agg[c].max()
    cat_agg[c+"_norm"] = (cat_agg[c]-mn)/(mx-mn) if mx > mn else 0.0
cat_agg["composite"] = (cat_agg["rev_B_norm"] + cat_agg["sold_mean_norm"]) / 2
top5_names = set(cat_agg.nlargest(5,"composite")["sub_category"])

lol_df = cat_agg.sort_values("composite").reset_index(drop=True)
colors_l = [CB_VERMIL if n in top5_names else CB_GRAY for n in lol_df["sub_category"]]

fig_lol = go.Figure()
for i, row in lol_df.iterrows():
    fig_lol.add_shape(type="line", x0=0, x1=row["composite"], y0=i, y1=i,
                      line=dict(color=CB_GRAY, width=1.5))
fig_lol.add_trace(go.Scatter(
    x=lol_df["composite"], y=list(range(len(lol_df))),
    mode="markers+text",
    marker=dict(size=14, color=colors_l),
    text=[f" {v:.3f}" for v in lol_df["composite"]],
    textposition="middle right", showlegend=False,
))
fig_lol.update_layout(
    title="Điểm tiềm năng tổng hợp (Composite = avg rank % DT + rank % Sold)",
    xaxis_title="Composite Score (0–1)",
    plot_bgcolor="white", height=600,
    margin=dict(l=160, r=120),
    yaxis=dict(tickvals=list(range(len(lol_df))),
               ticktext=lol_df["sub_category"].tolist()),
)
fig_lol.add_vline(x=0.5, line_dash="dash", line_color=CB_SKYBLUE, opacity=0.6,
                  annotation_text="Ngưỡng 0.5", annotation_position="top")
st.plotly_chart(fig_lol, use_container_width=True)
st.caption("Lollipop chart = biến thể bar chart giảm ink-to-data ratio (nguyên tắc Tufte). "
           "Màu đỏ cam = Top 5 tiềm năng nhất. "
           "Composite Score = trung bình cộng xếp hạng % doanh thu và % lượng bán.")

st.markdown("""
**Nhận xét MT8:**

**Từ BCG bubble chart:**

- **Ngôi sao (mặt nạ):** Nằm ở góc trên phải — sold TB cao nhất (~5,280) VÀ doanh thu cao nhất (~700B). Thị trường lớn nhất nhưng cạnh tranh cực cao. Người bán mới cần vốn lớn và chất lượng vượt trội mới nên gia nhập.

- **Sinh lời (kem dưỡng ẩm, serum, toner):** Sold dưới median nhưng doanh thu trên median — nhờ giá cao. Margin tốt hơn, ít phải cạnh tranh về volume. Phù hợp người bán có sản phẩm chất lượng cao hoặc thương hiệu riêng.

- **Phễu (sữa rửa mặt, kem chống nắng, dầu gội, sữa tắm):** Sold cao, doanh thu ở mức trung. Đây là chiến lược xây review và traffic ban đầu — dễ tạo đơn hàng, xây followers, làm nền tảng để pivot sang danh mục sinh lời.

- **Câu hỏi (các danh mục chăm sóc tóc: sáp vuốt tóc, kem nhuộm tóc, xịt phong tóc):** Sold thấp VÀ doanh thu thấp — rủi ro cao nhất khi gia nhập. Nên tránh nếu chưa có thương hiệu riêng.

**Từ lollipop chart:**

- **mặt nạ = 1.000** là outlier tuyệt đối — bỏ xa tất cả (sữa rửa mặt chỉ đạt 0.614). Không có danh mục nào gần bắt kịp mặt nạ về tổng thể cả 2 chiều.

- **Chỉ 2 danh mục vượt ngưỡng 0.5:** mặt nạ (1.000) và sữa rửa mặt (0.614) — xác nhận đây là 2 danh mục "an toàn nhất" để gia nhập thị trường mỹ phẩm Shopee.

- **Danh mục tóc chiếm phần lớn cuối bảng:** bút kẻ mắt (0.006), kem tay (0.017), kem lót (0.031) — composite cực thấp, xác nhận kết quả từ BCG.

- **dầu gội (0.421) và kem chống nắng (0.403)** nằm ở mức trung — phù hợp chiến lược phễu ban đầu.

**Kết luận chiến lược (từ cả 3 mục tiêu):**

1. **Người bán mới -> Bắt đầu từ ô Phễu:** Ưu tiên sữa rửa mặt hoặc kem chống nắng với discount 10–20% (mức tối ưu cho mid-low, phù hợp giá thực tế của 2 danh mục này). Mục tiêu: nhanh tích lũy reviews và followers.

2. **Sau 50+ reviews -> Pivot sang Sinh lời:** Chuyển sang serum hoặc kem dưỡng ẩm với discount thấp (<=20%) để tăng margin. Rating và followers đã tích lũy là lợi thế cạnh tranh quan trọng.

3. **Tránh hoàn toàn:** Danh mục chăm sóc tóc đặc thù (composite < 0.1) và không nên gia nhập mặt nạ nếu chưa có thương hiệu đủ mạnh — quá cạnh tranh.
""")
