"""
pages/22127254.py
==========================================================
PHÂN CÔNG: 22127254

Obj 1: Phân tích trung vị sold theo discount_pct x price_tier
        -> Xác định ngưỡng discount tối ưu cho từng phân khúc giá
Obj 2: So sánh hiệu quả bán hàng giữa 22 danh mục (sub_category) để xác định top 5 danh mục tiềm năng nhất

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

from utils.helpers import (
    load_data, get_active, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_GREEN, CB_BLUE, CB_VERMIL, CB_GRAY,
    SEQ_BLUES, add_trendline,
)

st.set_page_config(page_title="22127254", page_icon="", layout="wide")
setup_sidebar()

products, shops, reviews = load_data()
active = get_active(products)

# ====== Header ======
st.title("22127254")
st.caption("products.csv; 20,658 sản phẩm; crawl 18/3/2026")

# ====== Mục tiêu SMART ======
with st.expander("Mục tiêu SMART", expanded=True):
    st.markdown("""
    **Obj 1 — Ngưỡng discount tối ưu:**
    > Sử dụng `discount_pct`, `sold`, `price_tier` của 20,658 sản phẩm crawl 18/3/2026
    > để xác định khoảng `discount_pct` có **trung vị sold cao nhất** trong từng `price_tier`
    > (budget -> luxury), từ đó **đề xuất ngưỡng chiết khấu tối ưu cho mỗi phân khúc** —
    > để người bán tránh giảm giá tùy tiện: biết rõ ở phân khúc nào, mức chiết khấu nào
    > thực sự kích thích lượng bán thay vì chỉ làm giảm margin.

    **Obj 2 — ...:**
    > ...
    """)

st.markdown("---")

# ======================
# OBJ 1 — Ngưỡng discount tối ưu theo price_tier
# ======================
st.markdown("## Obj 1 — Ngưỡng chiết khấu tối ưu theo phân khúc giá")

# Chuẩn bị dữ liệu
work = active[active["discount_pct"] >= 0].copy()
work["discount_bin"] = pd.cut(
    work["discount_pct"],
    bins=[0, 10, 20, 30, 40, 50, 100],
    labels=["0–10%", "10–20%", "20–30%", "30–40%", "40–50%", ">50%"],
)
tier_order = [t for t in ["budget","mid-low","mid","premium","luxury"]
              if t in work["price_tier"].unique()]

# ====== Biểu đồ 1a: Heatmap ======
st.subheader("Biểu đồ 1a: Heatmap — Trung vị sold theo price_tier x discount_bin")

pivot = (
    work.groupby(["price_tier", "discount_bin"])["sold"]
    .median()
    .unstack(fill_value=0)
    .reindex(tier_order)
)

fig_heat = px.imshow(
    pivot,
    text_auto=".0f",
    color_continuous_scale="Blues",
    title="Trung vị lượng bán (sold) theo phân khúc giá x khoảng chiết khấu",
    labels={"color": "Median sold", "x": "Khoảng chiết khấu", "y": "Phân khúc giá"},
    aspect="auto",
)
fig_heat.update_layout(plot_bgcolor="white")
st.plotly_chart(fig_heat, use_container_width=True)
st.caption(
    "Ô màu đậm hơn = trung vị sold cao hơn. "
    "Mỗi hàng là một price_tier; mỗi cột là một khoảng discount. "
    "Tìm ô màu đậm nhất trong từng hàng để xác định ngưỡng discount tối ưu."
)

st.markdown("""
**Nhận xét Biểu đồ 1a:**
*...*
- **budget:** Khoảng discount có trung vị sold cao nhất là ... -> Ngưỡng tối ưu gợi ý: ...
- **mid-low:** ...
- **mid:** ...
- **premium:** ...
- **luxury:** ...
""")

# ====== Biểu đồ 1b: Box plot ==============================
st.subheader("Biểu đồ 1b: Box plot — Phân bố sold theo discount_bin")
st.info("""
...""")

# ====== Kết luận ======
st.markdown("### Kết luận — Đề xuất chiến lược")
st.markdown("""
**Chiến lược 1 — Tối ưu chiết khấu:**
- ...

**Chiến lược 2 — Lựa chọn danh mục:**
- ...
""")
