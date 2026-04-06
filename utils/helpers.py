"""
utils/helpers.py — Shared constants, data loaders, and utility functions.

Mọi page import từ đây.
"""

import re
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter

# ====================== Okabe-Ito colorblind-safe palette (2008) ======================
CB_ORANGE  = "#E69F00"
CB_SKYBLUE = "#56B4E9"
CB_GREEN   = "#009E73"
CB_YELLOW  = "#F0E442"
CB_BLUE    = "#0072B2"
CB_VERMIL  = "#D55E00"
CB_PURPLE  = "#CC79A7"
CB_GRAY    = "#999999"

SEQ_BLUES = px.colors.sequential.Blues
DIV_RDBU  = px.colors.diverging.RdBu

# ====================== Vietnamese stopwords ======================
STOPS = {
    "và","của","là","có","được","không","cho","với","trong","một","các","này","đã","để",
    "tôi","mình","sản","phẩm","shop","hàng","rất","nên","thì","khi","về","như","từ","ra",
    "lên","lại","cũng","nha","ạ","ơi","nè","nhé","đây","đó","thật","quá","lắm","siêu",
    "mua","dùng","thấy","bạn","bên","chị","em","cái","hay","vì","nếu","theo","thể","luôn",
    "còn","đều","vẫn","mà","sau","trước","đơn","đóng","gói","nhưng","giao","đặt",
}

# ====================== Data loader ================================================─
@st.cache_data
def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load products.csv, shops.csv, reviews.csv
    Cached — chỉ đọc file 1 lần mỗi session.
    """
    products = pd.read_csv("data/products.csv")
    shops = pd.read_csv("data/shops.csv")
    reviews = pd.read_csv("data/reviews.csv")
    return products, shops, reviews


def get_active(products: pd.DataFrame) -> pd.DataFrame:
    """Lọc sản phẩm có sold > 0 — dùng cho mọi phân tích doanh thu."""
    return products[products["sold"] > 0].copy()


# ====================== Text utilities ========================================================================================
def extract_free(text: str) -> str:
    """Lấy phần text tự do trong review (bỏ structured tags kiểu 'Key: value')."""
    if pd.isna(text):
        return ""
    lines = str(text).split("\n")
    return " ".join(l.strip() for l in lines if not re.match(r"^[^\n:]{1,30}:", l.strip()))


def tokenize(text: str) -> list[str]:
    """Tách từ tiếng Việt đơn giản, loại stopwords và ký tự số."""
    if not text:
        return []
    toks = re.findall(r"[\w\u00C0-\u024F\u1E00-\u1EFF]+", text.lower())
    return [t for t in toks if len(t) >= 3 and t not in STOPS and not t.isdigit()]


# ====================== Chart utilities ==================================================================─
def add_trendline(
    fig: go.Figure,
    x_s: pd.Series,
    y_s: pd.Series,
    log_x: bool = False,
    log_y: bool = False,
    color: str = CB_VERMIL,
) -> go.Figure:
    """Thêm đường xu hướng OLS (polynomial bậc 1) vào scatter plot."""
    xs = np.log10(x_s.clip(1)) if log_x else x_s.values.astype(float)
    ys = np.log10(y_s.clip(1)) if log_y else y_s.values.astype(float)
    mask = np.isfinite(xs) & np.isfinite(ys)
    z  = np.polyfit(xs[mask], ys[mask], 1)
    xl = np.linspace(xs[mask].min(), xs[mask].max(), 100)
    yl = np.polyval(z, xl)
    if log_x: xl = 10 ** xl
    if log_y: yl = 10 ** yl
    fig.add_trace(go.Scatter(
        x=xl, y=yl, mode="lines", name="Xu hướng",
        line=dict(color=color, width=2, dash="dot"),
        showlegend=True,
    ))
    return fig


# === Sidebar branding (call at top of every page) =========
TEAM_CSS = """
<style>
/* === Global typography & background === */
html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
.main .block-container { padding-top: 1.2rem; padding-bottom: 2rem; }

/* === Metric cards === */
div[data-testid="stMetric"] {
    background: #fff;
    border: 1px solid #ffe0d6;
    border-left: 4px solid #EE4D2D;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 2px 8px rgba(238,77,45,.08);
}
div[data-testid="stMetricValue"] { color: #EE4D2D; font-weight: 700; }

/* === Section headers === */
h2 { color: #1a1a2e !important; border-bottom: 2px solid #EE4D2D;
     padding-bottom: .25rem; margin-top: 1.8rem; }
h3 { color: #2d2d44 !important; }

/* === Member badge === */
.member-badge {
    display:inline-block; background:#fff3f0; color:#c0392b;
    border:1px solid #EE4D2D; border-radius:6px;
    padding:2px 10px; font-size:.78rem; font-weight:600;
    margin-bottom:.5rem;
}
/* === Conclusion box === */
.conclusion-box {
    background:#f0fff4; border-left:5px solid #27ae60;
    border-radius:0 8px 8px 0; padding:14px 18px;
    margin-top:1rem;
}
/* === Page divider ====== */
hr { border:none; border-top:2px solid #ffe0d6; margin:1.5rem 0; }

/* === Sidebar ============ */
section[data-testid="stSidebar"] { background: #1a1a2e; }
section[data-testid="stSidebar"] * { color: #f0f0f0 !important; }
section[data-testid="stSidebar"] a { color: #ffb09e !important; }
</style>
"""


def inject_css() -> None:
    """Inject shared Shopee-themed CSS. Call once per page after set_page_config."""
    st.markdown(TEAM_CSS, unsafe_allow_html=True)


def member_badge(mssv: str, obj: str) -> None:
    """Render a small attribution badge below a section header."""
    st.markdown(
        f'<span class="member-badge"> MSSV {mssv} · {obj}</span>',
        unsafe_allow_html=True,
    )


def conclusion_box(text: str) -> None:
    """Render a green conclusion/insight box."""
    st.markdown(
        f'<div class="conclusion-box">{text}</div>',
        unsafe_allow_html=True,
    )


def setup_sidebar(team_name: str = "Nhóm 01") -> None:
    """Render consistent sidebar branding across all pages."""
    st.sidebar.image(
        "https://upload.wikimedia.org/wikipedia/commons/thumb/f/fe/Shopee.svg/200px-Shopee.svg.png",
        width=100,
    )
    st.sidebar.markdown(f"### Lab 01 – {team_name}")
    st.sidebar.markdown("**Dữ liệu:** Shopee Việt Nam, 18–19/3/2026")
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
**Thành viên:**
| MSSV | Mục tiêu |
|------|-----------|
| 22127254 | 1, 2, 8 |
| 23127488 | 3, 9 |
| 23127361 | 5, 10 |
| 22127418 | 4, 7 |
""")
    st.sidebar.markdown("---")
    st.sidebar.caption(
        "`products.csv`, `shops.csv`, `reviews.csv` lưu trữ trong thư mục `data`"
    )