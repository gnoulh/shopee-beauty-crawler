"""
pages/02_Reviews.py — Đánh giá & Phân tích Cảm xúc Khách hàng
================================================================
PHÂN CÔNG:
  MT9 — 23127488: Trích xuất từ khóa + phân loại 4 nhóm nguyên nhân
  ML: TF-IDF + Logistic Regression phân loại review
================================================================
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import warnings; warnings.filterwarnings("ignore")

from utils.helpers import (
    inject_css, member_badge, conclusion_box,
    load_data, setup_sidebar,
    CB_ORANGE, CB_SKYBLUE, CB_BLUE, CB_VERMIL, CB_GRAY,
    extract_free, tokenize,
)

st.set_page_config(page_title="02 – Đánh giá & Cảm xúc", layout="wide")
inject_css(); setup_sidebar()

products, shops, reviews = load_data()

st.title("Đánh giá & Phân tích Cảm xúc Khách hàng")
st.caption("reviews.csv - 23,989 đánh giá - crawl 19/3/2026 - MSSV: 23127488")
member_badge("23127488", "MT9")

with st.expander("Mục tiêu SMART (MT9)", expanded=False):
    st.markdown("""
    > Trích xuất và phân tích **10 từ khóa phổ biến** trong các đánh giá 1–3 sao thu thập được
    > tính tới tháng 3/2026, phân loại theo **4 nhóm nguyên nhân** (sự cố sản phẩm, trải nghiệm KH,
    > chất lượng đóng gói, vận chuyển) — để đề xuất các giải pháp tương ứng giúp người bán
    > ngăn chặn hoặc thuyết phục khách hàng chỉnh sửa đánh giá tiêu cực.
    """)

st.markdown("---")

# === Prep ===
@st.cache_data
def prep(reviews_df):
    r = reviews_df.copy()
    r["free_text"] = r["review_text"].apply(extract_free)
    r["free_text_len"] = r["free_text"].str.len()
    pos_t, neg_t = [], []
    for t in r[r["rating"] == 5]["free_text"].dropna(): pos_t.extend(tokenize(t))
    for t in r[r["rating"] <= 3]["free_text"].dropna(): neg_t.extend(tokenize(t))
    return r, pd.DataFrame(Counter(pos_t).most_common(12), columns=["word","count"]), \
              pd.DataFrame(Counter(neg_t).most_common(12), columns=["word","count"])

rv, pos_top, neg_top = prep(reviews)
neg_reviews = rv[rv["rating"] <= 3]
pos_reviews = rv[rv["rating"] == 5]

# ============
# Biểu đồ 1: Phân bố rating + box độ dài review
# ============
st.subheader("Biểu đồ 1: Phân bố Rating – J-curve & Độ dài Review")
c1, c2 = st.columns(2)

with c1:
    rc = rv["rating"].value_counts().sort_index()
    bar_c = [CB_VERMIL, CB_VERMIL, CB_ORANGE, CB_SKYBLUE, CB_BLUE]
    fig = go.Figure()
    for i, (star, cnt) in enumerate(rc.items()):
        fig.add_bar(x=[star], y=[cnt], marker_color=bar_c[i],
                    text=[f"{cnt:,}<br>({cnt/len(rv)*100:.1f}%)"],
                    textposition="outside", name=f"{star} sao")
    fig.update_layout(showlegend=True, plot_bgcolor="white",
                      xaxis_title="Số sao", yaxis_title="Số đánh giá",
                      title=f"Phân bố rating – {len(rv):,} đánh giá")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    fig = go.Figure()
    for i, star in enumerate([1,2,3,4,5]):
        d = rv[rv["rating"]==star]["review_length"].dropna()
        if len(d):
            fig.add_trace(go.Box(y=d, name=f"{star} sao",
                                  marker_color=bar_c[i], boxmean=True))
    fig.update_layout(plot_bgcolor="white", yaxis_title="Độ dài review (ký tự)",
                      title="Độ dài review theo mức rating")
    st.plotly_chart(fig, use_container_width=True)

n_neg, n_pos = len(neg_reviews), len(pos_reviews)
st.success(
    f"**Nhận xét:** {n_pos:,}/{len(rv):,} ({n_pos/len(rv)*100:.1f}%) là 5 sao — **J-curve điển hình TMĐT**. "
    f"Chỉ {n_neg} review 1–3 sao -> dataset cực kỳ mất cân bằng, cần cân bằng trước khi dùng ML. "
    "Review tiêu cực thường ngắn hơn tích cực — người thất vọng ít giải thích."
)

# ============
# Biểu đồ 2: Top từ khóa 5 sao vs 1–3 sao (horizontal bar song song)
# ============
st.markdown("---")
st.subheader("Biểu đồ 2: Top 10 Từ khóa – So sánh 5 sao vs 1–3 sao")

c3, c4 = st.columns(2)
with c3:
    fig = px.bar(pos_top.head(10), x="count", y="word", orientation="h",
                 color="count", color_continuous_scale="Blues",
                 title="Top 10 từ khóa – Đánh giá 5 sao",
                 labels={"count":"Tần suất","word":""})
    fig.update_layout(yaxis={"categoryorder":"total ascending"}, plot_bgcolor="white",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

with c4:
    fig = px.bar(neg_top.head(10), x="count", y="word", orientation="h",
                 color="count", color_continuous_scale="Oranges",
                 title="Top 10 từ khóa – Đánh giá 1–3 sao",
                 labels={"count":"Tần suất","word":""})
    fig.update_layout(yaxis={"categoryorder":"total ascending"}, plot_bgcolor="white",
                      coloraxis_showscale=False)
    st.plotly_chart(fig, use_container_width=True)

st.info("**Nhận xét:** 5 sao nổi bật: 'nhanh' (giao hàng), 'thơm/mùi' (hương liệu), 'cẩn thận' (đóng gói). "
        "1–3 sao: 'màu', 'hình', 'giống' -> sản phẩm không đúng mô tả/ảnh.")

# ============
# Biểu đồ 3: Phân loại 4 nhóm nguyên nhân
# ============
st.markdown("---")
st.subheader("Biểu đồ 3: Phân loại Nguyên nhân Đánh giá Tiêu cực (4 nhóm)")

keywords_dict = {
    "Sự cố sản phẩm": ["dị ứng","ngứa","rát","mùi","fake","giả","hỏng","chất lượng","lỏng","đặc"],
    "Trải nghiệm KH": ["thái độ","không trả lời","phục vụ","tư vấn","chăm sóc","nhắn tin"],
    "Đóng gói": ["móp","méo","rách","vỡ","đổ","chảy","bẹp","đóng gói","gói hàng"],
    "Vận chuyển": ["chậm","lâu","nhầm","sai","thiếu","shipper","giao hàng"],
}
cat_colors = {"Sự cố sản phẩm":CB_VERMIL,"Trải nghiệm KH":CB_ORANGE,
              "Đóng gói":CB_SKYBLUE,"Vận chuyển":CB_BLUE}

@st.cache_data
def compute_categories(reviews_df):
    bad = reviews_df[reviews_df["rating"] <= 3]
    cat_counts = {}
    for cat, kws in keywords_dict.items():
        cnt = 0
        for text in bad["review_text"].dropna():
            tl = text.lower()
            cnt += sum(1 for kw in kws if kw in tl)
        cat_counts[cat] = cnt
    return pd.DataFrame(cat_counts.items(), columns=["Nhóm","Tần suất"])

cat_df = compute_categories(reviews)
cat_df["Pct"] = (cat_df["Tần suất"] / cat_df["Tần suất"].sum() * 100).round(1)

c5, c6 = st.columns(2)
with c5:
    fig = px.pie(cat_df, names="Nhóm", values="Tần suất", color="Nhóm",
                 color_discrete_map=cat_colors, hole=0.4,
                 title="Tỷ lệ nhóm nguyên nhân – Đánh giá 1–3 sao")
    fig.update_traces(textinfo="percent+label")
    st.plotly_chart(fig, use_container_width=True)

with c6:
    fig = px.bar(cat_df.sort_values("Tần suất"), x="Tần suất", y="Nhóm",
                 orientation="h", color="Nhóm", color_discrete_map=cat_colors,
                 text="Pct", title="Tần suất từ khóa theo nhóm nguyên nhân")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with st.expander("Phân tích chi tiết & Đề xuất chiến lược"):
    top_cat = cat_df.sort_values("Tần suất", ascending=False).iloc[0]
    st.markdown(f"""
    **Phát hiện chính:**
    - **Sự cố sản phẩm** chiếm tỷ trọng lớn nhất ({cat_df[cat_df['Nhóm']=='Sự cố sản phẩm']['Pct'].values[0]:.1f}%) -> vấn đề mô tả/màu sắc/mùi hương không đúng thực tế là nguyên nhân cốt lõi.
    - **Vận chuyển** đứng thứ 2 -> dù logistics Việt Nam đã cải thiện, giao trễ/nhầm vẫn là điểm trừ.
    - **Đóng gói & Trải nghiệm khách hàng** chiếm tỷ lệ nhỏ -> người bán VN đã làm tốt hai khía cạnh này.

    **Lưu ý về mẫu:** Chỉ có {n_neg} review 1–3 sao từ 800 sản phẩm crawl — con số rất nhỏ, kết luận mang tính định hướng, không đại diện cho toàn bộ thị trường.

    **Đề xuất:**
    1. **Ưu tiên #1:** Đảm bảo ảnh sản phẩm, màu sắc, mùi hương đúng với thực tế.
    2. **Ưu tiên #2:** Hợp tác đơn vị vận chuyển uy tín, cung cấp tracking rõ ràng.
    3. **Chính sách đổi trả linh hoạt:** Giảm lo ngại, tăng khả năng khách chỉnh sửa review tiêu cực.
    """)

# ============
# Biểu đồ 4: Tỷ lệ kèm ảnh
# ============
st.markdown("---")
st.subheader("Biểu đồ 4: Tỷ lệ Review kèm ảnh theo mức Rating")

if "has_image" in rv.columns:
    cross = pd.crosstab(rv["rating"], rv["has_image"], normalize="index") * 100
    fig = go.Figure()
    if 0 in cross.columns:
        fig.add_bar(name="Không kèm ảnh",
                    x=[f"{r} sao" for r in cross.index], y=cross[0], marker_color=CB_ORANGE)
    if 1 in cross.columns:
        fig.add_bar(name="Kèm ảnh",
                    x=[f"{r} sao" for r in cross.index], y=cross[1], marker_color=CB_BLUE)
    fig.update_layout(barmode="group", plot_bgcolor="white", yaxis_title="Tỷ lệ (%)",
                      title="Tỷ lệ review kèm ảnh theo mức đánh giá",
                      legend=dict(orientation="h", yanchor="bottom", y=1))
    st.plotly_chart(fig, use_container_width=True)
    ni = rv[rv["rating"]==5]["has_image"].mean()*100
    ng = rv[rv["rating"]<=3]["has_image"].mean()*100
    st.info(f"**3 đặc trưng phân biệt 1–3 sao vs 5 sao:** "
            f"(1) Tỷ lệ kèm ảnh: {ng:.1f}% vs {ni:.1f}% — chênh {ni-ng:.0f}pp "
            f"(2) Độ dài text tự do: {rv[rv['rating']<=3]['free_text_len'].mean():.0f} vs {rv[rv['rating']==5]['free_text_len'].mean():.0f} ký tự "
            "(3) Nội dung từ khóa: màu/hình (tiêu cực) vs nhanh/thơm (tích cực)")

# ============
# ML: TF-IDF + Logistic Regression
# ============
st.markdown("---")
st.subheader("Machine Learning: TF-IDF + Logistic Regression (Phân loại Cảm xúc)")

@st.cache_data
def train_model(reviews_df):
    r2 = reviews_df.copy()
    r2["ti"] = r2["review_text"].fillna("") + " " + r2["review_text"].apply(extract_free).fillna("")
    neg_ = r2[r2["rating"] <= 3].copy()
    pos_ = r2[r2["rating"] == 5].sample(min(len(neg_)*4, len(r2[r2["rating"]==5])), random_state=42)
    bal  = pd.concat([neg_, pos_]).reset_index(drop=True)
    bal["label"] = (bal["rating"] == 5).astype(int)
    tv  = TfidfVectorizer(max_features=800, ngram_range=(1,2), min_df=1)
    X   = tv.fit_transform(bal["ti"]); y = bal["label"]
    Xtr,Xte,ytr,yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)
    clf = LogisticRegression(max_iter=500, class_weight="balanced")
    clf.fit(Xtr, ytr); yp = clf.predict(Xte)
    cm = confusion_matrix(yte, yp); acc = accuracy_score(yte, yp)
    feat = tv.get_feature_names_out(); coefs = clf.coef_[0]; n = 10
    words  = list(feat[coefs.argsort()[:n]]) + list(feat[coefs.argsort()[-n:]])
    vals   = list(coefs[coefs.argsort()[:n]]) + list(coefs[coefs.argsort()[-n:]])
    colors = [CB_VERMIL]*n + [CB_BLUE]*n
    return cm, pd.DataFrame({"word":words,"coef":vals,"color":colors}), acc

cm, feat_df, acc = train_model(reviews)

c7, c8 = st.columns(2)
with c7:
    fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                    x=["Tiêu cực","Tích cực"], y=["Tiêu cực","Tích cực"],
                    title=f"Ma trận nhầm lẫn (Accuracy = {acc*100:.1f}%)",
                    labels={"x":"Dự đoán","y":"Thực tế"})
    st.plotly_chart(fig, use_container_width=True)

with c8:
    fs = feat_df.sort_values("coef")
    fig = go.Figure()
    fig.add_bar(x=fs["coef"], y=fs["word"], orientation="h",
                marker_color=fs["color"].tolist())
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(showlegend=False, plot_bgcolor="white",
                      xaxis_title="Hệ số TF-IDF (cam=tiêu cực - xanh=tích cực)",
                      title="Từ khóa quyết định phân loại",
                      yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

st.caption(f"Cân bằng dataset 4:1 -> TF-IDF (800 features, bigrams) -> LogReg (balanced). "
           f"Accuracy {acc*100:.1f}% trên tập cân bằng ý nghĩa hơn ~99% trên tập lệch. "
           "Màu Okabe-Ito: cam (#D55E00)=tiêu cực, xanh (#0072B2)=tích cực — colorblind-safe.")
