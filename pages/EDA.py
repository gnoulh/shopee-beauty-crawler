import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/vietnam_housing_dataset.csv")
    return df


df = load_data()

# ── Data Dictionary ──────────────────────────────────────
DATA_DICT = {
    "Address": (
        "Địa chỉ",
        "String",
        "Địa chỉ đầy đủ: tên dự án / đường, phường/xã, quận/huyện, tỉnh/thành phố",
    ),
    "Area": ("Diện tích", "Decimal", "Tổng diện tích bất động sản (m²)"),
    "Frontage": ("Mặt tiền", "Decimal", "Chiều rộng mặt tiền (m)"),
    "Access Road": ("Đường vào", "Decimal", "Độ rộng đường dẫn vào bất động sản (m)"),
    "House direction": (
        "Hướng nhà",
        "String",
        "Hướng chính mặt trước ngôi nhà quay về (Đông, Tây, Nam, Bắc,...)",
    ),
    "Balcony direction": ("Hướng ban công", "String", "Hướng ban công quay về"),
    "Floors": ("Số tầng", "Integer", "Tổng số tầng của bất động sản"),
    "Bedrooms": ("Số phòng ngủ", "Integer", "Số lượng phòng ngủ"),
    "Bathrooms": ("Số phòng tắm", "Integer", "Số lượng phòng tắm / nhà vệ sinh"),
    "Legal status": (
        "Tình trạng pháp lý",
        "String",
        "Trạng thái pháp lý: sổ đỏ/hồng, hợp đồng mua bán,...",
    ),
    "Furniture state": (
        "Tình trạng nội thất",
        "String",
        "Mức độ nội thất: Full (đầy đủ), Basic (cơ bản), trống",
    ),
    "Price": ("Giá bán", "Decimal", "Giá bán bất động sản (tỷ đồng VND)"),
}

st.title("📊 Mô tả dữ liệu gốc")
st.caption("Dữ liệu trước khi xử lý — hiển thị nguyên trạng để đánh giá chất lượng")
st.markdown("---")

# ── 1. Tổng quan ─────────────────────────────────────────
st.subheader("1. Tổng quan")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Số dòng", f"{df.shape[0]:,}")
c2.metric("Số cột", df.shape[1])
c3.metric("Missing cells", f"{df.isnull().sum().sum():,}")
c4.metric("Tỷ lệ missing", f"{df.isnull().sum().sum() / df.size * 100:.1f}%")

st.markdown("**Dữ liệu mẫu (10 dòng đầu)**")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# ── 2. Data Dictionary ───────────────────────────────────
st.subheader("2. Mô tả các cột (Data Dictionary)")
st.markdown("Ý nghĩa, kiểu dữ liệu và thống kê nhanh của từng cột trong dataset gốc.")

dict_rows = []
for col in df.columns:
    viet_name, dtype_label, description = DATA_DICT.get(
        col, (col, str(df[col].dtype), "—")
    )
    n_missing = df[col].isnull().sum()
    pct_missing = n_missing / len(df) * 100
    n_unique = df[col].nunique(dropna=True)

    if pd.api.types.is_numeric_dtype(df[col]):
        sample_vals = f"min {df[col].min():.2g} | median {df[col].median():.2g} | max {df[col].max():.2g}"
    else:
        top3 = df[col].value_counts(dropna=True).head(3).index.tolist()
        sample_vals = ", ".join(str(v) for v in top3)

    dict_rows.append(
        {
            "Cột": col,
            "Tên tiếng Việt": viet_name,
            "Kiểu": dtype_label,
            "Mô tả": description,
            "Unique": n_unique,
            "Missing": n_missing,
            "Missing (%)": round(pct_missing, 1),
            "Giá trị mẫu": sample_vals,
        }
    )

dict_df = pd.DataFrame(dict_rows)


# Highlight missing cao
def highlight_missing(val):
    if isinstance(val, float) and val > 50:
        return "background-color: #FDECEA; color: #C0392B; font-weight: bold"
    elif isinstance(val, float) and val > 20:
        return "background-color: #FEF9E7; color: #B7770D"
    return ""


styled = dict_df.style.map(highlight_missing, subset=["Missing (%)"]).set_properties(
    **{"text-align": "left"}
)
st.dataframe(styled, use_container_width=True, hide_index=True, height=420)

st.markdown("---")

# ── 3. Missing Values ────────────────────────────────────
st.subheader("3. Missing Values")

missing_df = (
    pd.DataFrame(
        {
            "Cột": df.columns,
            "Missing": df.isnull().sum().values,
            "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(1),
        }
    )
    .sort_values("Missing (%)", ascending=False)
    .query("`Missing` > 0")
)

fig_missing = px.bar(
    missing_df,
    x="Cột",
    y="Missing (%)",
    text="Missing (%)",
    color="Missing (%)",
    color_continuous_scale=["#AED6F1", "#2E86C1", "#1A5276"],
    title="Tỷ lệ missing value theo cột (%)",
    labels={"Cột": "Tên cột", "Missing (%)": "Tỷ lệ thiếu (%)"},
    height=380,
)
fig_missing.update_traces(texttemplate="%{text}%", textposition="outside")
fig_missing.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    coloraxis_showscale=False,
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8"),
    yaxis=dict(gridcolor="#E8E8E8"),
    margin=dict(t=50, b=20),
)
st.plotly_chart(fig_missing, use_container_width=True)

st.markdown("---")

# ── 4. Thống kê mô tả ────────────────────────────────────
st.subheader("4. Thống kê mô tả — Cột số")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
desc = (
    df[numeric_cols]
    .describe()
    .round(3)
    .T.rename(
        columns={
            "count": "Số giá trị",
            "mean": "Trung bình",
            "std": "Độ lệch chuẩn",
            "min": "Min",
            "25%": "Q1",
            "50%": "Median",
            "75%": "Q3",
            "max": "Max",
        }
    )
)
# Thêm cột tên tiếng Việt
desc.insert(0, "Tên tiếng Việt", [DATA_DICT.get(c, (c,))[0] for c in desc.index])
st.dataframe(desc, use_container_width=True)

st.markdown("---")

# ── 5. Phân phối cột số ──────────────────────────────────
st.subheader("5. Phân phối cột số")
st.markdown("Chọn một cột để xem histogram gốc và phiên bản log-transform.")

selected_col = st.selectbox(
    "Chọn cột:",
    numeric_cols,
    format_func=lambda c: f"{c}  —  {DATA_DICT.get(c, (c,))[0]}",
    index=numeric_cols.index("Price") if "Price" in numeric_cols else 0,
)

col_a, col_b = st.columns(2)
with col_a:
    fig_hist = px.histogram(
        df,
        x=selected_col,
        nbins=50,
        title=f"Phân phối: {selected_col}",
        labels={selected_col: DATA_DICT.get(selected_col, (selected_col,))[0]},
        color_discrete_sequence=["#2E86AB"],
        height=360,
    )
    fig_hist.update_layout(
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        font=dict(color="#2C3E50"),
        xaxis=dict(gridcolor="#E8E8E8"),
        yaxis=dict(gridcolor="#E8E8E8", title="Số lượng"),
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    if df[selected_col].dropna().min() > 0:
        fig_log = px.histogram(
            df,
            x=np.log1p(df[selected_col]),
            nbins=50,
            title=f"Phân phối log(1 + {selected_col})",
            color_discrete_sequence=["#F28E2B"],
            height=360,
        )
        fig_log.update_xaxes(
            title=f"log(1 + {DATA_DICT.get(selected_col,(selected_col,))[0]})"
        )
        fig_log.update_layout(
            plot_bgcolor="#FFFFFF",
            paper_bgcolor="#FFFFFF",
            font=dict(color="#2C3E50"),
            xaxis=dict(gridcolor="#E8E8E8"),
            yaxis=dict(gridcolor="#E8E8E8", title="Số lượng"),
        )
        st.plotly_chart(fig_log, use_container_width=True)
    else:
        st.info("Không thể log-transform (có giá trị ≤ 0)")

st.markdown("---")

# ── 6. Phân phối cột categorical ─────────────────────────
st.subheader("6. Phân phối cột categorical")
st.markdown("Chọn một cột để xem tần suất các giá trị xuất hiện.")

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
selected_cat = st.selectbox(
    "Chọn cột:", cat_cols, format_func=lambda c: f"{c}  —  {DATA_DICT.get(c, (c,))[0]}"
)

val_counts = df[selected_cat].value_counts(dropna=False).reset_index()
val_counts.columns = ["Giá trị", "Số lượng"]
val_counts["Giá trị"] = val_counts["Giá trị"].astype(str).replace("nan", "⚠️ Missing")

top_n = min(20, len(val_counts))
plot_df = val_counts.head(top_n).sort_values("Số lượng", ascending=True)
chart_height = max(400, top_n * 45)

fig_bar = px.bar(
    plot_df,
    x="Số lượng",
    y="Giá trị",
    orientation="h",
    title=f"Top {top_n} giá trị: {selected_cat} — {DATA_DICT.get(selected_cat, (selected_cat,))[0]}",
    text="Số lượng",
    height=chart_height,
    labels={
        "Giá trị": DATA_DICT.get(selected_cat, (selected_cat,))[0],
        "Số lượng": "Số lượng BĐS",
    },
)
fig_bar.update_traces(
    marker_color="#2E86AB",
    marker_line_color="#1A5276",
    marker_line_width=0.5,
    textposition="outside",
    textfont=dict(color="#2C3E50"),
)
fig_bar.update_layout(
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    font=dict(color="#2C3E50"),
    xaxis=dict(gridcolor="#E8E8E8", linecolor="#CCCCCC"),
    yaxis=dict(linecolor="#CCCCCC"),
    margin=dict(l=20, r=60, t=50, b=20),
    yaxis_title=None,
)
st.plotly_chart(fig_bar, use_container_width=True)
st.dataframe(val_counts, hide_index=True, use_container_width=True)
