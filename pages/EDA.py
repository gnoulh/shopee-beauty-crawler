import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go


# ── Load data ────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("data/vietnam_housing_dataset.csv")  # ← thay đường dẫn thật
    return df


df = load_data()

st.title("📊 Mô tả dữ liệu gốc")
st.caption("Dữ liệu trước khi xử lý — hiển thị nguyên trạng để đánh giá chất lượng")
st.markdown("---")

# ── 1. Shape & Sample ────────────────────────────────────
st.subheader("1. Tổng quan")
col1, col2, col3 = st.columns(3)
col1.metric("Số dòng", f"{df.shape[0]:,}")
col2.metric("Số cột", df.shape[1])
col3.metric("Missing cells", f"{df.isnull().sum().sum():,}")

st.markdown("**Dữ liệu mẫu (10 dòng đầu)**")
st.dataframe(df.head(10), use_container_width=True)

st.markdown("---")

# ── 2. Kiểu dữ liệu ─────────────────────────────────────
st.subheader("2. Kiểu dữ liệu từng cột")

dtype_df = pd.DataFrame(
    {
        "Cột": df.columns,
        "Kiểu": df.dtypes.astype(str).values,
        "Unique values": [df[c].nunique() for c in df.columns],
        "Missing": df.isnull().sum().values,
        "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2),
    }
)
st.dataframe(dtype_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── 3. Missing Values ────────────────────────────────────
st.subheader("3. Missing Values")

missing_df = pd.DataFrame(
    {
        "Cột": df.columns,
        "Missing": df.isnull().sum().values,
        "Missing (%)": (df.isnull().sum().values / len(df) * 100).round(2),
    }
).sort_values("Missing (%)", ascending=False)

fig_missing = px.bar(
    missing_df[missing_df["Missing"] > 0],
    x="Cột",
    y="Missing (%)",
    color="Missing (%)",
    color_continuous_scale="Reds",
    text="Missing (%)",
    title="Tỷ lệ missing value theo cột (%)",
)
fig_missing.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_missing.update_layout(showlegend=False, coloraxis_showscale=False)
st.plotly_chart(fig_missing, use_container_width=True)

st.markdown("---")

# ── 4. Thống kê mô tả ────────────────────────────────────
st.subheader("4. Thống kê mô tả — cột số")

numeric_cols = df.select_dtypes(include="number").columns.tolist()
st.dataframe(
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
    ),
    use_container_width=True,
)

st.markdown("---")

# ── 5. Phân phối từng cột số ──────────────────────────────
st.subheader("5. Phân phối cột số")

selected_col = st.selectbox(
    "Chọn cột:",
    numeric_cols,
    index=numeric_cols.index("Price") if "Price" in numeric_cols else 0,
)

col_a, col_b = st.columns(2)

with col_a:
    fig_hist = px.histogram(
        df,
        x=selected_col,
        nbins=50,
        title=f"Phân phối: {selected_col}",
        color_discrete_sequence=["#4C78A8"],
    )
    st.plotly_chart(fig_hist, use_container_width=True)

with col_b:
    # Log transform nếu là Price hoặc skewed
    if df[selected_col].min() > 0:
        fig_log = px.histogram(
            df,
            x=np.log1p(df[selected_col]),
            nbins=50,
            title=f"Phân phối log({selected_col})",
            color_discrete_sequence=["#F28E2B"],
        )
        fig_log.update_xaxes(title=f"log(1 + {selected_col})")
        st.plotly_chart(fig_log, use_container_width=True)
    else:
        st.info("Không thể log-transform (có giá trị ≤ 0)")

st.markdown("---")


# ── 6. Categorical columns ───────────────────────────────
st.subheader("6. Phân phối cột categorical")

cat_cols = df.select_dtypes(include=["object", "string"]).columns.tolist()
selected_cat = st.selectbox("Chọn cột:", cat_cols)

val_counts = df[selected_cat].value_counts(dropna=False).reset_index()
val_counts.columns = ["Giá trị", "Số lượng"]
val_counts["Giá trị"] = val_counts["Giá trị"].astype(str).replace("nan", "⚠️ Missing")

if val_counts.empty:
    st.warning(f"Cột `{selected_cat}` không có dữ liệu để hiển thị.")
else:
    top_n = min(20, len(val_counts))
    plot_df = val_counts.head(top_n).sort_values("Số lượng", ascending=True)

    # Fix: height đủ lớn, không dùng top_n * 30
    chart_height = max(400, top_n * 45)

    fig_bar = px.bar(
        plot_df,
        x="Số lượng",
        y="Giá trị",
        orientation="h",
        title=f"Top {top_n} giá trị: {selected_cat}",
        text="Số lượng",
        color="Số lượng",
        color_continuous_scale="Blues",
        height=chart_height,
    )
    fig_bar.update_traces(textposition="outside")
    fig_bar.update_layout(
        coloraxis_showscale=False,
        margin=dict(l=20, r=60, t=50, b=20),
        yaxis_title=None,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.dataframe(val_counts, hide_index=True, use_container_width=True)
