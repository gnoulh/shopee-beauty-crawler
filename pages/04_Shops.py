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

inject_css(); setup_sidebar()

products, shops, reviews = load_data()

st.title("Hiệu quả Cửa hàng & Chỉ số Lòng tin")
st.caption("shops.csv + products.csv - crawl 18/3/2026 - MSSV: 23127361")
member_badge("23127361", "MT5 & 9")

# CSS chung cho bảng 
def render_styled_table(df):
    """Hàm dùng chung để vẽ bảng có viền đen, header nền xám chữ giữa, data chữ trái"""
    styles = [
        dict(selector="table", props=[
            ('width', '100%'),
            ('border-collapse', 'collapse'),
            ('font-family', 'sans-serif')
        ]),
        dict(selector="th", props=[
            ('font-weight', 'bold'), 
            ('text-align', 'center !important'),
            ('background-color', '#f0f2f6'), 
            ('border', '1px solid #000000'),
            ('padding', '10px')
        ]),
        dict(selector="td", props=[
            ('text-align', 'left !important'), 
            ('border', '1px solid #000000'), 
            ('padding', '10px')
        ]) 
    ]
    
    html_table = df.style.hide(axis="index").set_table_styles(styles).to_html()
    st.markdown(html_table, unsafe_allow_html=True)

# SMART
with st.expander("Tiêu chí SMART"):
    smart_df = pd.DataFrame({
        "Tiêu chí": [
            "<b>Specific</b><br>(Cụ thể)", 
            "<b>Measurable</b><br>(Đo lường được)", 
            "<b>Achievable</b><br>(Khả thi)", 
            "<b>Relevant</b><br>(Liên quan)", 
            "<b>Time-bound</b><br>(Có thời hạn)"
        ],
        "MT9 – Phân cụm Shop": [
            "Phân các shop thành các nhóm theo 4 tiêu chí (theo dõi, số đánh giá, điểm đánh giá, tốc độ phản hồi).",
            "Đo bằng Silhouette score, tổng doanh thu ước tính và tổng lượt bán của từng nhóm shop.",
            "K-Means trên tập dữ liệu đã làm sạch",
            "Giúp đánh giá tác động của mức độ uy tín và chất lượng dịch vụ đến hiệu quả kinh doanh tổng thể của shop.",
            "Snapshot 18/3/2026",
        ],
        "MT5 – Tương quan": [
            "Tìm xem trong 3 yếu tố: Lượt theo dõi, Tốc độ phản hồi và Điểm sao, cái nào ảnh hưởng lớn nhất đến doanh thu của shop.",
            "Hệ số tương quan Pearson — chỉ số có |r| lớn nhất = quan trọng nhất",
            "Ma trận tương quan trên tập dữ liệu hiện có",
            "Cung cấp cơ sở dữ liệu để nhà bán hàng biết nên ưu tiên cải thiện chỉ số nào nhất nhằm tối ưu hóa doanh thu.",
            "Snapshot 18/3/2026",
        ],
    })
    render_styled_table(smart_df)

st.markdown("---")

# Preprocessing 
@st.cache_data
def prep_shop_data(products_df, shops_df):
    # Tính tổng doanh thu ước tính của mỗi shop từ products_df
    shop_revenue = products_df.groupby("shop_id")["revenue_est"].sum().reset_index()
    
    # Chọn các cột cần thiết từ shops_df để merge với doanh thu
    shop_cols = ["shop_id","shop_name","follower_count","rating_star",
                 "rating_count","response_rate","total_sold","is_mall"]
    avail = [c for c in shop_cols if c in shops_df.columns]
    df = shops_df[avail].copy().merge(shop_revenue, on="shop_id", how="left")

    # Điền NaN bằng 0 (shop không có sản phẩm đã bán hoặc doanh thu không xác định)
    df["revenue_est"] = df["revenue_est"].fillna(0)
    df["total_sold"] = df["total_sold"].fillna(0) if "total_sold" in df.columns else 0
    return df

df = prep_shop_data(products, shops)

# MỤC TIÊU 9
st.subheader("Mục tiêu 9 — K-Means phân cụm Cửa hàng theo Độ tin cậy và Hiệu quả")

# Chọn các chỉ số liên quan đến độ tin cậy và hiệu quả của shop để phân cụm
features = ["follower_count","rating_count","rating_star","response_rate"]
avail_f  = [f for f in features if f in df.columns]

@st.cache_data
def explore_kmeans(df_, features_):
    # Chuẩn bị dữ liệu: điền NaN bằng 0, chuẩn hóa
    df_c = df_.copy()
    df_c[features_] = df_c[features_].fillna(0)
    X_scaled = StandardScaler().fit_transform(df_c[features_])
    
    # Khảo sát Silhouette Score để chọn k tối ưu
    sil_scores = {}
    saved_labels = {}
    
    for k in range(2, 7):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil_scores[k] = silhouette_score(X_scaled, labels)
        saved_labels[k] = labels
        
    return df_c, sil_scores, saved_labels

df_c, sil_scores, all_labels = explore_kmeans(df, avail_f)

# Vẽ biểu đồ Silhouette Score để chọn k tối ưu
st.markdown("#### Biểu đồ 1a: Khảo sát Silhouette Score để tìm k tối ưu")
fig_sil = px.line(x=list(sil_scores.keys()), y=list(sil_scores.values()),
                  markers=True, labels={"x":"k","y":"Silhouette Score"},
                  title="Silhouette Score theo số cụm k",
                  color_discrete_sequence=[CB_BLUE])
fig_sil.update_layout(plot_bgcolor="white", margin=dict(t=50,l=10,r=10,b=10))
st.plotly_chart(fig_sil, use_container_width=True)

# Chọn k dựa trên biểu đồ Silhouette Score
selected_k = max(sil_scores, key=sil_scores.get)
st.success(f"**Kết luận:** Chọn k={selected_k} vì có Silhouette Score cao nhất ({sil_scores[selected_k]:.3f}), cho thấy phân cụm rõ ràng nhất.")

# Lấy nhãn của k đã chọn
df_c["Cluster"] = all_labels[selected_k]

# Tạo expender để hiển thị đặc trưng trung bình của từng nhóm và cơ sở đặt tên nhóm
with st.expander(f"Giải thích tên gọi của các nhóm", expanded=False):
    # Bảng số liệu trung bình
    st.markdown("**1. Thống kê chỉ số trung bình của từng nhóm:**")

    # Tính trung bình các chỉ số theo nhóm
    summary = df_c.groupby("Cluster")[avail_f].mean().reset_index()
    disp = summary.rename(columns={
        "follower_count":"TB Followers",
        "rating_count":"TB Lượt đánh giá",
        "rating_star":"TB Điểm sao",
        "response_rate":"TB Tốc độ phản hồi",
    })
    render_styled_table(disp)
    
    # Đặt tên nhóm dựa trên đặc điểm trung bình
    st.markdown("**2. Cơ sở nhận diện và Đặt tên nhóm:**")
    naming_df = pd.DataFrame({
        "Tên Nhóm": [
            "Shop Dẫn Đầu (cluster 1)", 
            "Shop Uy Tín (cluster 0)", 
            "Shop Mới / Ngủ đông (cluster 2)"
        ],
        "Đặc điểm nhận diện": [
            "Dẫn đầu về số lượng followers và lượt đánh giá, phản hồi nhanh (> 93%)",
            "Điểm sao cao nhất (4.87), số lượng followers và đánh giá trung bình, phản hồi tốt (> 83%)",
            "Chỉ số chạm đáy (0 sao, 0 đánh giá, phản hồi thấp (< 50%))"
        ],
        "Đánh giá": [
            "Các thương hiệu lớn dẫn dắt thị trường",
            "Shop tầm trung hoạt động cực kỳ hiệu quả, giữ chân khách tốt",
            "Shop mới lập chưa có đơn hoặc đã ngưng hoạt động"
        ]
    })
    render_styled_table(naming_df)

# Áp tên nhóm vào DataFrame gốc để vẽ biểu đồ
cluster_name_map = {
    1: "Shop Dẫn Đầu",
    0: "Shop Uy Tín",
    2: "Shop Mới / Ngủ Đông"
}
df_c["Nhóm"] = df_c["Cluster"].map(cluster_name_map)

# Cập nhật bảng màu Okabe-Ito cho các nhóm để đảm bảo trực quan và thân thiện với người mù màu
colorblind_palette = {
    "Shop Dẫn Đầu": "#D55E00", 
    "Shop Uy Tín":  "#0072B2",
    "Shop Mới / Ngủ Đông": "#999999"
}

st.markdown("#### Biểu đồ 1b: Phân bố số lượng shop, doanh thu và lượt bán")

st.markdown("---")
# Hiển thị bộ lọc để người dùng có thể chọn trạng thái shop và khoảng doanh thu ước tính
col_f1, col_f2 = st.columns([1.2, 1]) # Chỉnh lại tỷ lệ cột cho đẹp

# Bộ lọc quy mô shop theo lượng follower
with col_f1:
    follower_filter = st.radio(
        "**Quy mô Shop (Theo lượng Follower):**", 
        options=["Tất cả", "Shop Nhỏ (< 1k)", "Tầm Trung (1k - 10k)", "Khổng Lồ (> 10k)"], 
        horizontal=True,
        key="follower_filter_unique"
    )

# Bộ lọc khoảng doanh thu ước tính
with col_f2:
    # Đổi đơn vị doanh thu sang triệu để slider dễ điều chỉnh hơn
    min_rev = float(df_c["revenue_est"].min()) / 1_000_000
    max_rev = float(df_c["revenue_est"].max()) / 1_000_000
    
    if max_rev > min_rev:
        selected_rev_trieu = st.slider(
            "Khoảng Doanh Thu (VND):",
            min_value=min_rev, 
            max_value=max_rev, 
            value=(min_rev, max_rev), 
            format="%d Triệu", # Hiển thị số trên slider với đơn vị triệu
            key="revenue_slider_unique"
        )
        
        # Chuyển lại giá trị đã chọn về đơn vị VND gốc để lọc dữ liệu
        selected_rev = (selected_rev_trieu[0] * 1_000_000, selected_rev_trieu[1] * 1_000_000)
    else:
        selected_rev = (min_rev * 1_000_000, max_rev * 1_000_000)

filtered_df = df_c.copy()

# Xử lý logic lọc Follower
if follower_filter == "Shop Nhỏ (< 1k)":
    filtered_df = filtered_df[filtered_df["follower_count"] < 1000]
elif follower_filter == "Tầm Trung (1k - 10k)":
    filtered_df = filtered_df[(filtered_df["follower_count"] >= 1000) & (filtered_df["follower_count"] <= 10000)]
elif follower_filter == "Khổng Lồ (> 10k)":
    filtered_df = filtered_df[filtered_df["follower_count"] > 10000]

# Xử lý logic lọc Doanh thu
filtered_df = filtered_df[(filtered_df["revenue_est"] >= selected_rev[0]) & (filtered_df["revenue_est"] <= selected_rev[1])]
st.markdown("---")

# Tính tổng doanh thu và lượt bán theo nhóm để vẽ biểu đồ
revenue_sales = filtered_df.groupby("Nhóm")[["revenue_est","total_sold"]].sum().reset_index()
shop_counts = filtered_df["Nhóm"].value_counts().reset_index()
shop_counts.columns = ["Nhóm","Số lượng Shop"]

# Hiển thị biểu đồ phân bố số lượng shop, doanh thu và lượt bán theo nhóm
col1, col2, col3 = st.columns(3)

chart_layout = dict(plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                    margin=dict(t=50,l=10,r=10,b=10),
                    font=dict(family="Arial",size=13,color="#333"))

# Biểu đồ 1b.1: Tỷ trọng số lượng shop theo nhóm
with col1:
    fig = px.pie(shop_counts, names="Nhóm", values="Số lượng Shop",
                 color="Nhóm", color_discrete_map=colorblind_palette,
                 title="Tỷ trọng số lượng Shop", hole=0.4)
    fig.update_traces(textinfo="percent", textfont_size=12)
    fig.update_layout(showlegend=True,
                      legend=dict(orientation="h",yanchor="bottom",y=-0.2,xanchor="center",x=0.5),
                      margin=dict(t=50,l=10,r=10,b=10))
    st.plotly_chart(fig, use_container_width=True)

# Biểu đồ 1b.2: Tổng doanh thu ước tính theo nhóm
with col2:
    fig = px.bar(revenue_sales, x="Nhóm", y="revenue_est", color="Nhóm",
                 color_discrete_map=colorblind_palette,
                 title="Tổng Doanh thu Ước tính (VND)",
                 labels={"revenue_est":"Doanh thu (VND)","Nhóm":""}, text_auto=".2s")
    fig.update_layout(**chart_layout)
    fig.update_yaxes(showgrid=True, gridcolor="#E5E5E5")
    fig.update_traces(textposition="outside")
    st.plotly_chart(fig, use_container_width=True)

# Biểu đồ 1b.3: Tổng lượt bán theo nhóm
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

# Insights và chiến lược đề xuất
with st.expander("Phân tích Insight: Tại sao có sự chênh lệch về hiệu quả kinh doanh?"):
    st.markdown("""
        #### <b>Các phát hiện từ kết quả phân cụm</b>
        
        1. **Nghịch lý giữa Số lượng và Doanh thu (Quy tắc Pareto):**
            * Nhìn vào **Biểu đồ tròn**, nhóm **Shop Dẫn Đầu** chỉ chiếm một tỷ lệ rất nhỏ về số lượng. Tuy nhiên, trên **Biểu đồ Doanh thu**, nhóm này lại nắm giữ "miếng bánh" khổng lồ, chi phối phần lớn dòng tiền của toàn ngành mỹ phẩm.
            * **Insight:** Thị trường mỹ phẩm Shopee đang bị dẫn dắt bởi một số ít các "ông lớn". Quyền lực và doanh số tập trung cực mạnh vào nhóm tinh hoa này thay vì chia đều cho số đông.

        2. **Sự đồng nhất giữa Doanh thu và Lượt bán:**
            * Cả hai biểu đồ cột đều cho thấy xu hướng tương đồng: Nhóm có doanh thu cao nhất cũng chính là nhóm có tổng lượt bán khủng nhất. 
            * **Insight:** Điều này cho thấy các **Shop Dẫn Đầu** không chỉ bán các sản phẩm giá trị cao mà còn có khả năng "chốt đơn" hàng loạt một cách đều đặn.

        3. **Sức mạnh của số đông:**
            * Biểu đồ tròn cho thấy nhóm **Shop Uy Tín** chiếm số lượng áp đảo. Chính vì vậy mà **tổng doanh thu và lượt bán** của cả nhóm này cộng lại cực kỳ đáng nể.
            * **Insight:** Đây là phân khúc khách hàng dễ tiếp cận nhất. Tuy nhiên, vì tổng doanh thu lớn này bị chia nhỏ cho hàng nghìn shop nên hiệu quả kinh doanh trên mỗi đầu shop sẽ không thể cao bằng nhóm Dẫn Đầu.
                    
        4. **Shop Mới / Ngủ Đông:**
            * Tuy nhóm này chiếm số lượng không đáng kể nhưng **Lượt bán hoàn toàn bằng 0**. 
            * **Insight:** Đây là trạng thái "tê liệt doanh số" do thiếu hụt lòng tin trầm trọng. Khi **Lượt bán = 0 dẫn đến Lượt đánh giá = 0**, khách hàng sẽ không bao giờ là người "thử nghiệm" đầu tiên. Nếu không có chiến dịch mồi đơn, các shop này sẽ mãi bị kẹt trong vòng lặp: **Không có lượt bán -> Không có khách mua -> Không có doanh thu.**
            * **Về việc không có lượt bán mà có doanh thu, có thể do một vài lý do sau:**
                * **Độ trễ hệ thống (Hàng Pre-order/Đang giao):** Đơn hàng đã được thanh toán (tạo ra dòng tiền) nhưng chưa giao thành công nên Shopee chưa cộng vào tổng Lượt bán.
                * **Hệ quả của việc "Buff đơn" ảo:** Shop gian lận bị Shopee phát hiện và phạt "xóa trắng" lượt bán về 0, nhưng công cụ thu thập dữ liệu (Scraper) vẫn lưu lại được mức doanh thu cũ.
        ---

        #### <b>Chiến lược đề xuất</b>

        1. **Giai đoạn "Phá băng" (Dành cho Shop Mới):**
            * **Mục tiêu:** Phá vỡ "Bẫy số 0" về lượt bán và lượt đánh giá.
            * **Hành động:** Tập trung mọi nguồn lực để có **10-20 đơn hàng đầu tiên**. Có thể chấp nhận lỗ vốn thông qua các chương trình "Voucher hoàn xu" hoặc "Mua kèm deal sốc" để đổi lấy lượt mua và đánh giá. Không có đánh giá, shop sẽ mãi "vô hình" dù sản phẩm có tốt đến đâu.

        2. **Giai đoạn "Vượt rào" (Dành cho Shop Uy Tín):**
            * **Mục tiêu:** Thoát khỏi sự cạnh tranh để tiến lên nhóm Dẫn Đầu.
            * **Hành động:** Tối ưu hóa **Tỷ lệ phản hồi (> 90%)** - Dữ liệu cho thấy đây là ranh giới giữa chuyên nghiệp và nghiệp dư.
        """, unsafe_allow_html=True)
        
st.markdown("---")

# MỤC TIÊU 5
st.subheader("Mục tiêu 5 — Tương quan Chỉ số Cửa hàng -> Doanh thu Sản phẩm")

# Chọn các chỉ số liên quan đến độ tin cậy và hiệu quả của shop để phân tích tương quan với doanh thu
corr_features = [f for f in ["follower_count","response_rate","rating_star","revenue_est"]
                 if f in df.columns]
corr_matrix = df[corr_features].corr()

# Đặt lại tên cột để dễ hiểu hơn trên biểu đồ
labels_map = {
    "follower_count":"Lượt theo dõi",
    "response_rate":"Tỷ lệ phản hồi",
    "rating_star":"Điểm sao",
    "revenue_est":"Doanh thu"
}

# Vẽ ma trận tương quan với Plotly Express
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

revenue_corrs = corr_matrix["revenue_est"].drop("revenue_est") if "revenue_est" in corr_matrix else pd.Series()
if not revenue_corrs.empty:
    strongest = revenue_corrs.abs().idxmax()
    strongest_r = revenue_corrs[strongest]

# Insights và chiến lược đề xuất
with st.expander("Phân tích Insight: Yếu tố nào ảnh hưởng mạnh nhất đến doanh thu?"):
    st.markdown("#### Các phát hiện quan trọng:")
        
    st.markdown(f"""
    1. **Lượt theo dõi là yếu tố có tương quan mạnh nhất với doanh thu:** 
        * Với hệ số tương quan **r = 0.216**, Lượt theo dõi là yếu tố có liên kết chặt chẽ nhất với sự tăng trưởng doanh thu. 
        * **Insight:** Trên Shopee, tệp khách hàng trung thành (followers) chính là nguồn doanh thu bền vững nhất. Shop có càng nhiều người theo dõi thì xác suất phát sinh đơn hàng càng cao.

    2. **Sự nghịch lý của Điểm đánh giá:**
       * Hệ số tương quan giữa Sao và Doanh thu đang ở mức thấp (âm nhẹ). 
        * **Insight:** Điều này không có nghĩa là Sao không quan trọng, mà thực tế cho thấy các Shop có doanh thu cực lớn thường bán lượng hàng khổng lồ, dẫn đến việc khó tránh khỏi một vài đánh giá tiêu cực, làm kéo nhẹ điểm trung bình xuống so với các shop nhỏ mới mở (ít khách nhưng 5 sao tuyệt đối).

    3. **Mối quan hệ giữa Chăm sóc khách hàng và Lòng tin:**
       * Tỷ lệ phản hồi có tương quan thuận với Doanh thu nhưng không quá mạnh.
        * **Insight:** Phản hồi nhanh giúp cải thiện tỷ lệ chuyển đổi đơn hàng. Tuy nhiên để đẩy tổng doanh thu lên quy mô lớn, Shop vẫn cần ưu tiên Marketing để tăng lượt theo dõi (Followers).            
    ---
                
    #### Lời khuyên hành động:
    * **Ưu tiên số 1:** Đầu tư vào các chiến dịch tăng Follower (như Flash Sale, Game, Voucher Follow) vì đây là con đường trực tiếp dẫn đến doanh thu.
    * **Ưu tiên số 2:** Có thể duy trì Tỷ lệ phản hồi cao để tăng doanh thu. Tuy nhiên, trả lời nhanh chủ yếu chỉ giúp giữ khách nhưng sẽ ít thu hút khách mới nếu thiếu đi Marketing.

    **Kết luận hành động:** Thay vì dàn trải nguồn lực, nhà bán hàng nên ưu tiên nguồn lực theo thứ tự: **Xây dựng Đánh giá (để có lòng tin) ➔ Tăng tốc độ Phản hồi (để chuyên nghiệp) ➔ Tích lũy Follower (để bùng nổ doanh thu)**.
""")

# Kết thúc dashboard
st.markdown("---")
st.caption("Dashboard được thực hiện bởi thành viên Thạch Ngọc Hân - MSSV: 23127361")