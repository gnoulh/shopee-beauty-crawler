import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

st.set_page_config(page_title="23127361", layout="wide")

# Bảng màu Okabe-Ito - Đảm bảo độ tương phản cao và dễ nhìn cho người mù màu
OKABE_ITO = [
    '#0072B2', # Xanh dương
    '#D55E00', # Đỏ cam (Vermilion)
    '#009E73', # Xanh ngọc
    '#E69F00', # Cam
    '#56B4E9', # Xanh da trời
    '#CC79A7', # Hồng tím
    '#F0E442', # Vàng
    '#000000'  # Đen (hoặc có thể dùng xám đậm)
]

# Một số tùy chỉnh CSS để cải thiện giao diện
st.markdown("""
<style>
    /* Tăng khoảng cách giữa các phần tử và làm nổi bật tiêu đề chính */
    .block-container { padding-top: 2rem; padding-bottom: 2rem; }
    
    /* Tùy chỉnh tiêu đề chính */
    .main-title {
        text-align: center;
        color: #000080 !important;
        font-weight: bold;
        margin-bottom: 10px;
    }
    
    /* Tùy chỉnh các KPI metrics */
    div[data-testid="stMetric"] {
        border: 2px solid #000080;
        border-radius: 10px;
        padding: 10px;
        background-color: #F4F6F9;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Căn giữa nhãn và giá trị trong KPI */
    div[data-testid="stMetricLabel"] {
        display: flex !important;
        justify-content: center !important;
        width: 100%;
    }

    /* Tùy chỉnh font và màu sắc cho nhãn KPI */
    div[data-testid="stMetricLabel"] > div {
        color: #000080;
        font-weight: bold;
        font-size: 15px;
        text-align: center !important;
        width: 100%;
    }
    
    /* Căn giữa giá trị trong KPI và tăng kích thước font */
    div[data-testid="stMetricValue"] {
        display: flex !important;
        justify-content: center !important;
        width: 100%;
    }
    
    /* Tùy chỉnh font và kích thước cho giá trị KPI */
    div[data-testid="stMetricValue"] > div {
        font-size: 28px;
        font-weight: 900 !important; 
        text-align: center !important;
    }
    
    /* Tùy chỉnh tiêu đề các phần*/
    .section-header {
        color: #1f77b4;
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 5px;
        margin-top: 30px;
        margin-bottom: 15px;
        font-size: 22px;
        font-weight: bold;
    }

    /* CSS CẬP NHẬT: Màu xanh pastel & chữ xanh Navy cho bảng tổng hợp */
    .custom-table {
        width: 100%;
        border-collapse: collapse;
        margin: 10px 0;
        font-family: sans-serif;
        color: #000080 !important; /* Toàn bộ chữ trong bảng màu Xanh Navy */
    }
    
    /* Tiêu đề bảng với nền xanh pastel và chữ xanh Navy */
    .custom-table th {
        background-color: #D0E4F5 !important;
        color: #000080 !important;         
        font-weight: bold !important;     
        text-align: center !important;
        padding: 12px;
        border-bottom: 2px solid #000080 !important;
    }
            
    /* Các hàng của bảng với viền nhạt màu pastel và căn giữa nội dung */
    .custom-table td {
        text-align: center !important;
        padding: 10px;
        border-bottom: 1px solid #D0E4F5;
    }
            
    /* Hiệu ứng hover với nền xanh pastel nhạt hơn để làm nổi bật hàng đang di chuột */
    .custom-table tr:hover {
        background-color: #F4F9FF !important;
    }
</style>
""", unsafe_allow_html=True)

# TIÊU ĐỀ CHÍNH
st.markdown("<h1 class='main-title'>Chân Dung & Giá Trị Bất Động Sản</h1>", unsafe_allow_html=True)

# Hàm tải và làm sạch dữ liệu, được cache để tối ưu hiệu suất
@st.cache_data
def load_data():
    df = pd.read_csv("data/vietnam_housing_dataset_cleaned.csv")
    df.columns = df.columns.str.strip()
    
    # Chuyển đổi các cột số sang kiểu dữ liệu số, bỏ qua lỗi và thay thế bằng NaN nếu không thể chuyển đổi
    numeric_cols = ['Price', 'Area', 'Access Road', 'Frontage', 'Bedrooms', 'Floors']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            
    df = df[(df['Price'] > 0) & (df['Area'] > 0)]
    
    # Chuẩn hóa giá trị của cột 'House direction' để đảm bảo tính nhất quán
    direction_mapping = {
        "1": "Đông", "1.0": "Đông", "Đông": "Đông",
        "2": "Tây", "2.0": "Tây", "Tây": "Tây",
        "3": "Nam", "3.0": "Nam", "Nam": "Nam",
        "4": "Bắc", "4.0": "Bắc", "Bắc": "Bắc",
        "5": "Đông Nam", "5.0": "Đông Nam", "Đông Nam": "Đông Nam",
        "6": "Tây Bắc", "6.0": "Tây Bắc", "Tây Bắc": "Tây Bắc",
        "7": "Tây Nam", "7.0": "Tây Nam", "Tây Nam": "Tây Nam",
        "8": "Đông Bắc", "8.0": "Đông Bắc", "Đông Bắc": "Đông Bắc"
    }
    if 'House direction' in df.columns:
        df['House direction'] = df['House direction'].astype(str).str.strip().map(direction_mapping)

    # Phân loại đường vào thành các nhóm để phân tích sâu hơn
    def categorize_road(r):
        if pd.isna(r): return "Không xác định"
        if r < 3: return "1. Hẻm nhỏ (<3m)"
        elif r <= 5: return "2. Hẻm xe hơi (3-5m)"
        elif r <= 10: return "3. Đường nhỏ/vừa (5-10m)"
        else: return "4. Mặt tiền/Đường lớn (>10m)"
        
    if 'Access Road' in df.columns:
        df['Road Category'] = df['Access Road'].apply(categorize_road)

    return df

try:
    df = load_data()

    if df.empty:
        st.error("Dữ liệu trống. Vui lòng kiểm tra lại file CSV gốc.")
        st.stop()

    # Chừa một khoảng trống để hiển thị các KPI tổng quan, sẽ được cập nhật sau khi áp dụng bộ lọc
    kpi_placeholder = st.empty() 

    # Thiết lập các slider để lọc dữ liệu theo giá và diện tích, giới hạn tối đa ở mức 95% để tránh ảnh hưởng của outliers
    max_p_slider = df['Price'].quantile(0.95)
    max_a_slider = df['Area'].quantile(0.95)
    
    col_filter1, col_filter2 = st.columns(2)
    with col_filter1:
        price_range = st.slider("Khoảng giá (Tỷ VND):", float(df['Price'].min()), float(df['Price'].max()), (0.0, float(max_p_slider)))
    with col_filter2:
        area_range = st.slider("Diện tích (m²):", float(df['Area'].min()), float(df['Area'].max()), (0.0, float(max_a_slider)))

    df_filtered = df[
        (df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1]) &
        (df['Area'] >= area_range[0]) & (df['Area'] <= area_range[1])
    ].copy()

    # Thay đổi các KPI tổng quan dựa trên dữ liệu đã lọc
    with kpi_placeholder.container():
        kpi1, kpi2, kpi3 = st.columns(3)
        kpi1.metric("TỔNG SỐ BĐS", f"{len(df_filtered):,}")
        kpi2.metric("GIÁ TRUNG BÌNH", f"{df_filtered['Price'].mean():.2f} Tỷ" if not df_filtered.empty else "0")
        kpi3.metric("DIỆN TÍCH TRUNG BÌNH", f"{df_filtered['Area'].mean():.1f} m²" if not df_filtered.empty else "0")
        
        st.write("") 
        
        # Thêm phần mô tả mục tiêu phân tích
        with st.expander("Mục tiêu phân tích"):
            st.markdown("""
            - **Mục tiêu 1:** Phân tích giá trị của **Đường vào**, **Mặt tiền** và **Phong thủy** ảnh hưởng đến giá Bất động sản.
            - **Mục tiêu 2:** Phác họa **"Chân dung"** bất động sản điển hình theo từng phân khúc ngân sách.
            """)

    # Mục tiêu 1 
    # Hàng 1: Đường vào & Mặt tiền
    row1_col1, row1_col2 = st.columns(2)
    
    with row1_col1:
        if 'Road Category' in df_filtered.columns:
            df_road = df_filtered[df_filtered['Road Category'] != "Không xác định"].groupby('Road Category')['Price'].mean().reset_index()
            df_road = df_road.sort_values('Road Category') 
            
            fig_road = px.bar(
                df_road, x='Road Category', y='Price', 
                color='Road Category', 
                title="Giá trung bình theo Cấp độ Đường vào",
                labels={'Road Category': 'Loại đường', 'Price': 'Giá TB (Tỷ VND)'},
                template='plotly_white', height=350,
                color_discrete_sequence=OKABE_ITO
            )
            fig_road.update_xaxes(tickvals=df_road['Road Category'], ticktext=[x[3:] for x in df_road['Road Category']])
            fig_road.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_road, use_container_width=True)

    with row1_col2:
        if 'Frontage' in df_filtered.columns:
            fig_frontage = px.scatter(
                df_filtered, x='Frontage', y='Price', opacity=0.5,
                title="Tương quan giữa Mặt tiền (m) và Giá (Tỷ)",
                labels={'Frontage': 'Độ rộng mặt tiền (m)', 'Price': 'Giá (Tỷ VND)'},
                color_discrete_sequence=[OKABE_ITO[1]], # Lấy màu đỏ cam làm điểm nhấn
                template='plotly_white', height=350
            )
            fig_frontage.update_layout(xaxis_range=[0, df_filtered['Frontage'].quantile(0.98)], margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_frontage, use_container_width=True)

    # Hàng 2: Phong thủy
    row2_col1, row2_col2 = st.columns(2)
    
    with row2_col1:
        if 'House direction' in df_filtered.columns:
            df_dir = df_filtered['House direction'].dropna().value_counts().reset_index()
            df_dir.columns = ['Direction', 'Count']
            
            # Tạo biểu đồ tròn để thể hiện thị phần phân bổ theo hướng nhà
            fig_pie_dir = px.pie(
                df_dir, names='Direction', values='Count',
                title="Thị phần phân bổ theo Hướng nhà",
                color='Direction',
                template='plotly_white', height=350,
                color_discrete_sequence=OKABE_ITO
            )
            fig_pie_dir.update_traces(textposition='inside', textinfo='percent+label')
            fig_pie_dir.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_pie_dir, use_container_width=True)

    with row2_col2:
        if 'House direction' in df_filtered.columns:
            df_dir_price = df_filtered.dropna(subset=['House direction']).groupby('House direction')['Price'].mean().reset_index().sort_values('Price')
            
            fig_dir_price = px.bar(
                df_dir_price, x='Price', y='House direction', 
                color='House direction', 
                orientation='h',
                title="Xếp hạng Giá trung bình theo Hướng nhà",
                labels={'Price': 'Giá TB (Tỷ VND)', 'House direction': 'Hướng'},
                color_discrete_sequence=OKABE_ITO,
                template='plotly_white', height=350
            )
            fig_dir_price.update_layout(showlegend=False, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_dir_price, use_container_width=True)

    st.markdown("---")

    # Mục tiêu 2
    if not df_filtered.empty:
        bins = [0, 3, 5, 10, float('inf')]
        labels = ['Dưới 3 tỷ', '3 - 5 tỷ', '5 - 10 tỷ', 'Trên 10 tỷ']
        df_filtered['Phân khúc'] = pd.cut(df_filtered['Price'], bins=bins, labels=labels)
        
        segment_summary = df_filtered.groupby('Phân khúc', observed=False).agg(
            Số_lượng=('Price', 'count'),
            DT_TB=('Area', 'mean'),
            PN_TB=('Bedrooms', 'mean'),
            Tầng_TB=('Floors', 'mean')
        ).reset_index()

        table_display = segment_summary.copy()
        table_display['DT_TB'] = table_display['DT_TB'].round(1)
        table_display['PN_TB'] = table_display['PN_TB'].round(1)
        table_display['Tầng_TB'] = table_display['Tầng_TB'].round(1)
        table_display.columns = ['Phân khúc ngân sách', 'Số lượng BĐS', 'Diện tích TB (m²)', 'Phòng ngủ TB', 'Số tầng TB']

        with st.expander("Bảng tổng hợp các tiện ích trung bình theo phân khúc"):
            html_table = table_display.to_html(index=False, classes='custom-table')
            st.markdown(html_table, unsafe_allow_html=True)

        st.markdown("##### Quy mô (Diện tích, Phòng, Tầng) theo mức giá")
        fig_line = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_line.add_trace(
            go.Scatter(x=segment_summary['Phân khúc'], y=segment_summary['DT_TB'], 
                       name="Diện tích TB (m²)", mode='lines+markers', 
                       line=dict(color=OKABE_ITO[0], width=3), marker=dict(size=8)), 
            secondary_y=False,
        )
        fig_line.add_trace(
            go.Scatter(x=segment_summary['Phân khúc'], y=segment_summary['PN_TB'], 
                       name="Phòng ngủ TB", mode='lines+markers', 
                       line=dict(color=OKABE_ITO[1], width=3), marker=dict(size=8)), 
            secondary_y=True,
        )
        fig_line.add_trace(
            go.Scatter(x=segment_summary['Phân khúc'], y=segment_summary['Tầng_TB'], 
                       name="Số tầng TB", mode='lines+markers', 
                       line=dict(color=OKABE_ITO[2], width=3), marker=dict(size=8)), 
            secondary_y=True,
        )

        fig_line.update_layout(
            template='plotly_white',
            hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            height=450,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        fig_line.update_yaxes(title_text="<b>Diện tích (m²)</b>", secondary_y=False, color=OKABE_ITO[0])
        fig_line.update_yaxes(title_text="<b>Số lượng (Phòng/Tầng)</b>", secondary_y=True)
        
        st.plotly_chart(fig_line, use_container_width=True)

except Exception as e:
    st.error(f"Lỗi hệ thống: {e}")

st.markdown("---")
st.caption("Dashboard được thực hiện bởi thành viên Thạch Ngọc Hân - MSSV: 23127361")