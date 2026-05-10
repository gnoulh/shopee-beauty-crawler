"""
pages/Thu.py — Phân tích thị trường và Giá trị vô hình
================================================================
PHÂN CÔNG:
   Câu 1: Yếu tố định vị thương hiệu (Dự án vs Thổ cư lẻ): Việc một tài sản nằm trong một khu quy hoạch "Dự án"
     mang lại mức giá trị thặng dư (premium) chênh lệch bao nhiêu so với nhà thổ cư riêng lẻ bên ngoài? Tại điểm ngưỡng nào thì giá dự án bắt đầu bão hòa?
   Câu 2: Ảnh hưởng của pháp lý và nội thất: Trạng thái pháp lý (Legal status) và mức độ hoàn thiện nội thất (Furniture state)
     tác động ra sao đến tính thanh khoản và phân khúc giá?

================================================================
"""
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from utils.helpers import load_data
import datetime
from code_editor import code_editor
import plotly.express as px
import plotly.graph_objects as go

# --- 1. CẤU HÌNH TRANG ---
st.set_page_config(page_title="Định vị & Giá trị Vô hình", layout="wide", page_icon="📈")
st.title("Định vị & Giá trị Vô hình")


# Khởi tạo trạng thái AI (Human-in-the-loop)
if 'generated_code' not in st.session_state:
    st.session_state.generated_code = ""

# Khởi tạo trạng thái hệ thống
if 'ai_active' not in st.session_state:
    st.session_state.ai_active = False
if 'temp_code' not in st.session_state:
    st.session_state.temp_code = ""
if 'applied_code' not in st.session_state:
    st.session_state.applied_code = ""


# --- 2. LOAD DATA & FILTER DATA ---
try:
    df = load_data()
except Exception as e:
    st.error("Không thể tải dữ liệu. Vui lòng kiểm tra lại hàm load_data().")
    st.stop()

df['Loại hình'] = np.where(df['Is_Project'] == 1, "Dự án", "Thổ cư lẻ")

# --- 3. BỘ LỌC DỮ LIỆU (SIDEBAR) ---
st.header("🔍 Bộ Lọc Dữ Liệu")

# Lọc theo khoảng giá
min_price = float(df['Price'].min()) if not df['Price'].isnull().all() else 0.0
max_price = float(df['Price'].max()) if not df['Price'].isnull().all() else 100.0
price_range = st.slider(
    "Chọn khoảng giá (Tỷ VNĐ)",
    min_value=min_price, max_value=max_price,
    value=(min_price, max_price), step=0.5
)

# Áp dụng bộ lọc
filtered_df = df[(df['Price'] >= price_range[0]) & (df['Price'] <= price_range[1])]

st.markdown("---")
st.info(f"Đang hiển thị {len(filtered_df):,} giao dịch.")

# 4. GIAO DIỆN CHÍNH VỚI TRỢ LÝ AI (HUMAN-IN-THE-LOOP)

# st.write("<br>", unsafe_allow_html=True) # Căn chỉnh icon xuống dòng cho cân đối với tiêu đề
# # Sử dụng biểu tượng Robot để đại diện cho Trợ lý AI
# if st.button("🤖", help="Mở/Đóng Trợ lý AI Phân tích"):
#     st.session_state.ai_active = not st.session_state.ai_active
#     st.rerun()

# ==========================================
# KHU VỰC TRỢ LÝ AI (LUÔN KIỂM TRA TRẠNG THÁI)
# ==========================================
if st.session_state.get('ai_active', False):
    with st.container(border=True):
        st.subheader("🤖 Trợ lý AI - Soạn thảo phân tích")
        prompt = st.text_area("Yêu cầu AI thực hiện phân tích mới:", height=100)
        
        if st.button("🪄 Gen code & Biểu đồ xem trước"):
            # AI sinh code kèm comment giải thích [cite: 21, 55]
            st.session_state.temp_code = """# Đoạn code này sẽ vẽ Boxplot để so sánh thặng dư.
# Quy tắc 5s: Đưa thông tin quan trọng nhất lên tiêu đề.
fig, ax = plt.subplots(figsize=(10, 5))
sns.boxplot(data=df, x='Loại hình', y='Price', palette='Set2', ax=ax)
ax.set_title("XEM TRƯỚC: BIÊN ĐỘ THẶNG DƯ DỰ ÁN")
st.pyplot(fig)
"""

        # HIỂN THỊ KHU VỰC CHỈNH SỬA & XEM TRƯỚC
        if st.session_state.temp_code:
            st.markdown("#### 📝 Chỉnh sửa & Phê duyệt")
            
            # Bộ soạn thảo hỗ trợ phím tắt
            response = code_editor(st.session_state.temp_code, lang="python", height=[10, 20])
            current_code = response['text']
            
            # CẬP NHẬT: Thực thi xem trước ngay lập tức bên dưới editor [cite: 58, 59]
            st.info("🔍 Kết quả xem trước:")
            try:
                # Chạy trên môi trường local của người dùng
                exec_env = {"st": st, "pd": pd, "sns": sns, "plt": plt, "df": df}
                exec(current_code, exec_env)
            except Exception as e:
                st.error(f"Lỗi thực thi: {e}")

            # Nút Phê duyệt 
            if st.button("✅ PHÊ DUYỆT & APPLY", type="primary", use_container_width=True):
                st.session_state.applied_code = current_code
                st.session_state.ai_active = False # Đóng AI sau khi duyệt [cite: 27]
                st.rerun()

st.markdown("---")

# ==========================================
# KHU VỰC DASHBOARD CHÍNH
# ==========================================
if st.session_state.applied_code:
    st.success("✨ Đang hiển thị Dashboard tùy chỉnh từ AI")
    if st.button("⬅️ Trở về Dashboard mặc định"):
        st.session_state.applied_code = ""
        st.rerun()
    
    # Thực thi code đã được phê duyệt
    try:
        exec_env_main = {"st": st, "pd": pd, "sns": sns, "plt": plt, "df": df}
        exec(st.session_state.applied_code, exec_env_main)
    except Exception as e:
        st.error(f"Lỗi thực thi Dashboard: {e}")
        st.session_state.applied_code = ""

else:
    # Dashboard mặc định tuân thủ các quy tắc BI
    # st.write("Vui lòng sử dụng icon 🤖 để yêu cầu AI thực hiện phân tích chuyên sâu.")
    st.markdown("### 📌 Tổng quan thị trường")

    # Tính giá trung vị để tránh bị kéo lệch bởi outlier
    gia_du_an = filtered_df[filtered_df['Loại hình'].str.contains('Dự án', case=False, na=False)]['Price'].median()
    gia_tho_cu = filtered_df[filtered_df['Loại hình'].str.contains('Thổ cư|Nhà lẻ', case=False, na=False)]['Price'].median()

    # Xử lý trường hợp tính ra NaN
    gia_du_an = gia_du_an if pd.notna(gia_du_an) else 0
    gia_tho_cu = gia_tho_cu if pd.notna(gia_tho_cu) else 0
    chenh_lech = gia_du_an - gia_tho_cu

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric("Tổng số BĐS", f"{len(filtered_df):,}")
    kpi2.metric("Trung vị Giá Dự án", f"{gia_du_an:,.2f} Tỷ")
    kpi3.metric("Trung vị Giá Thổ cư", f"{gia_tho_cu:,.2f} Tỷ")
    # kpi4.metric("Giá trị thặng dư", f"{chenh_lech:,.2f} Tỷ", delta_color="normal")
    phan_tram = (chenh_lech / gia_tho_cu) * 100 if gia_tho_cu != 0 else 0

    kpi4.metric(
        label="Giá trị thặng dư", 
        value=f"{chenh_lech:,.2f} Tỷ", 
        delta=f"{phan_tram:+.2f}%", # Dấu + bên trong f-string sẽ tự hiển thị + hoặc -
        delta_color="normal"         # Tự động: Xanh nếu dương, Đỏ nếu âm
    )

    st.markdown("---")

    # --- 5. TỔ CHỨC CÁC PHẦN PHÂN TÍCH BẰNG TABS ---
    tab1, tab2 = st.tabs(["Dự án vs Thổ cư lẻ", "Pháp lý & Nội thất"])

    # ==========================================
    # TAB 1: YẾU TỐ ĐỊNH VỊ THƯƠNG HIỆU
    # ==========================================
    with tab1:
        with st.container(border=True):
            st.subheader("🏢 Yếu tố định vị thương hiệu")
            with st.expander("📌 Câu hỏi phân tích"):
            #     st.write("""**Nội dung:** Việc một tài sản nằm trong một khu quy hoạch "Dự án"
            #     mang lại mức giá trị thặng dư chênh lệch bao nhiêu so với nhà thổ cư riêng lẻ bên ngoài?
            # """)
                st.write("**Nội dung:** Mức độ Premium (thặng dư giá) của dự án biến thiên thế nào so với đất thổ cư, và điểm ngưỡng nào giá dự án bắt đầu bão hòa?")
        st.markdown("---") # Đường kẻ phân cách rõ ràng

        st.subheader(" Phân tích Thặng dư (Premium)")
        c1, c2 = st.columns([7, 3])
        with c1:
            fig1 = px.histogram(df, x="Price", color="Loại hình", marginal="box", 
                                 barmode="overlay", color_discrete_sequence=px.colors.qualitative.Set2)
            fig1.update_layout(plot_bgcolor="white", title="Phân phối mật độ giá & Điểm bão hòa")
            st.plotly_chart(fig1, use_container_width=True)
        with c2:
            # premium = df[df['Is_Project']==1]['Price'].median() - df[df['Is_Project']==0]['Price'].median()
            st.metric("Giá trị Thặng dư (Median)", f"{chenh_lech:,.2f} Tỷ", delta=f"{(chenh_lech/df[df['Is_Project']==0]['Price'].median())*100:.1f}%")
            st.write("**Giải thích:** Khoảng cách giữa hai đỉnh đồ thị thể hiện giá trị thương hiệu dự án.")

        st.markdown("#### Tác động của quy hoạch dự án đến giá trị Bất Động Sản")
        # st.write(f"Nhà trong khu dự án đang có mức giá cao hơn nhà thổ cư tự do khoảng **{chenh_lech:,.2f} tỷ VNĐ** (dựa trên trung vị).")

        col1_t1, col2_t1 = st.columns([6, 4])
        
        with col1_t1:
            # Biểu đồ phân tán giá
            fig1, ax1 = plt.subplots(figsize=(8, 5))
            sns.boxplot(data=filtered_df, x='Loại hình', y='Price', ax=ax1, palette='Set2')
            ax1.set_title("Phân phối giá bán: Dự án vs Thổ cư", fontsize=14, pad=15)
            ax1.set_ylabel("Giá bán (Tỷ VNĐ)")
            ax1.set_xlabel("")
            sns.despine()
            st.pyplot(fig1)
            
        with col2_t1:
            # Biểu đồ tỷ trọng (Thanh khoản/Nguồn cung)
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            sns.countplot(data=filtered_df, x='Loại hình', ax=ax2, palette='Set2')
            ax2.set_title("Nguồn cung (Thanh khoản) theo Loại hình", fontsize=14, pad=15)
            ax2.set_ylabel("Số lượng tin đăng")
            ax2.set_xlabel("")
            sns.despine()
            st.pyplot(fig2)

    # ==========================================
    # TAB 2: ẢNH HƯỞNG CỦA PHÁP LÝ VÀ NỘI THẤT
    # ==========================================
    with tab2:
        with st.container(border=True):
            st.subheader("⚖️ Tác động của Pháp lý & Nội thất")
            with st.expander("📌 Câu hỏi phân tích"):
            #     st.write("""**Nội dung:** Trạng thái pháp lý (Legal status) và mức độ hoàn thiện nội thất (Furniture state)
            #     tác động ra sao đến tính thanh khoản và phân khúc giá?
            # """)
                st.write("**Nội dung:** Trạng thái pháp lý và mức độ hoàn thiện nội thất tác động ra sao đến tính thanh khoản và phân khúc giá?")

        st.markdown("---") # Đường kẻ phân cách rõ ràng
        st.markdown("#### Đánh giá tính thanh khoản và phân khúc giá dựa trên Tình trạng")
        
        # Chia làm 2 hàng phân tích
        # HÀNG 1: Pháp lý
        st.subheader("Tương quan Đa nhân tố Pháp lý – Nội thất trong Định giá Tài sản")
        heatmap_data = df.groupby(['Legal status', 'Furniture state'])['Price'].median().reset_index()
        fig_heat = px.density_heatmap(heatmap_data, x="Legal status", y="Furniture state", z="Price",
                                        color_continuous_scale='YlOrRd', text_auto='.2f')
        st.plotly_chart(fig_heat, use_container_width=True)


        st.subheader("A. Trạng thái Pháp lý (Legal Status)")
        col1_t2, col2_t2 = st.columns(2)
        
        with col1_t2:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.violinplot(data=filtered_df, x='Legal status', y='Price', ax=ax3, palette='muted', inner='quartile')
            ax3.set_title("Phân khúc giá theo Pháp lý", pad=10)
            ax3.set_ylabel("Giá bán (Tỷ VNĐ)")
            ax3.set_xlabel("")
            plt.xticks(rotation=15)
            st.pyplot(fig3)
            
        with col2_t2:
            fig4, ax4 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=filtered_df, y='Legal status', ax=ax4, palette='muted', order=filtered_df['Legal status'].value_counts().index)
            ax4.set_title("Thanh khoản (Số lượng giao dịch) theo Pháp lý", pad=10)
            ax4.set_xlabel("Số lượng BĐS")
            ax4.set_ylabel("")
            st.pyplot(fig4)

        st.markdown("---")
        
        # HÀNG 2: Nội thất
        st.subheader("B. Mức độ hoàn thiện Nội thất (Furniture State)")
        col3_t2, col4_t2 = st.columns(2)
        
        with col3_t2:
            fig5, ax5 = plt.subplots(figsize=(8, 4))
            sns.boxplot(data=filtered_df, x='Furniture state', y='Price', ax=ax5, palette='pastel')
            ax5.set_title("Phân khúc giá theo Nội thất", pad=10)
            ax5.set_ylabel("Giá bán (Tỷ VNĐ)")
            ax5.set_xlabel("")
            plt.xticks(rotation=15)
            st.pyplot(fig5)
            
        with col4_t2:
            fig6, ax6 = plt.subplots(figsize=(8, 4))
            sns.countplot(data=filtered_df, x='Furniture state', ax=ax6, palette='pastel', order=filtered_df['Furniture state'].value_counts().index)
            ax6.set_title("Thanh khoản (Số lượng giao dịch) theo Nội thất", pad=10)
            ax6.set_ylabel("Số lượng BĐS")
            ax6.set_xlabel("")
            plt.xticks(rotation=15)
            st.pyplot(fig6)


        # st.subheader(" Trạng thái Pháp lý (Legal Status) và Mức độ hoàn thiện Nội thất (Furniture State)")
        # col1_t2, col2_t2 = st.columns(2)
        
        # with col1_t2:
        #     fig3, ax3 = plt.subplots(figsize=(8, 4))
        #     sns.violinplot(data=filtered_df, x='Legal status', y='Price', ax=ax3, palette='muted', inner='quartile')
        #     ax3.set_title("Phân khúc giá theo Pháp lý", pad=10)
        #     ax3.set_ylabel("Giá bán (Tỷ VNĐ)")
        #     ax3.set_xlabel("")
        #     plt.xticks(rotation=15)
        #     st.pyplot(fig3)
            
        # with col2_t2:
        #     fig6, ax6 = plt.subplots(figsize=(8, 4))
        #     sns.countplot(data=filtered_df, x='Furniture state', ax=ax6, palette='pastel', order=filtered_df['Furniture state'].value_counts().index)
        #     ax6.set_title("Thanh khoản (Số lượng giao dịch) theo Nội thất", pad=10)
        #     ax6.set_ylabel("Số lượng BĐS")
        #     ax6.set_xlabel("")
        #     plt.xticks(rotation=15)
        #     st.pyplot(fig6)

        # st.markdown("---")
    