import streamlit as st

st.title("🏠 House Price Prediction — Vietnam 2024")
st.markdown("---")

# ── Giới thiệu ───────────────────────────────────────────
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Giới thiệu dataset")
    st.markdown(
        """
    Bộ dữ liệu **House Price Prediction Dataset Vietnam 2024** được thu thập từ trang
    [batdongsan.vn](https://batdongsan.vn), phân phối qua Kaggle.

    Dataset chứa thông tin về **30.000+ bất động sản nhà ở** tại Việt Nam, bao gồm:
    - Vị trí địa lý (tỉnh, huyện, dự án)
    - Đặc điểm vật lý (diện tích, mặt tiền, số tầng, phòng ngủ...)
    - Tình trạng pháp lý & nội thất
    - Giá bán (đơn vị: **tỷ đồng VND**)
    """
    )

with col2:
    st.subheader("⚡ Thông tin nhanh")
    st.info("📦 **30.000+** bất động sản")
    st.info("📍 Trải rộng **toàn quốc**")
    st.info("📅 Thu thập năm **2024**")
    st.info("💰 Đơn vị giá: **tỷ đồng VND**")

st.markdown("---")

# ── Backstory ────────────────────────────────────────────
st.subheader("📖 Bối cảnh thị trường")

tab1, tab2 = st.tabs(["🌊 Làn sóng sốt đất 2020–2022", "🏘️ Thị trường hai phân khúc"])

with tab1:
    st.markdown(
        """
    Giai đoạn **2020–2022**, thị trường BĐS Việt Nam trải qua nhiều đợt tăng giá bất thường,
    đặc biệt là đất nền vùng ven. Nhiều khu vực ven Hà Nội và các tỉnh như **Bắc Giang, Bắc Ninh,
    Hòa Bình, Hưng Yên** ghi nhận giá đất tăng cục bộ **40–50%** so với trước dịch.
    Cuối năm 2022, thị trường *"đóng băng"*, nhà đầu tư buộc phải hạ giá **20–30%** để thanh khoản.
    """
    )
    st.info(
        """
    📌 **Liên hệ với dataset**

    Dataset được thu thập năm 2024 — sau chu kỳ sốt đất và giai đoạn đóng băng — nên phản ánh
    mặt bằng giá **đã được điều chỉnh**, không phải đỉnh sốt. Tuy nhiên, dấu vết vẫn có thể
    quan sát được: **Hưng Yên** xuất hiện dày đặc trong dataset với các dự án Vinhomes Ocean Park
    quy mô lớn, và `Price_per_m2` tại đây có thể cao bất thường so với các tỉnh lân cận
    cùng quy mô kinh tế.
    """
    )

    st.markdown("**📰 Nguồn tham khảo**")
    st.markdown(
        """
    - 📰 [VnExpress: Nhà nước can thiệp khi giá nhà đất tăng hơn 20% trong 3 tháng](https://vnexpress.net/nha-nuoc-se-can-thiep-khi-gia-nha-dat-tang-hon-20-trong-3-thang-4779496.html)
    - 📰 [VnExpress: Có hiệu ứng và thắc thức của bát động sản 2022](https://vnexpress.net/co-hoi-va-thach-thuc-cua-bat-dong-san-2022-4422754.html)
    - 📰 [VnEconomy: Giá đất vùng ven có xu hướng tiếm cản giá khu vực trung tâm](https://vneconomy.vn/gia-dat-vung-ven-co-xu-huong-tiem-can-gia-khu-vuc-trung-tam.htm)
    - 📰 [Vov: Kinh té bát động sản Hà Nội — Tung sót nhiều nhất miền Bác giải ra sao?](https://vov.vn/kinh-te/bat-dong-san-hung-yen-tung-sot-nhat-mien-bac-gio-ra-sao-post1215445.vov)
    """
    )

with tab2:
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### 🏗️ Nhà dự án (`Is_Project = 1`)")
        st.markdown(
            """
        Vinhomes, Him Lam, Sun Casa,...

        - Giá định sẵn theo chính sách chủ đầu tư
        - Hạ tầng đồng bộ, pháp lý rõ ràng
        - Ít biến động, dễ dự đoán hơn
        """
        )
    with c2:
        st.markdown("#### 🏠 Nhà phố / thổ cư (`Is_Project = 0`)")
        st.markdown(
            """
        Nhà riêng lẻ, đất thổ cư,...

        - Giá phụ thuộc vị trí, mặt tiền, thương lượng
        - Biến động cao hơn, khó dự đoán hơn
        - Đa dạng về pháp lý
        """
        )

    st.info(
        """
    📌 Hai phân khúc được phân biệt qua cột `Is_Project`, cho phép so sánh trực tiếp
    hành vi giá, mức độ hoàn thiện pháp lý (`Has_certificate`) và tương quan giữa
    đặc điểm vật lý với giá bán giữa hai nhóm.
    """
    )

st.markdown("---")

# ── Điểm bất ngờ trong dữ liệu ──────────────────────────
st.subheader("⚠️ Tính ngoại lệ & Bất ngờ trong dữ liệu")
st.markdown("Bốn điểm cần lưu ý trước khi phân tích:")

c1, c2 = st.columns(2)

with c1:
    st.warning(
        """
    **1. Phân phối giá tập trung — không right-skewed như kỳ vọng**

    Trái với kỳ vọng, phân phối `Price` khá đối xứng, tập trung trong khoảng
    **1–10 tỷ đồng**. Dữ liệu từ batdongsan.vn có thể bị giới hạn ngầm ở phân khúc
    nhà ở phổ thông — biệt thự, đất dự án giá trăm tỷ gần như vắng mặt.
    """
    )
    st.warning(
        """
    **2. Chênh lệch giá/m² giữa các tỉnh — hai thế giới trong cùng dataset**

    `Price_per_m2` tại trung tâm TP.HCM có thể cao hơn tỉnh vùng sâu **50–100 lần**.
    Phân phối địa lý tạo ra hai cụm tách biệt rõ ràng theo `Province`.
    """
    )

with c2:
    st.warning(
        """
    **3. Nghịch lý diện tích — nhà nhỏ nhưng đắt hơn/m²**

    BĐS diện tích nhỏ đôi khi có `Price_per_m2` cao hơn nhà lớn, do vị trí trung tâm
    hoặc tiện ích đặc biệt. Mối quan hệ `Area` vs `Price_per_m2` không tuyến tính thuần túy.
    """
    )
    st.warning(
        """
    **4. Dự án đôi khi đắt hơn nhà phố cùng diện tích**

    Phản ánh kỳ vọng vào hạ tầng, tiện ích và thương hiệu chủ đầu tư. Có thể kiểm chứng
    trực tiếp bằng cách so sánh `Price_per_m2` theo cột `Is_Project`.
    """
    )

st.markdown("---")
