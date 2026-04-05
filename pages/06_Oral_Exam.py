"""
pages/05_Oral_Exam.py — Chuẩn bị vấn đáp
Tổng hợp giải thích phương pháp, hạn chế, SMART, colorblind-safe.
Dùng để ôn tập trước buổi vấn đáp — không cần chỉnh gì.
"""

import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

import streamlit as st
from utils.helpers import setup_sidebar

st.set_page_config(page_title="Chuẩn bị vấn đáp", page_icon="📖", layout="wide")
setup_sidebar()

st.title("📖 Chuẩn bị Vấn đáp – Giải thích toàn bộ phân tích")
st.caption("Đọc kỹ trước buổi bảo vệ — mỗi expander = 1 câu hỏi khả năng cao được hỏi")

st.markdown("---")

with st.expander("🗃️ 1. Dữ liệu đến từ đâu? Tại sao không dùng Kaggle?", expanded=True):
    st.markdown("""
    **Nguồn:** Shopee Public API (api.shopee.vn) — gọi trực tiếp endpoint tìm kiếm sản phẩm,
    chi tiết shop, và danh sách review. Code trong `crawling/shopee_crawler.py`.

    **Tại sao crawl thay vì Kaggle?** Đề bài bắt buộc. Dữ liệu Kaggle là cũ, không phản ánh thị trường hiện tại.

    **Hạn chế snapshot:** Chụp tại một thời điểm (18–19/3/2026), không phải chuỗi thời gian.
    → KHÔNG thể kết luận "tăng X% so với tháng trước".
    → Khi viết SMART: dùng **"snapshot 18/3/2026"** thay vì "6 tháng gần nhất" (không có trong data).

    **Reviews:** Shopee giới hạn ~30 reviews/sản phẩm mới nhất
    → 800 sản phẩm × ~30 = 23,989 reviews (không phải toàn bộ reviews lịch sử).
    """)

with st.expander("💬 2. TV2 — J-curve, TF-IDF, và tại sao cân bằng dataset"):
    st.markdown("""
    **J-curve:** Trên sàn TMĐT, phân bố rating hình chữ J — hầu hết 5★ hoặc 1★, ít ở giữa.
    Lý do: người hài lòng trung bình không đánh giá; chỉ người rất hài lòng hoặc rất thất vọng mới để lại feedback.

    **TF-IDF (Term Frequency–Inverse Document Frequency):**
    - **TF**: từ này xuất hiện bao nhiêu lần trong document này
    - **IDF**: từ này ít xuất hiện ở các document khác → mang nhiều thông tin hơn
    - **TF-IDF cao** = từ quan trọng và đặc trưng cho document đó
    - Logistic Regression học hệ số cho mỗi từ → hệ số âm = dấu hiệu tiêu cực, dương = tích cực

    **Tại sao cân bằng 4:1?**
    Nếu giữ nguyên 23,888 tích cực vs 47 tiêu cực → model học "luôn đoán tích cực" đạt accuracy 99.8%
    nhưng vô dụng (không phát hiện được review tiêu cực).
    Lấy 47 âm + 188 dương → model học được cả hai phía.
    Accuracy 91.5% trên tập cân bằng có ý nghĩa hơn 99.8% trên tập lệch.

    **3 đặc trưng phân biệt (Obj 1 TV2):**
    1. Tỷ lệ kèm ảnh: ~32% (tiêu cực) vs ~95% (tích cực)
    2. Độ dài text tự do: ~67 ký tự vs ~157 ký tự
    3. Nội dung từ khóa: màu/hình (tiêu cực) vs nhanh/thơm/cẩn thận (tích cực)
    """)

with st.expander("🏪 3. TV3 — K-Means: tại sao log, tại sao StandardScaler, cluster naming"):
    st.markdown("""
    **K-Means hoạt động:**
    1. Khởi tạo k tâm cụm ngẫu nhiên
    2. Gán mỗi điểm vào tâm cụm gần nhất (Euclidean distance)
    3. Cập nhật tâm cụm = trung bình của các điểm trong cụm
    4. Lặp lại đến khi hội tụ

    **Tại sao log-normalize?**
    Price và sold có phân bố lệch phải rất mạnh — vài sản phẩm sold 100,000+, hầu hết <1,000.
    Nếu không log, K-Means bị dominated bởi outliers. `log1p(x) = ln(x+1)` đưa phân bố về dạng đối xứng hơn.

    **Tại sao StandardScaler sau đó?**
    Price ~100,000–500,000, sold ~10–10,000 → đơn vị khác nhau hoàn toàn.
    Nếu không scale, K-Means ưu tiên price (giá trị lớn hơn) → cluster không phản ánh cả 2 chiều.

    **Silhouette score:** Đo mức độ một điểm "gần" với cụm của nó hơn so với cụm hàng xóm.
    Score từ -1 đến +1, cao hơn = phân cụm tốt hơn.

    **Cluster naming:** Không phải K-Means đặt tên — ta tính mean price và mean sold của mỗi cluster,
    so sánh với median tổng thể → 4 góc phần tư (cao/thấp × bán chạy/chậm) → đặt tên chiến lược.

    **Counterintuitive insight:** shop_rating tương quan âm với revenue không phải vì rating tốt thì bán ít,
    mà vì **Simpson's Paradox** — shop nhỏ chuyên biệt có rating 5.0 nhưng ít sản phẩm,
    trong khi mega-shop có rating 4.7 nhưng tổng revenue khổng lồ.
    """)

with st.expander("🗺️ 4. TV4 — Bubble chart, Ma trận BCG, Lollipop chart, ink-to-data ratio"):
    st.markdown("""
    **Bubble Chart:** 4 chiều trong 1 biểu đồ:
    - X = avg_sold, Y = total_revenue, Size = n_products, Color = avg_price
    - **Nhược điểm:** area encoding (kích thước bong bóng) kém chính xác hơn length encoding (bar chart)
    - **Ưu điểm:** truyền đạt 4 chiều cùng lúc → phù hợp với mục đích định vị chiến lược (direction), không so sánh chính xác số liệu

    **4 ô chiến lược (tương tự BCG Matrix):**
    - ⭐ **Ngôi sao** = sold cao + revenue cao: gia nhập nếu có nguồn lực
    - 💎 **Sinh lời** = sold thấp + revenue cao: margin cao, ít cạnh tranh volume
    - 🔄 **Phễu** = sold cao + revenue thấp: bán nhiều nhưng lợi nhuận thấp → dùng để xây reviews
    - ❓ **Câu hỏi** = sold thấp + revenue thấp: rủi ro cao, tránh nếu mới vào thị trường

    **Lollipop chart** (Biểu đồ 3):
    Kết hợp điểm tròn và đường kẻ = biến thể của bar chart.
    Giảm **"ink-to-data ratio"** (nguyên tắc Tufte): cùng thông tin nhưng ít pixels hơn → sạch hơn.
    Dùng tốt nhất khi so sánh nhiều categories mà bar chart trở nên nặng về thị giác.

    **Viridis colorscale:** Perceptually uniform (mỗi bước đều nhau về nhận thức thị giác)
    VÀ colorblind-safe → phù hợp cho continuous variable (avg_price).
    """)

with st.expander("🎨 5. Colorblind-safe — tại sao và áp dụng thế nào"):
    st.markdown("""
    **Vấn đề:** ~8% nam giới và ~0.5% nữ giới bị mù màu đỏ-xanh (deuteranopia/protanopia).
    Biểu đồ dùng đỏ=xấu, xanh=tốt sẽ không đọc được với họ — hai màu trông giống nhau.

    **Okabe-Ito palette (2008):** 8 màu phân biệt rõ cho cả 3 loại mù màu phổ biến nhất.
    Dashboard này dùng:
    - `#0072B2` (CB_BLUE) = tích cực / tốt
    - `#E69F00` (CB_ORANGE) = trung tính / highlight
    - `#D55E00` (CB_VERMIL) = tiêu cực / cảnh báo *(đỏ cam — vẫn phân biệt được với xanh)*

    **3 loại colorscale:**
    - **Sequential** (Blues, YlOrRd): dữ liệu có thứ tự (nhiều→ít)
    - **Categorical** (Okabe-Ito): phân biệt nhóm không có thứ tự
    - **Diverging** (RdBu): giá trị âm và dương quanh điểm giữa 0 (như correlation matrix)
    - **Viridis**: sequential nhưng perceptually uniform — an toàn với tất cả loại mù màu

    **Trong bài:** Tránh dùng đỏ-xanh lá cùng nhau. Khi cần đối lập → xanh dương vs cam.
    """)

with st.expander("📐 6. SMART — giải thích từng chữ, áp dụng đúng với snapshot data"):
    st.markdown("""
    | Chữ | Ý nghĩa | Áp dụng trong bài |
    |-----|---------|-------------------|
    | **S**pecific | Nêu rõ trường dữ liệu, dataset, thao tác | "47 reviews 1–3★ trong reviews.csv crawl 19/3/2026" |
    | **M**easurable | Kết quả có con số hoặc tiêu chí đo được | "xác định ≥3 đặc trưng", "top 5 danh mục" |
    | **A**chievable | Thực sự làm được với dữ liệu đang có | Không viết "trend theo tháng" khi chỉ có 1 ngày |
    | **R**elevant | Trả lời được câu hỏi kinh doanh thực tế | Thêm "để người bán X biết Y và làm Z" |
    | **T**ime-bound | Gắn rõ thời điểm/thời hạn | "snapshot 18/3/2026" thay vì "gần đây" |

    **Sai phổ biến:** "phân tích thị trường mỹ phẩm Việt Nam" → quá rộng, không S, không M, không T.
    **Đúng:** "So sánh trung vị sold của sản phẩm có discount 0–10% vs 20–30% trong phân khúc budget
    từ 20,658 sản phẩm crawl 18/3/2026, để xác định ngưỡng chiết khấu kích thích lượng bán."
    """)

with st.expander("⚠️ 7. Hạn chế — chủ động nêu ra trong vấn đáp để gây ấn tượng"):
    st.markdown("""
    Nêu ra hạn chế = chứng tỏ hiểu sâu — **không làm giảm điểm, ngược lại tăng điểm**.

    1. **47 reviews tiêu cực quá ít** → kết quả phân tích TV2 mang tính tham khảo, không đại diện
    2. **K-Means giả định cluster hình cầu** → nếu cluster thực tế không tròn, kết quả sai. Giải pháp: thử DBSCAN
    3. **revenue_est = price × sold** → ước tính, không phải doanh thu thực (không có thuế, phí, trả hàng)
    4. **Snapshot bias** → ngày 18/3/2026 có thể không đại diện cho ngày khác (cuối tháng, flash sale)
    5. **Correlation ≠ Causation** → followers tương quan với revenue nhưng không chắc tăng followers → tăng revenue
    6. **Bubble chart** → area encoding kém chính xác hơn position/length; phù hợp định vị chiến lược, không so sánh số liệu chính xác
    7. **Tokenizer đơn giản (regex)** → không xử lý được từ ghép tiếng Việt phức tạp; giải pháp: dùng underthesea hoặc PhoBERT
    """)

st.markdown("---")
st.success("✅ Nếu đọc hiểu hết 7 phần trên, bạn đã sẵn sàng cho vấn đáp. Chúc bảo vệ tốt! 🎯")
