# Shopee Mỹ phẩm VN – Dashboard Phân tích Dữ liệu
> Lab 01 Trực quan hóa Dữ liệu | 23KHDL  
> Dataset: 20,658 SP - 5,746 shops - 23,989 reviews - Crawl: 18–19/3/2026

---

## Cấu trúc thư mục

```
shopee-beauty-analysis/
|- app.py                      <- Trang chủ + bản đồ objectives
|- data/
|   |- products.csv               
|	|- shops.csv
|	|- reviews.csv
|	|_ shopee_beauty.db
|- pages/
|	|- EDA.py               <- Tổng quan
|	|- 22127254.py       
|	|- 22127418.py       
|	|- 23127361.py
|	|_ 23127488.py
|- utils/
|	|- __init__.py
|	|_ helpers.py              <- Màu, load_data, tokenize, trendline (SHARED)
|- crawling/
|	|_ shopee_crawler.py       <- Code thu thập dữ liệu (nộp kèm)
|- requirements.txt
|_ README.md
```

---

## Setup & chạy

```bash
# 1. Clone repo
git clone <url>
cd shopee-beauty-analysis

# 2. Tạo venv
python -m venv venv
source venv/bin/activate   # Mac/Linux
venv\Scripts\activate      # Windows

# 3. Cài thư viện
pip install -r requirements.txt

# 4. Chạy
streamlit run app.py
```

---

## Phân công

| TV | Thành viên | File | Trạng thái |
|:--:|---|---|:--:|
| 22127254 | Trương Nguyễn Hiền Lương | `pages/22127254.py` | Cần hoàn thiện |
| 22127418 |  | `22127418.py` |  |
| 23127361 |  | `23127361.py` |  |
| 23127488 |  | `23127488.py` |  |

---

## Việc cần làm

Mở `pages/[MSSV].py` và làm theo thứ tự:

1. **Biểu đồ 1**
2. **Điền nhận xét Obj 1**
3. **Biểu đồ 2**
4. **Điền nhận xét Obj 2**
5. **Viết kết luận**

> Test trực tiếp bằng `streamlit run app.py`

---

## Git workflow

```bash
git checkout -b feature/[MSSV]      # tạo branch riêng
# ... làm xong ...
git add pages/[MSSV].py
git commit -m "[MSSV]: Add box plot + scatter + nhận xét Obj 1, 2"
git push origin feature/[MSSV]
# Tạo Pull Request -> merge vào main
```

**Quy tắc:** Mỗi TV chỉ commit file của mình — không đụng file TV khác.

---

## Nộp bài

```
[Nhom_01].zip
|- data/ (gồm link Google Drive)
|- crawling/shopee_crawler.py
|- app.py, pages/, utils/
|- requirements.txt
|_ [Nhom_01].pdf
```