"""
Crawler for Beauty & Personal Care
Local Chrome CDP approach.

1. SETUP:
    pip install playwright pandas tqdm
    playwright install chromium

2. CHROME CDP SETUP:
    1. Close all Chrome windows
    2. Run: chrome --remote-debugging-port=9222 --user-data-dir=C:/chrome_debug
       (Mac: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome)
       (Linux: google-chrome --remote-debugging-port=9222 ...)
    3. Log in to shopee.vn in that Chrome window, wait a bit then pass captcha yourself
    4. Run this script in another terminal

HOW TO RUN (divided):
    python shopee_beauty_crawler.py products M1
    python shopee_beauty_crawler.py categories M1
    python shopee_beauty_crawler.py products M2
    python shopee_beauty_crawler.py categories M2
    python shopee_beauty_crawler.py products M3
    python shopee_beauty_crawler.py categories M3
    python shopee_beauty_crawler.py products M4
    python shopee_beauty_crawler.py categories M4
    python shopee_beauty_crawler.py reviews        # after products done
    python shopee_beauty_crawler.py shops          # after products done
    python shopee_beauty_crawler.py export         # at the end
"""

import asyncio, sys, os, json, random, time, sqlite3
from datetime import datetime
from urllib.parse import quote
from playwright.async_api import async_playwright
import time as time_module
import pandas as pd

# ==============================================================================
# CONFIG
# ==============================================================================

DB_PATH       = "data/shopee_beauty.db"
CDP_URL       = "http://localhost:9222"
PARENT_CATID  = 11036279   # Sắc Đẹp top-level
PAGE_SIZE     = 60
DELAY         = (5.0, 8.0)

# --- Sub-categories — crawls assigned keywords + cats ---
MEMBER_CONFIG = {
    "M1": {
        "keywords": [
            "serum duong da", "serum vitamin c", "serum niacinamide",
            "serum retinol", "serum ha", "serum phuc hoi",
            "sua rua mat", "toner", "nuoc hoa hong",
            "nuoc tay trang", "tay te bao chet", "mat na",
        ],
        "catids": [11036328],  # Chăm sóc da mặt
    },
    "M2": {
        "keywords": [
            "kem chong nang", "kem duong am", "kem duong trang da",
            "kem mat", "kem cc", "kem bb",
        ],
        "catids": [11036280],  # Tắm và chăm sóc cơ thể
    },
    "M3": {
        "keywords": [
            "son moi", "phan nen", "mascara",
            "phan mat", "ke mat"
        ],
        "catids": [11036314],  # Trang điểm
    },
    "M4": {
        "keywords": [
            "dau goi", "kem u toc", "xit duong toc",
            "luoc", "bong tay trang", "keo dan mi"
        ],
        "catids": [11036297, 11036321],  # Chăm sóc tóc & Phụ kiện làm đẹp
    },
}

ALL_KEYWORDS = [kw for m in MEMBER_CONFIG.values() for kw in m["keywords"]]

MAX_PAGES_PER_KEYWORD = 17  
MAX_PAGES_PER_CAT     = 17

BRANDS = [
    # International
    "innisfree", "some by mi", "the ordinary", "cetaphil", "la roche-posay",
    "bioderma", "neutrogena", "olay", "loreal", "l'oreal", "maybelline",
    "mac", "3ce", "romand", "laneige", "cosrx", "klairs", "the face shop",
    "etude house", "nature republic", "hada labo", "pond's", "ponds",
    "nivea", "vaseline", "dove", "pantene", "ohui", "whoo", "sulwhasoo",
    "skinfood", "tony moly", "missha", "a'pieu", "purito", "anessa",
    "shiseido", "sk-ii", "skii", "nars", "estee lauder", "lancome",
    "dior", "chanel", "ysl", "jo malone", "paula's choice", "torriden",
    "axis-y", "isntree", "round lab", "skin1004", "beauty of joseon",
    "d'alba", "manyo", "medicube", "dr.jart", "belif", "re:erth",
    "kiehl's", "kiehls", "clinique", "urban decay", "benefit", "too faced",
    "garnier", "biore", "rohto", "hada labo", "mentholatum", "kose",
    # Vietnamese local brands
    "co mem", "thorakao", "cỏ mềm", "seventh generation", "x5", "olic",
    "lana", "vedette", "ecofa", "hasaki", "pharmacity", "watsons",
    "aderma", "senka", "sunplay", "rohto", "biore", "safi", "himalaya",
    "biotique", "lotus", "lakme", "pond", "camellia", "emmié", "emmie",
    "vt cosmetics", "vt", "tia mo", "tiamo", "cỏ mềm", "nina dreams",
    "sevenlab", "cos de baha", "cos de", "haruto", "herbario",
    "cocoon", "olic", "ohui", "bbia", "hera", "iope", "su:m37",
]

# ============================================================================
# DATABASE
# ============================================================================

SCHEMA = """
PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

CREATE TABLE IF NOT EXISTS products (
    id                INTEGER  PRIMARY KEY AUTOINCREMENT,
    item_id           INTEGER  NOT NULL,
    shop_id           INTEGER  NOT NULL,
    sub_category      TEXT,
    sort_source       TEXT,
    name              TEXT,
    brand             TEXT,
    brand_normalized  TEXT,
    price             INTEGER,
    original_price    INTEGER,
    discount_pct      REAL,
    price_tier        TEXT,
    sold              INTEGER,
    monthly_sold      INTEGER,
    stock             INTEGER,
    rating            REAL,
    rating_count      INTEGER,
    r5 INTEGER, r4 INTEGER, r3 INTEGER, r2 INTEGER, r1 INTEGER,
    liked             INTEGER,
    is_mall           INTEGER,
    has_flash_sale    INTEGER,
    is_free_ship      INTEGER,
    revenue_est       INTEGER,
    image_count       INTEGER,
    product_url       TEXT,
    crawled_at        TEXT     DEFAULT (datetime('now')),
    UNIQUE(item_id, shop_id)
);

CREATE TABLE IF NOT EXISTS reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    item_id         INTEGER NOT NULL,
    shop_id         INTEGER NOT NULL,
    reviewer_id     TEXT,
    rating          INTEGER,
    review_text     TEXT,
    review_length   INTEGER,
    has_image       INTEGER,
    has_video       INTEGER,
    helpful_count   INTEGER,
    variation       TEXT,
    reviewed_ts     INTEGER,
    crawled_at      TEXT DEFAULT (datetime('now')),
    UNIQUE(item_id, reviewer_id, reviewed_ts)
);

CREATE TABLE IF NOT EXISTS shops (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    shop_id         INTEGER UNIQUE NOT NULL,
    shop_name       TEXT,
    is_mall         INTEGER,
    is_verified     INTEGER,
    follower_count  INTEGER,
    rating_star     REAL,
    rating_count    INTEGER,
    response_rate   REAL,
    location        TEXT,
    total_products  INTEGER,
    total_sold      INTEGER,
    shop_url        TEXT,
    crawled_at      TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS crawl_log (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    task      TEXT,
    status    TEXT,
    count     INTEGER,
    note      TEXT,
    ts        TEXT DEFAULT (datetime('now'))
);

CREATE INDEX IF NOT EXISTS idx_prod_item   ON products(item_id);
CREATE INDEX IF NOT EXISTS idx_prod_brand  ON products(brand_normalized);
CREATE INDEX IF NOT EXISTS idx_rev_item    ON reviews(item_id);
"""

def get_conn():
    os.makedirs("data", exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)
    print(f"[DB] Ready: {DB_PATH}")

def insert_products(conn, rows: list[dict]) -> int:
    sql = """
    INSERT OR IGNORE INTO products
    (item_id, shop_id, sub_category, sort_source, name, brand, brand_normalized,
     price, original_price, discount_pct, price_tier, sold, monthly_sold, stock,
     rating, rating_count, r5, r4, r3, r2, r1, liked,
     is_mall, has_flash_sale, is_free_ship, revenue_est, image_count, product_url)
    VALUES
    (:item_id,:shop_id,:sub_category,:sort_source,:name,:brand,:brand_normalized,
     :price,:original_price,:discount_pct,:price_tier,:sold,:monthly_sold,:stock,
     :rating,:rating_count,:r5,:r4,:r3,:r2,:r1,:liked,
     :is_mall,:has_flash_sale,:is_free_ship,:revenue_est,:image_count,:product_url)
    """
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)

def insert_reviews(conn, rows: list[dict]) -> int:
    sql = """
    INSERT OR IGNORE INTO reviews
    (item_id, shop_id, reviewer_id, rating, review_text, review_length,
     has_image, has_video, helpful_count, variation, reviewed_ts)
    VALUES
    (:item_id,:shop_id,:reviewer_id,:rating,:review_text,:review_length,
     :has_image,:has_video,:helpful_count,:variation,:reviewed_ts)
    """
    conn.executemany(sql, rows)
    conn.commit()
    return len(rows)

def insert_shop(conn, d: dict):
    sql = """
    INSERT OR REPLACE INTO shops
    (shop_id, shop_name, is_mall, is_verified, follower_count,
     rating_star, rating_count, response_rate, location,
     total_products, total_sold, shop_url)
    VALUES
    (:shop_id,:shop_name,:is_mall,:is_verified,:follower_count,
     :rating_star,:rating_count,:response_rate,:location,
     :total_products,:total_sold,:shop_url)
    """
    conn.execute(sql, d)
    conn.commit()

# ==============================================================================
# HELPERS
# ==============================================================================

def extract_brand(name: str, desc: str = "") -> str:
    text = (name + " " + desc).lower()
    for b in BRANDS:
        if b.lower() in text:
            return b.lower()
    return "unknown"

def parse_price(raw) -> int:
    if not raw or raw <= 0:
        return 0
    return int(raw) // 100000

def price_tier(p: int) -> str:
    if p <= 0:           return "unknown"
    elif p < 100_000:    return "budget"
    elif p < 300_000:    return "mid-low"
    elif p < 700_000:    return "mid"
    elif p < 1_500_000:  return "premium"
    else:                return "luxury"

def parse_item(raw: dict, cat_name: str, sort_src: str = "") -> dict | None:
    d       = raw.get("item_basic", raw)
    item_id = d.get("itemid")
    shop_id = d.get("shopid")
    if not item_id or not shop_id:
        return None
    ip   = parse_price(d.get("price", 0))
    op   = parse_price(d.get("price_before_discount", 0) or d.get("price_max", 0))
    if op == 0 or op < ip:
        op = ip
    disc = round((op - ip) / op * 100, 1) if op > 0 and op > ip else 0.0
    sold = d.get("sold", 0) or 0
    rc   = (d.get("item_rating") or {}).get("rating_count", [0]*6) or [0]*6
    name = d.get("name", "")
    brand = extract_brand(name)
    return {
        "item_id":         item_id,
        "shop_id":         shop_id,
        "sub_category":    cat_name,
        "sort_source":     sort_src,
        "name":            name,
        "brand":           brand,
        "brand_normalized": brand.lower().strip(),
        "price":           ip,
        "original_price":  op,
        "discount_pct":    disc,
        "price_tier":      price_tier(ip),
        "sold":            sold,
        "monthly_sold":    d.get("monthly_sold", 0) or 0,
        "stock":           d.get("stock", 0) or 0,
        "rating":          (d.get("item_rating") or {}).get("rating_star", 0) or 0,
        "rating_count":    rc[0] if rc else 0,
        "r5": rc[5] if len(rc) > 5 else 0,
        "r4": rc[4] if len(rc) > 4 else 0,
        "r3": rc[3] if len(rc) > 3 else 0,
        "r2": rc[2] if len(rc) > 2 else 0,
        "r1": rc[1] if len(rc) > 1 else 0,
        "liked":           d.get("liked_count", 0) or 0,
        "is_mall":         1 if d.get("shopee_verified") else 0,
        "has_flash_sale":  1 if d.get("flash_sale") else 0,
        "is_free_ship":    1 if d.get("show_free_shipping") else 0,
        "revenue_est":     ip * sold,
        "image_count":     len(d.get("images") or []),
        "product_url":     f"https://shopee.vn/product/{shop_id}/{item_id}",
    }

# ============================================================================
# CONNECT TO CHROME
# ============================================================================

async def connect_chrome(pw):
    """Connect to running Chrome with remote debugging. Returns (browser, page)."""
    print(f"[CDP] Connecting to Chrome at {CDP_URL}...")
    try:
        browser = await pw.chromium.connect_over_cdp(CDP_URL)
    except Exception as e:
        print(f"\n[ERROR] Cannot connect to Chrome: {e}")
        print("Make sure Chrome is running with --remote-debugging-port=9222")
        print("Command: chrome --remote-debugging-port=9222")
        raise

    ctx  = browser.contexts[0]
    page = await ctx.new_page()

    # Verify Shopee session
    await page.goto("https://shopee.vn", wait_until="domcontentloaded", timeout=30000)
    await asyncio.sleep(2)
    cookies = await ctx.cookies()
    spc_u = next((c for c in cookies if c["name"] == "SPC_U"), None)
    if not spc_u:
        print("[ERROR] Not logged in to Shopee. Please log in first.")
        await browser.close()
        raise RuntimeError("Not logged in")
    print(f"[CDP] Connected. Logged in (SPC_U={spc_u['value'][:8]}...)")
    return browser, page

# ============================================================================
# PRODUCT CRAWL — KEYWORD SEARCH
# ============================================================================

async def crawl_keyword(page, keyword: str, max_pages: int, conn) -> int:
    """
    Crawl products by navigating to Shopee search pages and intercepting the API response
    """
    total    = 0
    kw_label = keyword.replace(" ", "_")

    for page_num in range(1, max_pages + 1):
        captured = []

        async def on_response(response):
            if "/api/v4/search/search_items" in response.url and response.status == 200:
                try:
                    data = await response.json()
                    if data and data.get("items"):
                        captured.append(data)
                except Exception:
                    pass

        page.on("response", on_response)
        try:
            await page.goto(
                f"https://shopee.vn/search?keyword={quote(keyword)}"
                f"&sortBy=sales&page={page_num}",
                wait_until="networkidle",
                timeout=25000,
            )
            await asyncio.sleep(random.uniform(*DELAY))
        except Exception as e:
            print(f"  {keyword} p{page_num}: nav error {e}")
            page.remove_listener("response", on_response)
            break
        finally:
            page.remove_listener("response", on_response)

        if not captured:
            print(f"  {keyword} p{page_num}: no API response — stopping")
            break

        items = captured[0].get("items", [])
        if not items:
            print(f"  {keyword} p{page_num}: empty — stopping")
            break

        rows = [r for raw in items if (r := parse_item(raw, keyword, "kw_sales")) is not None]
        inserted = insert_products(conn, rows)
        total   += len(rows)
        print(f"  {keyword} p{page_num}: {len(rows)} items  (total={total})")

    return total


async def crawl_category(page, catid: int, cat_name: str, max_pages: int, conn) -> int:
    """
    Crawl a category by navigating to Shopee category pages (with slug format).
    The category page triggers the same /api/v4/search/search_items API.
    """
    total = 0
    
    # Build category slug URL (Shopee uses slug + catid format)
    # Example: https://shopee.vn/Chăm-sóc-da-mặt-cat.11036279.11036328
    category_slug = cat_name.lower().replace(" ", "-")
    parent_catid = PARENT_CATID  # 11036279 (Sắc Đẹp)

    for sort_by in ["sales", "popular", "ctime"]:
        print(f"  Trying sort={sort_by}...")
        for page_num in range(1, max_pages + 1):
            captured = []

            async def on_response(response):
                if "/api/v4/search/search_items" in response.url and response.status == 200:
                    try:
                        data = await response.json()
                        if data and data.get("items"):
                            captured.append(data)
                    except Exception:
                        pass

            page.on("response", on_response)
            try:
                # Use the slug-based category URL format
                sort_map = {"sales": "sales", "popular": "pop", "ctime": "newest"}
                category_url = (
                    f"https://shopee.vn/{category_slug}-cat.{parent_catid}.{catid}"
                    f"?sortBy={sort_map.get(sort_by, 'sales')}&page={page_num}"
                )
                await page.goto(
                    category_url,
                    wait_until="networkidle",
                    timeout=25000,
                )
                await asyncio.sleep(random.uniform(*DELAY))
            except Exception as e:
                print(f"    {cat_name} p{page_num} sort={sort_by}: nav error {e}")
                page.remove_listener("response", on_response)
                break
            finally:
                page.remove_listener("response", on_response)

            if not captured:
                print(f"    {cat_name} p{page_num} sort={sort_by}: no API response")
                break

            items = captured[0].get("items", [])
            if not items:
                print(f"    {cat_name} p{page_num} sort={sort_by}: empty")
                break

            rows = [r for raw in items if (r := parse_item(raw, cat_name, f"cat_{sort_by}")) is not None]
            total += len(rows)
            insert_products(conn, rows)
            print(f"    {cat_name} p{page_num}: {len(rows)} items  (total={total})")

            await asyncio.sleep(random.uniform(1.5, 3.0))

        if total > 0:
            break

    return total

# ==============================================================================
# REVIEWS CRAWL
# ==============================================================================

JS_FETCH = (
    "async ([url]) => {"
    "  try {"
    "    const r = await fetch(url, {"
    "      credentials: 'include',"
    "      headers: {'accept':'application/json','x-api-source':'pc'}"
    "    });"
    "    if (!r.ok) return {error: r.status};"
    "    return await r.json();"
    "  } catch(e) { return {error: e.toString()}; }"
    "}"
)

def parse_review(r: dict, item_id: int, shop_id: int) -> dict:
    text = r.get("comment", "") or ""
    pi   = r.get("product_items") or []
    return {
        "item_id":       item_id,
        "shop_id":       shop_id,
        "reviewer_id":   str(r.get("userid", "")),
        "rating":        r.get("rating_star", 0),
        "review_text":   text,
        "review_length": len(text),
        "has_image":     1 if r.get("images") else 0,
        "has_video":     1 if r.get("videos") else 0,
        "helpful_count": r.get("helpful_count", 0) or 0,
        "variation":     pi[0].get("name", "") if pi else "",
        "reviewed_ts":   r.get("ctime", 0),
    }

async def crawl_reviews(page, max_products: int = 800, reviews_per_product: int = 30):
    """Fetch reviews for top-rated products using page.evaluate fetch."""
    conn = get_conn()

    # Target products with most reviews, sorted by rating count descending
    rows = conn.execute("""
        SELECT item_id, shop_id, name, rating_count
        FROM   products
        WHERE  rating_count > 5
        ORDER  BY rating_count DESC
        LIMIT  ?
    """, (max_products,)).fetchall()

    print(f"\n[REVIEWS] Targeting {len(rows)} products")
    total = 0

    for i, row in enumerate(rows):
        item_id      = row["item_id"]
        shop_id      = row["shop_id"]
        reviews_got  = []

        for offset in range(0, reviews_per_product, 6):
            url = (
                f"https://shopee.vn/api/v4/item/get_ratings"
                f"?itemid={item_id}&shopid={shop_id}"
                f"&limit=6&offset={offset}&type=0&filter=0"
            )
            
            # Navigate to a product page first to set proper context
            if offset == 0:
                try:
                    await page.goto(
                        f"https://shopee.vn/product/{shop_id}/{item_id}",
                        wait_until="domcontentloaded",
                        timeout=15000,
                    )
                    await asyncio.sleep(random.uniform(1.0, 2.0))
                except Exception:
                    pass
            
            data = await page.evaluate(JS_FETCH, [url])
            
            # error=0 means success, error!=0 means failure
            if not data or (data.get("error") and data.get("error") != 0):
                break

            batch = (data.get("data") or {}).get("ratings", [])
            if not batch:
                break

            reviews_got.extend([parse_review(r, item_id, shop_id) for r in batch])
            await asyncio.sleep(random.uniform(2.0, 3.0))

        if reviews_got:
            insert_reviews(conn, reviews_got)
            total += len(reviews_got)

        if (i + 1) % 50 == 0:
            print(f"  Progress: {i+1}/{len(rows)} products | {total:,} reviews saved")
        
        await asyncio.sleep(random.uniform(2.0, 4.0))

    conn.execute("INSERT INTO crawl_log (task,status,count) VALUES ('reviews','done',?)", (total,))
    conn.commit()
    conn.close()
    print(f"\n[REVIEWS DONE] {total:,} reviews saved.")

# ==============================================================================
# SHOPS CRAWL
# ==============================================================================

def parse_shop(s: dict) -> dict:
    shop_id = s.get("shopid")
    username = s.get("username") or s.get("shop_name", "").replace(" ", "-").lower()
    return {
        "shop_id":        shop_id,
        "shop_name":      s.get("name", ""),
        "is_mall":        1 if s.get("is_official_shop") else 0,
        "is_verified":    1 if s.get("is_seller_cbf") else 0,
        "follower_count": s.get("follower_count", 0) or 0,
        "rating_star":    s.get("rating_star", 0) or 0,
        "rating_count":   s.get("rating_normal", 0) or 0,
        "response_rate":  s.get("response_rate", 0) or 0,
        "location":       s.get("shop_location", ""),
        "total_products": s.get("item_count", 0) or 0,
        "total_sold":     s.get("total_sold", 0) or 0,
        "shop_url":       f"https://shopee.vn/shop/{shop_id}/{username}" if username else f"https://shopee.vn/shop/{shop_id}",
    }

async def crawl_shops(page):
    conn  = get_conn()
    rows  = conn.execute(
        "SELECT DISTINCT shop_id FROM products"
    ).fetchall()
    total = 0
    print(f"\n[SHOPS] Fetching {len(rows)} unique shops")

    for i, row in enumerate(rows):
        shop_id = row["shop_id"]
        url     = f"https://shopee.vn/api/v4/product/get_shop_info?shopid={shop_id}"
        data    = await page.evaluate(JS_FETCH, [url])

        # Check if error is None (success) and data has shopid
        if data and data.get("error") is None and data.get("shopid"):
            insert_shop(conn, parse_shop(data))
            total += 1

        if (i + 1) % 100 == 0:
            print(f"  Progress: {i+1}/{len(rows)} | saved {total}")

        await asyncio.sleep(random.uniform(0.8, 1.5))

    conn.execute("INSERT INTO crawl_log (task,status,count) VALUES ('shops','done',?)", (total,))
    conn.commit()
    conn.close()
    print(f"\n[SHOPS DONE] {total:,} shops saved.")

# ==============================================================================
# EXPORT
# ==============================================================================

def export():
    conn = sqlite3.connect(DB_PATH)

    products = pd.read_sql("""
        SELECT p.*,
               s.shop_name, s.is_mall AS shop_is_mall,
               s.is_verified AS shop_is_verified,
               s.follower_count AS shop_followers,
               s.rating_star AS shop_rating,
               s.response_rate AS shop_response_rate,
               s.location AS shop_location,
               s.total_sold AS shop_total_sold
        FROM   products p
        LEFT JOIN shops s ON p.shop_id = s.shop_id
    """, conn)

    reviews = pd.read_sql("SELECT * FROM reviews", conn)
    shops   = pd.read_sql("SELECT * FROM shops",   conn)
    conn.close()

    os.makedirs("data", exist_ok=True)
    products.to_csv("data/products.csv", index=False, encoding="utf-8-sig")
    reviews.to_csv ("data/reviews.csv",  index=False, encoding="utf-8-sig")
    shops.to_csv   ("data/shops.csv",    index=False, encoding="utf-8-sig")

    print("\n" + "="*55)
    print("  DATA EXPORT REPORT")
    print("="*55)
    print(f"  Products:        {len(products):,}")
    print(f"  Unique shops:    {products['shop_id'].nunique():,}")
    print(f"  Mall products:   {products['is_mall'].sum():,}  ({products['is_mall'].mean()*100:.1f}%)")
    print(f"  With discount:   {(products['discount_pct']>0).sum():,}  ({(products['discount_pct']>0).mean()*100:.1f}%)")
    print(f"  Price range:     {products[products['price']>0]['price'].min():,} – {products['price'].max():,} VND")
    s = products[products['sold']>0]['sold']
    print(f"  Sold CV:         {s.std()/s.mean():.2f}")
    print(f"\n  Reviews:         {len(reviews):,}")
    print(f"  With text:       {(reviews['review_length']>20).sum():,}  ({(reviews['review_length']>20).mean()*100:.1f}%)")
    print(f"\n  Shops:           {len(shops):,}")
    print(f"  Mall shops:      {shops['is_mall'].sum():,}")
    print(f"  Verified shops:  {shops['is_verified'].sum():,}")
    print("="*55)
    print("  data/products.csv")
    print("  data/reviews.csv")
    print("  data/shops.csv")

# ==============================================================================
# MAIN RUNNER
# ==============================================================================

async def run_products(member: str = None):
    start_time = time_module.time()
    init_db()
    conn = get_conn()

    if member and member in MEMBER_CONFIG:
        keywords = MEMBER_CONFIG[member]["keywords"]
        catids   = MEMBER_CONFIG[member]["catids"]
    else:
        keywords = ALL_KEYWORDS
        catids   = [c for m in MEMBER_CONFIG.values() for c in m["catids"]]

    print(f"\n{'='*55}")
    print(f"  PRODUCT CRAWL  member={member or 'ALL'}")
    print(f"  Keywords: {len(keywords)}  |  Categories: {len(catids)}")
    print(f"  Max pages per keyword: {MAX_PAGES_PER_KEYWORD}")
    print(f"{'='*55}\n")

    async with async_playwright() as pw:
        browser, page = await connect_chrome(pw)
        grand_total   = 0

        # ------- Keywords ------------------------------------------------
        print(f"\n--- KEYWORD SEARCH ({len(keywords)} keywords) -------------------")
        for kw in keywords:
            print(f"\n {kw}")
            n = await crawl_keyword(page, kw, MAX_PAGES_PER_KEYWORD, conn)
            grand_total += n
            conn.execute(
                "INSERT INTO crawl_log (task,status,count) VALUES (?,?,?)",
                (f"kw:{kw}", "done", n)
            )
            conn.commit()

        # ------- Categories --------------------------------------------
        if catids:
            print(f"\n--- CATEGORY CRAWL ({len(catids)} categories) -------------------")
            for catid in catids:
                cat_name = next(
                    (c for m in MEMBER_CONFIG.values()
                     for c in m["catids"] if c == catid),
                    str(catid)
                )
                print(f"\n catid={catid}")
                n = await crawl_category(page, catid, str(catid), MAX_PAGES_PER_CAT, conn)
                grand_total += n
                conn.execute(
                    "INSERT INTO crawl_log (task,status,count) VALUES (?,?,?)",
                    (f"cat:{catid}", "done", n)
                )
                conn.commit()

        await browser.close()
        conn.close()

    with get_conn() as c:
        total_db = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]
    
    elapsed = time_module.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    
    print(f"\n{'='*55}")
    print(f"  PRODUCT CRAWL DONE - member={member or 'ALL'}")
    print(f"  Added this run:    {grand_total:,}")
    print(f"  Total in DB:       {total_db:,}")
    print(f"  Duration:          {hours}h {minutes}m {seconds}s")
    print(f"{'='*55}")


async def run_reviews():
    start_time = time_module.time()
    async with async_playwright() as pw:
        browser, page = await connect_chrome(pw)
        await crawl_reviews(page)
        await browser.close()
    
    elapsed = time_module.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"  Duration:          {hours}h {minutes}m {seconds}s")


async def run_shops():
    start_time = time_module.time()
    async with async_playwright() as pw:
        browser, page = await connect_chrome(pw)
        await crawl_shops(page)
        await browser.close()
    
    elapsed = time_module.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"  Duration:          {hours}h {minutes}m {seconds}s")


# ==============================================================================
# CAPTCHA HANDLING
# ==============================================================================

async def handle_captcha(page, timeout: int = 60):
    """Wait to solve captcha if it appears."""
    try:
        captcha_selectors = [
            'iframe[src*="recaptcha"]',
            'iframe[src*="hcaptcha"]',
            '.captcha-container',
            '[data-captcha]'
        ]
        
        for selector in captcha_selectors:
            if await page.query_selector(selector):
                print("  CAPTCHA DETECTED — Please solve it in the Chrome window")
                print(f"  Waiting {timeout}s to complete captcha...")
                await asyncio.sleep(timeout)
                print("  Continuing...")
                return True
    except Exception:
        pass
    return False


# ============================================================================
# CATEGORY CRAWL
# ============================================================================

async def run_category_crawl(member: str = None):
    """Run ONLY category crawl."""
    start_time = time_module.time()
    init_db()
    conn = get_conn()

    if member and member in MEMBER_CONFIG:
        catids = MEMBER_CONFIG[member]["catids"]
    else:
        catids = [c for m in MEMBER_CONFIG.values() for c in m["catids"]]

    print(f"\n{'='*55}")
    print(f"  CATEGORY CRAWL  member={member or 'ALL'}")
    print(f"  Categories: {len(catids)}")
    print(f"  Max pages per category: {MAX_PAGES_PER_CAT}")
    print(f"{'='*55}\n")

    async with async_playwright() as pw:
        browser, page = await connect_chrome(pw)
        grand_total = 0

        for catid in catids:
            print(f"\n Category ID: {catid}")
            
            # Check for captcha before starting
            await handle_captcha(page, timeout=120)
            
            n = await crawl_category(page, catid, str(catid), MAX_PAGES_PER_CAT, conn)
            grand_total += n
            conn.execute(
                "INSERT INTO crawl_log (task,status,count) VALUES (?,?,?)",
                (f"cat:{catid}", "done", n)
            )
            conn.commit()

        await browser.close()
        conn.close()

    with get_conn() as c:
        total_db = c.execute("SELECT COUNT(*) FROM products").fetchone()[0]

    elapsed = time_module.time() - start_time
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)

    print(f"\n{'='*55}")
    print(f"  CATEGORY CRAWL DONE - member={member or 'ALL'}")
    print(f"  Added this run:    {grand_total:,}")
    print(f"  Total in DB:       {total_db:,}")
    print(f"  Duration:          {hours}h {minutes}m {seconds}s")
    print(f"{'='*55}")


# ============================================================================
# CLI ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python shopee_crawler.py products [M1|M2|M3|M4]   — Crawl keywords")
        print("  python shopee_crawler.py categories [M1|M2|M3|M4]  — Crawl categories")
        print("  python shopee_crawler.py reviews                   — Crawl reviews")
        print("  python shopee_crawler.py shops                     — Crawl shops")
        print("  python shopee_crawler.py export                    — Export to CSV")
        sys.exit(1)

    cmd = sys.argv[1].lower()

    if cmd == "products":
        member = sys.argv[2].upper() if len(sys.argv) > 2 else None
        asyncio.run(run_products(member))
    elif cmd == "categories":
        member = sys.argv[2].upper() if len(sys.argv) > 2 else None
        asyncio.run(run_category_crawl(member))
    elif cmd == "reviews":
        asyncio.run(run_reviews())
    elif cmd == "shops":
        asyncio.run(run_shops())
    elif cmd == "export":
        export()
    else:
        print(f"Unknown command: {cmd}")
        sys.exit(1)