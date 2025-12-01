# app/classify_service.py

from app.ai_utils import get_embedding, cosine_similarity
import difflib  # 用於模糊比對

# ===== 類別語意描述 =====
CATEGORY_DESCRIPTIONS = {
    "餐飲": "食物 飲料 餐廳 甜點 咖啡 早餐 午餐 晚餐 飲品 速食 咖啡店 手搖",
    "交通": "搭車 捷運 高鐵 公車 停車費 計程車 汽油 交通費 共享機車 租車",
    "娛樂": "電影 電玩 遊戲 唱歌 旅遊 玩樂 Netflix 音樂 演唱會 手遊",
    "日用品": "衛生紙 牙膏 洗髮精 肥皂 清潔 家用品 生活用品 保養品",
    "教育": "學費 補習 書籍 課程 學習 教材 學校",
    "購物": "衣服 鞋子 包包 電腦 手機 配件 百貨 購物 網購",
}

# ===== 品牌資料庫（可再擴充） =====
BRAND_DB = {
    "星巴克": ["星巴克", "星巴剋", "星巴客", "Starbucks", "sbux"],
    "7-11": ["7-11", "7/11", "711", "seven", "統一超商"],
    "全家": ["全家", "familymart", "famimart", "family mart"],
    "路易莎": ["路易莎", "louisa"],
    "麥當勞": ["麥當勞", "麥當牢", "mcdonald", "mcd"],
    "摩斯": ["摩斯", "mos", "mosburger"],
    "可口可樂": ["可口可樂", "可口可了", "cocacola"],
    "家樂福": ["家樂福", "carrefour"],
    "全聯": ["全聯", "pxmart"],
}

# 對應品牌 → 類別
BRAND_CATEGORY = {
    "星巴克": "餐飲",
    "7-11": "餐飲",
    "全家": "餐飲",
    "路易莎": "餐飲",
    "麥當勞": "餐飲",
    "摩斯": "餐飲",
    "可口可樂": "餐飲",
    "家樂福": "購物",
    "全聯": "購物",
}


# ===== 模糊比對函式 =====
def fuzzy_match(name: str, candidate: str):
    """回傳兩字串相似度（0~1），大於 0.6 視為相似"""
    return difflib.SequenceMatcher(None, name, candidate).ratio()


# ===== 主分類函式 =====
def classify_product(product_name: str):
    """商品名稱 → 分類（品牌 + 模糊比對 + 語意分類）"""

    if not product_name:
        return {
            "input": product_name,
            "category": "未知",
            "confidence": 0.0,
            "method": "none"
        }

    name_lower = product_name.lower()

    # ===== 1️⃣ 品牌 + 模糊比對 =====
    best_brand = None
    best_score = 0

    for brand, variants in BRAND_DB.items():
        for variant in variants:
            score = fuzzy_match(name_lower, variant.lower())
            if score > best_score:
                best_score = score
                best_brand = brand

    # 如果模糊比對 >= 0.6 → 高置信度品牌分類
    if best_score >= 0.6:
        category = BRAND_CATEGORY.get(best_brand, "餐飲")
        return {
            "input": product_name,
            "normalized": best_brand,
            "category": category,
            "confidence": float(best_score),
            "method": "brand_fuzzy"
        }

    # ===== 2️⃣ 語意分類（Embedding）=====
    input_vec = get_embedding(product_name)
    scores = {}

    for category, desc in CATEGORY_DESCRIPTIONS.items():
        cat_vec = get_embedding(desc)
        scores[category] = cosine_similarity(input_vec, cat_vec)

    best_cat = max(scores, key=scores.get)
    confidence = float(scores[best_cat])

    return {
        "input": product_name,
        "category": best_cat,
        "confidence": confidence,
        "method": "semantic"
    }
