# app/classify_service.py

from app.ai_utils import get_embedding, cosine_similarity

# ===== 類別語意描述（語意分類用） =====
CATEGORY_DESCRIPTIONS = {
    "餐飲": "食物 飲料 餐廳 甜點 咖啡 早餐 午餐 晚餐 飲品 速食",
    "交通": "搭車 捷運 高鐵 公車 停車費 計程車 汽油 交通費",
    "娛樂": "電影 電玩 遊戲 唱歌 旅遊 玩樂 Netflix 音樂 演唱會",
    "日用品": "衛生紙 牙膏 洗髮精 肥皂 清潔 家用品 生活用品",
    "教育": "學費 補習 書籍 課程 學習 教材 學校",
    "購物": "衣服 鞋子 包包 電腦 手機 配件 百貨 購物",
}

# ===== 品牌 / 關鍵字快速分類（高準確度） =====
BRAND_KEYWORDS = {
    "餐飲": [
        "星巴克", "可口可樂", "可樂", "麥當勞", "mcdonald", "摩斯", "mos",
        "7-11", "7/11", "711", "全家", "familymart", "頂呱呱", "伯朗咖啡",
        "丹堤", "路易莎", "手搖", "飲料", "珍奶", "滷味", "便當", "火鍋"
    ],
    "交通": ["uber", "台鐵", "高鐵", "公車", "捷運", "加油", "停車", "機車"],
    "娛樂": ["netflix", "youtube", "電影院", "遊樂園", "ps5", "switch"],
    "購物": ["momo", "蝦皮", "pchome", "家樂福", "全聯", "ikea"],
}

# ===== 主分類函式 =====
def classify_product(product_name: str):
    """商品名稱 → 分類（品牌辨識＋語意分類）"""

    if not product_name:
        return {
            "input": product_name,
            "category": "未知",
            "confidence": 0.0
        }

    name_lower = product_name.lower()

    # ===== 1️⃣ 品牌 / 關鍵字 快速分類（速度快、準確度高） =====
    for category, keywords in BRAND_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name_lower:  # 錯字容忍度高
                return {
                    "input": product_name,
                    "category": category,
                    "confidence": 1.0
                }

    # ===== 2️⃣ 語意分類（Embedding 相似度） =====
    input_vec = get_embedding(product_name)
    scores = {}

    for category, desc in CATEGORY_DESCRIPTIONS.items():
        cat_vec = get_embedding(desc)
        scores[category] = cosine_similarity(input_vec, cat_vec)

    # 找最佳分類
    best = max(scores, key=scores.get)
    confidence = float(scores[best])

    return {
        "input": product_name,
        "category": best,
        "confidence": confidence
    }
