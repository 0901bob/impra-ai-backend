from app.ai_utils import get_embedding, cosine_similarity

# 類別語意描述（可自行擴充）
CATEGORY_DESCRIPTIONS = {
    "餐飲": "食物、飲料、餐廳、咖啡、甜點、便當、早餐、午餐、晚餐、飲品",
    "交通": "搭車、捷運、高鐵、油錢、公車、停車費、計程車、車票、加油",
    "娛樂": "電影、遊戲、唱歌、旅遊、玩樂、Netflix、音樂、演唱會",
    "日用品": "衛生紙、牙膏、洗髮精、肥皂、清潔、家用品、生活用品",
    "教育": "學費、補習、書籍、課程、學習、學校、教材",
    "購物": "衣服、鞋子、包包、電腦、手機、配件、百貨"
}

def classify_product(product_name: str):
    """使用 Embedding 語意相似度分類（正式版）。"""
    input_vec = get_embedding(product_name)
    scores = {}

    for category, desc in CATEGORY_DESCRIPTIONS.items():
        cat_vec = get_embedding(desc)
        scores[category] = cosine_similarity(input_vec, cat_vec)

    best = max(scores, key=scores.get)
    confidence = float(scores[best])

    return {
        "input": product_name,
        "category": best,
        "confidence": confidence
    }


