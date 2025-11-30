import requests
import json

BASE_URL = "http://127.0.0.1:8000"

user_ids = ["child_test", "parent_demo"]

# 模擬一筆分類好的發票
sample_expense = {
    "date": "2025-11-10",
    "store": "7-ELEVEN",
    "product_name": "可口可樂",
    "amount": 45.0,
    "category": {               # ⬅️ 新增分類欄位
        "category": "餐飲類",
        "confidence": 0.98
    }
}

for uid in user_ids:
    payload = {
        "user_id": uid,
        **sample_expense
    }

    print(f"\n🚀 上傳發票給使用者：{uid}")
    resp = requests.post(f"{BASE_URL}/test_add_expense", json=payload)
    print("狀態碼：", resp.status_code)
    print("回傳結果：", json.dumps(resp.json(), indent=2, ensure_ascii=False))
