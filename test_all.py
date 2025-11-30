# test_all.py
"""
🔥 IMPRA AI Backend 測試工具
自動測試以下功能：
1️⃣ Firebase 初始化
2️⃣ Firestore 讀取
3️⃣ Firestore 寫入
4️⃣ OpenAI 聊天模型
5️⃣ 綜合分析（Firestore + AI）
"""

from app.firebase_utils import init_firebase, get_user_expenses, save_user_expense
from app.ai_utils import call_openai_chat
import datetime


def test_firebase_init():
    print("\n=== 1️⃣ 測試 Firebase 初始化 ===")
    db = init_firebase()
    print("✅ Firebase 初始化成功")
    return db


def test_firestore_read():
    print("\n=== 2️⃣ 測試 Firestore 讀取 ===")
    expenses = get_user_expenses("child_test")
    if not expenses:
        print("⚠️ 找不到任何消費紀錄。")
    else:
        print(f"📦 取得 {len(expenses)} 筆資料：")
        for e in expenses:
            print(e)
    return expenses


def test_firestore_write():
    print("\n=== 3️⃣ 測試 Firestore 寫入 ===")
    test_data = {
        "date": datetime.date.today().isoformat(),
        "store": "測試商店",
        "amount": "88",
        "product_name": "麵包"
    }
    category = {"category": "餐飲", "confidence": 0.95}
    save_user_expense("child_test", test_data, category)
    print("✅ 寫入完成，請到 Firestore 查看 users/child_test/expenses")


def test_openai_chat():
    print("\n=== 4️⃣ 測試 OpenAI 聊天模型 ===")
    messages = [
        {"role": "system", "content": "你是一位友善又簡潔的理財顧問，請控制回答在60字內"},
        {"role": "user", "content": "幫我分析昨天在全聯花了120元買水果的支出"}
    ]
    reply = call_openai_chat(messages)
    print("💬 AI 回覆：", reply)


def test_firestore_and_ai():
    print("\n=== 5️⃣ 綜合測試 Firestore + AI ===")
    expenses = get_user_expenses("child_test")

    if not expenses:
        print("⚠️ 沒有消費資料可分析。")
        return

    # 安全轉換，確保 None 或非數字字串不會出錯
    total = sum(
    float(e.get("amount")) if isinstance(e.get("amount"), (int, float, str)) and str(e.get("amount")).strip() else 0
    for e in expenses
    )

    text_summary = f"使用者 child_test 的總支出為 {total} 元，分類如下：\n"
    for e in expenses:
        text_summary += f"- {e.get('store')}：{e.get('amount')} 元（{e.get('category')}）\n"

    messages = [
        {"role": "system", "content": "你是一位理財顧問，請根據以下支出提供簡短建議（不超過80字）："},
        {"role": "user", "content": text_summary}
    ]

    print("📤 傳送摘要給 AI：\n", text_summary)
    reply = call_openai_chat(messages)
    print("💬 AI 建議：\n", reply)


if __name__ == "__main__":
    print("🚀 開始執行 IMPRA 全系統測試...\n")

    db = test_firebase_init()
    expenses = test_firestore_read()
    test_firestore_write()
    test_openai_chat()
    test_firestore_and_ai()

    print("\n✅ 所有測試已執行完畢。")
