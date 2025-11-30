# app/advice_service.py
from app.ai_utils import call_openai_chat

def generate_financial_advice(mode: str, user_data: dict):
    """
    根據使用者消費資料生成理財建議。
    mode 可為：
      - predict：預測未來支出趨勢
      - suggest：提供節省建議
      - analyze：總體支出分析
    """
    expenses = user_data.get("expenses", [])
    user_id = user_data.get("user_id", "unknown")

    if not expenses:
        return {
            "warning": f"⚠️ 找不到使用者 {user_id} 的消費資料。請先記錄一些支出。"
        }

    # 組合使用者支出摘要文字
    total = 0
    category_summary = {}
    for e in expenses:
        try:
            amount = float(e.get("amount", 0))
            total += amount
            cat = e.get("category", "未分類")
            category_summary[cat] = category_summary.get(cat, 0) + amount
        except Exception:
            continue

    summary_text = ", ".join([f"{k}：{int(v)}元" for k, v in category_summary.items()])
    base_summary = f"近期期支出總共約 {int(total)} 元，各分類金額為：{summary_text}。"

    # 根據 mode 產生不同的分析重點
    if mode == "predict":
        prompt = base_summary + "請用 2～3 句話預測他接下來一週可能的支出變化，語氣口語自然。"
    elif mode == "suggest":
        prompt = base_summary + "請用 2～3 句話給出具體的省錢與理財建議，避免條列，直接講重點。"
    else:  # analyze
        prompt = base_summary + "請用 2～3 句話說明目前的消費重點、哪一類支出比例較高，並給一個簡單建議。"

    messages = [
        {
            "role": "system",
            "content": (
                "你是一位專業但講話輕鬆的理財顧問。"
                "請根據提供的支出摘要給出簡短建議，控制在 80 字以內，約 2～3 句話即可。"
                "避免使用條列式，不要幫我下標題，也不要重複輸入的數字敘述太多次。"
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        reply = call_openai_chat(messages, max_tokens=200, temperature=0.4)
        return {
            "summary": base_summary,
            "advice": reply,
            "data_count": len(expenses)
        }
    except Exception as e:
        return {"error": str(e)}
