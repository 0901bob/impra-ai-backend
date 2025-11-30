# app/chat_service.py
from app.ai_utils import call_openai_chat
from app.firebase_utils import get_user_expenses
from datetime import datetime

def handle_chat_request(user_id: str, message: str, mode: str = "auto"):
    """
    處理對話或理財建議請求。
    mode:
        - "chat": 一般對話
        - "advice": 財務建議
        - "auto": 自動判斷（根據輸入文字）
    """

    # 1️⃣ 取得使用者消費紀錄
    expenses = get_user_expenses(user_id)

    # 格式化消費資料（僅取最近 5 筆）
    if expenses:
        formatted_expenses = "\n".join([
            f"- {e.get('date', '未知日期')} | {e.get('store', '未知商店')} | "
            f"{e.get('product_name', '未知品項')} | {e.get('amount', '0')} 元 | "
            f"{e.get('category', '未分類')}"
            for e in expenses[-5:]
        ])
        summary = f"共取得 {len(expenses)} 筆紀錄，以下是最近 5 筆：\n{formatted_expenses}"
    else:
        formatted_expenses = "(查無消費資料)"
        summary = "(目前沒有可供分析的紀錄)"

    # 2️⃣ 自動判斷模式
    if mode == "auto":
        keywords = ["分析", "理財", "消費", "支出", "建議", "省錢", "報告", "趨勢", "花太多"]
        mode = "advice" if any(k in message for k in keywords) else "chat"

    # 3️⃣ 根據模式組合 System Prompt
    if mode == "advice":
        system_prompt = (
            "你是一位輕鬆好聊的 AI 理財顧問，會參考使用者的消費紀錄，"
            "用 2～3 句話給出重點建議。"
            "請控制在 80 字以內，避免條列式，不要幫我下標題。"
        )
        max_tokens = 200
        temperature = 0.4
    else:
        system_prompt = (
            "你是一位親切的理財聊天助理。"
            "可以聊理財，也可以簡單回應日常問題。"
            "回覆要自然、簡潔、有生活感，控制在 60 字以內，避免太制式。"
        )
        max_tokens = 120
        temperature = 0.6

    # 4️⃣ 組合對話內容
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": (
                f"使用者 {user_id} 的最近消費紀錄如下：\n"
                f"{formatted_expenses}\n\n"
                f"使用者的問題是：「{message}」"
            ),
        },
    ]

    # 5️⃣ 呼叫 OpenAI API
    reply = call_openai_chat(messages, max_tokens=max_tokens, temperature=temperature)

    # 6️⃣ 回傳統一格式
    return {
        "user_id": user_id,
        "mode": mode,
        "summary": summary,
        "reply": reply
    }
