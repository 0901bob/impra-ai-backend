# app/voice_service.py
import os
from datetime import date

from app.ai_utils import transcribe_audio, llm_extract_fields, infer_date_from_text
from app.classify_service import classify_product
from app.firebase_utils import save_user_expense


def handle_voice_record(file_path: str, user_id: str):
    """
    語音記帳主流程：
    1) 語音檔 → Whisper 轉文字
    2) 使用 LLM 抽取 date / store / product_name / amount
    3) 若沒有明確日期，依文字內容推斷（今天 / 昨天 / 前天 ...）
    4) 透過分類服務決定消費類別
    5) 寫入 Firestore
    """

    print("🎙️ 開始語音記帳流程")

    # 1️⃣ 語音轉文字
    raw_text = transcribe_audio(file_path)
    print(f"🗣️ 語音辨識結果：{raw_text}")

    if raw_text.startswith("⚠️"):
        return {"error": raw_text}

    # 2️⃣ LLM 抽取欄位
    structured = llm_extract_fields(raw_text) or {}
    print("🧩 LLM 抽取資訊：", structured)

    # 3️⃣ 日期補全／推斷
    if not structured.get("date"):
        structured["date"] = infer_date_from_text(raw_text)
    # 如果 amount 是字串也無妨，Firestore 寫入時仍可視需要轉型

    # 4️⃣ 分類
    classification = classify_product(structured.get("product_name", "") or "")
    print("📊 分類結果：", classification)

    # 5️⃣ 寫入 Firestore（若有 user_id 且有金額才寫）
    if user_id and structured.get("amount"):
        try:
            save_user_expense(user_id, structured, classification)
        except Exception as e:
            print(f"⚠️ Firestore 寫入失敗：{e}")

    return {
        "mode": "record",
        "raw_text": raw_text,
        "structured": structured,
        "category": classification
    }
