# app/ai_utils.py
import os
import json
import re
import math
from datetime import datetime, timedelta

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===== 通用 LLM 呼叫 =====
def call_openai_chat(messages, max_tokens: int = 400, temperature: float = 0.5) -> str:
    """
    呼叫 OpenAI Chat API。

    - 預設走「簡潔回答」路線，max_tokens 可依需求縮短或放寬
    - messages 為 OpenAI 標準格式
    """
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ OpenAI 呼叫失敗: {e}"


# ===== 語音轉文字 =====
def transcribe_audio(file_path: str) -> str:
    """
    使用 OpenAI Whisper 將語音檔轉成文字（支援自然語言敘述）。
    """
    try:
        with open(file_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="gpt-4o-mini-transcribe",
                file=audio_file
            )
        return transcript.text.strip()
    except Exception as e:
        return f"⚠️ 語音轉文字失敗: {e}"


# ===== 發票 / 語音欄位萃取 =====
def llm_extract_fields(raw_text: str):
    """
    使用 GPT 將 OCR 或語音辨識結果整理成結構化資料。

    輸入文字可能是：
    - 發票內容
    - 自然語言記帳，例如：「昨天在全聯買零食花了 120 元」

    要求輸出 JSON 物件，欄位：
    - date: 若是「昨天、前天、上週五」等相對日期就原樣保留；若有完整日期就用 YYYY-MM-DD
    - store: 商店名稱
    - product_name: 商品或消費項目名稱
    - amount: 只輸出阿拉伯數字金額（例如 120），不要加「元」或其他字
    未找到則給 null。
    """
    if not raw_text:
        return {}

    system_prompt = (
        "你是一個記帳助手，輸入內容可能是發票文字或自然語言描述。"
        "請從文字中萃取這些欄位：date, store, product_name, amount。"
        "規則："
        "1) 若文字只有相對日期（例如『昨天』『上週五』），就直接把那個詞放到 date。"
        "2) 若有完整日期（西元或民國），轉成 YYYY-MM-DD 放到 date。"
        "3) amount 一律轉成阿拉伯數字（例如『一百二十元』→ 120），不要附加單位。"
        "4) 若某欄位沒有資訊，請給 null。"
        "只輸出一個 JSON 物件，不要任何解釋文字。"
    )
    user_prompt = f"請從以下文字中抽取資料並依規則輸出 JSON：\n{raw_text}"

    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.3,
            max_tokens=300,
        )
        content = completion.choices[0].message.content.strip()

        # 安全解析 JSON（允許前後多餘字元）
        json_match = re.search(r"\{[\s\S]*\}", content)
        if json_match:
            json_str = json_match.group(0)
            return json.loads(json_str)
        else:
            print("⚠️ 無法解析 GPT 回傳內容:", content)
            return {}
    except Exception as e:
        print(f"❌ LLM 抽取失敗：{e}")
        return {}


# ===== 📅 日期推算模組 =====
def infer_date_from_text(text: str) -> str:
    """
    從語句中推斷日期：
    - 今天 / 昨天 / 前天 / 上週五 / 上個月
    回傳格式：YYYY-MM-DD
    """
    today = datetime.now()

    patterns = {
        "今天": 0,
        "昨日": -1,
        "昨天": -1,
        "前天": -2,
        "大前天": -3,
        "明天": 1,
        "後天": 2,
    }

    for key, delta in patterns.items():
        if key in text:
            target = today + timedelta(days=delta)
            return target.strftime("%Y-%m-%d")

    date_match = re.search(r"(\d{1,2})月(\d{1,2})[日号號]?", text)
    if date_match:
        month, day = map(int, date_match.groups())
        year = today.year
        if month > today.month + 1:
            year -= 1
        return f"{year}-{month:02d}-{day:02d}"

    return today.strftime("%Y-%m-%d")


# ===== 🔹 Embedding 與相似度函式（給 classify_service 用） =====
def get_embedding(text: str, model: str = "text-embedding-3-small"):
    """
    取得文字向量（embedding）
    """
    if not text.strip():
        return []
    try:
        resp = client.embeddings.create(model=model, input=text)
        return resp.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"❌ 取得 embedding 失敗：{e}")


def cosine_similarity(vec_a, vec_b) -> float:
    """
    計算兩個向量的餘弦相似度
    """
    if not vec_a or not vec_b:
        return 0.0
    dot, na, nb = 0.0, 0.0, 0.0
    for x, y in zip(vec_a, vec_b):
        dot += x * y
        na += x * x
        nb += y * y
    na, nb = math.sqrt(na), math.sqrt(nb)
    return dot / (na * nb) if na and nb else 0.0
