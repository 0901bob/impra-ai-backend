# app/main.py
import os
import tempfile
import base64
import json
import re
from datetime import date, datetime

from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from openai import OpenAI

# IMPRA 模組
from app.ai_utils import llm_extract_fields, call_openai_chat, transcribe_audio, infer_date_from_text
from app.classify_service import classify_product
from app.ocr_service import extract_invoice_data
from app.firebase_utils import init_firebase, save_user_expense
from app.advice_service import generate_financial_advice
from app.chat_service import handle_chat_request

app = FastAPI(
    title="IMPRA AI Backend",
    version="1.4",
    description="IMPRA AI 理財關聯系統後端（支援 OCR / 語音記帳 / AI 聊天）"
)

@app.on_event("startup")
def startup_event():
    try:
        init_firebase()
        print("✅ Firebase 初始化完成")
    except Exception as e:
        print(f"⚠️ Firebase 初始化失敗：{e}")

@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

@app.get("/health")
def health():
    return {"status": "ok"}


# ===== 文字分類 =====
class Item(BaseModel):
    product_name: str

@app.post("/classify")
def classify(item: Item):
    return classify_product(item.product_name)


# ===== OCR 辨識 + Firestore 入庫 =====
@app.post("/ocr")
async def ocr_upload(
    file: UploadFile = File(...),
    user_id: str = Body(None),
):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        # Step 1️⃣ OCR
        ocr_result = extract_invoice_data(tmp_path)
        raw_text = ocr_result.get("raw_text", "").strip()

        # Step 2️⃣ 前贅詞
        prompt_hint = "這是一張電子發票的 OCR 結果，請協助整理發票資訊。"
        full_text = f"{prompt_hint}\n{raw_text}" if raw_text else prompt_hint

        # Step 3️⃣ LLM 抽取欄位
        structured = llm_extract_fields(full_text) or {
            "date": None,
            "store": None,
            "product_name": None,
            "amount": None,
        }

        # Step 4️⃣ AI 分類
        classification = classify_product(structured.get("product_name", "") or "")

        # Step 5️⃣ Firestore 寫入
        if user_id and structured.get("amount"):
            try:
                save_user_expense(user_id, structured, classification)
            except Exception as e:
                print(f"⚠️ Firestore 寫入失敗：{e}")

        return {
            "message": "✅ 發票辨識完成並寫入 Firestore",
            "structured": structured,
            "classification": classification,
        }

    finally:
        os.remove(tmp_path)


# ===== 語音端點 =====
@app.post("/voice")
async def voice_input(
    file: UploadFile = File(...),
    mode: str = Body("record", description="record=記帳 / chat=對話"),
    user_id: str = Body(None),
):
    suffix = os.path.splitext(file.filename)[1] or ".m4a"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        raw_text = transcribe_audio(tmp_path)

        if mode == "record":
            # 1️⃣ LLM 抽取欄位
            structured = llm_extract_fields(raw_text) or {}

            # 2️⃣ 日期推斷／補全
            if not structured.get("date"):
                structured["date"] = infer_date_from_text(raw_text)

            # 3️⃣ 類別分類
            category = classify_product(structured.get("product_name", "") or {})

            # 4️⃣ Firestore 寫入
            if user_id and structured.get("amount"):
                save_user_expense(user_id, structured, category)

            return {
                "mode": "record",
                "raw_text": raw_text,
                "structured": structured,
                "category": category
            }

        elif mode == "chat":
            # 使用語音內容當作聊天輸入
            response = call_openai_chat(
                [
                    {
                        "role": "system",
                        "content": (
                            "你是一位理財聊天助理，會用輕鬆口語的方式回應使用者。"
                            "回覆請控制在 60 字以內，不要條列式，也不要太正式。"
                        ),
                    },
                    {"role": "user", "content": raw_text},
                ],
                max_tokens=120,
                temperature=0.6,
            )
            return {"mode": "chat", "raw_text": raw_text, "response": response}

        return {"error": f"未知模式：{mode}"}

    finally:
        os.remove(tmp_path)


# ===== 理財建議 =====
@app.post("/advice")
def get_advice(
    mode: str = Body(...),
    user_id: str = Body(...),
):
    from app.firebase_utils import get_user_expenses
    expenses = get_user_expenses(user_id)
    if not expenses:
        return {"warning": f"⚠️ 找不到使用者 {user_id} 的資料"}
    return generate_financial_advice(mode, {"user_id": user_id, "expenses": expenses})


# ===== Chat =====
class ChatRequest(BaseModel):
    user_id: str
    message: str
    mode: str = "auto"  # chat / advice / auto


@app.get("/ping")
def ping():
    return {"status": "alive"}


import asyncio
from fastapi import HTTPException

async def run_chat_async(user_id: str, message: str, mode: str):
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        None,
        lambda: handle_chat_request(user_id, message, mode)
    )


@app.post("/chat", summary="Chat Endpoint")
async def chat_endpoint(req: ChatRequest):

    try:
        reply = await asyncio.wait_for(
            run_chat_async(req.user_id, req.message, req.mode),
            timeout=25
        )
        return reply

    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="AI 回覆逾時，請稍後再試")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



# ===== OpenAI Vision OCR =====
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/ocr_vision")
async def ocr_vision(file: UploadFile = File(...), user_id: str = Body(None)):
    suffix = os.path.splitext(file.filename)[1] or ".jpg"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(await file.read())
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            image_b64 = base64.b64encode(f.read()).decode("utf-8")

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一位發票分析助手，請辨識這張發票並輸出 JSON，包含日期、商店名稱、商品名稱、金額。"
                    "請盡量轉成 YYYY-MM-DD 與阿拉伯數字金額。"
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "請幫我分析這張電子發票"},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"},
                    },
                ],
            },
        ]
        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, max_tokens=500
        )
        content = response.choices[0].message.content
        json_match = re.search(r"\{[\s\S]*\}", content)
        if not json_match:
            raise ValueError(f"⚠️ 模型未回傳有效 JSON：{content}")

        raw = json.loads(json_match.group(0))
        structured = {
            "date": raw.get("日期") or raw.get("發票日期"),
            "store": raw.get("商店名稱") or raw.get("店名"),
            "product_name": raw.get("商品名稱") or raw.get("品名"),
            "amount": raw.get("金額") or raw.get("總金額"),
        }
        if structured["date"]:
            try:
                structured["date"] = datetime.strptime(
                    str(structured["date"]).split()[0], "%Y-%m-%d"
                ).strftime("%Y-%m-%d")
            except Exception:
                pass

        classification = classify_product(structured.get("product_name", "") or {})
        if user_id:
            save_user_expense(user_id, structured, classification)

        return {
            "message": "✅ Vision 模型辨識完成並寫入 Firestore",
            "structured": structured,
            "classification": classification,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}
    finally:
        os.remove(tmp_path)
        
