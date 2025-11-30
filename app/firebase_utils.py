# app/firebase_utils.py
import os
import re
from datetime import datetime

import firebase_admin
from firebase_admin import credentials, firestore

# ✅ 可選：自動載入 .env
try:
    from dotenv import load_dotenv, find_dotenv
    _dotenv_path = find_dotenv()
    if _dotenv_path:
        load_dotenv(_dotenv_path, override=False)
except Exception:
    pass


def init_firebase():
    """
    初始化 Firebase Admin SDK。
    以 .env 的 GOOGLE_APPLICATION_CREDENTIALS 指向 service account JSON。
    """
    if firebase_admin._apps:
        # 已初始化過
        return firebase_admin.get_app()

    cred_path = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")
    if cred_path:
        cred_path = cred_path.strip().strip('"').replace("\\", "/")

    if not cred_path:
        raise FileNotFoundError(
            "❌ 找不到 Firebase 憑證路徑。請在 .env 設定 GOOGLE_APPLICATION_CREDENTIALS，"
            "或於 PowerShell 設定：$env:GOOGLE_APPLICATION_CREDENTIALS='C:/.../impra-firebase-key.json'"
        )
    if not os.path.exists(cred_path):
        raise FileNotFoundError(
            f"❌ 指定的憑證檔不存在：{cred_path}\n"
            "請確認路徑是否正確（建議使用正斜線 / 並不要加引號）。"
        )

    cred = credentials.Certificate(cred_path)
    app = firebase_admin.initialize_app(cred)
    return app


def get_db():
    """確保已初始化後，回傳 Firestore client。"""
    if not firebase_admin._apps:
        init_firebase()
    return firestore.client()


def add_expense(user_id: str, expense_data: dict):
    """（非必要但可保留）寫入指定使用者支出資料。"""
    db = get_db()
    return db.collection("users").document(user_id).collection("expenses").add(expense_data)


def get_expenses(user_id: str):
    """（非必要但可保留）讀取指定使用者支出資料。"""
    db = get_db()
    docs = db.collection("users").document(user_id).collection("expenses").stream()
    return [doc.to_dict() for doc in docs]


def save_user_expense(user_id: str, data: dict, category: dict = None):
    """統一的寫入函式（主程式在用這個）。"""
    db = get_db()
    doc_ref = db.collection("users").document(user_id).collection("expenses").document()
    doc_ref.set({
        "date": data.get("date"),
        "store": data.get("store"),
        "amount": data.get("amount"),
        "product_name": data.get("product_name"),
        "category": category.get("category") if category else None,
        "confidence": category.get("confidence") if category else None,
        "created_at": firestore.SERVER_TIMESTAMP
    })
    print(f"✅ 已成功寫入 Firestore：users/{user_id}/expenses/{doc_ref.id}")


def get_user_expenses(user_id: str):
    """給 /advice、/chat 等流程讀資料用。"""
    db = get_db()
    expenses_ref = db.collection("users").document(user_id).collection("expenses")
    docs = expenses_ref.stream()
    expenses = []
    for doc in docs:
        data = doc.to_dict()
        # 時間格式可序列化
        for k, v in list(data.items()):
            if hasattr(v, "isoformat"):
                data[k] = v.isoformat()
        expenses.append(data)
    print(f"📦 讀取到 {len(expenses)} 筆 {user_id} 的消費資料。")
    return expenses


def clean_expense_data(data: dict):
    """（選用）統一欄位格式。"""
    # 統一金額
    if data.get("amount"):
        try:
            data["amount"] = float(re.sub(r"[^\d.]", "", str(data["amount"])))
        except Exception:
            data["amount"] = 0.0
    # 日期轉成 YYYY-MM-DD
    if isinstance(data.get("date"), datetime):
        data["date"] = data["date"].strftime("%Y-%m-%d")
    return data
