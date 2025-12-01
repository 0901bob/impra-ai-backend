# app/analysis_service.py

from datetime import datetime
import pandas as pd
from prophet import Prophet
from app.firebase_utils import get_user_expenses

# =============================
# 主功能：月度 AI 分析（摘要 + 預測）
# =============================

def monthly_analysis(user_id: str):

    expenses = get_user_expenses(user_id)

    if not expenses:
        return {
            "user_id": user_id,
            "message": "沒有任何消費紀錄",
            "summary": {},
            "forecast": [],
            "status": "empty"
        }

    # ======= 資料整理 =======
    df = []
    for exp in expenses:
        try:
            df.append({
                "ds": datetime.strptime(exp["date"], "%Y-%m-%d"),
                "y": float(exp["amount"])
            })
        except:
            continue

    df = pd.DataFrame(df).sort_values("ds")

    if len(df) < 3:
        return {
            "user_id": user_id,
            "message": "資料量不足，無法進行預測",
            "summary": build_summary(df),
            "forecast": [],
            "status": "too_small"
        }

    # ======= Prophet 預測 =======
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True
    )
    model.fit(df)

    # 預測未來 30 天
    future = model.make_future_dataframe(periods=30)
    forecast = model.predict(future)

    forecast_data = forecast[["ds", "yhat"]].tail(30).to_dict(orient="records")

    return {
        "user_id": user_id,
        "summary": build_summary(df),
        "forecast": [
            {"date": x["ds"].strftime("%Y-%m-%d"), "predict": round(x["yhat"], 2)}
            for x in forecast_data
        ],
        "status": "ok"
    }


# =============================
# 副功能：月度摘要（AI 分析）
# =============================

def build_summary(df: pd.DataFrame):

    if df.empty:
        return {}

    total = df["y"].sum()
    avg = df["y"].mean()
    max_day = df.iloc[df["y"].idxmax()]

    summary = {
        "total_spending": round(total, 2),
        "average_daily": round(avg, 2),
        "max_spending_date": max_day["ds"].strftime("%Y-%m-%d"),
        "max_spending_amount": round(max_day["y"], 2)
    }

    # ======= 異常偵測：大於平均兩倍 =======
    threshold = avg * 2
    anomalies = df[df["y"] > threshold]

    summary["anomaly_days"] = [
        {
            "date": row["ds"].strftime("%Y-%m-%d"),
            "amount": row["y"]
        }
        for _, row in anomalies.iterrows()
    ]

    return summary
