# app/stats_service.py

from datetime import datetime, timedelta
from app.firebase_utils import get_user_expenses

# ----- 1. 每日支出折線圖 -----
def get_daily_stats(user_id: str):
    expenses = get_user_expenses(user_id)
    daily = {}

    for exp in expenses:
        date = exp.get("date")
        amount = exp.get("amount") or 0

        if date:
            daily[date] = daily.get(date, 0) + amount

    # 排序後輸出
    return [
        {"date": date, "total": total}
        for date, total in sorted(daily.items())
    ]


# ----- 2. 類別圓餅圖 -----
def get_category_stats(user_id: str):
    expenses = get_user_expenses(user_id)
    cat_total = {}

    for exp in expenses:
        cat = exp.get("category") or "其他"
        amount = exp.get("amount") or 0

        cat_total[cat] = cat_total.get(cat, 0) + amount

    return [
        {"category": cat, "total": total}
        for cat, total in cat_total.items()
    ]


# ----- 3. 本週 vs 上週比較 -----
def get_week_compare_stats(user_id: str):
    today = datetime.today()
    start_of_week = today - timedelta(days=today.weekday())

    last_week_start = start_of_week - timedelta(days=7)
    last_week_end = start_of_week - timedelta(days=1)

    expenses = get_user_expenses(user_id)

    this_week = 0
    last_week = 0

    for exp in expenses:
        date_str = exp.get("date")
        if not date_str:
            continue

        amount = exp.get("amount") or 0
        date = datetime.strptime(date_str, "%Y-%m-%d")

        # 本週
        if date >= start_of_week:
            this_week += amount

        # 上週
        elif last_week_start <= date <= last_week_end:
            last_week += amount

    # 百分比變化
    if last_week == 0:
        change = "N/A"
    else:
        change_value = (this_week - last_week) / last_week * 100
        change = f"{change_value:+.1f}%"

    return {
        "this_week": this_week,
        "last_week": last_week,
        "change": change
    }
