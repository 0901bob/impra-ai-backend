from PIL import Image, ImageOps, ImageFilter
import pytesseract
import re
import os
from datetime import datetime

# 1) 指定 tesseract.exe 位置（若安裝目錄不同請修改）
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


def _available_langs_ok() -> bool:
    """檢查是否有 chi_tra 語言包（繁中）"""
    try:
        langs = pytesseract.get_languages(config="")
        return "chi_tra" in langs
    except Exception:
        return False


def _preprocess(img: Image.Image) -> Image.Image:
    """影像前處理：灰階、放大、對比、銳利、簡單二值化"""
    img = img.convert("L")
    w, h = img.size
    img = img.resize((int(w * 1.5), int(h * 1.5)))
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.point(lambda x: 255 if x > 160 else 0, mode="1").convert("L")
    return img


def _norm_iso_date(text_line: str) -> str | None:
    """找西元日期→轉 YYYY-MM-DD；支援 YYYY-MM-DD / YYYY/MM/DD / YYYY.MM.DD"""
    m = re.search(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})", text_line)
    if not m:
        return None
    y, mo, d = m.groups()
    try:
        dt = datetime(int(y), int(mo), int(d))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return None


def extract_invoice_data(image_path: str):
    """
    從發票圖片擷取文字 + 規則抽取（無 LLM）
    回傳：date, store, amount, product_name, raw_text, warnings
    """
    warnings = []
    try:
        if not _available_langs_ok():
            warnings.append("chi_tra 語言包未安裝，中文辨識會不準。")

        img = Image.open(image_path)
        proc = _preprocess(img)

        text = pytesseract.image_to_string(
            proc,
            lang="chi_tra+eng",
            config="--oem 1 --psm 6 -c preserve_interword_spaces=1"
        )
        lines = [ln.strip() for ln in text.split("\n") if ln.strip()]

        data = {
            "raw_text": text,
            "date": None,
            "store": None,
            "amount": None,
            "product_name": None,
            "warnings": warnings or None
        }

        # 先抓日期（優先西元 yyyy-mm-dd）
        for ln in lines:
            iso = _norm_iso_date(ln)
            if iso:
                data["date"] = iso
                break

        # 再抓店家、金額、商品
        for ln in lines:
            # 店家（公司 / 店 / 名稱 / 行號）
            if data["store"] is None and any(k in ln for k in ["公司", "店", "名稱", "行號"]):
                data["store"] = ln
                continue

            # 金額（$、合計、總計、應收、應付）
            if data["amount"] is None and re.search(
                r"(\$[\d,]+)|([合總]計[:：]?\s*\$?\s*[\d,]+)|(應(收|付).*\$?[\d,]+)",
                ln
            ):
                data["amount"] = ln
                continue

            # 商品（包含關鍵字，但排除抬頭行）
            if data["product_name"] is None:
                if re.search(r"(品名|商品|明細|餐|飲|蛋糕|奶|茶|飯|麵|票|項|咖啡)", ln):
                    if not any(skip in ln for skip in ["電子發票", "買受人", "交易明細", "統編", "格式", "隨機碼"]):
                        data["product_name"] = ln
                        continue

        # 若仍沒有商品，從含食品關鍵字的行找
        if data["product_name"] is None:
            for ln in lines:
                if any(k in ln for k in ["蛋糕", "茶", "奶", "飯", "餐", "咖啡", "飲", "甜", "便當"]):
                    data["product_name"] = ln
                    break

        # 最後 fallback：取最長一行（多半是商品+金額那行）
        if data["product_name"] is None and lines:
            data["product_name"] = max(lines, key=len)

        return data

    except Exception as e:
        return {"error": str(e), "warnings": warnings or None}
