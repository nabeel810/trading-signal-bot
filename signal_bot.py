# -*- coding: utf-8 -*-
"""
بوت إشارات تداول - استراتيجية Trend Following متعددة الفريمات (ICT)
=====================================================================
- الفريم 1H: تحديد الاتجاه العام (EMA20/EMA50 + ADX/DMI)
- الفريم 5M: نقطة الدخول (BOS + FVG + Order Block + Liquidity Sweep)
- فلتر الجلسات: لندن ونيويورك فقط
- يرسل إشارات فقط عبر تلكرام - لا تنفيذ تلقائي للصفقات
- مصمم للعمل عبر GitHub Actions (تشغيل مجدول كل عدة دقائق)
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==================== الإعدادات (تُقرأ من متغيرات البيئة / GitHub Secrets) ====================
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")
PAIR = os.environ.get("PAIR", "EUR/USD")   # صيغة Twelve Data: EUR/USD

STATE_FILE = "state.json"          # لتخزين آخر إشارة تم إرسالها (لتجنب التكرار)
ADX_THRESHOLD = 20                  # الحد الأدنى لقوة الترند
RISK_REWARD_RATIO = 2.0             # نسبة المخاطرة إلى الربح 1:2
ATR_PERIOD = 14

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"


# ==================== دوال جلب البيانات ====================
def fetch_candles(pair, interval, count=150):
    """جلب الشموع من Twelve Data وتحويلها إلى DataFrame"""
    params = {
        "symbol": pair,
        "interval": interval,      # مثال: "1h" أو "5min"
        "outputsize": count,
        "apikey": TWELVEDATA_API_KEY,
        "order": "ASC",
    }
    response = requests.get(TWELVEDATA_URL, params=params, timeout=20)
    data = response.json()

    if "values" not in data:
        raise RuntimeError(f"خطأ من Twelve Data: {data}")

    rows = []
    for c in data["values"]:
        rows.append({
            "time": c["datetime"],
            "open": float(c["open"]),
            "high": float(c["high"]),
            "low": float(c["low"]),
            "close": float(c["close"]),
        })
    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    return df


# ==================== المؤشرات الفنية ====================
def calculate_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calculate_atr(df, period=ATR_PERIOD):
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def calculate_adx_dmi(df, period=14):
    """حساب ADX و +DI و -DI"""
    up_move = df["high"].diff()
    down_move = -df["low"].diff()

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    atr = calculate_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm, index=df.index).rolling(period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).rolling(period).mean() / atr

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx, plus_di, minus_di


# ==================== تحديد الاتجاه العام (فريم 1H) ====================
def get_h1_bias(df_h1):
    """يحدد الاتجاه العام: bullish / bearish / neutral"""
    ema20 = calculate_ema(df_h1["close"], 20)
    ema50 = calculate_ema(df_h1["close"], 50)
    adx, plus_di, minus_di = calculate_adx_dmi(df_h1)

    last_adx = adx.iloc[-1]
    last_ema20 = ema20.iloc[-1]
    last_ema50 = ema50.iloc[-1]

    if last_adx < ADX_THRESHOLD:
        return "neutral", last_adx

    if last_ema20 > last_ema50:
        return "bullish", last_adx
    elif last_ema20 < last_ema50:
        return "bearish", last_adx
    return "neutral", last_adx


# ==================== أدوات ICT (فريم 5M) ====================
def find_pivots(df, lookback=5):
    """تحديد القمم والقيعان (Pivot High / Pivot Low)"""
    highs, lows = [], []
    for i in range(lookback, len(df) - lookback):
        window = df.iloc[i - lookback:i + lookback + 1]
        if df["high"].iloc[i] == window["high"].max():
            highs.append(i)
        if df["low"].iloc[i] == window["low"].min():
            lows.append(i)
    return highs, lows


def detect_bos(df, highs, lows):
    """كسر البنية (Break of Structure) - يعيد 'bullish' / 'bearish' / None"""
    if not highs or not lows:
        return None
    last_high_idx = highs[-1]
    last_low_idx = lows[-1]
    last_close = df["close"].iloc[-1]

    if last_close > df["high"].iloc[last_high_idx]:
        return "bullish"
    if last_close < df["low"].iloc[last_low_idx]:
        return "bearish"
    return None


def detect_fvg(df, direction, lookback=10):
    """اكتشاف الفجوة السعرية (Fair Value Gap) في آخر عدة شموع"""
    zones = []
    n = len(df)
    start = max(2, n - lookback)
    for i in range(start, n):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if direction == "bullish" and c1["high"] < c3["low"]:
            zones.append((c1["high"], c3["low"]))
        elif direction == "bearish" and c1["low"] > c3["high"]:
            zones.append((c3["high"], c1["low"]))
    return zones


def detect_order_block(df, bos_index, direction):
    """آخر شمعة معاكسة قبل حركة الاختراق (Order Block)"""
    search_range = df.iloc[max(0, bos_index - 6):bos_index]
    if direction == "bullish":
        bearish_candles = search_range[search_range["close"] < search_range["open"]]
        if bearish_candles.empty:
            return None
        ob = bearish_candles.iloc[-1]
        return (ob["low"], ob["high"])
    else:
        bullish_candles = search_range[search_range["close"] > search_range["open"]]
        if bullish_candles.empty:
            return None
        ob = bullish_candles.iloc[-1]
        return (ob["low"], ob["high"])


def detect_liquidity_sweep(df, lookback=20):
    """اكتشاف اختراق سيولة (Stop Hunt): اختراق قمة/قاع سابق ثم الإغلاق داخل النطاق"""
    recent = df.iloc[-lookback:-1]
    last = df.iloc[-1]

    prev_high = recent["high"].max()
    prev_low = recent["low"].min()

    swept_high = last["high"] > prev_high and last["close"] < prev_high
    swept_low = last["low"] < prev_low and last["close"] > prev_low

    if swept_high:
        return "bearish"  # سيولة علوية تم اختراقها -> احتمال حركة بيعية
    if swept_low:
        return "bullish"  # سيولة سفلية تم اختراقها -> احتمال حركة شرائية
    return None


# ==================== فلتر الجلسات ====================
def in_active_session():
    """يسمح فقط بجلستي لندن ونيويورك (بتوقيت UTC)"""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    london = 7 <= hour < 16
    new_york = 12 <= hour < 21
    return london or new_york


# ==================== إدارة الحالة (لتجنب تكرار الإشارة) ====================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"last_signal_time": None}


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)


# ==================== إرسال إشارة تلكرام ====================
def send_telegram_signal(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    response = requests.post(url, data=payload, timeout=15)
    response.raise_for_status()


def format_signal_message(pair, direction, entry, sl, tp, adx_value):
    arrow = "🟢 شراء (BUY)" if direction == "bullish" else "🔴 بيع (SELL)"
    return (
        f"<b>إشارة تداول جديدة</b>\n"
        f"الزوج: {pair.replace('_', '/')}\n"
        f"النوع: {arrow}\n"
        f"سعر الدخول: {entry:.5f}\n"
        f"إيقاف الخسارة (SL): {sl:.5f}\n"
        f"جني الربح (TP): {tp:.5f}\n"
        f"قوة الترند (ADX): {adx_value:.1f}\n"
        f"الوقت (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
    )


# ==================== المنطق الرئيسي ====================
def run():
    if not in_active_session():
        print("خارج أوقات جلسة لندن/نيويورك - لا يوجد فحص")
        return

    df_h1 = fetch_candles(PAIR, "1h", count=100)
    df_m5 = fetch_candles(PAIR, "5min", count=150)

    if len(df_h1) < 60 or len(df_m5) < 40:
        print("بيانات غير كافية")
        return

    bias, adx_value = get_h1_bias(df_h1)
    if bias == "neutral":
        print(f"لا يوجد ترند واضح (ADX={adx_value:.1f}) - لا إشارة")
        return

    highs, lows = find_pivots(df_m5)
    bos = detect_bos(df_m5, highs, lows)

    if bos != bias:
        print("لا يوجد BOS متوافق مع اتجاه الفريم الأعلى")
        return

    sweep = detect_liquidity_sweep(df_m5)
    fvg_zones = detect_fvg(df_m5, bos)
    bos_index = len(df_m5) - 1
    order_block = detect_order_block(df_m5, bos_index, bos)

    # شرط الدخول: BOS متوافق + وجود منطقة FVG أو Order Block لدعم الإشارة
    if not fvg_zones and not order_block:
        print("لا توجد منطقة FVG أو Order Block لدعم الإشارة")
        return

    state = load_state()
    current_candle_time = str(df_m5["time"].iloc[-1])
    if state.get("last_signal_time") == current_candle_time:
        print("تم إرسال إشارة لهذه الشمعة مسبقاً")
        return

    entry_price = df_m5["close"].iloc[-1]
    atr = calculate_atr(df_m5).iloc[-1]

    if bos == "bullish":
        sl = entry_price - atr
        tp = entry_price + (atr * RISK_REWARD_RATIO)
    else:
        sl = entry_price + atr
        tp = entry_price - (atr * RISK_REWARD_RATIO)

    message = format_signal_message(PAIR, bos, entry_price, sl, tp, adx_value)
    send_telegram_signal(message)
    print("تم إرسال الإشارة بنجاح")

    state["last_signal_time"] = current_candle_time
    save_state(state)


if __name__ == "__main__":
    run()
