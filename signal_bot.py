# -*- coding: utf-8 -*-
"""
بوت إشارات تداول - استراتيجية Trend Following متعددة الفريمات (ICT)
=====================================================================
- الفريم 1H: تحديد الاتجاه العام (EMA20/EMA50 + ADX/DMI)
- فريم الدخول: 15 دقيقة للأزواج السريعة، 5 دقائق للأزواج البطيئة
- الفلاتر: BOS + Liquidity Sweep (إلزامي) + FVG/Order Block (بحد أدنى للحجم)
           + فلتر المسافة عن المنطقة + فلتر الأخبار الاقتصادية عالية التأثير
- يحسب نسبة ثقة تقنية لكل إشارة، ويحدد هدفين (TP1/TP2)
- يراقب الصفقات المفتوحة بعد الإرسال وينبّه عند وجوب نقل الستوب لوس
- فلتر الجلسات: لندن ونيويورك فقط (لفتح صفقات جديدة فقط - المراقبة مستمرة دائماً)
- يرسل إشارات وتنبيهات فقط عبر تلكرام - لا تنفيذ تلقائي للصفقات
- مصمم للعمل عبر GitHub Actions (تشغيل مجدول كل عدة دقائق)
"""

import os
import json
import time
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timezone

# ==================== الإعدادات (تُقرأ من متغيرات البيئة / GitHub Secrets) ====================
TWELVEDATA_API_KEY = os.environ.get("TWELVEDATA_API_KEY")
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

# تصنيف الأزواج: بطيئة -> فريم دخول 5 دقائق | سريعة -> فريم دخول 15 دقيقة
SLOW_PAIRS = ["EUR/USD", "USD/CHF", "EUR/CHF", "EUR/GBP"]
FAST_PAIRS = [
    "GBP/USD", "USD/JPY", "AUD/USD", "USD/CAD", "NZD/USD",
    "EUR/JPY", "GBP/JPY", "EUR/AUD", "AUD/JPY", "GBP/CHF",
    "XAU/USD",   # الذهب - تقلب عالٍ، يُعامل كزوج سريع
]

_pairs_env = os.environ.get("PAIRS")
if _pairs_env:
    _custom = [p.strip() for p in _pairs_env.split(",")]
    PAIRS_CONFIG = {p: ("5min" if p in SLOW_PAIRS else "15min") for p in _custom}
else:
    PAIRS_CONFIG = {}
    for p in SLOW_PAIRS:
        PAIRS_CONFIG[p] = "5min"
    for p in FAST_PAIRS:
        PAIRS_CONFIG[p] = "15min"

STATE_FILE = "state.json"
ADX_THRESHOLD = 20                     # الحد الأدنى لقوة الترند
TP1_RR = 1.0                            # نسبة الهدف الأول (1:1)
TP2_RR = 2.0                            # نسبة الهدف الثاني (1:2)
ATR_PERIOD = 14
DELAY_BETWEEN_PAIRS = 8                 # ثوانٍ انتظار بين كل زوج وآخر (حد الطلبات المجاني)
MIN_ZONE_ATR_RATIO = 0.25               # الحد الأدنى لحجم FVG/Order Block نسبة إلى ATR
MAX_ZONE_DISTANCE_ATR = 1.5             # أقصى مسافة مسموحة بين السعر الحالي والمنطقة (نسبة ATR)
NEWS_WINDOW_MINUTES = 30                # نافذة تجنب الأخبار عالية التأثير (دقيقة قبل/بعد)

TWELVEDATA_URL = "https://api.twelvedata.com/time_series"
NEWS_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"


# ==================== دوال جلب البيانات ====================
def fetch_candles(pair, interval, count=150):
    """جلب الشموع من Twelve Data وتحويلها إلى DataFrame"""
    params = {
        "symbol": pair,
        "interval": interval,
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


# ==================== أدوات ICT (فريم الدخول) ====================
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


def detect_fvg(df, direction, atr_value, lookback=10):
    """اكتشاف الفجوة السعرية (FVG) بحد أدنى للحجم نسبة إلى ATR"""
    zones = []
    n = len(df)
    start = max(2, n - lookback)
    min_size = atr_value * MIN_ZONE_ATR_RATIO
    for i in range(start, n):
        c1, c3 = df.iloc[i - 2], df.iloc[i]
        if direction == "bullish" and c1["high"] < c3["low"]:
            size = c3["low"] - c1["high"]
            if size >= min_size:
                zones.append((c1["high"], c3["low"]))
        elif direction == "bearish" and c1["low"] > c3["high"]:
            size = c1["low"] - c3["high"]
            if size >= min_size:
                zones.append((c3["high"], c1["low"]))
    return zones


def detect_order_block(df, bos_index, direction, atr_value):
    """آخر شمعة معاكسة قبل حركة الاختراق (Order Block) بحد أدنى للحجم"""
    search_range = df.iloc[max(0, bos_index - 6):bos_index]
    min_size = atr_value * MIN_ZONE_ATR_RATIO
    if direction == "bullish":
        candles = search_range[search_range["close"] < search_range["open"]]
    else:
        candles = search_range[search_range["close"] > search_range["open"]]

    if candles.empty:
        return None
    ob = candles.iloc[-1]
    if (ob["high"] - ob["low"]) < min_size:
        return None
    return (ob["low"], ob["high"])


def detect_liquidity_sweep(df, atr_value, lookback=20):
    """اكتشاف اختراق سيولة (Stop Hunt). يعيد (direction, magnitude) أو (None, 0)"""
    recent = df.iloc[-lookback:-1]
    last = df.iloc[-1]

    prev_high = recent["high"].max()
    prev_low = recent["low"].min()

    swept_high = last["high"] > prev_high and last["close"] < prev_high
    swept_low = last["low"] < prev_low and last["close"] > prev_low

    if swept_high:
        magnitude = (last["high"] - prev_high) / atr_value if atr_value else 0
        return "bearish", magnitude
    if swept_low:
        magnitude = (prev_low - last["low"]) / atr_value if atr_value else 0
        return "bullish", magnitude
    return None, 0


def zone_distance_ok(entry_price, zone, atr_value):
    """يتحقق أن السعر الحالي ليس بعيداً جداً عن منطقة FVG/Order Block"""
    if zone is None or not atr_value:
        return False
    low, high = zone
    if entry_price < low:
        distance = low - entry_price
    elif entry_price > high:
        distance = entry_price - high
    else:
        distance = 0
    return distance <= (atr_value * MAX_ZONE_DISTANCE_ATR)


# ==================== فلتر الجلسات ====================
def in_active_session():
    """يسمح فقط بجلستي لندن ونيويورك (بتوقيت UTC) - لفتح صفقات جديدة فقط"""
    now_utc = datetime.now(timezone.utc)
    hour = now_utc.hour
    london = 7 <= hour < 16
    new_york = 12 <= hour < 21
    return london or new_york


# ==================== فلتر الأخبار الاقتصادية عالية التأثير ====================
def is_near_high_impact_news(pair):
    """
    يتحقق من وجود خبر اقتصادي عالي التأثير قريب زمنياً لعملات الزوج.
    عند أي فشل في الجلب أو التحليل، يفشل بأمان (لا يمنع الإشارة) حتى لا يتعطل البوت.
    """
    try:
        currencies = [c for c in pair.replace("XAU", "XAU").split("/")]
        response = requests.get(NEWS_CALENDAR_URL, timeout=10)
        events = response.json()
        now = datetime.now(timezone.utc)

        for event in events:
            impact = str(event.get("impact", "")).lower()
            if impact != "high":
                continue
            currency = event.get("country") or event.get("currency") or ""
            if currency not in currencies:
                continue

            event_time_raw = event.get("date") or event.get("dateline")
            if not event_time_raw:
                continue
            try:
                if isinstance(event_time_raw, (int, float)):
                    event_time = datetime.fromtimestamp(event_time_raw, tz=timezone.utc)
                else:
                    event_time = pd.to_datetime(event_time_raw, utc=True).to_pydatetime()
            except Exception:
                continue

            diff_minutes = abs((event_time - now).total_seconds()) / 60
            if diff_minutes <= NEWS_WINDOW_MINUTES:
                return True, currency
        return False, None
    except Exception as e:
        print(f"تحذير: تعذر التحقق من الأخبار الاقتصادية ({e}) - سيتم تجاوز هذا الفلتر لهذه الدورة")
        return False, None


# ==================== إدارة الحالة ====================
def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            state = json.load(f)
    else:
        state = {}
    # تحصين ضد ملف state.json بتنسيق قديم أو ناقص المفاتيح
    state.setdefault("last_signal", {})
    state.setdefault("open_trades", {})
    return state


def save_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, default=str)


# ==================== إرسال تلكرام ====================
def send_telegram_message(message):
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message, "parse_mode": "HTML"}
    response = requests.post(url, data=payload, timeout=15)
    response.raise_for_status()


# ==================== حساب نسبة الثقة ====================
def calculate_confidence(adx_value, has_fvg, has_ob, sweep_magnitude, zone_dist_score):
    score = 0

    # قوة الترند (ADX)
    if adx_value >= 35:
        score += 30
    elif adx_value >= 25:
        score += 20
    else:
        score += 10

    # وجود FVG و/أو Order Block معاً
    if has_fvg and has_ob:
        score += 25
    elif has_fvg or has_ob:
        score += 15

    # قرب السعر من المنطقة
    score += zone_dist_score  # 25 أو 10 أو 0

    # قوة اختراق السيولة
    if sweep_magnitude >= 0.3:
        score += 20
    else:
        score += 10

    score = min(score, 100)

    if score >= 80:
        label = "عالية جداً 🟢🟢"
    elif score >= 60:
        label = "عالية 🟢"
    elif score >= 40:
        label = "متوسطة 🟡"
    else:
        label = "منخفضة 🔴"

    return score, label


def format_signal_message(pair, direction, entry, sl, tp1, tp2, adx_value,
                           conditions_met, confidence_score, confidence_label,
                           signals_this_run):
    arrow = "🟢 شراء (BUY)" if direction == "bullish" else "🔴 بيع (SELL)"
    bias_text = "صاعد ⬆️" if direction == "bullish" else "هابط ⬇️"

    conditions_text = "\n".join([f"✅ {c}" for c in conditions_met])

    note = ""
    if signals_this_run > 0:
        note = f"\n⚠️ ملاحظة: تم إرسال {signals_this_run} إشارة أخرى في نفس هذه الدورة - انتبه لتعدد التعرض على أزواج مترابطة."

    return (
        f"<b>📊 إشارة تداول جديدة</b>\n"
        f"الزوج: <b>{pair}</b>\n"
        f"النوع: {arrow}\n\n"
        f"<b>اتفاق الفريمين:</b> الفريم الكبير (1H) {bias_text} ومتوافق مع فريم الدخول ✅\n\n"
        f"<b>الشروط المحققة:</b>\n{conditions_text}\n\n"
        f"<b>نسبة الثقة التقنية:</b> {confidence_score}/100 ({confidence_label})\n\n"
        f"<b>سعر الدخول:</b> {entry:.5f}\n"
        f"<b>إيقاف الخسارة (SL):</b> {sl:.5f}\n"
        f"<b>الهدف الأول (TP1):</b> {tp1:.5f}\n"
        f"<b>الهدف الثاني (TP2):</b> {tp2:.5f}\n\n"
        f"<b>قوة الترند (ADX):</b> {adx_value:.1f}\n"
        f"الوقت (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}"
        f"{note}"
    )


def format_monitor_message(pair, event, direction, entry, sl, tp1, tp2, price):
    arrow = "🟢 شراء" if direction == "bullish" else "🔴 بيع"
    lines = [f"<b>🔔 تحديث صفقة</b>", f"الزوج: <b>{pair}</b> ({arrow})"]

    if event == "tp1_hit":
        lines.append("✅ تم الوصول للهدف الأول (TP1)")
        lines.append("👉 انصح بنقل الستوب لوس إلى نقطة الدخول (Breakeven) لتأمين الصفقة")
        lines.append(f"الستوب لوس الجديد المقترح: {entry:.5f}")
    elif event == "tp2_hit":
        lines.append("🎯 تم الوصول للهدف الثاني (TP2) - تم إغلاق متابعة الصفقة")
    elif event == "sl_hit":
        lines.append("🛑 تم الوصول لمستوى الستوب لوس - تم إغلاق متابعة الصفقة")

    lines.append(f"السعر الحالي: {price:.5f}")
    lines.append(f"الوقت (UTC): {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')}")
    return "\n".join(lines)


# ==================== مراقبة صفقة مفتوحة ====================
def monitor_open_trade(pair, interval, trade):
    """يفحص آخر سعر لزوج لديه صفقة مفتوحة، وينبّه عند TP1/TP2/SL. يعيد الصفقة المحدثة أو None إذا أُغلقت"""
    try:
        df = fetch_candles(pair, interval, count=5)
    except Exception as e:
        print(f"[{pair}] فشل جلب السعر للمراقبة: {e}")
        return trade

    last = df.iloc[-1]
    high, low = last["high"], last["low"]
    direction = trade["direction"]

    # فحص TP1 (لتنبيه نقل الستوب لوس فقط)
    if not trade.get("tp1_hit"):
        hit_tp1 = (direction == "bullish" and high >= trade["tp1"]) or \
                  (direction == "bearish" and low <= trade["tp1"])
        if hit_tp1:
            send_telegram_message(format_monitor_message(
                pair, "tp1_hit", direction, trade["entry"], trade["sl"],
                trade["tp1"], trade["tp2"], last["close"]))
            trade["tp1_hit"] = True
            trade["sl"] = trade["entry"]  # نقل الستوب لوس إلى نقطة الدخول

    # فحص TP2 (إغلاق المتابعة)
    hit_tp2 = (direction == "bullish" and high >= trade["tp2"]) or \
              (direction == "bearish" and low <= trade["tp2"])
    if hit_tp2:
        send_telegram_message(format_monitor_message(
            pair, "tp2_hit", direction, trade["entry"], trade["sl"],
            trade["tp1"], trade["tp2"], last["close"]))
        return None

    # فحص الستوب لوس (بعد التحديث المحتمل إلى Breakeven)
    hit_sl = (direction == "bullish" and low <= trade["sl"]) or \
             (direction == "bearish" and high >= trade["sl"])
    if hit_sl:
        send_telegram_message(format_monitor_message(
            pair, "sl_hit", direction, trade["entry"], trade["sl"],
            trade["tp1"], trade["tp2"], last["close"]))
        return None

    return trade


# ==================== فحص زوج جديد (بحث عن إشارة دخول) ====================
def check_pair(pair, interval, state, signals_this_run):
    try:
        df_h1 = fetch_candles(pair, "1h", count=100)
        df_entry = fetch_candles(pair, interval, count=150)
    except Exception as e:
        print(f"[{pair}] فشل جلب البيانات: {e}")
        return False

    if len(df_h1) < 60 or len(df_entry) < 40:
        print(f"[{pair}] بيانات غير كافية")
        return False

    bias, adx_value = get_h1_bias(df_h1)
    if bias == "neutral":
        print(f"[{pair}] لا يوجد ترند واضح (ADX={adx_value:.1f})")
        return False

    highs, lows = find_pivots(df_entry)
    bos = detect_bos(df_entry, highs, lows)
    if bos != bias:
        print(f"[{pair}] لا يوجد BOS متوافق مع اتجاه الفريم الأعلى")
        return False

    atr_value = calculate_atr(df_entry).iloc[-1]

    # فلتر إلزامي: اختراق السيولة (Liquidity Sweep)
    sweep_dir, sweep_magnitude = detect_liquidity_sweep(df_entry, atr_value)
    if sweep_dir != bos:
        print(f"[{pair}] لا يوجد اختراق سيولة متوافق قبل BOS - تُرفض الإشارة")
        return False

    # فلتر FVG أو Order Block (بحد أدنى للحجم)
    fvg_zones = detect_fvg(df_entry, bos, atr_value)
    bos_index = len(df_entry) - 1
    order_block = detect_order_block(df_entry, bos_index, bos, atr_value)

    if not fvg_zones and not order_block:
        print(f"[{pair}] لا توجد منطقة FVG أو Order Block كافية الحجم")
        return False

    entry_price = df_entry["close"].iloc[-1]

    # فلتر المسافة عن المنطقة
    zone_to_check = fvg_zones[-1] if fvg_zones else order_block
    if not zone_distance_ok(entry_price, zone_to_check, atr_value):
        print(f"[{pair}] السعر بعيد جداً عن منطقة الدخول")
        return False

    # فلتر الأخبار الاقتصادية عالية التأثير
    near_news, news_currency = is_near_high_impact_news(pair)
    if near_news:
        print(f"[{pair}] تم رفض الإشارة - خبر اقتصادي عالي التأثير قريب ({news_currency})")
        return False

    # فلتر عدم التكرار لنفس الشمعة
    current_candle_time = str(df_entry["time"].iloc[-1])
    if state["last_signal"].get(pair) == current_candle_time:
        print(f"[{pair}] تم إرسال إشارة لهذه الشمعة مسبقاً")
        return False

    # منع فتح صفقة جديدة إن كانت هناك صفقة مفتوحة أصلاً لنفس الزوج
    if pair in state["open_trades"]:
        print(f"[{pair}] توجد صفقة مفتوحة بالفعل - تخطي")
        return False

    # ==================== حساب المستويات ====================
    if bos == "bullish":
        sl = entry_price - atr_value
        tp1 = entry_price + (atr_value * TP1_RR)
        tp2 = entry_price + (atr_value * TP2_RR)
    else:
        sl = entry_price + atr_value
        tp1 = entry_price - (atr_value * TP1_RR)
        tp2 = entry_price - (atr_value * TP2_RR)

    # ==================== نسبة الثقة ====================
    zone_distance = 0
    low_z, high_z = zone_to_check
    if entry_price < low_z:
        zone_distance = low_z - entry_price
    elif entry_price > high_z:
        zone_distance = entry_price - high_z
    zone_dist_ratio = zone_distance / atr_value if atr_value else 1
    zone_dist_score = 25 if zone_dist_ratio <= 0.3 else (10 if zone_dist_ratio <= 0.7 else 0)

    confidence_score, confidence_label = calculate_confidence(
        adx_value, bool(fvg_zones), bool(order_block), sweep_magnitude, zone_dist_score
    )

    conditions_met = [
        f"اتفاق الاتجاه بين الفريم الكبير (1H) وفريم الدخول ({interval})",
        "كسر بنية (BOS) في اتجاه الترند",
        "اختراق سيولة (Liquidity Sweep) قبل الاختراق",
    ]
    if fvg_zones:
        conditions_met.append("وجود فجوة سعرية (FVG) كافية الحجم")
    if order_block:
        conditions_met.append("وجود منطقة Order Block كافية الحجم")
    conditions_met.append("لا يوجد خبر اقتصادي عالي التأثير قريب")
    conditions_met.append(f"قوة الترند ADX = {adx_value:.1f} (أعلى من الحد الأدنى {ADX_THRESHOLD})")

    message = format_signal_message(
        pair, bos, entry_price, sl, tp1, tp2, adx_value,
        conditions_met, confidence_score, confidence_label, signals_this_run
    )
    send_telegram_message(message)
    print(f"[{pair}] تم إرسال إشارة جديدة بنجاح (ثقة: {confidence_score})")

    state["last_signal"][pair] = current_candle_time
    state["open_trades"][pair] = {
        "direction": bos, "entry": entry_price, "sl": sl,
        "tp1": tp1, "tp2": tp2, "tp1_hit": False,
    }
    return True


# ==================== المنطق الرئيسي ====================
def run():
    state = load_state()
    session_active = in_active_session()
    signals_this_run = 0

    for i, (pair, interval) in enumerate(PAIRS_CONFIG.items()):
        if pair in state["open_trades"]:
            # مراقبة الصفقة المفتوحة تعمل دائماً بغض النظر عن الجلسة
            updated_trade = monitor_open_trade(pair, interval, state["open_trades"][pair])
            if updated_trade is None:
                del state["open_trades"][pair]
            else:
                state["open_trades"][pair] = updated_trade
        elif session_active:
            sent = check_pair(pair, interval, state, signals_this_run)
            if sent:
                signals_this_run += 1
        else:
            print(f"[{pair}] خارج أوقات الجلسة النشطة - تخطي البحث عن إشارة جديدة")

        if i < len(PAIRS_CONFIG) - 1:
            time.sleep(DELAY_BETWEEN_PAIRS)

    save_state(state)


if __name__ == "__main__":
    run()
