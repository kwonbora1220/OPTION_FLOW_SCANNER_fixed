from __future__ import annotations

import json
import os
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(BASE_DIR, "..")
)

RESULT_DIR = os.path.join(
    ROOT_DIR,
    "03_RESULTS",
    "daily"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ------------------------------------------------------------
# AM MARKET WATCHLIST
# ------------------------------------------------------------

MARKET_SYMBOLS = {
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",
    "SOXX": "SOXX",
    "VIX": "^VIX",
}


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):
    try:
        return float(value)
    except Exception:
        return None


def get_history(symbol):
    try:
        ticker = yf.Ticker(symbol)

        df = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False
        )

        if df is None or df.empty:
            return None

        return df

    except Exception as e:
        print(
            f"⚠️ {symbol} 데이터 조회 실패: {e}"
        )
        return None


def get_last_price(symbol):
    df = get_history(symbol)

    if df is None:
        return None

    try:
        close = df["Close"].dropna()

        if close.empty:
            return None

        return float(close.iloc[-1])

    except Exception:
        return None


def get_daily_change(symbol):
    df = get_history(symbol)

    if df is None:
        return None, None

    try:
        close = df["Close"].dropna()

        if len(close) < 2:
            return None, None

        previous = float(close.iloc[-2])
        current = float(close.iloc[-1])

        if previous == 0:
            return current, None

        change_pct = (
            (current / previous) - 1
        ) * 100

        return current, change_pct

    except Exception:
        return None, None


# ============================================================
# MARKET DATA
# ============================================================

def collect_market_data():

    result = {}

    for name, symbol in MARKET_SYMBOLS.items():

        price, change_pct = get_daily_change(
            symbol
        )

        result[name] = {
            "symbol": symbol,
            "price": price,
            "change_pct": change_pct,
        }

        print(
            f"📊 {name}: "
            f"{price if price is not None else 'N/A'} "
            f"({change_pct:+.2f}%"
            f")"
            if change_pct is not None
            else
            f"📊 {name}: "
            f"{price if price is not None else 'N/A'}"
        )

    return result


# ============================================================
# MARKET DIRECTION
# ============================================================

def calculate_market_score(data):

    score = 50.0

    bullish = 0
    bearish = 0

    reasons = []

    # --------------------------------------------------------
    # SPY
    # --------------------------------------------------------

    spy_change = (
        data.get("SPY", {})
        .get("change_pct")
    )

    if spy_change is not None:

        if spy_change >= 1.0:

            score += 12
            bullish += 1

            reasons.append(
                "SPY 강세"
            )

        elif spy_change >= 0.30:

            score += 6
            bullish += 1

            reasons.append(
                "SPY 소폭 강세"
            )

        elif spy_change <= -1.0:

            score -= 12
            bearish += 1

            reasons.append(
                "SPY 약세"
            )

        elif spy_change <= -0.30:

            score -= 6
            bearish += 1

            reasons.append(
                "SPY 소폭 약세"
            )


    # --------------------------------------------------------
    # QQQ
    # --------------------------------------------------------

    qqq_change = (
        data.get("QQQ", {})
        .get("change_pct")
    )

    if qqq_change is not None:

        if qqq_change >= 1.0:

            score += 12
            bullish += 1

            reasons.append(
                "QQQ 강세"
            )

        elif qqq_change >= 0.30:

            score += 6
            bullish += 1

            reasons.append(
                "QQQ 소폭 강세"
            )

        elif qqq_change <= -1.0:

            score -= 12
            bearish += 1

            reasons.append(
                "QQQ 약세"
            )

        elif qqq_change <= -0.30:

            score -= 6
            bearish += 1

            reasons.append(
                "QQQ 소폭 약세"
            )


    # --------------------------------------------------------
    # IWM
    # --------------------------------------------------------

    iwm_change = (
        data.get("IWM", {})
        .get("change_pct")
    )

    if iwm_change is not None:

        if iwm_change >= 0.75:

            score += 5
            bullish += 1

            reasons.append(
                "IWM 강세"
            )

        elif iwm_change <= -0.75:

            score -= 5
            bearish += 1

            reasons.append(
                "IWM 약세"
            )


    # --------------------------------------------------------
    # SOXX
    # --------------------------------------------------------

    soxx_change = (
        data.get("SOXX", {})
        .get("change_pct")
    )

    if soxx_change is not None:

        if soxx_change >= 0.75:

            score += 6
            bullish += 1

            reasons.append(
                "SOXX 강세"
            )

        elif soxx_change <= -0.75:

            score -= 6
            bearish += 1

            reasons.append(
                "SOXX 약세"
            )


    # --------------------------------------------------------
    # VIX
    # --------------------------------------------------------

    vix_change = (
        data.get("VIX", {})
        .get("change_pct")
    )

    if vix_change is not None:

        if vix_change <= -8:

            score += 8

            reasons.append(
                "VIX 급락"
            )

        elif vix_change <= -3:

            score += 4

            reasons.append(
                "VIX 하락"
            )

        elif vix_change >= 8:

            score -= 10
            bearish += 1

            reasons.append(
                "VIX 급등"
            )

        elif vix_change >= 3:

            score -= 5
            bearish += 1

            reasons.append(
                "VIX 상승"
            )


    # --------------------------------------------------------
    # LIMIT
    # --------------------------------------------------------

    score = max(
        0,
        min(
            100,
            score
        )
    )


    # --------------------------------------------------------
    # DIRECTION
    # --------------------------------------------------------

    if bullish >= bearish + 2:

        direction = "BULLISH"

    elif bearish >= bullish + 2:

        direction = "BEARISH"

    else:

        direction = "NEUTRAL"


    # --------------------------------------------------------
    # REGIME
    # --------------------------------------------------------

    if score >= 70:

        regime = "🟢 BULLISH"

    elif score <= 35:

        regime = "🔴 BEARISH"

    else:

        regime = "🟡 NEUTRAL"


    return {
        "score": round(score, 2),
        "direction": direction,
        "regime": regime,
        "bullish_signals": bullish,
        "bearish_signals": bearish,
        "reasons": reasons,
    }


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_telegram_message(
    data,
    market
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🌅 <b>AM MARKET CONTEXT</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    lines.append(
        f"🎯 MARKET REGIME: "
        f"<b>{market['regime']}</b>"
    )

    lines.append(
        f"📊 Market Score: "
        f"<b>{market['score']:.1f}</b>"
    )

    lines.append(
        f"📈 Direction: "
        f"<b>{market['direction']}</b>"
    )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 <b>MARKET DATA</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for name in [
        "SPY",
        "QQQ",
        "IWM",
        "SOXX",
        "VIX"
    ]:

        item = data.get(
            name,
            {}
        )

        price = item.get(
            "price"
        )

        change = item.get(
            "change_pct"
        )

        if price is None:

            price_text = "N/A"

        else:

            price_text = (
                f"${price:.2f}"
            )

        if change is None:

            change_text = "N/A"

        else:

            change_text = (
                f"{change:+.2f}%"
            )

        lines.append(
            f"{name:<5} "
            f"{price_text:<12} "
            f"{change_text}"
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>MARKET SIGNALS</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if market["reasons"]:

        for reason in market["reasons"]:

            lines.append(
                f"• {reason}"
            )

    else:

        lines.append(
            "• 뚜렷한 시장 신호 없음"
        )

    lines.append("")

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔗 <b>OPTION SEARCH INPUT</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if market["direction"] == "BULLISH":

        lines.append(
            "🟢 Bullish 옵션 후보 우선"
        )

        lines.append(
            "→ 시장 방향과 일치하는 Flow 가점 대상"
        )

    elif market["direction"] == "BEARISH":

        lines.append(
            "🔴 Bullish 옵션 후보 주의"
        )

        lines.append(
            "→ 시장 방향과 반대되는 Flow 감점 대상"
        )

    else:

        lines.append(
            "🟡 시장 방향 중립"
        )

        lines.append(
            "→ 개별 옵션 Flow 중심 판단"
        )

    lines.append("")

    lines.append(
        "⏰ "
        + datetime.now(
            timezone.utc
        ).strftime(
            "%Y-%m-%d %H:%M UTC"
        )
    )

    return "\n".join(lines)


# ============================================================
# SAVE JSON
# ============================================================

def save_market_context(
    data,
    market
):

    today = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d"
    )

    json_path = os.path.join(
        RESULT_DIR,
        f"AM_MARKET_CONTEXT_{today}.json"
    )

    payload = {
        "date": today,
        "created_at_utc":
            datetime.now(
                timezone.utc
            ).isoformat(),

        "market_data": data,

        "market_context": market,
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            payload,
            f,
            ensure_ascii=False,
            indent=2
        )

    print(
        f"💾 AM MARKET CONTEXT 저장: "
        f"{json_path}"
    )

    return json_path


# ============================================================
# SAVE TXT
# ============================================================

def save_text(
    message
):

    today = datetime.now(
        timezone.utc
    ).strftime(
        "%Y%m%d"
    )

    path = os.path.join(
        RESULT_DIR,
        f"AM_MARKET_CONTEXT_{today}.txt"
    )

    # Telegram HTML 태그 제거
    text = (
        message
        .replace("<b>", "")
        .replace("</b>", "")
    )

    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)

    print(
        f"💾 AM 보고서 저장: {path}"
    )

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🌅 AM MARKET CONTEXT START"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    data = collect_market_data()

    market = calculate_market_score(
        data
    )

    message = build_telegram_message(
        data,
        market
    )

    save_market_context(
        data,
        market
    )

    save_text(
        message
    )

    print("")
    print(message)
    print("")

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "✅ AM MARKET CONTEXT COMPLETE"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


if __name__ == "__main__":
    main()
