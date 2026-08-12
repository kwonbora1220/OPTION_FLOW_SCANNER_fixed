from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import yfinance as yf


# ============================================================
# PM VOLATILITY UPDATE
# ============================================================
#
# 기준 시간:
#   🇺🇸 America/New_York
#
# 정규장:
#   09:30 ~ 16:00 ET
#
# PM 분석:
#   장 종료 후 실행
#
# IMPORTANT:
#   기존 OPTION SEARCH 로직과 완전히 독립적으로 동작
# ============================================================


ET = ZoneInfo("America/New_York")


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

ROOT_DIR = os.path.abspath(
    os.path.join(
        BASE_DIR,
        ".."
    )
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


# ============================================================
# MARKET SYMBOLS
# ============================================================

MARKET_SYMBOLS = {

    # Broad Market
    "SPY": "SPY",
    "QQQ": "QQQ",
    "IWM": "IWM",

    # Semiconductor
    "SOXX": "SOXX",

    # Volatility
    "VIX": "^VIX",
}


# ============================================================
# TIME
# ============================================================

def get_et_now():

    return datetime.now(
        ET
    )


def get_market_date():

    return get_et_now().strftime(
        "%Y%m%d"
    )


# ============================================================
# SAFE FLOAT
# ============================================================

def safe_float(
    value
):

    try:

        if value is None:
            return None

        return float(
            value
        )

    except Exception:

        return None


# ============================================================
# INTRADAY DATA
# ============================================================

def get_intraday_data(
    symbol
):

    try:

        ticker = yf.Ticker(
            symbol
        )

        df = ticker.history(
            period="1d",
            interval="5m",
            auto_adjust=False
        )

        if (
            df is None
            or df.empty
        ):

            return None


        # ----------------------------------------------------
        # Remove invalid rows
        # ----------------------------------------------------

        df = df.dropna(
            subset=[
                "Open",
                "High",
                "Low",
                "Close"
            ]
        )

        if df.empty:

            return None


        # ----------------------------------------------------
        # Prices
        # ----------------------------------------------------

        first_open = safe_float(
            df["Open"].iloc[0]
        )

        last_close = safe_float(
            df["Close"].iloc[-1]
        )

        day_high = safe_float(
            df["High"].max()
        )

        day_low = safe_float(
            df["Low"].min()
        )


        # ----------------------------------------------------
        # Change from open
        # ----------------------------------------------------

        change_from_open_pct = None

        if (
            first_open is not None
            and last_close is not None
            and first_open != 0
        ):

            change_from_open_pct = (
                (
                    last_close
                    / first_open
                )
                - 1
            ) * 100


        # ----------------------------------------------------
        # High / Low distance
        # ----------------------------------------------------

        high_distance_pct = None

        low_distance_pct = None

        if (
            last_close is not None
            and day_high is not None
            and day_high != 0
        ):

            high_distance_pct = (
                (
                    day_high
                    / last_close
                )
                - 1
            ) * 100


        if (
            last_close is not None
            and day_low is not None
            and day_low != 0
        ):

            low_distance_pct = (
                (
                    last_close
                    / day_low
                )
                - 1
            ) * 100


        # ----------------------------------------------------
        # Volume
        # ----------------------------------------------------

        total_volume = None

        if "Volume" in df.columns:

            volume = (
                df["Volume"]
                .dropna()
            )

            if not volume.empty:

                total_volume = int(
                    volume.sum()
                )


        return {

            "first_open":
                first_open,

            "last_close":
                last_close,

            "day_high":
                day_high,

            "day_low":
                day_low,

            "change_from_open_pct":
                change_from_open_pct,

            "high_distance_pct":
                high_distance_pct,

            "low_distance_pct":
                low_distance_pct,

            "total_volume":
                total_volume,

        }


    except Exception as e:

        print(
            f"⚠️ {symbol} 데이터 조회 실패: {e}"
        )

        return None


# ============================================================
# COLLECT MARKET DATA
# ============================================================

def collect_market_data():

    result = {}


    for name, symbol in MARKET_SYMBOLS.items():

        print(
            f"📊 PM 데이터 수집: {name}"
        )

        data = get_intraday_data(
            symbol
        )

        result[name] = {

            "symbol":
                symbol,

            "data":
                data,

        }


    return result


# ============================================================
# VOLATILITY / MARKET REGIME
# ============================================================

def calculate_market_regime(
    data
):

    bullish = 0

    bearish = 0

    neutral = 0

    reasons = []


    # ========================================================
    # SPY
    # ========================================================

    spy = (
        data
        .get("SPY", {})
        .get("data")
    )

    if spy:

        change = spy.get(
            "change_from_open_pct"
        )

        if change is not None:

            if change >= 0.50:

                bullish += 1

                reasons.append(
                    "SPY 장중 강세"
                )

            elif change <= -0.50:

                bearish += 1

                reasons.append(
                    "SPY 장중 약세"
                )

            else:

                neutral += 1


    # ========================================================
    # QQQ
    # ========================================================

    qqq = (
        data
        .get("QQQ", {})
        .get("data")
    )

    if qqq:

        change = qqq.get(
            "change_from_open_pct"
        )

        if change is not None:

            if change >= 0.50:

                bullish += 1

                reasons.append(
                    "QQQ 장중 강세"
                )

            elif change <= -0.50:

                bearish += 1

                reasons.append(
                    "QQQ 장중 약세"
                )

            else:

                neutral += 1


    # ========================================================
    # IWM
    # ========================================================

    iwm = (
        data
        .get("IWM", {})
        .get("data")
    )

    if iwm:

        change = iwm.get(
            "change_from_open_pct"
        )

        if change is not None:

            if change >= 0.75:

                bullish += 1

                reasons.append(
                    "IWM 강세"
                )

            elif change <= -0.75:

                bearish += 1

                reasons.append(
                    "IWM 약세"
                )

            else:

                neutral += 1


    # ========================================================
    # SOXX
    # ========================================================

    soxx = (
        data
        .get("SOXX", {})
        .get("data")
    )

    if soxx:

        change = soxx.get(
            "change_from_open_pct"
        )

        if change is not None:

            if change >= 0.75:

                bullish += 1

                reasons.append(
                    "SOXX 강세"
                )

            elif change <= -0.75:

                bearish += 1

                reasons.append(
                    "SOXX 약세"
                )

            else:

                neutral += 1


    # ========================================================
    # VIX
    # ========================================================

    vix = (
        data
        .get("VIX", {})
        .get("data")
    )

    vix_change = None

    if vix:

        vix_change = vix.get(
            "change_from_open_pct"
        )


        if vix_change is not None:

            if vix_change >= 8:

                bearish += 2

                reasons.append(
                    "VIX 급등 → 위험회피"
                )

            elif vix_change >= 3:

                bearish += 1

                reasons.append(
                    "VIX 상승"
                )

            elif vix_change <= -8:

                bullish += 2

                reasons.append(
                    "VIX 급락 → 위험선호"
                )

            elif vix_change <= -3:

                bullish += 1

                reasons.append(
                    "VIX 하락"
                )


    # ========================================================
    # FINAL REGIME
    # ========================================================

    if (
        bullish
        >= bearish + 2
    ):

        direction = "BULLISH"

        regime = "🟢 BULLISH"


    elif (
        bearish
        >= bullish + 2
    ):

        direction = "BEARISH"

        regime = "🔴 BEARISH"


    else:

        direction = "NEUTRAL"

        regime = "🟡 NEUTRAL"


    # ========================================================
    # VOLATILITY LEVEL
    # ========================================================

    if vix_change is None:

        volatility = "UNKNOWN"


    elif vix_change >= 8:

        volatility = "🔴 HIGH"


    elif vix_change >= 3:

        volatility = "🟠 ELEVATED"


    elif vix_change <= -5:

        volatility = "🟢 FALLING"


    else:

        volatility = "🟡 NORMAL"


    return {

        "direction":
            direction,

        "regime":
            regime,

        "volatility":
            volatility,

        "bullish_signals":
            bullish,

        "bearish_signals":
            bearish,

        "neutral_signals":
            neutral,

        "reasons":
            reasons,

        "vix_change_pct":
            vix_change,

    }


# ============================================================
# BUILD TELEGRAM MESSAGE
# ============================================================

def build_telegram_message(
    data,
    regime
):

    now_et = get_et_now()


    lines = []


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🌙 <b>PM VOLATILITY UPDATE</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")


    lines.append(
        "🇺🇸 <b>US EASTERN TIME</b>"
    )

    lines.append(
        now_et.strftime(
            "%Y-%m-%d %H:%M ET"
        )
    )

    lines.append("")


    # ========================================================
    # MARKET REGIME
    # ========================================================

    lines.append(
        "🎯 <b>MARKET REGIME</b>"
    )

    lines.append(
        f"{regime['regime']}"
    )

    lines.append(
        f"Direction: "
        f"<b>{regime['direction']}</b>"
    )

    lines.append(
        f"Volatility: "
        f"<b>{regime['volatility']}</b>"
    )

    lines.append("")


    # ========================================================
    # MARKET DATA
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 <b>MARKET CLOSE / INTRADAY</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    for name in [
        "SPY",
        "QQQ",
        "IWM",
        "SOXX",
        "VIX",
    ]:

        item = data.get(
            name,
            {}
        )

        x = item.get(
            "data"
        )


        if not x:

            lines.append(
                f"{name}: N/A"
            )

            continue


        close = x.get(
            "last_close"
        )

        change = x.get(
            "change_from_open_pct"
        )

        high = x.get(
            "day_high"
        )

        low = x.get(
            "day_low"
        )


        close_text = (
            "N/A"
            if close is None
            else f"${close:.2f}"
        )


        change_text = (
            "N/A"
            if change is None
            else f"{change:+.2f}%"
        )


        high_text = (
            "N/A"
            if high is None
            else f"${high:.2f}"
        )


        low_text = (
            "N/A"
            if low is None
            else f"${low:.2f}"
        )


        lines.append(
            f"<b>{name}</b>"
        )

        lines.append(
            f"종가: {close_text}"
        )

        lines.append(
            f"장중 변화: {change_text}"
        )

        lines.append(
            f"High: {high_text} / "
            f"Low: {low_text}"
        )

        lines.append("")


    # ========================================================
    # SIGNALS
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>PM SIGNALS</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    if regime["reasons"]:

        for reason in regime[
            "reasons"
        ]:

            lines.append(
                f"• {reason}"
            )

    else:

        lines.append(
            "• 뚜렷한 PM 신호 없음"
        )


    lines.append("")


    # ========================================================
    # OPTION SEARCH RELATION
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔗 <b>OPTION FLOW CHECK</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    if (
        regime["direction"]
        == "BULLISH"
    ):

        lines.append(
            "🟢 시장 방향: BULLISH"
        )

        lines.append(
            "→ Bullish Option Flow "
            "일치 여부 확인"
        )


    elif (
        regime["direction"]
        == "BEARISH"
    ):

        lines.append(
            "🔴 시장 방향: BEARISH"
        )

        lines.append(
            "→ Bullish Option Flow "
            "충돌 여부 확인"
        )


    else:

        lines.append(
            "🟡 시장 방향: NEUTRAL"
        )

        lines.append(
            "→ 개별 Option Flow 중심"
        )


    lines.append("")


    lines.append(
        "⚠️ PM 데이터는 "
        "장 마감 후 확정값 기준"
    )


    return "\n".join(
        lines
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    data,
    regime
):

    market_date = get_market_date()


    path = os.path.join(
        RESULT_DIR,
        f"PM_VOLATILITY_UPDATE_{market_date}.json"
    )


    payload = {

        "market_date_et":
            market_date,

        "created_at_et":
            get_et_now().isoformat(),

        "timezone":
            "America/New_York",

        "market_data":
            data,

        "pm_context":
            regime,

    }


    with open(
        path,
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
        f"💾 PM JSON 저장: {path}"
    )


    return path


# ============================================================
# SAVE TXT
# ============================================================

def save_text(
    message
):

    market_date = get_market_date()


    path = os.path.join(
        RESULT_DIR,
        f"PM_VOLATILITY_UPDATE_{market_date}.txt"
    )


    text = (
        message
        .replace(
            "<b>",
            ""
        )
        .replace(
            "</b>",
            ""
        )
    )


    with open(
        path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            text
        )


    print(
        f"💾 PM TXT 저장: {path}"
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
        "🌙 PM VOLATILITY UPDATE START"
    )

    print(
        "🇺🇸 TIMEZONE: America/New_York"
    )

    print(
        f"📅 MARKET DATE: "
        f"{get_market_date()}"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    data = collect_market_data()


    regime = calculate_market_regime(
        data
    )


    message = build_telegram_message(
        data,
        regime
    )


    save_json(
        data,
        regime
    )


    save_text(
        message
    )


    print("")

    print(
        message
    )

    print("")


    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "✅ PM VOLATILITY UPDATE COMPLETE"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


if __name__ == "__main__":

    main()
