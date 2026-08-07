import os
import time
import pandas as pd

from selected_symbols import SELECTED_SYMBOLS

from option_search import (
    analyze_ticker,
    send_telegram
)


# ============================================================
# CONFIG
# ============================================================

TOP_ENTRY = 5

ENTRY_SCORE = 70
WATCH_SCORE = 40


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

RANKING_FILE = os.path.join(
    RESULT_DIR,
    "OPTION_FINAL_RANKING.csv"
)


# ============================================================
# FORMAT
# ============================================================

def format_money(x):

    try:

        x = float(x)

    except Exception:

        return "$0"

    sign = ""

    if x < 0:

        sign = "-"
        x = abs(x)

    if x >= 1_000_000:

        return (
            f"{sign}${x / 1_000_000:.2f}M"
        )

    if x >= 1_000:

        return (
            f"{sign}${x / 1_000:.1f}K"
        )

    return (
        f"{sign}${x:.0f}"
    )


# ============================================================
# SCORE
# ============================================================

def calculate_score(
    df,
    greeks
):

    score = 50.0

    reasons = []

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    # ========================================================
    # PREMIUM
    # ========================================================

    call_premium = calls[
        "premium_flow"
    ].sum()

    put_premium = puts[
        "premium_flow"
    ].sum()

    total_premium = (
        call_premium
        + put_premium
    )

    if total_premium > 0:

        ratio = (
            call_premium
            / total_premium
        )

        if ratio >= 0.60:

            score += 10

            reasons.append(
                "Call Premium 강세"
            )

        elif ratio >= 0.55:

            score += 6

            reasons.append(
                "Call Premium 우세"
            )

        elif ratio <= 0.40:

            score -= 10

            reasons.append(
                "Put Premium 강세"
            )

        elif ratio <= 0.45:

            score -= 6

            reasons.append(
                "Put Premium 우세"
            )

    # ========================================================
    # VOLUME
    # ========================================================

    call_volume = calls[
        "volume"
    ].sum()

    put_volume = puts[
        "volume"
    ].sum()

    total_volume = (
        call_volume
        + put_volume
    )

    if total_volume > 0:

        ratio = (
            call_volume
            / total_volume
        )

        if ratio >= 0.60:

            score += 8

            reasons.append(
                "Call 거래량 우세"
            )

        elif ratio <= 0.40:

            score -= 8

            reasons.append(
                "Put 거래량 우세"
            )

    # ========================================================
    # OI
    # ========================================================

    call_oi = calls[
        "openInterest"
    ].sum()

    put_oi = puts[
        "openInterest"
    ].sum()

    total_oi = (
        call_oi
        + put_oi
    )

    if total_oi > 0:

        ratio = (
            call_oi
            / total_oi
        )

        if ratio >= 0.60:

            score += 6

            reasons.append(
                "Call OI 우세"
            )

        elif ratio <= 0.40:

            score -= 6

            reasons.append(
                "Put OI 우세"
            )

    # ========================================================
    # DTE
    # ========================================================

    dte_0_7 = df[
        (df["DTE"] >= 0)
        &
        (df["DTE"] <= 7)
    ]

    dte_8_30 = df[
        (df["DTE"] >= 8)
        &
        (df["DTE"] <= 30)
    ]

    dte_31_60 = df[
        (df["DTE"] >= 31)
        &
        (df["DTE"] <= 60)
    ]

    dte_61_180 = df[
        (df["DTE"] >= 61)
        &
        (df["DTE"] <= 180)
    ]

    if (
        not dte_0_7.empty
        and dte_8_30.empty
        and dte_31_60.empty
        and dte_61_180.empty
    ):

        score -= 6

        reasons.append(
            "초단기 DTE 집중"
        )

    if not dte_8_30.empty:

        score += 2

        reasons.append(
            "8~30DTE 구조"
        )

    if not dte_31_60.empty:

        score += 4

        reasons.append(
            "31~60DTE 구조"
        )

    if not dte_61_180.empty:

        score += 2

        reasons.append(
            "61~180DTE 구조"
        )

    long_calls = calls[
        calls["DTE"] >= 30
    ]

    if not long_calls.empty:

        score += 3

        reasons.append(
            "30D+ Call 구조 존재"
        )

    # ========================================================
    # DELTA
    # ========================================================

    delta = greeks.get(
        "Delta",
        0
    )

    if abs(delta) > 5_000_000:

        if delta > 0:

            score += 10

            reasons.append(
                "Delta Exposure 강한 상방"
            )

        else:

            score -= 10

            reasons.append(
                "Delta Exposure 강한 하방"
            )

    elif delta > 0:

        score += 6

        reasons.append(
            "Delta Exposure 상방"
        )

    elif delta < 0:

        score -= 6

        reasons.append(
            "Delta Exposure 하방"
        )

    # ========================================================
    # GEX
    # ========================================================

    gex = greeks.get(
        "GEX",
        0
    )

    if gex > 0:

        score += 4

        reasons.append(
            "GEX Positive"
        )

    elif gex < 0:

        score -= 4

        reasons.append(
            "GEX Negative"
        )

    # ========================================================
    # HIRO
    # ========================================================

    hiro = greeks.get(
        "HIRO",
        0
    )

    if hiro > 0:

        score += 4

        reasons.append(
            "HIRO Proxy Positive"
        )

    elif hiro < 0:

        score -= 4

        reasons.append(
            "HIRO Proxy Negative"
        )

    # ========================================================
    # VANNA
    # ========================================================

    vanna = greeks.get(
        "Vanna",
        0
    )

    if vanna > 0:

        score += 3

        reasons.append(
            "Vanna 상방"
        )

    elif vanna < 0:

        score -= 3

        reasons.append(
            "Vanna 하방"
        )

    # ========================================================
    # IV
    # ========================================================

    iv = greeks.get(
        "IV",
        0
    )

    try:

        iv_pct = float(iv) * 100

    except Exception:

        iv_pct = 0

    if iv_pct <= 40:

        score += 4

        reasons.append(
            "IV 낮음"
        )

    elif iv_pct <= 70:

        score += 2

        reasons.append(
            "IV 적정"
        )

    elif iv_pct <= 100:

        score -= 2

        reasons.append(
            "IV 높음"
        )

    elif iv_pct <= 150:

        score -= 5

        reasons.append(
            "IV 과열"
        )

    else:

        score -= 8

        reasons.append(
            "IV 극단적"
        )

    # ========================================================
    # SIGNAL CONFLICT
    # ========================================================

    bullish = 0
    bearish = 0

    if delta > 0:

        bullish += 1

    elif delta < 0:

        bearish += 1

    if gex > 0:

        bullish += 1

    elif gex < 0:

        bearish += 1

    if hiro > 0:

        bullish += 1

    elif hiro < 0:

        bearish += 1

    if vanna > 0:

        bullish += 1

    elif vanna < 0:

        bearish += 1

    if bullish >= 3 and bearish >= 1:

        score -= 8

        reasons.append(
            "⚠️ Signal Conflict"
        )

    # ========================================================
    # LIMIT
    # ========================================================

    score = max(
        0,
        min(
            100,
            score
        )
    )

    # ========================================================
    # DIRECTION
    # ========================================================

    if score >= 70:

        direction = "BULLISH"

    elif score >= 55:

        direction = "SLIGHT BULLISH"

    elif score >= 45:

        direction = "NEUTRAL"

    elif score >= 30:

        direction = "SLIGHT BEARISH"

    else:

        direction = "BEARISH"

    return (
        score,
        direction,
        reasons
    )


# ============================================================
# CATEGORY
# ============================================================

def classify(score):

    if score >= ENTRY_SCORE:

        return "🟢 오늘 진입 후보"

    if score >= WATCH_SCORE:

        return "🟡 관망"

    return "🔴 회피"


# ============================================================
# ANALYZE RESULT
# ============================================================

def make_final_result(
    analysis
):

    ticker = analysis["ticker"]

    df = analysis["df"]

    greeks = analysis["greeks"]

    (
        score,
        direction,
        reasons
    ) = calculate_score(
        df,
        greeks
    )

    category = classify(
        score
    )

    return {
        "ticker": ticker,
        "current_price":
            analysis["current_price"],
        "score": score,
        "direction": direction,
        "category": category,
        "reasons": reasons,
        "delta":
            greeks.get("Delta", 0),
        "gex":
            greeks.get("GEX", 0),
        "hiro":
            greeks.get("HIRO", 0),
        "vanna":
            greeks.get("Vanna", 0),
        "iv":
            greeks.get("IV", 0),
        "call_wall":
            analysis["call_wall"],
        "put_wall":
            analysis["put_wall"]
    }


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_ranking(results):

    rows = []

    for r in results:

        rows.append(
            {
                "ticker":
                    r["ticker"],

                "current_price":
                    r["current_price"],

                "score":
                    r["score"],

                "direction":
                    r["direction"],

                "category":
                    r["category"],

                "reasons":
                    " | ".join(
                        r["reasons"]
                    ),

                "delta":
                    r["delta"],

                "gex":
                    r["gex"],

                "hiro":
                    r["hiro"],

                "vanna":
                    r["vanna"],

                "iv":
                    r["iv"],

                "call_wall":
                    r["call_wall"],

                "put_wall":
                    r["put_wall"]
            }
        )

    df = pd.DataFrame(
        rows
    )

    df.to_csv(
        RANKING_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("")
    print(
        "💾 FINAL CSV 저장:"
    )

    print(
        RANKING_FILE
    )

    print(
        f"✅ 최종 종목 수: "
        f"{len(df)}"
    )


# ============================================================
# FINAL TELEGRAM MESSAGE
# ============================================================

def build_final_message(
    results
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>오늘의 OPTION FINAL RANKING</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    # ========================================================
    # ENTRY TOP 5
    # ========================================================

    entry = [
        r
        for r in results
        if r["score"] >= ENTRY_SCORE
    ][:TOP_ENTRY]

    lines.append(
        "🟢 <b>오늘 살 만한 후보 TOP 5</b>"
    )

    lines.append("")

    if entry:

        for i, r in enumerate(
            entry,
            1
        ):

            lines.append(
                f"<b>{i}. {r['ticker']}</b> "
                f"| {r['score']:.1f}점 "
                f"| {r['direction']}"
            )

            lines.append(
                "   → "
                + ", ".join(
                    r["reasons"][:5]
                )
            )

            lines.append("")

    else:

        lines.append(
            "오늘 진입 후보 없음"
        )

        lines.append("")

    # ========================================================
    # WATCH
    # ========================================================

    watch = [
        r
        for r in results
        if (
            WATCH_SCORE
            <= r["score"]
            < ENTRY_SCORE
        )
    ]

    lines.append(
        "🟡 <b>관망</b>"
    )

    lines.append("")

    if watch:

        for r in watch:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f}점 "
                f"| {r['direction']}"
            )

    else:

        lines.append(
            "관망 종목 없음"
        )

    lines.append("")

    # ========================================================
    # AVOID
    # ========================================================

    avoid = [
        r
        for r in results
        if r["score"] < WATCH_SCORE
    ]

    lines.append(
        "🔴 <b>회피</b>"
    )

    lines.append("")

    if avoid:

        for r in avoid:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f}점 "
                f"| {r['direction']}"
            )

    else:

        lines.append(
            "회피 종목 없음"
        )

    lines.append("")

    # ========================================================
    # ALL RANKING
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 <b>전체 종목 순위</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    for i, r in enumerate(
        results,
        1
    ):

        lines.append(
            f"{i}. "
            f"<b>{r['ticker']}</b> "
            f"| {r['score']:.1f}점 "
            f"| {r['category']}"
        )

    lines.append("")

    # ========================================================
    # TOP 5 DETAIL
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 <b>TOP 5 구조 상세</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    for r in entry:

        lines.append(
            f"📌 <b>{r['ticker']}</b> "
            f"${r['current_price']:.2f}"
        )

        if r["call_wall"] is not None:

            distance = (
                (
                    r["call_wall"]
                    - r["current_price"]
                )
                / r["current_price"]
                * 100
            )

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:g} "
                f"(+{distance:.1f}%)"
            )

        else:

            lines.append(
                "📈 Call Wall N/A"
            )

        if r["put_wall"] is not None:

            distance = (
                (
                    r["current_price"]
                    - r["put_wall"]
                )
                / r["current_price"]
                * 100
            )

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:g} "
                f"(-{distance:.1f}%)"
            )

        else:

            lines.append(
                "📉 Put Wall N/A"
            )

        try:

            iv_text = (
                f"{float(r['iv']) * 100:.1f}%"
            )

        except Exception:

            iv_text = "N/A"

        lines.append(
            f"IV {iv_text}"
        )

        lines.append(
            f"Delta "
            f"{format_money(r['delta'])}"
        )

        lines.append(
            f"GEX "
            f"{format_money(r['gex'])}"
        )

        lines.append("")

    # ========================================================
    # FOOTER
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "⚠️ Greeks/GEX/Wall은 옵션 데이터 기반 계산값입니다."
    )

    lines.append(
        "⚠️ HIRO는 yfinance 환경의 Proxy입니다."
    )

    lines.append(
        "⚠️ 옵션 거래량만으로 실제 BUY/SELL을 확정할 수 없습니다."
    )

    return "\n".join(
        lines
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("")
    print("=" * 70)

    print(
        "🔥 OPTION FLOW SCANNER V1"
    )

    print("=" * 70)

    print("")
    print(
        f"📊 검색 종목: "
        f"{len(SELECTED_SYMBOLS)}개"
    )

    print("")
    print(
        "📌 처리 방식:"
    )

    print(
        "1. 종목별 OPTION SEARCH"
    )

    print(
        "2. 종목별 Telegram"
    )

    print(
        "3. 종목별 CSV 저장"
    )

    print(
        "4. 같은 분석 데이터를 즉시 FINAL SCORE 계산"
    )

    print(
        "5. 전체 Ranking"
    )

    print(
        "6. 살 만한 후보 TOP 5"
    )

    print(
        "7. 최종 Telegram"
    )

    print("")

    results = []

    total = len(
        SELECTED_SYMBOLS
    )

    for i, ticker in enumerate(
        SELECTED_SYMBOLS,
        1
    ):

        ticker = (
            ticker
            .upper()
            .strip()
        )

        print("")
        print("=" * 70)

        print(
            f"🔥 {i}/{total} "
            f"{ticker}"
        )

        print("=" * 70)

        try:

            analysis = analyze_ticker(
                ticker
            )

            if analysis is None:

                print(
                    f"❌ {ticker} 분석 실패"
                )

                continue

            final_result = (
                make_final_result(
                    analysis
                )
            )

            results.append(
                final_result
            )

            print("")

            print(
                f"🎯 {ticker} "
                f"FINAL SCORE: "
                f"{final_result['score']:.1f}"
            )

            print(
                f"   방향: "
                f"{final_result['direction']}"
            )

            print(
                f"   판정: "
                f"{final_result['category']}"
            )

        except Exception as e:

            print("")

            print(
                f"❌ {ticker} 전체 분석 실패"
            )

            print(
                f"   {type(e).__name__}: {e}"
            )

        if i < total:

            print("")

            print(
                "⏳ 다음 종목 준비..."
            )

            time.sleep(3)

    print("")
    print("=" * 70)

    print(
        "📊 ALL OPTION SEARCH FINISHED"
    )

    print("=" * 70)

    print("")

    print(
        f"✅ 최종 분석 완료: "
        f"{len(results)}개"
    )

    if not results:

        print("")
        print(
            "❌ 분석 결과가 하나도 없습니다."
        )

        raise SystemExit(1)

    # ========================================================
    # SORT
    # ========================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_ranking(
        results
    )

    # ========================================================
    # FINAL MESSAGE
    # ========================================================

    final_message = (
        build_final_message(
            results
        )
    )

    print("")
    print("=" * 70)

    print(
        "🧠 FINAL OPTION RANKING"
    )

    print("=" * 70)

    print("")
    print(
        final_message
    )

    # ========================================================
    # FINAL TELEGRAM
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "📱 FINAL TELEGRAM"
    )

    print("=" * 70)

    telegram_ok = send_telegram(
        final_message
    )

    if telegram_ok:

        print(
            "✅ 최종 Telegram 전송 완료"
        )

    else:

        print(
            "⚠️ 최종 Telegram 전송 실패"
        )

    print("")
    print("=" * 70)

    print(
        "🔥 OPTION FLOW SCANNER V1 COMPLETE"
    )

    print("=" * 70)

    print("")

    print(
        "✅ 개별 OPTION SEARCH 완료"
    )

    print(
        "✅ 개별 Telegram 완료"
    )

    print(
        "✅ 개별 CSV 저장 완료"
    )

    print(
        "✅ FINAL SCORE 완료"
    )

    print(
        "✅ FINAL RANKING 완료"
    )

    print(
        "✅ 살 만한 후보 TOP 5 완료"
    )

    print(
        "✅ 최종 Telegram 완료"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
