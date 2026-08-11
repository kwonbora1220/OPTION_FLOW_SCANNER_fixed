import os
import time
import pandas as pd

from selected_symbols import SELECTED_SYMBOLS

from option_search import (
    analyze_ticker,
    send_telegram
)

from oi_history import (
    calculate_oi_change,
    save_oi_snapshot,
    format_oi_change
)

from signal_history import (
    record_signal,
    update_signal_results,
    get_signal_stats,
    format_signal_stats
)

# ============================================================
# CONFIG
# ============================================================

TOP_ENTRY = 5

ENTRY_SCORE = 70
WATCH_SCORE = 45


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

if os.path.basename(BASE_DIR).upper() == "02_PROGRAM":

    RESULT_DIR = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "..",
            "03_RESULTS",
            "daily"
        )
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

    if x >= 1_000_000_000:

        return (
            f"{sign}${x / 1_000_000_000:.2f}B"
        )

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
    analysis
):

    df = analysis["df"]

    greeks = analysis["greeks"]

    flow = analysis["flow"]

    walls = analysis["walls"]

    quality = analysis["quality"]

    current_price = analysis[
        "current_price"
    ]

    score = 50.0

    reasons = []

    bullish_signals = 0
    bearish_signals = 0

    # ========================================================
    # 1. CALL / PUT VOLUME
    # ========================================================

    volume_ratio = (
        flow["call_volume_ratio"]
    )

    if volume_ratio >= 0.60:

        score += 6

        bullish_signals += 1

        reasons.append(
            "Call 거래량 우세"
        )

    elif volume_ratio >= 0.55:

        score += 3

        bullish_signals += 1

        reasons.append(
            "Call 거래량 소폭 우세"
        )

    elif volume_ratio <= 0.40:

        score -= 6

        bearish_signals += 1

        reasons.append(
            "Put 거래량 우세"
        )

    elif volume_ratio <= 0.45:

        score -= 3

        bearish_signals += 1

        reasons.append(
            "Put 거래량 소폭 우세"
        )

    # ========================================================
    # 2. TRADED PREMIUM PROXY
    # ========================================================
    #
    # 실제 BUY/SELL이 아니므로 영향도를 낮춘다.
    #

    premium_ratio = (
        flow["call_premium_ratio"]
    )

    if premium_ratio >= 0.60:

        score += 5

        bullish_signals += 1

        reasons.append(
            "Call 거래대금 Proxy 우세"
        )

    elif premium_ratio >= 0.55:

        score += 2

        reasons.append(
            "Call 거래대금 Proxy 소폭 우세"
        )

    elif premium_ratio <= 0.40:

        score -= 5

        bearish_signals += 1

        reasons.append(
            "Put 거래대금 Proxy 우세"
        )

    elif premium_ratio <= 0.45:

        score -= 2

        reasons.append(
            "Put 거래대금 Proxy 소폭 우세"
        )

    # ========================================================
    # 3. DELTA EXPOSURE PROXY
    # ========================================================

    delta = float(
        greeks.get(
            "Delta",
            0
        )
    )

    if delta > 0:

        score += 7

        bullish_signals += 1

        reasons.append(
            "OI Delta Exposure Proxy 상방"
        )

    elif delta < 0:

        score -= 7

        bearish_signals += 1

        reasons.append(
            "OI Delta Exposure Proxy 하방"
        )

    # ========================================================
    # 4. VANNA
    # ========================================================

    vanna = float(
        greeks.get(
            "Vanna",
            0
        )
    )

    if vanna > 0:

        score += 3

        bullish_signals += 1

        reasons.append(
            "Vanna 상방"
        )

    elif vanna < 0:

        score -= 3

        bearish_signals += 1

        reasons.append(
            "Vanna 하방"
        )

    # ========================================================
    # 5. HIRO-LIKE PROXY
    # ========================================================
    #
    # 실제 HIRO가 아니므로 ±2만 반영.
    #

    hiro = float(
        greeks.get(
            "HIRO",
            0
        )
    )

    if hiro > 0:

        score += 2

        bullish_signals += 1

        reasons.append(
            "체결방향 Flow Proxy 상방"
        )

    elif hiro < 0:

        score -= 2

        bearish_signals += 1

        reasons.append(
            "체결방향 Flow Proxy 하방"
        )

    # ========================================================
    # 6. GEX REGIME
    # ========================================================
    #
    # GEX는 방향성으로 사용하지 않는다.
    #
    # Positive GEX = 상대적으로 가격 안정/감쇠 가능성
    # Negative GEX = 변동성 확대 가능성
    #

    gex = float(
        greeks.get(
            "GEX",
            0
        )
    )

    if gex > 0:

        reasons.append(
            "Positive GEX Regime"
        )

    elif gex < 0:

        reasons.append(
            "Negative GEX Regime"
        )

    # ========================================================
    # 7. ATM IV
    # ========================================================

    atm_iv = float(
        flow.get(
            "atm_iv",
            0
        )
    )

    iv_pct = (
        atm_iv * 100
    )

    if iv_pct <= 40:

        score += 3

        reasons.append(
            "ATM IV 낮음"
        )

    elif iv_pct <= 70:

        score += 1

        reasons.append(
            "ATM IV 적정"
        )

    elif iv_pct <= 100:

        score -= 2

        reasons.append(
            "ATM IV 높음"
        )

    elif iv_pct <= 150:

        score -= 5

        reasons.append(
            "ATM IV 과열"
        )

    else:

        score -= 8

        reasons.append(
            "ATM IV 극단적"
        )

    # ========================================================
    # 8. DTE STRUCTURE
    # ========================================================

    dte = flow["dte_buckets"]

    if dte["31_60"] > 0:

        score += 2

        reasons.append(
            "31~60DTE 구조 존재"
        )

    if dte["61_180"] > 0:

        score += 1

        reasons.append(
            "61~180DTE 구조 존재"
        )

    # ========================================================
    # 9. WALL / PRICE LOCATION
    # ========================================================

    call_wall = walls.get(
        "call_wall"
    )

    put_wall = walls.get(
        "put_wall"
    )

    call_distance = None
    put_distance = None

    if call_wall is not None:

        call_distance = (
            (
                call_wall
                - current_price
            )
            / current_price
            * 100
        )

        # 현재가가 Call Wall 바로 아래라면
        # bullish 방향이어도 진입에는 불리할 수 있다.

        if (
            0 <= call_distance <= 3
        ):

            score -= 5

            reasons.append(
                "Call Wall 바로 아래"
            )

        elif call_distance >= 8:

            score += 2

            reasons.append(
                "상방 여유 구간"
            )

    if put_wall is not None:

        put_distance = (
            (
                current_price
                - put_wall
            )
            / current_price
            * 100
        )

        if (
            0 <= put_distance <= 3
        ):

            score -= 5

            reasons.append(
                "Put Wall 바로 위"
            )

        elif put_distance >= 8:

            score += 2

            reasons.append(
                "하방 완충 여유"
            )

    # ========================================================
    # 10. SIGNAL CONFLICT
    # ========================================================

    if (
        bullish_signals >= 3
        and bearish_signals >= 2
    ):

        score -= 7

        reasons.append(
            "⚠️ Signal Conflict"
        )

    elif (
        bearish_signals >= 3
        and bullish_signals >= 2
    ):

        score += 0

        reasons.append(
            "⚠️ Bearish Signal Conflict"
        )

    # ========================================================
    # 11. DATA QUALITY
    # ========================================================

    quality_score = float(
        quality.get(
            "score",
            0
        )
    )

    if quality_score < 40:

        score -= 5

        reasons.append(
            "⚠️ 낮은 데이터 품질"
        )

    elif quality_score >= 75:

        reasons.append(
            "데이터 품질 양호"
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

    if (
        bullish_signals
        >= bearish_signals + 2
    ):

        direction = "BULLISH"

    elif (
        bearish_signals
        >= bullish_signals + 2
    ):

        direction = "BEARISH"

    else:

        direction = "NEUTRAL"

    # ========================================================
    # STRUCTURE
    # ========================================================

    if gex > 0:

        structure = "STABLE_GEX"

    elif gex < 0:

        structure = "HIGH_VOL_GEX"

    else:

        structure = "NEUTRAL_GEX"

    # ========================================================
    # FINAL ACTION
    # ========================================================

    # Extreme IV에서는 단순 bullish만으로
    # 진입시키지 않는다.

    if (
        score >= ENTRY_SCORE
        and direction == "BULLISH"
        and iv_pct < 150
        and quality_score >= 40
    ):

        category = "🟢 오늘 진입 후보"

    elif (
        score <= 35
        or direction == "BEARISH"
    ):

        category = "🔴 회피"

    else:

        category = "🟡 관망"

    return {
        "score": score,
        "direction": direction,
        "structure": structure,
        "category": category,
        "reasons": reasons,
        "call_distance": call_distance,
        "put_distance": put_distance,
        "iv_pct": iv_pct,
        "bullish_signals": bullish_signals,
        "bearish_signals": bearish_signals
    }


# ============================================================
# FINAL RESULT
# ============================================================

def make_final_result(
    analysis
):

    score_data = calculate_score(
        analysis
    )

    greeks = analysis[
        "greeks"
    ]

    flow = analysis[
        "flow"
    ]

    walls = analysis[
        "walls"
    ]

    quality = analysis[
        "quality"
    ]

    return {

        "ticker":
            analysis["ticker"],

        "current_price":
            analysis["current_price"],

        "score":
            score_data["score"],

        "direction":
            score_data["direction"],

        "structure":
            score_data["structure"],

        "category":
            score_data["category"],

        "reasons":
            score_data["reasons"],

        "delta":
            greeks.get(
                "Delta",
                0
            ),

        "gex":
            greeks.get(
                "GEX",
                0
            ),

        "vanna":
            greeks.get(
                "Vanna",
                0
            ),

        "charm":
            greeks.get(
                "Charm",
                0
            ),

        "vega":
            greeks.get(
                "Vega",
                0
            ),

        "hiro":
            greeks.get(
                "HIRO",
                0
            ),

        "atm_iv":
            flow.get(
                "atm_iv",
                0
            ),

        "call_volume_ratio":
            flow.get(
                "call_volume_ratio",
                0.5
            ),

        "call_oi_ratio":
            flow.get(
                "call_oi_ratio",
                0.5
            ),

        "call_premium_ratio":
            flow.get(
                "call_premium_ratio",
                0.5
            ),

        "call_wall":
            walls.get(
                "call_wall"
            ),

        "put_wall":
            walls.get(
                "put_wall"
            ),

        "call_wall_gex":
            walls.get(
                "call_wall_gex",
                0
            ),

        "put_wall_gex":
            walls.get(
                "put_wall_gex",
                0
            ),

        "quality":
            quality.get(
                "score",
                0
            ),

        "bullish_signals":
            score_data[
                "bullish_signals"
            ],

        "bearish_signals":
            score_data[
                "bearish_signals"
            ]
    }


# ============================================================
# SAVE FINAL CSV
# ============================================================

def save_ranking(
    results
):

    rows = []

    for r in results:

        rows.append({

            "ticker":
                r["ticker"],

            "current_price":
                r["current_price"],

            "score":
                r["score"],

            "direction":
                r["direction"],

            "structure":
                r["structure"],

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

            "vanna":
                r["vanna"],

            "charm":
                r["charm"],

            "vega":
                r["vega"],

            "hiro":
                r["hiro"],

            "atm_iv":
                r["atm_iv"],

            "call_volume_ratio":
                r["call_volume_ratio"],

            "call_oi_ratio":
                r["call_oi_ratio"],

            "call_premium_ratio":
                r["call_premium_ratio"],

            "call_wall":
                r["call_wall"],

            "put_wall":
                r["put_wall"],

            "call_wall_gex":
                r["call_wall_gex"],

            "put_wall_gex":
                r["put_wall_gex"],

            "data_quality":
                r["quality"],

            "bullish_signals":
                r["bullish_signals"],

            "bearish_signals":
                r["bearish_signals"]
        })

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
        f"💾 FINAL CSV 저장:"
    )

    print(
        RANKING_FILE
    )

    print(
        f"✅ 최종 종목 수: "
        f"{len(df)}"
    )


# ============================================================
# FINAL TELEGRAM
# ============================================================

def build_final_message(
    results
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>오늘의 PORTFOLIO "
        "OPTION RANKING</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    # ========================================================
    # ENTRY
    # ========================================================

    entry = [
        r
        for r in results
        if r["category"]
        == "🟢 오늘 진입 후보"
    ][:TOP_ENTRY]

    lines.append(
        "🟢 <b>진입 후보 TOP 5</b>"
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
                    r["reasons"][:4]
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
        if r["category"]
        == "🟡 관망"
    ]

    lines.append(
        "🟡 <b>관망</b>"
    )

    lines.append("")

    if watch:

        for r in watch:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f} "
                f"| {r['direction']} "
                f"| {r['structure']}"
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
        if r["category"]
        == "🔴 회피"
    ]

    lines.append(
        "🔴 <b>회피</b>"
    )

    lines.append("")

    if avoid:

        for r in avoid:

            lines.append(
                f"• {r['ticker']} "
                f"| {r['score']:.1f} "
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
            f"{i}. <b>{r['ticker']}</b> "
            f"| {r['score']:.1f} "
            f"| {r['direction']} "
            f"| {r['category']}"
        )

    lines.append("")

    # ========================================================
    # TOP DETAIL
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

    for r in results[:TOP_ENTRY]:

        lines.append(
            f"📌 <b>{r['ticker']}</b> "
            f"${r['current_price']:.2f}"
        )

        lines.append(
            f"🎯 Score "
            f"{r['score']:.1f} "
            f"| {r['direction']}"
        )

        lines.append(
            f"🏗 Structure "
            f"{r['structure']}"
        )

        try:

            iv_text = (
                f"{float(r['atm_iv']) * 100:.1f}%"
            )

        except Exception:

            iv_text = "N/A"

        lines.append(
            f"IV {iv_text}"
        )

        lines.append(
            f"GEX "
            f"{format_money(r['gex'])}"
        )

        lines.append(
            f"Delta Proxy "
            f"{format_money(r['delta'])}"
        )

        lines.append(
            f"Vanna "
            f"{format_money(r['vanna'])}"
        )

        if r["call_wall"] is not None:

            lines.append(
                f"📈 Call Wall "
                f"${r['call_wall']:g}"
            )

        else:

            lines.append(
                "📈 Call Wall N/A"
            )

        if r["put_wall"] is not None:

            lines.append(
                f"📉 Put Wall "
                f"${r['put_wall']:g}"
            )

        else:

            lines.append(
                "📉 Put Wall N/A"
            )

        lines.append("")

    lines.append(
        "⚠️ GEX / Delta / Vanna는 "
        "OI 기반 Proxy입니다."
    )

    lines.append(
        "⚠️ 거래대금은 실제 Buy/Sell Flow가 아닙니다."
    )

    lines.append(
        "⚠️ 무료 yfinance 데이터에는 "
        "실제 체결 방향 정보가 없습니다."
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
        "🔥 PORTFOLIO OPTION SCANNER V2"
    )

    print("=" * 70)

    print("")

    print(
        f"📊 분석 종목: "
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
        "2. 종목별 CSV"
    )

    print(
        "3. 종목별 Telegram"
    )

    print(
        "4. Direction / Structure / IV 분석"
    )

    print(
        "5. FINAL SCORE"
    )

    print(
        "6. 전체 Ranking"
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
            f"🔥 {i}/{total} {ticker}"
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
                f"SCORE: "
                f"{final_result['score']:.1f}"
            )

            print(
                f"   방향: "
                f"{final_result['direction']}"
            )

            print(
                f"   구조: "
                f"{final_result['structure']}"
            )

            print(
                f"   판정: "
                f"{final_result['category']}"
            )

        except Exception as e:

            print("")

            print(
                f"❌ {ticker} 분석 실패"
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

    # ========================================================
    # CHECK
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "📊 ALL OPTION SEARCH FINISHED"
    )

    print("=" * 70)

    print("")

    print(
        f"✅ 분석 완료: "
        f"{len(results)}개"
    )

    if not results:

        print(
            "❌ 분석 결과가 없습니다."
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
        "🧠 FINAL PORTFOLIO RANKING"
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

    # ========================================================
    # DONE
    # ========================================================

    print("")
    print("=" * 70)

    print(
        "🔥 PORTFOLIO OPTION SCANNER V2 COMPLETE"
    )

    print("=" * 70)


if __name__ == "__main__":

    main()
