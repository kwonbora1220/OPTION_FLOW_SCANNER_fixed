import os
import time
import pandas as pd

from selected_symbols import SELECTED_SYMBOLS

from option_search import (
    run,
    calculate_aggregate_greeks,
)


# ============================================================
# CONFIG
# ============================================================

TOP_ENTRY = 5

ENTRY_SCORE = 70
WATCH_SCORE = 40

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

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
# SCORE
# ============================================================

def calculate_score(
    df,
    greeks,
    current_price
):

    score = 50.0

    reasons = []

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    # ========================================================
    # CALL / PUT PREMIUM
    # ========================================================

    if not calls.empty and not puts.empty:

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

            call_ratio = (
                call_premium
                / total_premium
            )

            if call_ratio >= 0.60:

                score += 10

                reasons.append(
                    "Call Premium 강세"
                )

            elif call_ratio >= 0.55:

                score += 6

                reasons.append(
                    "Call Premium 우세"
                )

            elif call_ratio <= 0.40:

                score -= 10

                reasons.append(
                    "Put Premium 강세"
                )

            elif call_ratio <= 0.45:

                score -= 6

                reasons.append(
                    "Put Premium 우세"
                )

    # ========================================================
    # VOLUME
    # ========================================================

    if not calls.empty and not puts.empty:

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

            call_ratio = (
                call_volume
                / total_volume
            )

            if call_ratio >= 0.60:

                score += 8

                reasons.append(
                    "Call 거래량 우세"
                )

            elif call_ratio <= 0.40:

                score -= 8

                reasons.append(
                    "Put 거래량 우세"
                )

    # ========================================================
    # OPEN INTEREST
    # ========================================================

    if not calls.empty and not puts.empty:

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

            call_ratio = (
                call_oi
                / total_oi
            )

            if call_ratio >= 0.60:

                score += 6

                reasons.append(
                    "Call OI 우세"
                )

            elif call_ratio <= 0.40:

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

    long_puts = puts[
        puts["DTE"] >= 30
    ]

    if not long_puts.empty:

        reasons.append(
            "30D+ Put 구조 존재"
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

    if (
        bullish >= 3
        and bearish >= 1
    ):

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
# LOAD INDIVIDUAL OPTION SEARCH RESULT
# ============================================================

def load_individual_result(ticker):

    filename = os.path.join(
        RESULT_DIR,
        f"{ticker}_OPTION_SEARCH.csv"
    )

    if not os.path.exists(filename):

        return None

    try:

        df = pd.read_csv(
            filename
        )

        if df.empty:

            return None

        return df

    except Exception as e:

        print(
            f"❌ {ticker} CSV 읽기 실패: {e}"
        )

        return None


# ============================================================
# CURRENT PRICE
# ============================================================

def get_price_from_df(df):

    try:

        # OPTION SEARCH CSV에는
        # strike 등이 있으므로
        # 현재가는 별도 계산하지 않는다.
        #
        # 마지막으로 저장된 옵션의
        # 현재가를 직접 가져올 수 없으므로
        # option_search의 함수 사용

        from option_search import (
            get_current_price
        )

        return get_current_price(
            df["_ticker"].iloc[0]
        )

    except Exception:

        return None


# ============================================================
# INDIVIDUAL ANALYSIS
# ============================================================

def analyze_individual_csv(
    ticker,
    df
):

    try:

        from option_search import (
            get_current_price
        )

        current_price = get_current_price(
            ticker
        )

        if current_price is None:

            return None

        # ----------------------------------------------------
        # option_search.py에서 이미 계산된 컬럼을
        # 사용하는 구조
        # ----------------------------------------------------

        greeks = calculate_aggregate_greeks(
            df
        )

        score, direction, reasons = (
            calculate_score(
                df,
                greeks,
                current_price
            )
        )

        category = classify(
            score
        )

        call_wall = None
        put_wall = None

        # ----------------------------------------------------
        # CALL WALL
        # ----------------------------------------------------

        active = df[
            df["volume"] > 0
        ].copy()

        if not active.empty:

            active["distance_pct"] = (
                abs(
                    active["strike"]
                    - current_price
                )
                / current_price
                * 100
            )

            active = active[
                active["distance_pct"] <= 30
            ].copy()

        if not active.empty:

            calls = active[
                (
                    active["option_type"]
                    == "CALL"
                )
                &
                (
                    active["strike"]
                    > current_price
                )
            ]

            if not calls.empty:

                group = (
                    calls
                    .groupby("strike")["GEX"]
                    .sum()
                )

                if not group.empty:

                    call_wall = float(
                        group.idxmax()
                    )

            puts = active[
                (
                    active["option_type"]
                    == "PUT"
                )
                &
                (
                    active["strike"]
                    < current_price
                )
            ]

            if not puts.empty:

                group = (
                    puts
                    .groupby("strike")["GEX"]
                    .sum()
                )

                if not group.empty:

                    put_wall = float(
                        group.idxmin()
                    )

        return {
            "ticker": ticker,
            "current_price": current_price,
            "score": score,
            "direction": direction,
            "category": category,
            "reasons": reasons,
            "delta": greeks.get(
                "Delta",
                0
            ),
            "gex": greeks.get(
                "GEX",
                0
            ),
            "hiro": greeks.get(
                "HIRO",
                0
            ),
            "vanna": greeks.get(
                "Vanna",
                0
            ),
            "iv": greeks.get(
                "IV",
                0
            ),
            "call_wall": call_wall,
            "put_wall": put_wall
        }

    except Exception as e:

        print(
            f"❌ {ticker} 최종 분석 오류: {e}"
        )

        return None


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

    pd.DataFrame(
        rows
    ).to_csv(
        RANKING_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("")
    print(
        f"💾 FINAL CSV 저장: "
        f"{RANKING_FILE}"
    )

    if os.path.exists(
        RANKING_FILE
    ):

        size = os.path.getsize(
            RANKING_FILE
        )

        print(
            f"✅ CSV 생성 확인 "
            f"({size:,} bytes)"
        )


# ============================================================
# BUILD FINAL TELEGRAM
# ============================================================

def build_final_message(
    results
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 오늘의 OPTION FINAL RANKING"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    # ========================================================
    # ENTRY
    # ========================================================

    entry = [
        x
        for x in results
        if x["score"] >= ENTRY_SCORE
    ][:TOP_ENTRY]

    lines.append(
        "🟢 오늘 진입 후보 TOP 5"
    )

    lines.append("")

    if entry:

        for i, r in enumerate(
            entry,
            1
        ):

            lines.append(
                f"{i}. "
                f"{r['ticker']} | "
                f"{r['score']:.1f}점 | "
                f"{r['direction']}"
            )

            if r["reasons"]:

                lines.append(
                    "   → "
                    + ", ".join(
                        r["reasons"]
                    )
                )

    else:

        lines.append(
            "오늘 진입 후보 없음"
        )

    lines.append("")

    # ========================================================
    # WATCH
    # ========================================================

    watch = [
        x
        for x in results
        if (
            WATCH_SCORE
            <= x["score"]
            < ENTRY_SCORE
        )
    ]

    lines.append(
        "🟡 관망"
    )

    lines.append("")

    if watch:

        for r in watch:

            lines.append(
                f"• {r['ticker']} | "
                f"{r['score']:.1f}점 | "
                f"{r['direction']}"
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
        x
        for x in results
        if x["score"] < WATCH_SCORE
    ]

    lines.append(
        "🔴 회피"
    )

    lines.append("")

    if avoid:

        for r in avoid:

            lines.append(
                f"• {r['ticker']} | "
                f"{r['score']:.1f}점 | "
                f"{r['direction']}"
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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 전체 종목 점수"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    for i, r in enumerate(
        results,
        1
    ):

        lines.append(
            f"{i}. "
            f"{r['ticker']:<6} "
            f"{r['score']:>5.1f}점 "
            f"{r['category']}"
        )

    lines.append("")

    # ========================================================
    # TOP DETAIL
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 TOP 5 구조 상세"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    if entry:

        for r in entry:

            lines.append(
                f"📌 {r['ticker']} "
                f"${r['current_price']:.2f}"
            )

            if r["call_wall"] is not None:

                distance = (
                    r["call_wall"]
                    - r["current_price"]
                ) / r["current_price"] * 100

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
                    r["current_price"]
                    - r["put_wall"]
                ) / r["current_price"] * 100

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

            lines.append("")

    # ========================================================
    # FOOTER
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
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
# TELEGRAM FINAL
# ============================================================

def send_final_telegram(
    message
):

    try:

        from option_search import (
            send_telegram
        )

        ok = send_telegram(
            message
        )

        if ok:

            print(
                "✅ FINAL Telegram 전송 완료"
            )

        else:

            print(
                "❌ FINAL Telegram 전송 실패"
            )

    except Exception as e:

        print(
            f"❌ FINAL Telegram 오류: {e}"
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
        "📌 실행 순서:"
    )

    print(
        "1. 개별 OPTION SEARCH"
    )

    print(
        "2. 개별 Telegram 전송"
    )

    print(
        "3. 개별 CSV 저장"
    )

    print(
        "4. 전체 종목 최종 Ranking"
    )

    print(
        "5. OPTION_FINAL_RANKING.csv 저장"
    )

    print(
        "6. 최종 Telegram 전송"
    )

    print("")

    results = []

    # ========================================================
    # STEP 1
    # 개별 종목 OPTION SEARCH
    # ========================================================

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
            f"🔥 INDIVIDUAL OPTION SEARCH "
            f"{i}/{len(SELECTED_SYMBOLS)}"
        )

        print(
            f"📌 {ticker}"
        )

        print("=" * 70)

        try:

            # ------------------------------------------------
            # 핵심
            #
            # option_search.py의 run(ticker)를 호출한다.
            #
            # 여기서:
            #
            # 현재가 조회
            # ↓
            # 옵션 전체 만기 수집
            # ↓
            # Greeks/GEX 계산
            # ↓
            # OPTION SEARCH 리포트 생성
            # ↓
            # 개별 Telegram 전송
            # ↓
            # {TICKER}_OPTION_SEARCH.csv 저장
            #
            # ------------------------------------------------

            telegram_ok = run(
                ticker
            )

            if telegram_ok:

                print(
                    f"✅ {ticker} "
                    "개별 OPTION SEARCH + Telegram 완료"
                )

            else:

                print(
                    f"⚠️ {ticker} "
                    "분석은 완료되었으나 Telegram 확인 필요"
                )

        except Exception as e:

            print("")
            print(
                f"❌ {ticker} "
                f"OPTION SEARCH 실패"
            )

            print(
                f"   {e}"
            )

            continue

        # ----------------------------------------------------
        # 개별 CSV 확인
        # ----------------------------------------------------

        csv_file = os.path.join(
            RESULT_DIR,
            f"{ticker}_OPTION_SEARCH.csv"
        )

        if not os.path.exists(
            csv_file
        ):

            print("")
            print(
                f"❌ {ticker} "
                "개별 CSV가 없습니다."
            )

            continue

        print("")
        print(
            f"✅ 개별 CSV 확인: "
            f"{csv_file}"
        )

        # ----------------------------------------------------
        # CSV 다시 읽기
        # ----------------------------------------------------

        try:

            df = pd.read_csv(
                csv_file
            )

            if df.empty:

                print(
                    f"⚠️ {ticker} "
                    "CSV가 비어 있습니다."
                )

                continue

            # ------------------------------------------------
            # FINAL SCORE
            # ------------------------------------------------

            final_result = (
                analyze_individual_csv(
                    ticker,
                    df
                )
            )

            if final_result:

                results.append(
                    final_result
                )

                print("")
                print(
                    f"🎯 {ticker} "
                    f"최종 점수: "
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

            print(
                f"❌ {ticker} "
                f"최종 점수 계산 실패: {e}"
            )

        # ----------------------------------------------------
        # 다음 종목
        # ----------------------------------------------------

        if i < len(
            SELECTED_SYMBOLS
        ):

            print("")
            print(
                "⏳ 다음 종목 준비..."
            )

            time.sleep(3)

    # ========================================================
    # STEP 2
    # ALL RESULT CHECK
    # ========================================================

    print("")
    print("=" * 70)
    print(
        "📊 ALL INDIVIDUAL SEARCH FINISHED"
    )
    print("=" * 70)

    print("")
    print(
        f"✅ 분석 완료 종목: "
        f"{len(results)}"
    )

    if not results:

        print("")
        print(
            "❌ 최종 Ranking을 만들 수 없습니다."
        )

        raise SystemExit(1)

    # ========================================================
    # STEP 3
    # SORT
    # ========================================================

    results = sorted(
        results,
        key=lambda x: x["score"],
        reverse=True
    )

    # ========================================================
    # STEP 4
    # FINAL CSV
    # ========================================================

    print("")
    print("=" * 70)
    print(
        "💾 FINAL RANKING CSV"
    )
    print("=" * 70)

    save_ranking(
        results
    )

    # ========================================================
    # STEP 5
    # FINAL MESSAGE
    # ========================================================

    print("")
    print("=" * 70)
    print(
        "🧠 FINAL OPTION RANKING"
    )
    print("=" * 70)

    final_message = (
        build_final_message(
            results
        )
    )

    print("")
    print(
        final_message
    )

    # ========================================================
    # STEP 6
    # FINAL TELEGRAM
    # ========================================================

    print("")
    print("=" * 70)
    print(
        "📱 FINAL TELEGRAM"
    )
    print("=" * 70)

    send_final_telegram(
        final_message
    )

    # ========================================================
    # DONE
    # ========================================================

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
        "✅ 개별 Telegram 전송 완료"
    )

    print(
        "✅ 개별 CSV 생성 완료"
    )

    print(
        "✅ 최종 Ranking 완료"
    )

    print(
        "✅ OPTION_FINAL_RANKING.csv 생성 완료"
    )

    print(
        "✅ 최종 Telegram 전송 완료"
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
