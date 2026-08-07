import os
import requests
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    "7729872113"
)


# ============================================================
# RESULT FILE
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_FILE = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily",
    "OPTION_FINAL_RANKING.csv"
)


# ============================================================
# TELEGRAM
# ============================================================

def send_message(text):

    if not BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN이 없습니다."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    max_length = 4000

    chunks = [
        text[i:i + max_length]
        for i in range(
            0,
            len(text),
            max_length
        )
    ]

    for chunk in chunks:

        response = requests.post(
            url,
            data={
                "chat_id": CHAT_ID,
                "text": chunk
            },
            timeout=30
        )

        response.raise_for_status()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "📱 TELEGRAM V1 RESULT SENDER"
    )
    print("=" * 70)

    if not BOT_TOKEN:

        print(
            "❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다."
        )

        raise SystemExit(1)

    print(
        "✅ Telegram Bot Token 확인"
    )

    # --------------------------------------------------------
    # RESULT FILE
    # --------------------------------------------------------

    if not os.path.isfile(
        RESULT_FILE
    ):

        print(
            "❌ OPTION_FINAL_RANKING.csv를 찾을 수 없습니다."
        )

        print("")
        print(
            f"찾는 위치: {RESULT_FILE}"
        )

        raise SystemExit(1)

    print(
        f"✅ 결과 파일 발견: {RESULT_FILE}"
    )

    # --------------------------------------------------------
    # LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        RESULT_FILE
    )

    if df.empty:

        print(
            "❌ 결과 CSV가 비어 있습니다."
        )

        raise SystemExit(1)

    print(
        f"📊 종목 수: {len(df)}"
    )

    # --------------------------------------------------------
    # RANKING
    # --------------------------------------------------------

    message = (
        "🔥 OPTION FLOW SCANNER V1\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 전체 종목 점수\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for i, (_, row) in enumerate(
        df.head(20).iterrows(),
        1
    ):

        ticker = str(
            row.get(
                "ticker",
                row.get(
                    "symbol",
                    "UNKNOWN"
                )
            )
        )

        try:

            score = float(
                row.get(
                    "score",
                    0
                )
            )

        except Exception:

            score = 0

        if score >= 70:

            status = "🟢 오늘 진입 후보"

        elif score >= 40:

            status = "🟡 관망"

        else:

            status = "🔴 회피"

        message += (
            f"{i}. {ticker:<6} "
            f"{score:.1f}점 "
            f"{status}\n"
        )

    send_message(
        message
    )

    print(
        "✅ 전체 순위 Telegram 전송 완료"
    )

    # --------------------------------------------------------
    # TOP 5
    # --------------------------------------------------------

    detail = (
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🎯 TOP 5 구조 상세\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for _, row in df.head(5).iterrows():

        ticker = str(
            row.get(
                "ticker",
                "UNKNOWN"
            )
        )

        price = row.get(
            "current_price",
            "-"
        )

        call_wall = row.get(
            "call_wall",
            "-"
        )

        put_wall = row.get(
            "put_wall",
            "-"
        )

        iv = row.get(
            "iv",
            "-"
        )

        try:

            price_text = (
                f"${float(price):.2f}"
            )

        except Exception:

            price_text = str(price)

        try:

            call_text = (
                f"${float(call_wall):g}"
            )

        except Exception:

            call_text = str(call_wall)

        try:

            put_text = (
                f"${float(put_wall):g}"
            )

        except Exception:

            put_text = str(put_wall)

        try:

            iv_value = float(iv)

            if iv_value < 5:

                iv_text = (
                    f"{iv_value * 100:.1f}%"
                )

            else:

                iv_text = (
                    f"{iv_value:.1f}%"
                )

        except Exception:

            iv_text = str(iv)

        detail += (
            f"📌 {ticker} {price_text}\n"
            f"📈 Call Wall: {call_text}\n"
            f"📉 Put Wall: {put_text}\n"
            f"IV: {iv_text}\n\n"
        )

    send_message(
        detail
    )

    print(
        "✅ TOP 5 상세 Telegram 전송 완료"
    )

    print("")
    print("=" * 70)
    print(
        "🔥 TELEGRAM V1 완료"
    )
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
