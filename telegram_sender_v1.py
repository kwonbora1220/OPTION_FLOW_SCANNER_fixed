import os
import requests
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")

# 실제 Telegram Chat ID
CHAT_ID = "7729872113"

# ============================================================
# RESULT FILE SEARCH
# ============================================================

def find_result_file():

    target = "OPTION_FINAL_RANKING.csv"

    for root, dirs, files in os.walk("."):

        if target in files:

            return os.path.join(
                root,
                target
            )

    return None


# ============================================================
# TELEGRAM
# ============================================================

def send_message(text):

    if not BOT_TOKEN:

        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN이 없습니다."
        )

    url = (
        f"https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    response = requests.post(

        url,

        data={
            "chat_id": CHAT_ID,
            "text": text
        },

        timeout=30
    )

    response.raise_for_status()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("📱 TELEGRAM V1 RESULT SENDER")
    print("=" * 70)

    # --------------------------------------------------------
    # TOKEN
    # --------------------------------------------------------

    if not BOT_TOKEN:

        print(
            "❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다."
        )

        raise SystemExit(1)

    print("✅ Telegram Bot Token 확인")

    # --------------------------------------------------------
    # RESULT FILE SEARCH
    # --------------------------------------------------------

    result_file = find_result_file()

    if not result_file:

        print(
            "❌ OPTION_FINAL_RANKING.csv를 찾을 수 없습니다."
        )

        print("")
        print("현재 저장소의 CSV 파일:")

        for root, dirs, files in os.walk("."):

            for file in files:

                if file.endswith(".csv"):

                    print(
                        os.path.join(
                            root,
                            file
                        )
                    )

        raise SystemExit(1)

    print(
        f"✅ 결과 파일 발견: {result_file}"
    )

    # --------------------------------------------------------
    # CSV LOAD
    # --------------------------------------------------------

    df = pd.read_csv(
        result_file
    )

    print(
        f"📊 종목 수: {len(df)}"
    )

    # --------------------------------------------------------
    # 전체 순위
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

        score = row.get(
            "score",
            0
        )

        try:

            score = float(score)

        except:

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

    # --------------------------------------------------------
    # Telegram 전송
    # --------------------------------------------------------

    send_message(
        message
    )

    print(
        "✅ 전체 순위 Telegram 전송 완료"
    )

    # --------------------------------------------------------
    # TOP 5 상세
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
                row.get(
                    "symbol",
                    "UNKNOWN"
                )
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

            iv_value = float(iv)

            if iv_value < 5:

                iv_text = (
                    f"{iv_value * 100:.1f}%"
                )

            else:

                iv_text = (
                    f"{iv_value:.1f}%"
                )

        except:

            iv_text = str(iv)

        detail += (
            f"📌 {ticker} ${price}\n"
            f"📈 Call Wall: ${call_wall}\n"
            f"📉 Put Wall: ${put_wall}\n"
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
    print("🔥 TELEGRAM V1 완료")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()
