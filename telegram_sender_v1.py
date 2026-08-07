import os
import requests
import pandas as pd

# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT_ID = "7729872113"

RESULT_FILE = os.path.join(
    "03_RESULTS",
    "daily",
    "OPTION_FINAL_RANKING.csv"
)

# ============================================================
# TELEGRAM
# ============================================================

def send_message(text):
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendMessage"

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

    if not BOT_TOKEN:
        print("❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다.")
        return

    if not os.path.exists(RESULT_FILE):
        print(f"❌ 결과 파일이 없습니다: {RESULT_FILE}")
        return

    df = pd.read_csv(RESULT_FILE)

    print(f"📊 결과 파일: {RESULT_FILE}")
    print(f"📊 종목 수: {len(df)}")

    # --------------------------------------------------------
    # 전체 순위
    # --------------------------------------------------------

    message = (
        "🔥 OPTION FLOW SCANNER V1\n\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "📊 전체 종목 점수\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n"
    )

    for i, (_, row) in enumerate(df.head(20).iterrows(), 1):

        ticker = str(
            row.get("symbol",
            row.get("ticker", "UNKNOWN"))
        )

        score = row.get(
            "score",
            row.get("final_score", 0)
        )

        try:
            score = float(score)
        except:
            score = 0

        if score >= 80:
            status = "🟢 오늘 진입 후보"
        elif score >= 60:
            status = "🟡 관망"
        else:
            status = "🔴 회피"

        message += (
            f"{i}. {ticker:<6} "
            f"{score:.1f}점 {status}\n"
        )

    send_message(message)

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
            row.get("symbol",
            row.get("ticker", "UNKNOWN"))
        )

        price = row.get(
            "current_price",
            row.get("price", "-")
        )

        call_wall = row.get(
            "call_wall",
            row.get("Call Wall", "-")
        )

        put_wall = row.get(
            "put_wall",
            row.get("Put Wall", "-")
        )

        iv = row.get(
            "iv",
            row.get("IV", "-")
        )

        detail += (
            f"📌 {ticker} ${price}\n"
            f"📈 Call Wall: ${call_wall}\n"
            f"📉 Put Wall: ${put_wall}\n"
            f"IV: {iv}%\n\n"
        )

    send_message(detail)

    print("")
    print("✅ Telegram V1 결과 전송 완료")


if __name__ == "__main__":
    main()
