import os
from datetime import date

import pandas as pd
import yfinance as yf


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


SIGNAL_FILE = os.path.join(
    RESULT_DIR,
    "SIGNAL_HISTORY.csv"
)


def _safe_float(x):

    try:
        return float(x)
    except Exception:
        return None


def record_signal(
    ticker,
    score,
    direction,
    category,
    current_price
):

    if category != "🟢 오늘 진입 후보":
        return

    today = date.today().isoformat()

    row = {
        "signal_date": today,
        "ticker": ticker,
        "score": _safe_float(score),
        "direction": direction,
        "category": category,
        "entry_price": _safe_float(
            current_price
        ),

        "price_1d": None,
        "price_3d": None,
        "price_5d": None,

        "return_1d": None,
        "return_3d": None,
        "return_5d": None,

        "win_1d": None,
        "win_3d": None,
        "win_5d": None
    }

    if os.path.exists(
        SIGNAL_FILE
    ):

        history = pd.read_csv(
            SIGNAL_FILE
        )

        # 같은 날 같은 종목 중복 제거
        history = history[
            ~(
                (history["signal_date"] == today)
                &
                (history["ticker"] == ticker)
            )
        ]

        history = pd.concat(
            [
                history,
                pd.DataFrame([row])
            ],
            ignore_index=True
        )

    else:

        history = pd.DataFrame([row])

    history.to_csv(
        SIGNAL_FILE,
        index=False
    )


def update_signal_results():

    if not os.path.exists(
        SIGNAL_FILE
    ):
        return

    history = pd.read_csv(
        SIGNAL_FILE
    )

    if history.empty:
        return

    history["signal_date"] = pd.to_datetime(
        history["signal_date"],
        errors="coerce"
    )

    today = pd.Timestamp(
        date.today()
    )

    for idx, row in history.iterrows():

        ticker = row["ticker"]
        signal_date = row["signal_date"]

        if pd.isna(signal_date):
            continue

        entry_price = _safe_float(
            row["entry_price"]
        )

        if (
            entry_price is None
            or entry_price <= 0
        ):
            continue

        try:

            start = (
                signal_date
                - pd.Timedelta(days=2)
            )

            end = today + pd.Timedelta(days=1)

            hist = yf.Ticker(
                ticker
            ).history(
                start=start.strftime(
                    "%Y-%m-%d"
                ),
                end=end.strftime(
                    "%Y-%m-%d"
                ),
                auto_adjust=False
            )

            if hist.empty:
                continue

            closes = hist["Close"].dropna()

            future = closes[
                closes.index > signal_date
            ]

            targets = {
                1: "price_1d",
                3: "price_3d",
                5: "price_5d"
            }

            returns = {
                1: "return_1d",
                3: "return_3d",
                5: "return_5d"
            }

            wins = {
                1: "win_1d",
                3: "win_3d",
                5: "win_5d"
            }

            for days in [1, 3, 5]:

                if len(future) < days:
                    continue

                price = float(
                    future.iloc[
                        days - 1
                    ]
                )

                ret = (
                    price
                    / entry_price
                    - 1
                ) * 100

                history.at[
                    idx,
                    targets[days]
                ] = price

                history.at[
                    idx,
                    returns[days]
                ] = ret

                history.at[
                    idx,
                    wins[days]
                ] = (
                    1
                    if ret > 0
                    else 0
                )

        except Exception as e:

            print(
                f"⚠️ {ticker} "
                f"signal 업데이트 실패: {e}"
            )

    history.to_csv(
        SIGNAL_FILE,
        index=False
    )


def get_signal_stats(
    ticker,
    score,
    direction
):

    if not os.path.exists(
        SIGNAL_FILE
    ):
        return None

    history = pd.read_csv(
        SIGNAL_FILE
    )

    if history.empty:
        return None

    history["score"] = pd.to_numeric(
        history["score"],
        errors="coerce"
    )

    score = float(score)

    # 현재 신호와 유사한 과거 신호
    similar = history[
        (
            history["ticker"] == ticker
        )
        &
        (
            history["direction"]
            == direction
        )
        &
        (
            history["score"]
            >= score - 5
        )
        &
        (
            history["score"]
            <= score + 5
        )
    ]

    # 미래 결과가 실제 계산된 것만 사용
    stats = {}

    for days in [1, 3, 5]:

        col = f"win_{days}d"

        valid = similar[
            similar[col].notna()
        ]

        if len(valid) < 5:

            stats[days] = {
                "count": len(valid),
                "win_rate": None
            }

        else:

            stats[days] = {
                "count": len(valid),
                "win_rate": (
                    valid[col]
                    .astype(float)
                    .mean()
                    * 100
                )
            }

    total_samples = max(
        [
            stats[d]["count"]
            for d in stats
        ]
    )

    if total_samples >= 30:
        confidence = "HIGH"
    elif total_samples >= 10:
        confidence = "MEDIUM"
    elif total_samples >= 5:
        confidence = "LOW"
    else:
        confidence = "INSUFFICIENT"

    return {
        "samples": total_samples,
        "stats": stats,
        "confidence": confidence
    }


def format_signal_stats(
    stats
):

    if not stats:
        return (
            "🧠 <b>SIGNAL BACKTEST</b>\n"
            "아직 과거 유사 신호 데이터가 부족합니다."
        )

    lines = [
        "🧠 <b>SIGNAL BACKTEST</b>"
    ]

    for days in [1, 3, 5]:

        item = stats["stats"][days]

        if item["win_rate"] is None:

            lines.append(
                f"{days}D 승률: 데이터 부족 "
                f"({item['count']}회)"
            )

        else:

            lines.append(
                f"{days}D 승률: "
                f"{item['win_rate']:.1f}% "
                f"({item['count']}회)"
            )

    lines.append(
        f"유사 신호 최대 표본: "
        f"{stats['samples']}회"
    )

    lines.append(
        f"신뢰도: {stats['confidence']}"
    )

    lines.append(
        "⚠️ 승률은 과거 유사 신호의 "
        "실제 가격 결과 기반"
    )

    return "\n".join(lines)
