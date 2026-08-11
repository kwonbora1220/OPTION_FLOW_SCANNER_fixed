import os
import glob
from datetime import date, datetime

import pandas as pd


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


OI_HISTORY_FILE = os.path.join(
    RESULT_DIR,
    "OI_HISTORY.csv"
)


def _safe_float(value):
    try:
        return float(value)
    except Exception:
        return 0.0


def get_oi_summary(df):

    work = df.copy()

    if "option_type" not in work.columns:
        return None

    if "openInterest" not in work.columns:
        return None

    call_oi = _safe_float(
        work.loc[
            work["option_type"] == "CALL",
            "openInterest"
        ].sum()
    )

    put_oi = _safe_float(
        work.loc[
            work["option_type"] == "PUT",
            "openInterest"
        ].sum()
    )

    total_oi = call_oi + put_oi

    call_ratio = (
        call_oi / total_oi
        if total_oi > 0
        else 0
    )

    return {
        "call_oi": call_oi,
        "put_oi": put_oi,
        "total_oi": total_oi,
        "call_ratio": call_ratio
    }


def save_oi_snapshot(
    ticker,
    df
):

    summary = get_oi_summary(df)

    if summary is None:
        return None

    today = date.today().isoformat()

    row = {
        "date": today,
        "ticker": ticker,
        **summary
    }

    if os.path.exists(OI_HISTORY_FILE):

        history = pd.read_csv(
            OI_HISTORY_FILE
        )

        history = history[
            ~(
                (history["date"] == today)
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
        OI_HISTORY_FILE,
        index=False
    )

    return summary


def get_previous_oi(
    ticker
):

    if not os.path.exists(
        OI_HISTORY_FILE
    ):
        return None

    history = pd.read_csv(
        OI_HISTORY_FILE
    )

    if history.empty:
        return None

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    rows = history[
        history["ticker"] == ticker
    ].sort_values("date")

    if len(rows) < 1:
        return None

    today = pd.Timestamp(
        date.today()
    )

    previous = rows[
        rows["date"] < today
    ]

    if previous.empty:
        return None

    return previous.iloc[-1].to_dict()


def calculate_oi_change(
    ticker,
    current_df
):

    current = get_oi_summary(
        current_df
    )

    if current is None:
        return None

    previous = get_previous_oi(
        ticker
    )

    if previous is None:
        return {
            "available": False,
            "current": current
        }

    def pct_change(
        current_value,
        previous_value
    ):

        if previous_value == 0:
            return None

        return (
            (
                current_value
                / previous_value
            ) - 1
        ) * 100

    return {
        "available": True,

        "current": current,

        "previous": {
            "call_oi": _safe_float(
                previous["call_oi"]
            ),
            "put_oi": _safe_float(
                previous["put_oi"]
            ),
            "total_oi": _safe_float(
                previous["total_oi"]
            ),
            "call_ratio": _safe_float(
                previous["call_ratio"]
            )
        },

        "call_change": (
            current["call_oi"]
            - _safe_float(previous["call_oi"])
        ),

        "put_change": (
            current["put_oi"]
            - _safe_float(previous["put_oi"])
        ),

        "total_change": (
            current["total_oi"]
            - _safe_float(previous["total_oi"])
        ),

        "call_change_pct": pct_change(
            current["call_oi"],
            _safe_float(previous["call_oi"])
        ),

        "put_change_pct": pct_change(
            current["put_oi"],
            _safe_float(previous["put_oi"])
        ),

        "total_change_pct": pct_change(
            current["total_oi"],
            _safe_float(previous["total_oi"])
        ),

        "call_ratio_change": (
            current["call_ratio"]
            - _safe_float(previous["call_ratio"])
        )
    }


def format_oi_change(
    ticker,
    oi_change
):

    if not oi_change:
        return ""

    if not oi_change.get(
        "available",
        False
    ):

        return (
            "📊 OI STRUCTURE\n"
            "전일 데이터 없음\n"
            "→ 오늘부터 비교 데이터 축적"
        )

    current = oi_change["current"]
    previous = oi_change["previous"]

    call_pct = oi_change[
        "call_change_pct"
    ]

    put_pct = oi_change[
        "put_change_pct"
    ]

    total_pct = oi_change[
        "total_change_pct"
    ]

    call_ratio_delta = (
        oi_change[
            "call_ratio_change"
        ] * 100
    )

    if (
        call_ratio_delta > 1
        and
        oi_change["put_change"] < 0
    ):
        structure = "🟢 Bullish 강화"

    elif (
        call_ratio_delta < -1
        and
        oi_change["put_change"] > 0
    ):
        structure = "🔴 Bearish 강화"

    else:
        structure = "🟡 OI 구조 큰 변화 없음"

    return (
        f"📊 <b>OI STRUCTURE</b>\n"
        f"CALL OI "
        f"{previous['call_oi']:,.0f}"
        f" → "
        f"{current['call_oi']:,.0f}"
        f" ({call_pct:+.1f}%)\n"
        f"PUT OI "
        f"{previous['put_oi']:,.0f}"
        f" → "
        f"{current['put_oi']:,.0f}"
        f" ({put_pct:+.1f}%)\n"
        f"TOTAL OI "
        f"{previous['total_oi']:,.0f}"
        f" → "
        f"{current['total_oi']:,.0f}"
        f" ({total_pct:+.1f}%)\n"
        f"Call 비중 변화 "
        f"{call_ratio_delta:+.1f}%p\n"
        f"→ {structure}"
    )
