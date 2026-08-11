import os
from datetime import date

import pandas as pd


# ============================================================
# CONFIG
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


OI_HISTORY_FILE = os.path.join(
    RESULT_DIR,
    "OI_HISTORY.csv"
)


# ============================================================
# HELPERS
# ============================================================

def _safe_float(value):

    try:
        if pd.isna(value):
            return 0.0

        return float(value)

    except Exception:

        return 0.0


# ============================================================
# OI SUMMARY
# ============================================================

def get_oi_summary(df):

    if df is None:
        return None

    try:
        work = df.copy()

    except Exception:
        return None

    if "option_type" not in work.columns:
        return None

    if "openInterest" not in work.columns:
        return None

    work["option_type"] = (
        work["option_type"]
        .astype(str)
        .str.upper()
    )

    work["openInterest"] = pd.to_numeric(
        work["openInterest"],
        errors="coerce"
    ).fillna(0)

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

    total_oi = (
        call_oi
        + put_oi
    )

    call_ratio = (
        call_oi / total_oi
        if total_oi > 0
        else 0.0
    )

    put_ratio = (
        put_oi / total_oi
        if total_oi > 0
        else 0.0
    )

    return {

        "call_oi": call_oi,

        "put_oi": put_oi,

        "total_oi": total_oi,

        "call_ratio": call_ratio,

        "put_ratio": put_ratio
    }


# ============================================================
# SAVE OI SNAPSHOT
# ============================================================

def save_oi_snapshot(
    ticker,
    df,
    snapshot_date=None
):

    summary = get_oi_summary(
        df
    )

    if summary is None:
        return None

    if snapshot_date is None:

        snapshot_date = date.today()

    if hasattr(
        snapshot_date,
        "date"
    ):

        snapshot_date = (
            snapshot_date.date()
        )

    snapshot_date = str(
        snapshot_date
    )

    row = {

        "date": snapshot_date,

        "ticker": str(
            ticker
        ).upper(),

        **summary
    }

    # --------------------------------------------------------
    # 기존 HISTORY
    # --------------------------------------------------------

    if os.path.exists(
        OI_HISTORY_FILE
    ):

        try:

            history = pd.read_csv(
                OI_HISTORY_FILE
            )

        except Exception:

            history = pd.DataFrame()

    else:

        history = pd.DataFrame()

    # --------------------------------------------------------
    # 날짜/티커 컬럼 보정
    # --------------------------------------------------------

    if not history.empty:

        if "date" not in history.columns:

            history["date"] = ""

        if "ticker" not in history.columns:

            history["ticker"] = ""

        history["date"] = (
            history["date"]
            .astype(str)
        )

        history["ticker"] = (
            history["ticker"]
            .astype(str)
            .str.upper()
        )

        # 오늘 같은 종목의 기존 snapshot 제거
        history = history[
            ~(
                (history["date"] == snapshot_date)
                &
                (
                    history["ticker"]
                    == str(ticker).upper()
                )
            )
        ]

    # --------------------------------------------------------
    # APPEND
    # --------------------------------------------------------

    history = pd.concat(
        [
            history,
            pd.DataFrame([row])
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # SORT
    # --------------------------------------------------------

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    history = history.sort_values(
        [
            "ticker",
            "date"
        ]
    )

    history["date"] = (
        history["date"]
        .dt.strftime("%Y-%m-%d")
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    history.to_csv(
        OI_HISTORY_FILE,
        index=False
    )

    return summary


# ============================================================
# PREVIOUS OI
# ============================================================

def get_previous_oi(
    ticker,
    current_date=None
):

    if not os.path.exists(
        OI_HISTORY_FILE
    ):

        return None

    try:

        history = pd.read_csv(
            OI_HISTORY_FILE
        )

    except Exception:

        return None

    if history.empty:
        return None

    required = {
        "date",
        "ticker"
    }

    if not required.issubset(
        history.columns
    ):

        return None

    history["date"] = pd.to_datetime(
        history["date"],
        errors="coerce"
    )

    history["ticker"] = (
        history["ticker"]
        .astype(str)
        .str.upper()
    )

    ticker = str(
        ticker
    ).upper()

    rows = history[
        history["ticker"] == ticker
    ].copy()

    rows = rows.dropna(
        subset=["date"]
    )

    if rows.empty:
        return None

    # --------------------------------------------------------
    # 현재 날짜
    # --------------------------------------------------------

    if current_date is None:

        current_date = pd.Timestamp(
            date.today()
        )

    else:

        current_date = pd.Timestamp(
            current_date
        )

    # --------------------------------------------------------
    # 현재 snapshot보다 이전인 가장 최근 데이터
    # --------------------------------------------------------

    previous = rows[
        rows["date"] < current_date
    ].sort_values(
        "date"
    )

    if previous.empty:
        return None

    return previous.iloc[-1].to_dict()


# ============================================================
# PERCENT CHANGE
# ============================================================

def _pct_change(
    current_value,
    previous_value
):

    current_value = _safe_float(
        current_value
    )

    previous_value = _safe_float(
        previous_value
    )

    if previous_value == 0:
        return None

    return (
        (
            current_value
            / previous_value
        ) - 1
    ) * 100


# ============================================================
# CALCULATE OI CHANGE
# ============================================================

def calculate_oi_change(
    ticker,
    current_df,
    current_date=None
):

    current = get_oi_summary(
        current_df
    )

    if current is None:
        return None

    if current_date is None:

        current_date = date.today()

    if hasattr(
        current_date,
        "date"
    ):

        current_date = (
            current_date.date()
        )

    previous = get_previous_oi(
        ticker,
        current_date=current_date
    )

    # --------------------------------------------------------
    # 전일 데이터 없음
    # --------------------------------------------------------

    if previous is None:

        return {

            "available": False,

            "current": current,

            "previous": None
        }

    previous_call = _safe_float(
        previous.get(
            "call_oi",
            0
        )
    )

    previous_put = _safe_float(
        previous.get(
            "put_oi",
            0
        )
    )

    previous_total = _safe_float(
        previous.get(
            "total_oi",
            0
        )
    )

    previous_call_ratio = _safe_float(
        previous.get(
            "call_ratio",
            0
        )
    )

    previous_put_ratio = _safe_float(
        previous.get(
            "put_ratio",
            0
        )
    )

    call_change = (
        current["call_oi"]
        - previous_call
    )

    put_change = (
        current["put_oi"]
        - previous_put
    )

    total_change = (
        current["total_oi"]
        - previous_total
    )

    call_ratio_change = (
        current["call_ratio"]
        - previous_call_ratio
    )

    put_ratio_change = (
        current["put_ratio"]
        - previous_put_ratio
    )

    # --------------------------------------------------------
    # 구조 판정
    # --------------------------------------------------------

    if (
        call_ratio_change > 0.01
        and
        put_change < 0
    ):

        structure = (
            "🟢 BULLISH 강화"
        )

    elif (
        call_ratio_change < -0.01
        and
        put_change > 0
    ):

        structure = (
            "🔴 BEARISH 강화"
        )

    elif (
        call_change > 0
        and
        put_change > 0
    ):

        structure = (
            "🟡 양쪽 OI 증가"
        )

    elif (
        call_change < 0
        and
        put_change < 0
    ):

        structure = (
            "🟡 양쪽 OI 감소"
        )

    else:

        structure = (
            "🟡 OI 구조 큰 변화 없음"
        )

    return {

        "available": True,

        "current": current,

        "previous": {

            "call_oi": previous_call,

            "put_oi": previous_put,

            "total_oi": previous_total,

            "call_ratio":
                previous_call_ratio,

            "put_ratio":
                previous_put_ratio
        },

        "call_change":
            call_change,

        "put_change":
            put_change,

        "total_change":
            total_change,

        "call_change_pct":
            _pct_change(
                current["call_oi"],
                previous_call
            ),

        "put_change_pct":
            _pct_change(
                current["put_oi"],
                previous_put
            ),

        "total_change_pct":
            _pct_change(
                current["total_oi"],
                previous_total
            ),

        "call_ratio_change":
            call_ratio_change,

        "put_ratio_change":
            put_ratio_change,

        "structure":
            structure
    }


# ============================================================
# FORMAT OI CHANGE
# ============================================================

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
            "📊 <b>OI STRUCTURE</b>\n"
            "전일 데이터 없음\n"
            "→ 오늘부터 비교 데이터 축적"
        )

    current = (
        oi_change["current"]
    )

    previous = (
        oi_change["previous"]
    )

    call_pct = (
        oi_change.get(
            "call_change_pct"
        )
    )

    put_pct = (
        oi_change.get(
            "put_change_pct"
        )
    )

    total_pct = (
        oi_change.get(
            "total_change_pct"
        )
    )

    call_ratio_delta = (
        oi_change.get(
            "call_ratio_change",
            0
        ) * 100
    )

    structure = oi_change.get(
        "structure",
        "🟡 OI 구조 큰 변화 없음"
    )

    call_pct_text = (
        f"{call_pct:+.1f}%"
        if call_pct is not None
        else "N/A"
    )

    put_pct_text = (
        f"{put_pct:+.1f}%"
        if put_pct is not None
        else "N/A"
    )

    total_pct_text = (
        f"{total_pct:+.1f}%"
        if total_pct is not None
        else "N/A"
    )

    return (
        "📊 <b>OI STRUCTURE</b>\n"

        f"CALL OI "
        f"{previous['call_oi']:,.0f}"
        f" → "
        f"{current['call_oi']:,.0f}"
        f" ({call_pct_text})\n"

        f"PUT OI "
        f"{previous['put_oi']:,.0f}"
        f" → "
        f"{current['put_oi']:,.0f}"
        f" ({put_pct_text})\n"

        f"TOTAL OI "
        f"{previous['total_oi']:,.0f}"
        f" → "
        f"{current['total_oi']:,.0f}"
        f" ({total_pct_text})\n"

        f"Call 비중 변화 "
        f"{call_ratio_delta:+.1f}%p\n"

        f"→ {structure}"
    )
