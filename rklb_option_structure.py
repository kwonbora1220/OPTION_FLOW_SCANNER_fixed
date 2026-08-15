import os
import time
import math
import traceback
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, date


# ============================================================
# RKLB OPTION STRUCTURE SCANNER
# ============================================================
#
# 독립 실행형
#
# 실행:
#     python rklb_option_structure.py
#
# 환경변수:
#     SYMBOL=RKLB
#     MIN_STRIKE=80
#     MAX_STRIKE=100
#     MAX_DTE=180
#
# Telegram:
#     TELEGRAM_BOT_TOKEN
#     TELEGRAM_CHAT_ID
#
# 출력:
#     rklb_option_structure/
#         report.md
#         summary.csv
#         contracts.csv
#         strike_structure.csv
#         top_contracts.csv
#         oi_history/
#
# ============================================================


# ============================================================
# CONFIG
# ============================================================

SYMBOL = os.environ.get(
    "SYMBOL",
    "RKLB"
).upper().strip()


MIN_STRIKE = float(
    os.environ.get(
        "MIN_STRIKE",
        "80"
    )
)


MAX_STRIKE = float(
    os.environ.get(
        "MAX_STRIKE",
        "100"
    )
)


MAX_DTE = int(
    os.environ.get(
        "MAX_DTE",
        "180"
    )
)


BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)


CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)


CONTRACT_MULTIPLIER = 100


# BAR
OI_BAR_WIDTH = 10
VOLUME_BAR_WIDTH = 10


# 최대 BAR 대상 strike 수
OI_BAR_MAX_STRIKES = 30


# strike별 OI 구조는 전체 범위 사용
OI_BAR_RANGE_PCT = 100


# Telegram 한 메시지 최대 길이
TELEGRAM_MAX_LENGTH = 3800


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "rklb_option_structure"
)


OI_HISTORY_DIR = os.path.join(
    OUTPUT_DIR,
    "oi_history"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


os.makedirs(
    OI_HISTORY_DIR,
    exist_ok=True
)


# ============================================================
# UTILS
# ============================================================

def safe_float(value, default=0.0):

    try:

        value = float(value)

        if math.isnan(value):
            return default

        if math.isinf(value):
            return default

        return value

    except Exception:

        return default


def safe_int(value, default=0):

    try:

        return int(float(value))

    except Exception:

        return default


def format_money(value):

    value = safe_float(value)

    sign = ""

    if value < 0:

        sign = "-"

        value = abs(value)

    if value >= 1_000_000_000:

        return (
            f"{sign}"
            f"${value / 1_000_000_000:.2f}B"
        )

    if value >= 1_000_000:

        return (
            f"{sign}"
            f"${value / 1_000_000:.2f}M"
        )

    if value >= 1_000:

        return (
            f"{sign}"
            f"${value / 1_000:.1f}K"
        )

    return (
        f"{sign}"
        f"${value:.0f}"
    )


def format_number(value):

    return f"{safe_int(value):,}"


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

    if not BOT_TOKEN:

        print(
            "⚠️ TELEGRAM_BOT_TOKEN 없음"
        )

        return False


    if not CHAT_ID:

        print(
            "⚠️ TELEGRAM_CHAT_ID 없음"
        )

        return False


    if not text:

        return False


    chunks = []


    if len(text) <= TELEGRAM_MAX_LENGTH:

        chunks = [text]

    else:

        current = ""

        for line in text.splitlines(
            True
        ):

            if (
                len(current)
                + len(line)
                <= TELEGRAM_MAX_LENGTH
            ):

                current += line

            else:

                if current:

                    chunks.append(
                        current
                    )

                while (
                    len(line)
                    > TELEGRAM_MAX_LENGTH
                ):

                    chunks.append(
                        line[
                            :TELEGRAM_MAX_LENGTH
                        ]
                    )

                    line = line[
                        TELEGRAM_MAX_LENGTH:
                    ]

                current = line


        if current:

            chunks.append(
                current
            )


    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )


    success = True


    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        if len(chunks) > 1:

            chunk = (
                f"📨 PART "
                f"{index}/{len(chunks)}\n\n"
                + chunk
            )


        payload = {

            "chat_id": CHAT_ID,

            "text": chunk
        }


        try:

            response = requests.post(
                url,
                data=payload,
                timeout=30
            )


            print(
                f"📨 Telegram "
                f"{index}/{len(chunks)}: "
                f"{response.status_code}"
            )


            if not response.ok:

                print(
                    response.text[:1000]
                )

                success = False


            time.sleep(0.3)


        except Exception as e:

            print(
                f"❌ Telegram 오류: {e}"
            )

            success = False


    return success


# ============================================================
# MARKET PRICE
# ============================================================

def get_market_price(ticker):

    print("")
    print(
        f"💰 {ticker} 현재가 조회"
    )


    for attempt in range(1, 4):

        try:

            t = yf.Ticker(ticker)


            hist = t.history(
                period="5d",
                interval="1d",
                auto_adjust=False
            )


            if hist.empty:

                raise ValueError(
                    "가격 데이터 없음"
                )


            close = (
                hist["Close"]
                .dropna()
            )


            if close.empty:

                raise ValueError(
                    "종가 데이터 없음"
                )


            regular_close = float(
                close.iloc[-1]
            )


            print(
                f"💰 정규장 종가: "
                f"${regular_close:.2f}"
            )


            # 시간외 가격은 참고용
            after_hours = None


            try:

                intraday = t.history(
                    period="1d",
                    interval="1m",
                    prepost=True,
                    auto_adjust=False
                )


                if not intraday.empty:

                    latest = (
                        intraday["Close"]
                        .dropna()
                    )


                    if not latest.empty:

                        latest_price = float(
                            latest.iloc[-1]
                        )


                        if (
                            abs(
                                latest_price
                                - regular_close
                            )
                            > 0.000001
                        ):

                            after_hours = (
                                latest_price
                            )


            except Exception as e:

                print(
                    f"⚠️ 시간외 가격 조회 실패: {e}"
                )


            ah_pct = None


            if (
                after_hours is not None
                and regular_close != 0
            ):

                ah_pct = (
                    after_hours
                    / regular_close
                    - 1
                ) * 100


            if after_hours is not None:

                print(
                    f"🌙 시간외 현재가: "
                    f"${after_hours:.2f}"
                    + (
                        f" ({ah_pct:+.2f}%)"
                        if ah_pct is not None
                        else ""
                    )
                )

            else:

                print(
                    "🌙 시간외 현재가: N/A"
                )


            return {

                "regular_close":
                regular_close,

                "after_hours":
                after_hours,

                "after_hours_pct":
                ah_pct
            }


        except Exception as e:

            print(
                f"⚠️ 가격 조회 실패 "
                f"({attempt}/3): {e}"
            )


            if attempt < 3:

                time.sleep(2)


    return None


# ============================================================
# EXPIRATIONS
# ============================================================

def get_valid_expirations(ticker):

    t = yf.Ticker(ticker)


    expirations = list(
        t.options
    )


    if not expirations:

        raise ValueError(
            "옵션 만기 데이터가 없습니다."
        )


    today = date.today()


    valid = []


    for expiration in expirations:

        try:

            exp_date = datetime.strptime(
                expiration,
                "%Y-%m-%d"
            ).date()


            dte = (
                exp_date
                - today
            ).days


            if (
                0 <= dte <= MAX_DTE
            ):

                valid.append(
                    (
                        expiration,
                        dte
                    )
                )


        except Exception:

            continue


    valid.sort(
        key=lambda x: x[1]
    )


    print(
        f"📅 전체 만기: "
        f"{len(expirations):,}"
    )


    print(
        f"📅 0~{MAX_DTE} DTE 만기: "
        f"{len(valid):,}"
    )


    if not valid:

        raise ValueError(
            f"0~{MAX_DTE} DTE 만기가 없습니다."
        )


    return valid


# ============================================================
# OPTION COLLECTION
# ============================================================

def collect_options(ticker):

    t = yf.Ticker(ticker)


    expirations = (
        get_valid_expirations(
            ticker
        )
    )


    rows = []


    for expiration, dte in expirations:

        print(
            f"   수집: "
            f"{expiration} "
            f"| DTE {dte}"
        )


        try:

            chain = t.option_chain(
                expiration
            )


            calls = chain.calls.copy()

            puts = chain.puts.copy()


            calls["option_type"] = (
                "CALL"
            )

            puts["option_type"] = (
                "PUT"
            )


            calls["expiration"] = (
                expiration
            )

            puts["expiration"] = (
                expiration
            )


            calls["DTE"] = dte

            puts["DTE"] = dte


            rows.append(
                calls
            )

            rows.append(
                puts
            )


        except Exception as e:

            print(
                f"⚠️ {expiration} "
                f"수집 실패: {e}"
            )


    if not rows:

        raise ValueError(
            "옵션 데이터를 수집하지 못했습니다."
        )


    df = pd.concat(
        rows,
        ignore_index=True
    )


    print(
        f"📊 전체 옵션 계약: "
        f"{len(df):,}"
    )


    return df


# ============================================================
# NORMALIZE
# ============================================================

def normalize_options(df):

    df = df.copy()


    numeric_columns = [

        "strike",

        "lastPrice",

        "bid",

        "ask",

        "volume",

        "openInterest",

        "impliedVolatility",

        "DTE"
    ]


    for column in numeric_columns:

        if column not in df.columns:

            df[column] = 0


        df[column] = pd.to_numeric(
            df[column],
            errors="coerce"
        ).fillna(0)


    df["strike"] = (
        df["strike"]
        .clip(lower=0)
    )


    df["volume"] = (
        df["volume"]
        .clip(lower=0)
    )


    df["openInterest"] = (
        df["openInterest"]
        .clip(lower=0)
    )


    df["DTE"] = (
        df["DTE"]
        .clip(lower=0)
    )


    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )


    df["expiration"] = (
        df["expiration"]
        .astype(str)
        .str.strip()
    )


    return df


# ============================================================
# STRIKE FILTER
# ============================================================

def filter_strike_range(df):

    filtered = df[
        (
            df["strike"]
            >= MIN_STRIKE
        )
        &
        (
            df["strike"]
            <= MAX_STRIKE
        )
        &
        (
            df["DTE"]
            >= 0
        )
        &
        (
            df["DTE"]
            <= MAX_DTE
        )
    ].copy()


    filtered = filtered[
        filtered["option_type"].isin(
            [
                "CALL",
                "PUT"
            ]
        )
    ]


    print("")
    print(
        "🎯 STRIKE FILTER"
    )


    print(
        f"   Strike: "
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )


    print(
        f"   DTE: "
        f"0 ~ {MAX_DTE}"
    )


    print(
        f"   필터 후 계약: "
        f"{len(filtered):,}"
    )


    return filtered


# ============================================================
# OI SNAPSHOT
# ============================================================

def get_snapshot_path(
    ticker,
    snapshot_date=None
):

    if snapshot_date is None:

        snapshot_date = date.today()


    return os.path.join(

        OI_HISTORY_DIR,

        (
            f"{ticker}_OI_"
            f"{snapshot_date.strftime('%Y%m%d')}"
            f".csv"
        )
    )


def build_snapshot(df):

    snapshot = df[
        [
            "expiration",
            "DTE",
            "strike",
            "option_type",
            "openInterest"
        ]
    ].copy()


    snapshot["strike"] = pd.to_numeric(
        snapshot["strike"],
        errors="coerce"
    )


    snapshot["DTE"] = pd.to_numeric(
        snapshot["DTE"],
        errors="coerce"
    )


    snapshot["openInterest"] = pd.to_numeric(
        snapshot["openInterest"],
        errors="coerce"
    ).fillna(0)


    snapshot = snapshot.dropna(
        subset=[
            "expiration",
            "strike"
        ]
    )


    snapshot = snapshot[
        snapshot["option_type"].isin(
            [
                "CALL",
                "PUT"
            ]
        )
    ]


    keys = [
        "expiration",
        "strike",
        "option_type"
    ]


    snapshot = (
        snapshot
        .groupby(
            keys,
            as_index=False
        )
        .agg(
            DTE=("DTE", "max"),
            openInterest=(
                "openInterest",
                "max"
            )
        )
    )


    return snapshot


def find_previous_snapshot(ticker):

    pattern = os.path.join(

        OI_HISTORY_DIR,

        f"{ticker}_OI_*.csv"
    )


    today = date.today()


    candidates = []


    for path in os.listdir(
        OI_HISTORY_DIR
    ):

        if not path.startswith(
            f"{ticker}_OI_"
        ):

            continue


        if not path.endswith(
            ".csv"
        ):

            continue


        date_text = (
            path
            .replace(
                f"{ticker}_OI_",
                ""
            )
            .replace(
                ".csv",
                ""
            )
        )


        try:

            snapshot_date = (
                datetime.strptime(
                    date_text,
                    "%Y%m%d"
                ).date()
            )

        except Exception:

            continue


        if snapshot_date < today:

            candidates.append(
                (
                    snapshot_date,
                    os.path.join(
                        OI_HISTORY_DIR,
                        path
                    )
                )
            )


    if not candidates:

        return None, None


    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )


    return candidates[0]


def calculate_oi_delta(
    df,
    ticker
):

    current = df.copy()


    current_snapshot = (
        build_snapshot(
            current
        )
    )


    previous_date, previous_path = (
        find_previous_snapshot(
            ticker
        )
    )


    current["previous_openInterest"] = (
        0.0
    )

    current["oi_delta"] = 0.0

    current["oi_delta_available"] = (
        False
    )


    if previous_path is None:

        print(
            "⚠️ 이전 OI snapshot 없음"
        )

        return (
            current,
            None
        )


    print(
        f"📂 이전 OI snapshot: "
        f"{previous_path}"
    )


    try:

        previous = pd.read_csv(
            previous_path,
            encoding="utf-8-sig"
        )


        previous = build_snapshot(
            previous
        )


        keys = [
            "expiration",
            "strike",
            "option_type"
        ]


        previous = previous[
            keys
            + [
                "openInterest"
            ]
        ].rename(
            columns={
                "openInterest":
                "previous_openInterest"
            }
        )


        merged = pd.merge(

            current_snapshot[
                keys
                + [
                    "openInterest"
                ]
            ],

            previous,

            on=keys,

            how="outer"
        )


        merged[
            "openInterest"
        ] = pd.to_numeric(
            merged["openInterest"],
            errors="coerce"
        ).fillna(0)


        merged[
            "previous_openInterest"
        ] = pd.to_numeric(
            merged[
                "previous_openInterest"
            ],
            errors="coerce"
        ).fillna(0)


        merged["oi_delta"] = (

            merged[
                "openInterest"
            ]

            -

            merged[
                "previous_openInterest"
            ]
        )


        current["strike"] = pd.to_numeric(
            current["strike"],
            errors="coerce"
        )


        current = current.merge(

            merged[
                keys
                + [
                    "previous_openInterest",
                    "oi_delta"
                ]
            ],

            on=keys,

            how="left",

            suffixes=(
                "",
                "_merged"
            )
        )


        if (
            "previous_openInterest_merged"
            in current.columns
        ):

            current[
                "previous_openInterest"
            ] = current[
                "previous_openInterest_merged"
            ]


        if (
            "oi_delta_merged"
            in current.columns
        ):

            current[
                "oi_delta"
            ] = current[
                "oi_delta_merged"
            ]


        current[
            "previous_openInterest"
        ] = pd.to_numeric(
            current[
                "previous_openInterest"
            ],
            errors="coerce"
        ).fillna(0)


        current["oi_delta"] = pd.to_numeric(
            current["oi_delta"],
            errors="coerce"
        ).fillna(0)


        current[
            "oi_delta_available"
        ] = True


        print(
            f"📈 OI 비교 기준: "
            f"{previous_date}"
        )


        print(
            f"📈 OI 증가 계약: "
            f"{int((current['oi_delta'] > 0).sum()):,}"
        )


        print(
            f"📉 OI 감소 계약: "
            f"{int((current['oi_delta'] < 0).sum()):,}"
        )


        print(
            f"📊 OI Δ: "
            f"{current['oi_delta'].sum():+,.0f}"
        )


        return (
            current,
            previous_date
        )


    except Exception as e:

        print(
            f"⚠️ OI 비교 실패: {e}"
        )


        return (
            df,
            None
        )


def save_oi_snapshot(
    df,
    ticker
):

    snapshot = build_snapshot(
        df
    )


    path = get_snapshot_path(
        ticker
    )


    snapshot.to_csv(
        path,
        index=False,
        encoding="utf-8-sig"
    )


    print(
        f"💾 OI snapshot 저장: "
        f"{path}"
    )


    print(
        f"   계약 수: "
        f"{len(snapshot):,}"
    )


    print(
        f"   총 OI: "
        f"{snapshot['openInterest'].sum():,.0f}"
    )


    return path


# ============================================================
# STRIKE STRUCTURE
# ============================================================

def build_strike_structure(df):

    calls = df[
        df["option_type"]
        == "CALL"
    ]


    puts = df[
        df["option_type"]
        == "PUT"
    ]


    call_oi = (
        calls
        .groupby("strike")[
            "openInterest"
        ]
        .sum()
        .rename("call_oi")
    )


    put_oi = (
        puts
        .groupby("strike")[
            "openInterest"
        ]
        .sum()
        .rename("put_oi")
    )


    call_volume = (
        calls
        .groupby("strike")[
            "volume"
        ]
        .sum()
        .rename("call_volume")
    )


    put_volume = (
        puts
        .groupby("strike")[
            "volume"
        ]
        .sum()
        .rename("put_volume")
    )


    call_oi_delta = (
        calls
        .groupby("strike")[
            "oi_delta"
        ]
        .sum()
        .rename("call_oi_delta")
        if "oi_delta" in calls.columns
        else pd.Series(
            dtype=float,
            name="call_oi_delta"
        )
    )


    put_oi_delta = (
        puts
        .groupby("strike")[
            "oi_delta"
        ]
        .sum()
        .rename("put_oi_delta")
        if "oi_delta" in puts.columns
        else pd.Series(
            dtype=float,
            name="put_oi_delta"
        )
    )


    structure = pd.concat(
        [
            call_oi,
            put_oi,
            call_volume,
            put_volume,
            call_oi_delta,
            put_oi_delta
        ],
        axis=1
    ).fillna(0)


    structure = (
        structure
        .reset_index()
        .rename(
            columns={
                "strike":
                "strike"
            }
        )
    )


    structure["total_oi"] = (
        structure["call_oi"]
        + structure["put_oi"]
    )


    structure["total_volume"] = (
        structure["call_volume"]
        + structure["put_volume"]
    )


    structure["oi_difference"] = (
        structure["call_oi"]
        - structure["put_oi"]
    )


    structure["volume_difference"] = (
        structure["call_volume"]
        - structure["put_volume"]
    )


    structure["call_oi_ratio"] = 0.5


    valid_oi = (
        structure["total_oi"]
        > 0
    )


    structure.loc[
        valid_oi,
        "call_oi_ratio"
    ] = (
        structure.loc[
            valid_oi,
            "call_oi"
        ]
        /
        structure.loc[
            valid_oi,
            "total_oi"
        ]
    )


    structure["call_volume_ratio"] = 0.5


    valid_volume = (
        structure["total_volume"]
        > 0
    )


    structure.loc[
        valid_volume,
        "call_volume_ratio"
    ] = (
        structure.loc[
            valid_volume,
            "call_volume"
        ]
        /
        structure.loc[
            valid_volume,
            "total_volume"
        ]
    )


    structure = structure.sort_values(
        "strike"
    )


    return structure


# ============================================================
# BAR
# ============================================================

def make_bar(
    value,
    maximum,
    width,
    positive_symbol,
    negative_symbol=""
):

    value = max(
        0,
        safe_float(value)
    )


    maximum = max(
        0,
        safe_float(maximum)
    )


    if maximum <= 0:

        return ""


    length = int(
        round(
            value
            / maximum
            * width
        )
    )


    length = max(
        0,
        min(
            width,
            length
        )
    )


    if length <= 0:

        return ""


    return (
        positive_symbol
        * length
    )


def make_call_put_bar(
    call_value,
    put_value,
    width=10
):

    call_value = max(
        0,
        safe_float(call_value)
    )


    put_value = max(
        0,
        safe_float(put_value)
    )


    maximum = max(
        call_value,
        put_value,
        1
    )


    call_len = int(
        round(
            call_value
            / maximum
            * width
        )
    )


    put_len = int(
        round(
            put_value
            / maximum
            * width
        )
    )


    call_len = max(
        0,
        min(width, call_len)
    )


    put_len = max(
        0,
        min(width, put_len)
    )


    return (
        "🟩"
        * call_len
        +
        "🟥"
        * put_len
    )


def make_structure_bar(
    call_value,
    put_value,
    width=10
):

    call_value = max(
        0,
        safe_float(call_value)
    )


    put_value = max(
        0,
        safe_float(put_value)
    )


    maximum = max(
        call_value,
        put_value,
        1
    )


    call_len = int(
        round(
            call_value
            / maximum
            * width
        )
    )


    put_len = int(
        round(
            put_value
            / maximum
            * width
        )
    )


    call_len = max(
        0,
        min(width, call_len)
    )


    put_len = max(
        0,
        min(width, put_len)
    )


    return (
        "🟩"
        * call_len
        +
        "🟥"
        * put_len
    )


# ============================================================
# OI BAR REPORT
# ============================================================

def build_oi_bar_report(
    structure,
    current_price
):

    lines = []


    lines.append(
        "📊 CALL / PUT BAR STRUCTURE"
    )


    lines.append(
        "🟩 CALL   🟥 PUT"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append("")


    lines.append(
        "🟢 OI STRUCTURE"
    )


    lines.append("")


    if structure.empty:

        lines.append(
            "데이터 없음"
        )

        return "\n".join(lines)


    display = structure.copy()


    display["distance_pct"] = (
        abs(
            display["strike"]
            - current_price
        )
        / current_price
        * 100
    )


    display = display[
        display["distance_pct"]
        <= OI_BAR_RANGE_PCT
    ]


    if display.empty:

        lines.append(
            "해당 strike 데이터 없음"
        )

        return "\n".join(lines)


    display = display.sort_values(
        "strike"
    )


    # OI가 실제로 존재하는 strike만
    display = display[
        display["total_oi"] > 0
    ]


    # 최대 30개
    if len(display) > OI_BAR_MAX_STRIKES:

        # 현재가와 가까운 순
        display = (
            display
            .assign(
                _distance=abs(
                    display["strike"]
                    - current_price
                )
            )
            .sort_values(
                "_distance"
            )
            .head(
                OI_BAR_MAX_STRIKES
            )
            .sort_values(
                "strike"
            )
        )


    if display.empty:

        lines.append(
            "OI가 존재하는 strike 없음"
        )

        return "\n".join(lines)


    for _, row in display.iterrows():

        strike = safe_float(
            row["strike"]
        )


        call_oi = safe_float(
            row["call_oi"]
        )


        put_oi = safe_float(
            row["put_oi"]
        )


        bar = make_structure_bar(
            call_oi,
            put_oi,
            OI_BAR_WIDTH
        )


        lines.append(
            f"🎯 ${strike:g}  "
            f"{bar} "
            f"C {call_oi:,.0f} "
            f"/ P {put_oi:,.0f}"
        )


    return "\n".join(lines)


# ============================================================
# VOLUME BAR REPORT
# ============================================================

def build_volume_bar_report(
    structure,
    current_price
):

    lines = []


    lines.append(
        "📊 CALL / PUT VOLUME BAR"
    )


    lines.append(
        "🟩 CALL   🟥 PUT"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append("")


    lines.append(
        "🔥 VOLUME STRUCTURE"
    )


    lines.append("")


    display = structure.copy()


    display["distance_pct"] = (
        abs(
            display["strike"]
            - current_price
        )
        / current_price
        * 100
    )


    display = display[
        display["distance_pct"]
        <= OI_BAR_RANGE_PCT
    ]


    display = display[
        display["total_volume"] > 0
    ]


    if display.empty:

        lines.append(
            "거래량 데이터 없음"
        )

        return "\n".join(lines)


    if len(display) > OI_BAR_MAX_STRIKES:

        display = (
            display
            .assign(
                _distance=abs(
                    display["strike"]
                    - current_price
                )
            )
            .sort_values(
                "_distance"
            )
            .head(
                OI_BAR_MAX_STRIKES
            )
            .sort_values(
                "strike"
            )
        )


    for _, row in display.iterrows():

        strike = safe_float(
            row["strike"]
        )


        call_volume = safe_float(
            row["call_volume"]
        )


        put_volume = safe_float(
            row["put_volume"]
        )


        bar = make_structure_bar(
            call_volume,
            put_volume,
            VOLUME_BAR_WIDTH
        )


        lines.append(
            f"🎯 ${strike:g}  "
            f"{bar} "
            f"C {call_volume:,.0f} "
            f"/ P {put_volume:,.0f}"
        )


    return "\n".join(lines)


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(df):

    active = df[
        (
            df["volume"] > 0
        )
        |
        (
            df["openInterest"] > 0
        )
    ].copy()


    if active.empty:

        return active


    active["premium_proxy"] = (
        active["lastPrice"]
        * active["volume"]
        * CONTRACT_MULTIPLIER
    )


    active["oi_value_proxy"] = (
        active["openInterest"]
        * active["lastPrice"]
        * CONTRACT_MULTIPLIER
    )


    active["volume_oi_ratio"] = 0.0


    valid = (
        active["openInterest"]
        > 0
    )


    active.loc[
        valid,
        "volume_oi_ratio"
    ] = (
        active.loc[
            valid,
            "volume"
        ]
        /
        active.loc[
            valid,
            "openInterest"
        ]
    )


    active = active.sort_values(
        [
            "premium_proxy",
            "volume",
            "openInterest"
        ],
        ascending=False
    )


    return active.head(20)


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    ticker,
    current_price,
    df,
    structure,
    previous_oi_date
):

    calls = df[
        df["option_type"]
        == "CALL"
    ]


    puts = df[
        df["option_type"]
        == "PUT"
    ]


    call_oi = safe_float(
        calls["openInterest"].sum()
    )


    put_oi = safe_float(
        puts["openInterest"].sum()
    )


    call_volume = safe_float(
        calls["volume"].sum()
    )


    put_volume = safe_float(
        puts["volume"].sum()
    )


    total_oi = (
        call_oi
        + put_oi
    )


    total_volume = (
        call_volume
        + put_volume
    )


    call_oi_ratio = (

        call_oi
        / total_oi

        if total_oi > 0

        else 0.5
    )


    call_volume_ratio = (

        call_volume
        / total_volume

        if total_volume > 0

        else 0.5
    )


    oi_difference = (
        call_oi
        - put_oi
    )


    volume_difference = (
        call_volume
        - put_volume
    )


    total_oi_delta = 0


    call_oi_delta = 0


    put_oi_delta = 0


    if "oi_delta" in df.columns:

        call_oi_delta = safe_float(
            calls["oi_delta"].sum()
        )


        put_oi_delta = safe_float(
            puts["oi_delta"].sum()
        )


        total_oi_delta = (
            call_oi_delta
            + put_oi_delta
        )


    max_call_oi = 0

    max_put_oi = 0

    max_call_oi_strike = None

    max_put_oi_strike = None


    if not structure.empty:

        call_idx = (
            structure["call_oi"]
            .idxmax()
        )


        put_idx = (
            structure["put_oi"]
            .idxmax()
        )


        max_call_oi = safe_float(
            structure.loc[
                call_idx,
                "call_oi"
            ]
        )


        max_put_oi = safe_float(
            structure.loc[
                put_idx,
                "put_oi"
            ]
        )


        max_call_oi_strike = (
            safe_float(
                structure.loc[
                    call_idx,
                    "strike"
                ]
            )
        )


        max_put_oi_strike = (
            safe_float(
                structure.loc[
                    put_idx,
                    "strike"
                ]
            )
        )


    return {

        "ticker":
        ticker,

        "current_price":
        current_price,

        "min_strike":
        MIN_STRIKE,

        "max_strike":
        MAX_STRIKE,

        "max_dte":
        MAX_DTE,

        "contracts":
        len(df),

        "strikes":
        len(structure),

        "call_oi":
        call_oi,

        "put_oi":
        put_oi,

        "total_oi":
        total_oi,

        "call_volume":
        call_volume,

        "put_volume":
        put_volume,

        "total_volume":
        total_volume,

        "call_oi_ratio":
        call_oi_ratio,

        "call_volume_ratio":
        call_volume_ratio,

        "oi_difference":
        oi_difference,

        "volume_difference":
        volume_difference,

        "call_oi_delta":
        call_oi_delta,

        "put_oi_delta":
        put_oi_delta,

        "total_oi_delta":
        total_oi_delta,

        "max_call_oi":
        max_call_oi,

        "max_call_oi_strike":
        max_call_oi_strike,

        "max_put_oi":
        max_put_oi,

        "max_put_oi_strike":
        max_put_oi_strike,

        "previous_oi_date":
        (
            previous_oi_date.isoformat()
            if previous_oi_date
            else ""
        )
    }


# ============================================================
# REPORT
# ============================================================

def build_report(
    summary,
    structure,
    top_contracts
):

    ticker = summary["ticker"]

    current_price = (
        summary["current_price"]
    )


    lines = []


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        f"🔥 {ticker} OPTION STRUCTURE"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append("")


    lines.append(
        f"💰 현재가: "
        f"${current_price:.2f}"
    )


    lines.append(
        f"🎯 Strike Range: "
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )


    lines.append(
        f"📅 DTE: "
        f"0 ~ {MAX_DTE}"
    )


    lines.append(
        f"📊 옵션 계약: "
        f"{summary['contracts']:,}"
    )


    lines.append(
        f"🎯 Strike 수: "
        f"{summary['strikes']:,}"
    )


    if summary[
        "previous_oi_date"
    ]:

        lines.append(
            f"📈 OI 비교: "
            f"{summary['previous_oi_date']}"
        )

    else:

        lines.append(
            "📈 OI 비교: "
            "이전 snapshot 없음"
        )


    lines.append("")


    # ========================================================
    # OI SUMMARY
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "🟢 OI SUMMARY"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        f"CALL OI: "
        f"{summary['call_oi']:,.0f}"
    )


    lines.append(
        f"PUT OI: "
        f"{summary['put_oi']:,.0f}"
    )


    lines.append(
        f"TOTAL OI: "
        f"{summary['total_oi']:,.0f}"
    )


    lines.append(
        f"CALL OI Ratio: "
        f"{summary['call_oi_ratio'] * 100:.1f}%"
    )


    lines.append(
        f"CALL OI Δ: "
        f"{summary['call_oi_delta']:+,.0f}"
    )


    lines.append(
        f"PUT OI Δ: "
        f"{summary['put_oi_delta']:+,.0f}"
    )


    lines.append(
        f"TOTAL OI Δ: "
        f"{summary['total_oi_delta']:+,.0f}"
    )


    lines.append("")


    # ========================================================
    # VOLUME SUMMARY
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "🔥 VOLUME SUMMARY"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        f"CALL Volume: "
        f"{summary['call_volume']:,.0f}"
    )


    lines.append(
        f"PUT Volume: "
        f"{summary['put_volume']:,.0f}"
    )


    lines.append(
        f"TOTAL Volume: "
        f"{summary['total_volume']:,.0f}"
    )


    lines.append(
        f"CALL Volume Ratio: "
        f"{summary['call_volume_ratio'] * 100:.1f}%"
    )


    lines.append("")


    # ========================================================
    # BAR STRUCTURE
    # ========================================================

    bar_report = build_oi_bar_report(
        structure,
        current_price
    )


    lines.append(
        bar_report
    )


    lines.append("")


    volume_report = (
        build_volume_bar_report(
            structure,
            current_price
        )
    )


    lines.append(
        volume_report
    )


    lines.append("")


    # ========================================================
    # MAX OI
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "🏆 MAX OI"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    if (
        summary["max_call_oi_strike"]
        is not None
    ):

        lines.append(
            f"🟩 CALL "
            f"${summary['max_call_oi_strike']:g}"
            f" | OI "
            f"{summary['max_call_oi']:,.0f}"
        )


    if (
        summary["max_put_oi_strike"]
        is not None
    ):

        lines.append(
            f"🟥 PUT "
            f"${summary['max_put_oi_strike']:g}"
            f" | OI "
            f"{summary['max_put_oi']:,.0f}"
        )


    lines.append("")


    # ========================================================
    # TOP CONTRACTS
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "🔥 TOP CONTRACTS"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    if top_contracts.empty:

        lines.append(
            "없음"
        )

    else:

        for _, row in (
            top_contracts.head(10)
            .iterrows()
        ):

            option_type = (
                row["option_type"]
            )


            icon = (
                "🟩"
                if option_type == "CALL"
                else "🟥"
            )


            premium = safe_float(
                row.get(
                    "premium_proxy",
                    0
                )
            )


            lines.append(
                f"{icon} "
                f"${safe_float(row['strike']):g}"
                f" | "
                f"{row['expiration']}"
                f" | "
                f"DTE {safe_int(row['DTE'])}"
                f" | Vol "
                f"{safe_int(row['volume']):,}"
                f" | OI "
                f"{safe_int(row['openInterest']):,}"
                f" | "
                f"{format_money(premium)}"
            )


    lines.append("")


    # ========================================================
    # DATA NOTE
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "⚠️ DATA NOTE"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    lines.append(
        "• Yahoo Finance 무료 옵션 데이터 기반"
    )


    lines.append(
        "• OI는 실제 Long/Short 방향을 의미하지 않음"
    )


    lines.append(
        "• Volume은 실제 Buy/Sell 방향이 아님"
    )


    lines.append(
        "• OI Δ는 snapshot 간 차이"
    )


    lines.append(
        "• BAR는 CALL/PUT 상대 크기 시각화"
    )


    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    return "\n".join(lines)


# ============================================================
# SAVE FILES
# ============================================================

def save_outputs(
    ticker,
    df,
    structure,
    summary,
    top_contracts,
    report
):

    # --------------------------------------------------------
    # report.md
    # --------------------------------------------------------

    report_path = os.path.join(
        OUTPUT_DIR,
        "report.md"
    )


    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report
        )


    # --------------------------------------------------------
    # summary.csv
    # --------------------------------------------------------

    summary_path = os.path.join(
        OUTPUT_DIR,
        "summary.csv"
    )


    pd.DataFrame(
        [summary]
    ).to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # contracts.csv
    # --------------------------------------------------------

    contracts_path = os.path.join(
        OUTPUT_DIR,
        "contracts.csv"
    )


    df.to_csv(
        contracts_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # strike_structure.csv
    # --------------------------------------------------------

    structure_path = os.path.join(
        OUTPUT_DIR,
        "strike_structure.csv"
    )


    structure.to_csv(
        structure_path,
        index=False,
        encoding="utf-8-sig"
    )


    # --------------------------------------------------------
    # top_contracts.csv
    # --------------------------------------------------------

    top_path = os.path.join(
        OUTPUT_DIR,
        "top_contracts.csv"
    )


    top_contracts.to_csv(
        top_path,
        index=False,
        encoding="utf-8-sig"
    )


    print("")
    print(
        "💾 OUTPUT"
    )


    print(
        f"   {report_path}"
    )


    print(
        f"   {summary_path}"
    )


    print(
        f"   {contracts_path}"
    )


    print(
        f"   {structure_path}"
    )


    print(
        f"   {top_path}"
    )


    return {

        "report":
        report_path,

        "summary":
        summary_path,

        "contracts":
        contracts_path,

        "structure":
        structure_path,

        "top":
        top_path
    }


# ============================================================
# MAIN ANALYSIS
# ============================================================

def analyze():

    ticker = SYMBOL


    print("")
    print(
        "=" * 70
    )


    print(
        f"🔥 {ticker} OPTION STRUCTURE"
    )


    print(
        "=" * 70
    )


    print(
        f"🎯 Strike: "
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )


    print(
        f"📅 DTE: "
        f"0 ~ {MAX_DTE}"
    )


    print(
        "=" * 70
    )


    # ========================================================
    # PRICE
    # ========================================================

    price_context = (
        get_market_price(
            ticker
        )
    )


    if price_context is None:

        raise RuntimeError(
            "가격 조회 실패"
        )


    current_price = (
        price_context[
            "regular_close"
        ]
    )


    # ========================================================
    # OPTIONS
    # ========================================================

    print("")
    print(
        "📡 옵션 데이터 수집"
    )


    df = collect_options(
        ticker
    )


    df = normalize_options(
        df
    )


    # ========================================================
    # STRIKE FILTER
    # ========================================================

    df = filter_strike_range(
        df
    )


    if df.empty:

        raise RuntimeError(
            "설정된 Strike / DTE 범위에 "
            "옵션 데이터가 없습니다."
        )


    # ========================================================
    # OI DELTA
    # ========================================================

    print("")
    print(
        "📈 OI Delta 계산"
    )


    df, previous_oi_date = (
        calculate_oi_delta(
            df,
            ticker
        )
    )


    # ========================================================
    # STRIKE STRUCTURE
    # ========================================================

    print("")
    print(
        "🏗️ Strike 구조 계산"
    )


    structure = (
        build_strike_structure(
            df
        )
    )


    # ========================================================
    # TOP CONTRACTS
    # ========================================================

    top_contracts = (
        build_top_contracts(
            df
        )
    )


    # ========================================================
    # SUMMARY
    # ========================================================

    summary = build_summary(
        ticker,
        current_price,
        df,
        structure,
        previous_oi_date
    )


    # ========================================================
    # REPORT
    # ========================================================

    report = build_report(
        summary,
        structure,
        top_contracts
    )


    # ========================================================
    # DISPLAY
    # ========================================================

    print("")
    print(report)
    print("")


    # ========================================================
    # SAVE
    # ========================================================

    output_files = save_outputs(
        ticker,
        df,
        structure,
        summary,
        top_contracts,
        report
    )


    # ========================================================
    # SAVE OI SNAPSHOT
    # ========================================================

    oi_snapshot_path = (
        save_oi_snapshot(
            df,
            ticker
        )
    )


    # ========================================================
    # TELEGRAM
    # ========================================================

    telegram_ok = send_telegram(
        report
    )


    # ========================================================
    # RESULT
    # ========================================================

    print("")
    print(
        "=" * 70
    )


    print(
        "🔥 OPTION STRUCTURE 완료"
    )


    print(
        "=" * 70
    )


    print(
        f"📌 Symbol: "
        f"{ticker}"
    )


    print(
        f"📌 Price: "
        f"${current_price:.2f}"
    )


    print(
        f"📌 Strike: "
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )


    print(
        f"📌 Contracts: "
        f"{len(df):,}"
    )


    print(
        f"📌 Strikes: "
        f"{len(structure):,}"
    )


    print(
        f"📌 Previous OI: "
        f"{previous_oi_date}"
    )


    print(
        f"📌 Telegram: "
        f"{'SUCCESS' if telegram_ok else 'SKIPPED/FAILED'}"
    )


    print(
        "=" * 70
    )


    return {

        "ticker":
        ticker,

        "current_price":
        current_price,

        "df":
        df,

        "structure":
        structure,

        "summary":
        summary,

        "report":
        report,

        "previous_oi_date":
        previous_oi_date,

        "oi_snapshot":
        oi_snapshot_path,

        "telegram_ok":
        telegram_ok,

        "output_files":
        output_files
    }


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        analyze()

        print("")
        print(
            "✅ SUCCESS"
        )

    except Exception as e:

        print("")
        print(
            "❌ OPTION STRUCTURE FAILED"
        )


        print(
            f"❌ {e}"
        )


        traceback.print_exc()


        raise
