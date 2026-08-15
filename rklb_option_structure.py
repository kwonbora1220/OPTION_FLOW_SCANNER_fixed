
import os
import sys
import time
import argparse
import urllib.parse
import urllib.request

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SYMBOL = os.getenv(
    "SYMBOL",
    "RKLB"
).upper()

DEFAULT_PRICE = os.getenv(
    "PRICE",
    ""
)

MIN_STRIKE = float(
    os.getenv(
        "MIN_STRIKE",
        "80"
    )
)

MAX_STRIKE = float(
    os.getenv(
        "MAX_STRIKE",
        "100"
    )
)

MAX_DTE = int(
    os.getenv(
        "MAX_DTE",
        "180"
    )
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "rklb_option_structure"
)


# ============================================================
# AUTO KEY STRIKE CONFIG
# ============================================================

# 자동으로 몇 개의 핵심 Strike를 뽑을지
KEY_STRIKE_COUNT = int(
    os.getenv(
        "KEY_STRIKE_COUNT",
        "5"
    )
)

# 현재가와 너무 먼 Strike를 Key Strike로 선택하지 않도록
KEY_STRIKE_MAX_DISTANCE = float(
    os.getenv(
        "KEY_STRIKE_MAX_DISTANCE",
        "0.30"
    )
)

# 현재가 근처에서 자동으로 찾을 때 사용할 최소 OI
KEY_STRIKE_MIN_OI = float(
    os.getenv(
        "KEY_STRIKE_MIN_OI",
        "0"
    )
)

# Strike 간격이 너무 촘촘하게 중복 선택되는 것을 방지
KEY_STRIKE_MIN_GAP = float(
    os.getenv(
        "KEY_STRIKE_MIN_GAP",
        "2"
    )
)


# ============================================================
# BAR CONFIG
# ============================================================

BAR_WIDTH = 10
BAR_MIN_WIDTH = 1


# ============================================================
# DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# TIMEZONE
# ============================================================

US_EASTERN = ZoneInfo(
    "America/New_York"
)


# ============================================================
# DATE
# ============================================================

def market_today():

    return datetime.now(
        US_EASTERN
    ).date()


# ============================================================
# HELPERS
# ============================================================

def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):

            return value

    except Exception:

        pass

    return np.nan


def numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def fmt_money(value):

    value = safe_float(value)

    if not np.isfinite(value):

        return "N/A"

    sign = (
        "-"
        if value < 0
        else ""
    )

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
        f"${value:,.0f}"
    )


def fmt_number(value):

    value = safe_float(value)

    if not np.isfinite(value):

        return "N/A"

    return f"{value:,.0f}"


def fmt_pct(value):

    value = safe_float(value)

    if not np.isfinite(value):

        return "N/A"

    return f"{value:.1f}%"


def fmt_iv(value):

    value = safe_float(value)

    if not np.isfinite(value):

        return "N/A"

    if value < 2:

        value *= 100

    return f"{value:.1f}%"


# ============================================================
# DYNAMIC BAR
# ============================================================

def make_dynamic_dual_bar(
    call_value,
    put_value,
    reference_total,
    max_width=BAR_WIDTH
):

    call_value = safe_float(call_value)
    put_value = safe_float(put_value)
    reference_total = safe_float(reference_total)

    if (
        not np.isfinite(call_value)
        or call_value < 0
    ):
        call_value = 0.0

    if (
        not np.isfinite(put_value)
        or put_value < 0
    ):
        put_value = 0.0

    if (
        not np.isfinite(reference_total)
        or reference_total <= 0
    ):
        reference_total = 0.0

    total = call_value + put_value

    if total <= 0:

        return "·"

    scale_ratio = (
        total / reference_total
        if reference_total > 0
        else 1.0
    )

    scale_ratio = max(
        0.0,
        min(
            scale_ratio,
            1.0
        )
    )

    bar_length = int(
        round(
            scale_ratio * max_width
        )
    )

    if bar_length < BAR_MIN_WIDTH:

        bar_length = BAR_MIN_WIDTH

    bar_length = min(
        bar_length,
        max_width
    )

    call_ratio = (
        call_value / total
    )

    call_width = int(
        round(
            call_ratio * bar_length
        )
    )

    put_width = (
        bar_length - call_width
    )

    if (
        call_value > 0
        and put_value > 0
    ):

        if call_width <= 0:

            call_width = 1
            put_width = bar_length - 1

        elif put_width <= 0:

            put_width = 1
            call_width = bar_length - 1

    call_width = max(
        0,
        min(
            call_width,
            bar_length
        )
    )

    put_width = max(
        0,
        min(
            put_width,
            bar_length - call_width
        )
    )

    return (
        "🟩" * call_width
        +
        "🟥" * put_width
    )


def make_dual_bar(
    call_value,
    put_value,
    width=BAR_WIDTH
):

    call_value = safe_float(call_value)
    put_value = safe_float(put_value)

    if not np.isfinite(call_value):
        call_value = 0

    if not np.isfinite(put_value):
        put_value = 0

    return make_dynamic_dual_bar(
        call_value,
        put_value,
        call_value + put_value,
        max_width=width
    )


def make_dual_bar_line(
    call_value,
    put_value,
    reference_total,
    width=BAR_WIDTH
):

    bar = make_dynamic_dual_bar(
        call_value,
        put_value,
        reference_total,
        width
    )

    return (
        f"{bar} "
        f"C {fmt_number(call_value)} "
        f"/ "
        f"P {fmt_number(put_value)}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Yahoo Finance Independent "
            "Full Option Structure Scanner"
        )
    )

    parser.add_argument(
        "symbol",
        nargs="?",
        default=DEFAULT_SYMBOL
    )

    parser.add_argument(
        "price",
        nargs="?",
        default=DEFAULT_PRICE
    )

    parser.add_argument(
        "--min-strike",
        type=float,
        default=MIN_STRIKE
    )

    parser.add_argument(
        "--max-strike",
        type=float,
        default=MAX_STRIKE
    )

    parser.add_argument(
        "--max-dte",
        type=int,
        default=MAX_DTE
    )

    parser.add_argument(
        "--output",
        default=OUTPUT_DIR
    )

    return parser.parse_args()


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(
    ticker,
    manual_price=None
):

    print()
    print("=" * 70)
    print("FETCH CURRENT PRICE")
    print("=" * 70)

    if manual_price is not None:

        manual_price = safe_float(
            manual_price
        )

        if (
            np.isfinite(manual_price)
            and manual_price > 0
        ):

            print(
                f"MANUAL PRICE: "
                f"${manual_price:.2f}"
            )

            return manual_price

    try:

        history = ticker.history(
            period="1d",
            interval="1m",
            prepost=True
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                price = float(
                    close.iloc[-1]
                )

                print(
                    f"YAHOO PRICE: "
                    f"${price:.2f}"
                )

                return price

    except Exception as exc:

        print(
            "1m price error:",
            repr(exc)
        )

    try:

        history = ticker.history(
            period="5d"
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                price = float(
                    close.iloc[-1]
                )

                print(
                    f"YAHOO PRICE: "
                    f"${price:.2f}"
                )

                return price

    except Exception as exc:

        print(
            "5d price error:",
            repr(exc)
        )

    raise RuntimeError(
        "Unable to determine current price."
    )


# ============================================================
# DTE
# ============================================================

def calculate_dte(expiration):

    try:

        expiry = pd.Timestamp(
            expiration
        ).date()

        return (
            expiry - market_today()
        ).days

    except Exception:

        return np.nan


# ============================================================
# PREMIUM
# ============================================================

def calculate_premium(
    volume,
    bid,
    ask,
    last_price
):

    volume = safe_float(volume)
    bid = safe_float(bid)
    ask = safe_float(ask)
    last_price = safe_float(last_price)

    if (
        not np.isfinite(volume)
        or volume <= 0
    ):

        return 0.0

    if (
        np.isfinite(bid)
        and np.isfinite(ask)
        and bid >= 0
        and ask >= bid
        and ask > 0
    ):

        mid = (
            bid + ask
        ) / 2

    elif (
        np.isfinite(last_price)
        and last_price > 0
    ):

        mid = last_price

    else:

        return 0.0

    return (
        volume
        * mid
        * 100
    )


# ============================================================
# GEX PROXY
# ============================================================

def calculate_gex(
    gamma,
    open_interest,
    spot,
    option_type
):

    gamma = safe_float(gamma)
    open_interest = safe_float(open_interest)
    spot = safe_float(spot)

    if (
        not np.isfinite(gamma)
        or not np.isfinite(open_interest)
        or not np.isfinite(spot)
    ):

        return np.nan

    if (
        gamma <= 0
        or open_interest <= 0
        or spot <= 0
    ):

        return 0.0

    gex = (
        gamma
        * open_interest
        * 100
        * spot
        * spot
        * 0.01
    )

    if option_type == "PUT":

        gex *= -1

    return gex


# ============================================================
# FETCH ALL OPTIONS
# ============================================================

def fetch_options(
    symbol,
    manual_price=None
):

    print()
    print("=" * 70)
    print("FETCH YAHOO FINANCE FULL OPTION DATA")
    print("=" * 70)

    ticker = yf.Ticker(
        symbol
    )

    spot = get_current_price(
        ticker,
        manual_price
    )

    try:

        expirations = list(
            ticker.options
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to get Yahoo option expirations."
        ) from exc

    print(
        f"TOTAL EXPIRATIONS FOUND: "
        f"{len(expirations)}"
    )

    if not expirations:

        raise RuntimeError(
            "Yahoo returned ZERO expirations."
        )

    rows = []

    successful = 0
    failed = 0

    for index, expiration in enumerate(
        expirations,
        start=1
    ):

        print()
        print(
            f"[{index}/{len(expirations)}] "
            f"{expiration}"
        )

        try:

            chain = ticker.option_chain(
                expiration
            )

        except Exception as exc:

            failed += 1

            print(
                "❌ FAILED:",
                repr(exc)
            )

            continue

        calls = (
            chain.calls
            if chain.calls is not None
            else pd.DataFrame()
        )

        puts = (
            chain.puts
            if chain.puts is not None
            else pd.DataFrame()
        )

        print(
            f"CALL: {len(calls):,} | "
            f"PUT: {len(puts):,}"
        )

        if not calls.empty:

            frame = calls.copy()

            frame["option_type"] = "CALL"
            frame["expiration"] = expiration

            rows.append(frame)

        if not puts.empty:

            frame = puts.copy()

            frame["option_type"] = "PUT"
            frame["expiration"] = expiration

            rows.append(frame)

        if (
            not calls.empty
            or not puts.empty
        ):

            successful += 1

        time.sleep(
            0.25
        )

    if not rows:

        raise RuntimeError(
            "No option rows collected."
        )

    data = pd.concat(
        rows,
        ignore_index=True
    )

    print()
    print("=" * 70)
    print("FULL YAHOO COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Successful expirations: "
        f"{successful}"
    )

    print(
        f"Failed expirations: "
        f"{failed}"
    )

    print(
        f"RAW ROWS: "
        f"{len(data):,}"
    )

    return data, spot


# ============================================================
# NORMALIZE
# ============================================================

def normalize(data):

    data = data.copy()

    numeric_columns = [
        "strike",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "gamma",
        "delta",
        "vega"
    ]

    for column in numeric_columns:

        if column in data.columns:

            data[column] = numeric(
                data[column]
            )

        else:

            data[column] = np.nan

    data["DTE"] = (
        data["expiration"]
        .apply(
            calculate_dte
        )
    )

    data = data[
        data["strike"].notna()
    ].copy()

    data = data[
        data["option_type"].isin(
            [
                "CALL",
                "PUT"
            ]
        )
    ].copy()

    return data


# ============================================================
# FILTER
# ============================================================

def apply_filters(
    data,
    min_strike,
    max_strike,
    max_dte
):

    print()
    print("=" * 70)
    print("ANALYSIS FILTER")
    print("=" * 70)

    raw_count = len(data)

    data = data[
        data["DTE"].notna()
    ].copy()

    data = data[
        (
            data["DTE"] > 0
        )
        &
        (
            data["DTE"] <= max_dte
        )
    ].copy()

    print(
        f"DTE 1~{max_dte}: "
        f"{len(data):,}"
    )

    data = data[
        data["strike"].between(
            min_strike,
            max_strike
        )
    ].copy()

    print(
        f"Strike ${min_strike:g}"
        f"~${max_strike:g}: "
        f"{len(data):,}"
    )

    print(
        f"Removed from raw: "
        f"{raw_count - len(data):,}"
    )

    return data


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    data,
    spot
):

    data = data.copy()

    data["premium_proxy"] = (
        data.apply(
            lambda row:
            calculate_premium(
                row["volume"],
                row["bid"],
                row["ask"],
                row["lastPrice"]
            ),
            axis=1
        )
    )

    data["gex"] = (
        data.apply(
            lambda row:
            calculate_gex(
                row["gamma"],
                row["openInterest"],
                spot,
                row["option_type"]
            ),
            axis=1
        )
    )

    data["volume_oi"] = np.where(
        data["openInterest"] > 0,
        data["volume"]
        /
        data["openInterest"],
        np.nan
    )

    data["distance_pct"] = (
        (
            data["strike"]
            -
            spot
        )
        /
        spot
        *
        100
    )

    return data


# ============================================================
# STRIKE STRUCTURE
# ============================================================

def build_strike_table(data):

    rows = []

    for strike, frame in (
        data.groupby(
            "strike"
        )
    ):

        calls = frame[
            frame["option_type"]
            == "CALL"
        ]

        puts = frame[
            frame["option_type"]
            == "PUT"
        ]

        cv = (
            calls["volume"]
            .fillna(0)
            .sum()
        )

        pv = (
            puts["volume"]
            .fillna(0)
            .sum()
        )

        coi = (
            calls["openInterest"]
            .fillna(0)
            .sum()
        )

        poi = (
            puts["openInterest"]
            .fillna(0)
            .sum()
        )

        cp = (
            calls["premium_proxy"]
            .fillna(0)
            .sum()
        )

        pp = (
            puts["premium_proxy"]
            .fillna(0)
            .sum()
        )

        cg = calls["gex"].sum(
            min_count=1
        )

        pg = puts["gex"].sum(
            min_count=1
        )

        tv = cv + pv
        toi = coi + poi
        tp = cp + pp

        rows.append(
            {
                "strike": strike,

                "call_volume": cv,
                "put_volume": pv,
                "total_volume": tv,

                "call_oi": coi,
                "put_oi": poi,
                "total_oi": toi,

                "call_premium": cp,
                "put_premium": pp,
                "total_premium": tp,

                "call_volume_oi":
                    cv / coi
                    if coi > 0
                    else np.nan,

                "put_volume_oi":
                    pv / poi
                    if poi > 0
                    else np.nan,

                "call_volume_ratio":
                    cv / tv * 100
                    if tv > 0
                    else np.nan,

                "call_oi_ratio":
                    coi / toi * 100
                    if toi > 0
                    else np.nan,

                "call_premium_ratio":
                    cp / tp * 100
                    if tp > 0
                    else np.nan,

                "call_gex": cg,
                "put_gex": pg,

                "net_gex":
                    cg + pg
                    if np.isfinite(cg)
                    and np.isfinite(pg)
                    else np.nan
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return result

    return (
        result
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================
# ⭐ AUTO KEY STRIKE DETECTION
# ============================================================

def detect_key_strikes(
    strike_table,
    spot,
    count=KEY_STRIKE_COUNT
):

    if strike_table.empty:

        return pd.DataFrame()

    table = strike_table.copy()

    table["distance_pct"] = (
        (
            table["strike"]
            -
            spot
        )
        /
        spot
        *
        100
    )

    table["distance_abs_pct"] = (
        table["distance_pct"]
        .abs()
    )

    table["total_oi"] = (
        table["total_oi"]
        .fillna(0)
        .clip(lower=0)
    )

    table["total_volume"] = (
        table["total_volume"]
        .fillna(0)
        .clip(lower=0)
    )

    table["total_premium"] = (
        table["total_premium"]
        .fillna(0)
        .clip(lower=0)
    )

    # --------------------------------------------------------
    # 현재가에서 너무 먼 Strike 제외
    # --------------------------------------------------------

    table = table[
        table["distance_abs_pct"]
        <=
        KEY_STRIKE_MAX_DISTANCE * 100
    ].copy()

    if table.empty:

        return pd.DataFrame()

    table = table[
        table["total_oi"]
        >=
        KEY_STRIKE_MIN_OI
    ].copy()

    if table.empty:

        return pd.DataFrame()

    # --------------------------------------------------------
    # OI 기반 점수
    #
    # OI가 핵심
    # Volume / Premium은 보조
    # 현재가와 가까울수록 약간 가산
    # --------------------------------------------------------

    oi_score = np.log1p(
        table["total_oi"]
    )

    volume_score = np.log1p(
        table["total_volume"]
    )

    premium_score = np.log1p(
        table["total_premium"]
    )

    distance_score = (
        1
        /
        (
            1
            +
            table["distance_abs_pct"]
            /
            10
        )
    )

    table["key_score"] = (
        oi_score * 0.60
        +
        volume_score * 0.15
        +
        premium_score * 0.15
        +
        distance_score * 10 * 0.10
    )

    # --------------------------------------------------------
    # OI 순위
    # --------------------------------------------------------

    table["oi_rank"] = (
        table["total_oi"]
        .rank(
            ascending=False,
            method="min"
        )
    )

    # --------------------------------------------------------
    # Key Type 자동 해석
    # --------------------------------------------------------

    def classify_key(row):

        strike = row["strike"]

        if strike < spot:

            return "SUPPORT"

        if strike > spot:

            return "RESISTANCE"

        return "ATM"

    table["key_type"] = (
        table.apply(
            classify_key,
            axis=1
        )
    )

    # --------------------------------------------------------
    # 강도
    # --------------------------------------------------------

    oi_max = table["total_oi"].max()

    if oi_max > 0:

        table["oi_strength_pct"] = (
            table["total_oi"]
            /
            oi_max
            *
            100
        )

    else:

        table["oi_strength_pct"] = 0

    # --------------------------------------------------------
    # 핵심 Strike 선택
    #
    # 너무 가까운 Strike끼리 중복 선택 방지
    # --------------------------------------------------------

    candidates = (
        table
        .sort_values(
            [
                "key_score",
                "total_oi"
            ],
            ascending=False
        )
        .reset_index(drop=True)
    )

    selected = []

    for _, row in candidates.iterrows():

        strike = float(
            row["strike"]
        )

        too_close = False

        for selected_strike in selected:

            if abs(
                strike
                -
                selected_strike
            ) < KEY_STRIKE_MIN_GAP:

                too_close = True
                break

        if too_close:

            continue

        selected.append(
            strike
        )

        if len(selected) >= count:

            break

    if not selected:

        return pd.DataFrame()

    result = candidates[
        candidates["strike"].isin(
            selected
        )
    ].copy()

    # --------------------------------------------------------
    # 현재가 기준 순서
    # --------------------------------------------------------

    result["distance_from_spot"] = (
        result["strike"]
        -
        spot
    )

    result = (
        result
        .sort_values(
            "distance_from_spot"
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 자동 설명
    # --------------------------------------------------------

    descriptions = []

    for _, row in result.iterrows():

        strike = float(
            row["strike"]
        )

        if strike < spot:

            diff_pct = (
                (spot - strike)
                /
                spot
                *
                100
            )

            descriptions.append(
                f"${strike:g} "
                f"지지 후보 "
                f"(현재가 대비 -{diff_pct:.1f}%)"
            )

        elif strike > spot:

            diff_pct = (
                (strike - spot)
                /
                spot
                *
                100
            )

            descriptions.append(
                f"${strike:g} "
                f"저항 후보 "
                f"(현재가 대비 +{diff_pct:.1f}%)"
            )

        else:

            descriptions.append(
                f"${strike:g} ATM 핵심"
            )

    result["description"] = descriptions

    return result


# ============================================================
# AUTO PRICE SCENARIO
# ============================================================

def build_price_scenario(
    key_strikes,
    spot
):

    if key_strikes.empty:

        return []

    below = (
        key_strikes[
            key_strikes["strike"] < spot
        ]
        .sort_values(
            "strike",
            ascending=False
        )
    )

    above = (
        key_strikes[
            key_strikes["strike"] > spot
        ]
        .sort_values(
            "strike"
        )
    )

    scenarios = []

    # --------------------------------------------------------
    # 가장 가까운 아래 Strike
    # --------------------------------------------------------

    if not below.empty:

        row = below.iloc[0]

        scenarios.append(
            {
                "strike": row["strike"],
                "label": "🛡 방어선",
                "description": (
                    "현재가 아래에서 "
                    "OI가 가장 강한 지지 후보"
                )
            }
        )

    # --------------------------------------------------------
    # 위쪽 Strike들
    # --------------------------------------------------------

    above_rows = list(
        above.head(4).iterrows()
    )

    labels = [
        "🟢 1차 돌파",
        "🔥 핵심 저항",
        "🚀 상승 확인",
        "🎯 다음 목표"
    ]

    descriptions = [
        "첫 번째 OI 집중 돌파 후보",
        "OI 집중도가 높은 핵심 저항",
        "추가 상승 확인 구간",
        "상승 시 다음 옵션 집중 구간"
    ]

    for index, (_, row) in enumerate(
        above_rows
    ):

        scenarios.append(
            {
                "strike": row["strike"],
                "label": labels[
                    min(
                        index,
                        len(labels) - 1
                    )
                ],
                "description": descriptions[
                    min(
                        index,
                        len(descriptions) - 1
                    )
                ]
            }
        )

    return scenarios


# ============================================================
# DYNAMIC BAR STRUCTURE
# ============================================================

def build_bar_structure(
    strike_table
):

    if strike_table.empty:

        return []

    lines = []

    sorted_table = (
        strike_table
        .sort_values(
            "strike"
        )
        .reset_index(
            drop=True
        )
    )

    oi_reference = (
        (
            sorted_table[
                "call_oi"
            ].fillna(0)
            +
            sorted_table[
                "put_oi"
            ].fillna(0)
        )
        .max()
    )

    volume_reference = (
        (
            sorted_table[
                "call_volume"
            ].fillna(0)
            +
            sorted_table[
                "put_volume"
            ].fillna(0)
        )
        .max()
    )

    premium_reference = (
        (
            sorted_table[
                "call_premium"
            ].fillna(0)
            +
            sorted_table[
                "put_premium"
            ].fillna(0)
        )
        .max()
    )

    lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 CALL / PUT BAR STRUCTURE",
        "🟩 CALL   🟥 PUT",
        f"📏 BAR MAX: {BAR_WIDTH}칸",
        "📐 규모가 작으면 BAR도 짧게 표시",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🟢 OI STRUCTURE",
        ""
    ]

    for _, row in sorted_table.iterrows():

        lines.append(
            f"🎯 ${row['strike']:g}   "
            +
            make_dual_bar_line(
                row["call_oi"],
                row["put_oi"],
                oi_reference
            )
        )

    lines += [
        "",
        "🔥 VOLUME STRUCTURE",
        ""
    ]

    for _, row in sorted_table.iterrows():

        lines.append(
            f"🎯 ${row['strike']:g}   "
            +
            make_dual_bar_line(
                row["call_volume"],
                row["put_volume"],
                volume_reference
            )
        )

    lines += [
        "",
        "💰 PREMIUM STRUCTURE",
        ""
    ]

    for _, row in sorted_table.iterrows():

        bar = make_dynamic_dual_bar(
            row["call_premium"],
            row["put_premium"],
            premium_reference
        )

        lines.append(
            f"🎯 ${row['strike']:g}   "
            f"{bar} "
            f"C {fmt_money(row['call_premium'])} "
            f"/ "
            f"P {fmt_money(row['put_premium'])}"
        )

    return lines


# ============================================================
# EXPIRATION STRUCTURE
# ============================================================

def build_expiration_structure(
    data
):

    rows = []

    if data.empty:

        return pd.DataFrame()

    for expiration, frame in (
        data.groupby(
            "expiration"
        )
    ):

        calls = frame[
            frame["option_type"]
            == "CALL"
        ]

        puts = frame[
            frame["option_type"]
            == "PUT"
        ]

        cv = (
            calls["volume"]
            .fillna(0)
            .sum()
        )

        pv = (
            puts["volume"]
            .fillna(0)
            .sum()
        )

        coi = (
            calls["openInterest"]
            .fillna(0)
            .sum()
        )

        poi = (
            puts["openInterest"]
            .fillna(0)
            .sum()
        )

        cp = (
            calls["premium_proxy"]
            .fillna(0)
            .sum()
        )

        pp = (
            puts["premium_proxy"]
            .fillna(0)
            .sum()
        )

        rows.append(
            {
                "expiration": expiration,
                "DTE": calculate_dte(expiration),
                "call_volume": cv,
                "put_volume": pv,
                "total_volume": cv + pv,
                "call_oi": coi,
                "put_oi": poi,
                "total_oi": coi + poi,
                "call_premium": cp,
                "put_premium": pp,
                "total_premium": cp + pp,
                "call_volume_ratio":
                    cv / (cv + pv) * 100
                    if cv + pv > 0
                    else np.nan,
                "call_oi_ratio":
                    coi / (coi + poi) * 100
                    if coi + poi > 0
                    else np.nan
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:

        return result

    total_oi = result[
        "total_oi"
    ].sum()

    result[
        "total_oi_concentration_pct"
    ] = (
        result["total_oi"]
        /
        total_oi
        *
        100
        if total_oi > 0
        else np.nan
    )

    return (
        result
        .sort_values("DTE")
        .reset_index(drop=True)
    )


# ============================================================
# STRIKE × EXPIRATION
# ============================================================

def build_strike_expiration_structure(
    data,
    key_strikes
):

    rows = []

    if data.empty or key_strikes.empty:

        return pd.DataFrame()

    for target in key_strikes:

        strike_data = data[
            abs(
                data["strike"]
                -
                target
            ) < 0.001
        ]

        if strike_data.empty:

            continue

        for expiration, frame in (
            strike_data.groupby(
                "expiration"
            )
        ):

            calls = frame[
                frame["option_type"]
                == "CALL"
            ]

            puts = frame[
                frame["option_type"]
                == "PUT"
            ]

            cv = (
                calls["volume"]
                .fillna(0)
                .sum()
            )

            pv = (
                puts["volume"]
                .fillna(0)
                .sum()
            )

            coi = (
                calls["openInterest"]
                .fillna(0)
                .sum()
            )

            poi = (
                puts["openInterest"]
                .fillna(0)
                .sum()
            )

            cp = (
                calls["premium_proxy"]
                .fillna(0)
                .sum()
            )

            pp = (
                puts["premium_proxy"]
                .fillna(0)
                .sum()
            )

            rows.append(
                {
                    "strike": target,
                    "expiration": expiration,
                    "DTE":
                        calculate_dte(
                            expiration
                        ),
                    "call_volume": cv,
                    "put_volume": pv,
                    "total_volume": cv + pv,
                    "call_oi": coi,
                    "put_oi": poi,
                    "total_oi": coi + poi,
                    "call_premium": cp,
                    "put_premium": pp,
                    "total_premium": cp + pp
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:

        return result

    for strike in result[
        "strike"
    ].unique():

        mask = (
            result["strike"]
            ==
            strike
        )

        total_oi = (
            result.loc[
                mask,
                "total_oi"
            ].sum()
        )

        call_oi = (
            result.loc[
                mask,
                "call_oi"
            ].sum()
        )

        put_oi = (
            result.loc[
                mask,
                "put_oi"
            ].sum()
        )

        if total_oi > 0:

            result.loc[
                mask,
                "total_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "total_oi"
                ]
                /
                total_oi
                *
                100
            )

        else:

            result.loc[
                mask,
                "total_oi_pct"
            ] = np.nan

        if call_oi > 0:

            result.loc[
                mask,
                "call_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "call_oi"
                ]
                /
                call_oi
                *
                100
            )

        else:

            result.loc[
                mask,
                "call_oi_pct"
            ] = np.nan

        if put_oi > 0:

            result.loc[
                mask,
                "put_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "put_oi"
                ]
                /
                put_oi
                *
                100
            )

        else:

            result.loc[
                mask,
                "put_oi_pct"
            ] = np.nan

    return (
        result
        .sort_values(
            [
                "strike",
                "total_oi"
            ],
            ascending=[
                True,
                False
            ]
        )
        .reset_index(drop=True)
    )


# ============================================================
# KEY STRIKE SUMMARY
# ============================================================

def build_key_strike_summary(
    strike_expiration
):

    if strike_expiration.empty:

        return pd.DataFrame()

    rows = []

    for strike, frame in (
        strike_expiration.groupby(
            "strike"
        )
    ):

        frame = frame.sort_values(
            "total_oi",
            ascending=False
        )

        top = frame.iloc[0]

        rows.append(
            {
                "strike": strike,

                "total_oi":
                    frame[
                        "total_oi"
                    ].sum(),

                "call_oi":
                    frame[
                        "call_oi"
                    ].sum(),

                "put_oi":
                    frame[
                        "put_oi"
                    ].sum(),

                "top_expiration":
                    top["expiration"],

                "top_DTE":
                    top["DTE"],

                "top_expiration_total_oi":
                    top["total_oi"],

                "top_expiration_call_oi":
                    top["call_oi"],

                "top_expiration_put_oi":
                    top["put_oi"],

                "top_expiration_oi_pct":
                    top.get(
                        "total_oi_pct",
                        np.nan
                    )
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(
    data
):

    if data.empty:

        return pd.DataFrame()

    result = data.copy()

    result["importance"] = (
        np.log1p(
            result[
                "premium_proxy"
            ]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result[
                "volume"
            ]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result[
                "openInterest"
            ]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result[
                "gex"
            ]
            .fillna(0)
            .abs()
            .clip(lower=0)
        )
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False
        )
        .head(50)
    )


# ============================================================
# WALL
# ============================================================

def find_wall(
    strike_table,
    spot,
    option_type
):

    if strike_table.empty:

        return None

    if option_type == "CALL":

        candidates = strike_table[
            strike_table["strike"]
            >=
            spot
        ].copy()

        candidates["oi"] = (
            candidates["call_oi"]
        )

        candidates["gex_abs"] = (
            candidates["call_gex"].abs()
        )

        candidates["volume"] = (
            candidates["call_volume"]
        )

    else:

        candidates = strike_table[
            strike_table["strike"]
            <=
            spot
        ].copy()

        candidates["oi"] = (
            candidates["put_oi"]
        )

        candidates["gex_abs"] = (
            candidates["put_gex"].abs()
        )

        candidates["volume"] = (
            candidates["put_volume"]
        )

    if candidates.empty:

        return None

    candidates["distance"] = (
        (
            candidates["strike"]
            -
            spot
        ).abs()
        /
        spot
    )

    candidates = candidates[
        candidates["distance"]
        <=
        0.20
    ].copy()

    if candidates.empty:

        return None

    candidates["score"] = (
        np.log1p(
            candidates["oi"]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            candidates["gex_abs"]
            .fillna(0)
            .clip(lower=0)
        )
        +
        0.25
        *
        np.log1p(
            candidates["volume"]
            .fillna(0)
            .clip(lower=0)
        )
    )

    candidates["score"] += (
        3
        /
        (
            1
            +
            candidates["distance"]
            * 20
        )
    )

    return (
        candidates
        .sort_values(
            "score",
            ascending=False
        )
        .iloc[0]
    )


# ============================================================
# REPORT
# ============================================================

def build_report(
    data,
    strike_table,
    expiration_structure,
    strike_expiration,
    key_strike_summary,
    top_contracts,
    key_strikes,
    price_scenarios,
    spot,
    symbol,
    min_strike,
    max_strike,
    max_dte,
    started
):

    calls = data[
        data["option_type"]
        == "CALL"
    ]

    puts = data[
        data["option_type"]
        == "PUT"
    ]

    cv = (
        calls["volume"]
        .fillna(0)
        .sum()
    )

    pv = (
        puts["volume"]
        .fillna(0)
        .sum()
    )

    coi = (
        calls["openInterest"]
        .fillna(0)
        .sum()
    )

    poi = (
        puts["openInterest"]
        .fillna(0)
        .sum()
    )

    cp = (
        calls["premium_proxy"]
        .fillna(0)
        .sum()
    )

    pp = (
        puts["premium_proxy"]
        .fillna(0)
        .sum()
    )

    total_volume = cv + pv
    total_oi = coi + poi
    total_premium = cp + pp

    cv_ratio = (
        cv / total_volume * 100
        if total_volume > 0
        else np.nan
    )

    coi_ratio = (
        coi / total_oi * 100
        if total_oi > 0
        else np.nan
    )

    cp_ratio = (
        cp / total_premium * 100
        if total_premium > 0
        else np.nan
    )

    cv_oi = (
        cv / coi
        if coi > 0
        else np.nan
    )

    pv_oi = (
        pv / poi
        if poi > 0
        else np.nan
    )

    tgex = calls["gex"].sum(
        min_count=1
    )

    pgex = puts["gex"].sum(
        min_count=1
    )

    if (
        np.isfinite(tgex)
        and np.isfinite(pgex)
    ):

        net_gex = tgex + pgex

    elif np.isfinite(tgex):

        net_gex = tgex

    elif np.isfinite(pgex):

        net_gex = pgex

    else:

        net_gex = np.nan

    temp = data.copy()

    temp["atm_distance"] = (
        temp["strike"] - spot
    ).abs()

    atm_iv = (
        temp
        .sort_values(
            "atm_distance"
        )
        .head(10)[
            "impliedVolatility"
        ]
        .dropna()
        .mean()
    )

    call_wall = find_wall(
        strike_table,
        spot,
        "CALL"
    )

    put_wall = find_wall(
        strike_table,
        spot,
        "PUT"
    )

    report = []

    # ========================================================
    # HEADER
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"🔥 {symbol} OPTION STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "",
        f"💰 현재가: ${spot:.2f}",
        (
            f"🎯 분석 Strike: "
            f"${min_strike:g} ~ ${max_strike:g}"
        ),
        (
            f"📅 분석 DTE: "
            f"1 ~ {max_dte}"
        ),
        (
            f"📊 분석 행수: "
            f"{len(data):,}"
        ),
        ""
    ]

    # ========================================================
    # FLOW
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 1. OPTION FLOW",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f"CALL Volume: {cv:,.0f}",
        f"PUT Volume : {pv:,.0f}",
        f"CALL Volume Ratio: {fmt_pct(cv_ratio)}",
        "",
        f"CALL OI: {coi:,.0f}",
        f"PUT OI : {poi:,.0f}",
        f"CALL OI Ratio: {fmt_pct(coi_ratio)}",
        "",
        f"CALL Premium Proxy: {fmt_money(cp)}",
        f"PUT Premium Proxy : {fmt_money(pp)}",
        f"CALL Premium Ratio: {fmt_pct(cp_ratio)}",
        ""
    ]

    # ========================================================
    # V/OI
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 2. VOLUME / OI",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        (
            f"CALL Volume/OI: "
            f"{cv_oi:.3f}"
            if np.isfinite(cv_oi)
            else "CALL Volume/OI: N/A"
        ),
        (
            f"PUT Volume/OI : "
            f"{pv_oi:.3f}"
            if np.isfinite(pv_oi)
            else "PUT Volume/OI : N/A"
        ),
        ""
    ]

    # ========================================================
    # AUTO KEY PRICE STRUCTURE
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 3. KEY PRICE STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        (
            "📌 종목별 OI 집중도를 자동 분석하여 "
            "핵심 Strike를 선정"
        ),
        ""
    ]

    if key_strikes.empty:

        report.append(
            "⚠️ 자동 Key Strike를 찾지 못했습니다."
        )

    else:

        # ----------------------------------------------------
        # KEY STRIKE TABLE
        # ----------------------------------------------------

        for _, row in key_strikes.iterrows():

            strike = safe_float(
                row["strike"]
            )

            key_type = row[
                "key_type"
            ]

            total_oi_key = safe_float(
                row["total_oi"]
            )

            call_oi_key = safe_float(
                row["call_oi"]
            )

            put_oi_key = safe_float(
                row["put_oi"]
            )

            call_volume_key = safe_float(
                row["call_volume"]
            )

            put_volume_key = safe_float(
                row["put_volume"]
            )

            oi_strength = safe_float(
                row["oi_strength_pct"]
            )

            if key_type == "SUPPORT":

                emoji = "🛡"

            elif key_type == "RESISTANCE":

                emoji = "🎯"

            else:

                emoji = "🔥"

            report += [
                (
                    f"{emoji} ${strike:g} "
                    f"| {key_type}"
                ),
                (
                    f"   OI: "
                    f"{total_oi_key:,.0f}"
                    f" "
                    f"({oi_strength:.1f}% of max)"
                ),
                (
                    f"   CALL OI: "
                    f"{call_oi_key:,.0f}"
                    f" / "
                    f"PUT OI: "
                    f"{put_oi_key:,.0f}"
                ),
                (
                    f"   CALL Vol: "
                    f"{call_volume_key:,.0f}"
                    f" / "
                    f"PUT Vol: "
                    f"{put_volume_key:,.0f}"
                ),
                ""
            ]

    # ========================================================
    # AUTO PRICE SCENARIO
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📈 5. PRICE SCENARIO",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if not price_scenarios:

        report.append(
            "⚠️ 가격 시나리오를 생성하지 못했습니다."
        )

    else:

        for scenario in price_scenarios:

            strike = safe_float(
                scenario["strike"]
            )

            report.append(
                f"{scenario['label']} "
                f"${strike:g} → "
                f"{scenario['description']}"
            )

    report.append("")

    # ========================================================
    # BAR
    # ========================================================

    report.extend(
        build_bar_structure(
            strike_table
        )
    )

    report.append("")

    # ========================================================
    # WALL / GEX
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🧱 WALL / GEX",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if call_wall is not None:

        report.append(
            f"📈 Call Wall: "
            f"${call_wall['strike']:g}"
            f" | OI "
            f"{call_wall['call_oi']:,.0f}"
        )

    else:

        report.append(
            "📈 Call Wall: N/A"
        )

    if put_wall is not None:

        report.append(
            f"📉 Put Wall: "
            f"${put_wall['strike']:g}"
            f" | OI "
            f"{put_wall['put_oi']:,.0f}"
        )

    else:

        report.append(
            "📉 Put Wall: N/A"
        )

    report += [
        "",
        f"CALL GEX Proxy: {fmt_money(tgex)}",
        f"PUT GEX Proxy : {fmt_money(pgex)}",
        f"NET GEX Proxy : {fmt_money(net_gex)}",
        f"ATM IV: {fmt_iv(atm_iv)}",
        ""
    ]

    # ========================================================
    # STRIKE TABLE
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 STRIKE STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        (
            "STRIKE | C-VOL | P-VOL | "
            "C-OI | P-OI | C-PREM | P-PREM"
        ),
        "────────────────────────────────────────"
    ]

    for _, row in (
        strike_table
        .sort_values("strike")
        .iterrows()
    ):

        report.append(
            f"${row['strike']:g} | "
            f"{row['call_volume']:,.0f} | "
            f"{row['put_volume']:,.0f} | "
            f"{row['call_oi']:,.0f} | "
            f"{row['put_oi']:,.0f} | "
            f"{fmt_money(row['call_premium'])} | "
            f"{fmt_money(row['put_premium'])}"
        )

    report.append("")

    # ========================================================
    # HIGH OI
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 HIGH OI STRIKES",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    for _, row in (
        strike_table
        .sort_values(
            "total_oi",
            ascending=False
        )
        .head(10)
        .iterrows()
    ):

        report.append(
            f"${row['strike']:g}"
            f" | Total OI "
            f"{row['total_oi']:,.0f}"
            f" | C "
            f"{row['call_oi']:,.0f}"
            f" / P "
            f"{row['put_oi']:,.0f}"
            f" | GEX "
            f"{fmt_money(row['net_gex'])}"
        )

    report.append("")

    # ========================================================
    # EXPIRATION
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📅 EXPIRATION STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if expiration_structure.empty:

        report.append(
            "⚠️ 만기 구조 데이터가 없습니다."
        )

    else:

        report += [
            (
                "DTE | EXPIRATION | C-OI | P-OI | "
                "TOTAL OI | OI %"
            ),
            "────────────────────────────────────────"
        ]

        for _, row in (
            expiration_structure
            .iterrows()
        ):

            dte = safe_float(
                row["DTE"]
            )

            report.append(
                f"{int(dte) if np.isfinite(dte) else 'N/A'} | "
                f"{row['expiration']} | "
                f"{row['call_oi']:,.0f} | "
                f"{row['put_oi']:,.0f} | "
                f"{row['total_oi']:,.0f} | "
                f"{fmt_pct(row['total_oi_concentration_pct'])}"
            )

    report.append("")

    # ========================================================
    # AUTO KEY STRIKE × EXPIRATION
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 KEY STRIKE × EXPIRATION OI",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if key_strike_summary.empty:

        report.append(
            "N/A"
        )

    else:

        for _, row in (
            key_strike_summary
            .iterrows()
        ):

            dte = safe_float(
                row["top_DTE"]
            )

            report += [
                f"💥 ${row['strike']:g}",
                (
                    f"   Total OI: "
                    f"{row['total_oi']:,.0f}"
                ),
                (
                    f"   CALL OI: "
                    f"{row['call_oi']:,.0f}"
                ),
                (
                    f"   PUT OI: "
                    f"{row['put_oi']:,.0f}"
                ),
                (
                    f"   🏆 최대 집중: "
                    f"{row['top_expiration']}"
                    f" | DTE "
                    f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                ),
                (
                    f"   OI: "
                    f"{row['top_expiration_total_oi']:,.0f}"
                    f" | "
                    f"{fmt_pct(row['top_expiration_oi_pct'])}"
                ),
                (
                    f"   C-OI: "
                    f"{row['top_expiration_call_oi']:,.0f}"
                    f" | "
                    f"P-OI: "
                    f"{row['top_expiration_put_oi']:,.0f}"
                ),
                ""
            ]

    # ========================================================
    # DETAILED EXPIRATION
    # ========================================================

    if not strike_expiration.empty:

        report += [
            "📌 상세 만기 분포",
            ""
        ]

        for strike in (
            key_strikes["strike"]
            .tolist()
            if not key_strikes.empty
            else []
        ):

            frame = strike_expiration[
                abs(
                    strike_expiration["strike"]
                    -
                    strike
                ) < 0.001
            ]

            if frame.empty:

                continue

            report.append(
                f"━━ ${strike:g} ━━"
            )

            for _, row in (
                frame
                .sort_values(
                    "total_oi",
                    ascending=False
                )
                .head(8)
                .iterrows()
            ):

                dte = safe_float(
                    row["DTE"]
                )

                report.append(
                    f"DTE "
                    f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                    f" | "
                    f"{row['expiration']}"
                    f" | C-OI "
                    f"{row['call_oi']:,.0f}"
                    f" | P-OI "
                    f"{row['put_oi']:,.0f}"
                    f" | TOTAL "
                    f"{row['total_oi']:,.0f}"
                    f" | "
                    f"{fmt_pct(row.get('total_oi_pct', np.nan))}"
                )

            report.append("")

    # ========================================================
    # TOP CONTRACTS
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 TOP OPTION CONTRACTS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if top_contracts.empty:

        report.append(
            "N/A"
        )

    else:

        for _, row in (
            top_contracts
            .head(20)
            .iterrows()
        ):

            dte = safe_float(
                row["DTE"]
            )

            volume_oi = safe_float(
                row["volume_oi"]
            )

            report.append(
                f"{row['option_type']:4s} "
                f"${row['strike']:g}"
                f" | DTE "
                f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                f" | Vol "
                f"{fmt_number(row['volume'])}"
                f" | OI "
                f"{fmt_number(row['openInterest'])}"
                f" | Premium "
                f"{fmt_money(row['premium_proxy'])}"
                f" | V/OI "
                f"{volume_oi:.2f}"
                if np.isfinite(volume_oi)
                else
                f"{row['option_type']:4s} "
                f"${row['strike']:g}"
                f" | DTE "
                f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                f" | Vol "
                f"{fmt_number(row['volume'])}"
                f" | OI "
                f"{fmt_number(row['openInterest'])}"
                f" | Premium "
                f"{fmt_money(row['premium_proxy'])}"
                f" | V/OI N/A"
            )

    # ========================================================
    # FINAL STRUCTURE
    # ========================================================

    report += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🧠 FINAL STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if cv > pv:

        report.append(
            "🟢 CALL Volume 우세"
        )

    elif pv > cv:

        report.append(
            "🔴 PUT Volume 우세"
        )

    if coi > poi:

        report.append(
            "🟢 CALL OI 우세"
        )

    elif poi > coi:

        report.append(
            "🔴 PUT OI 우세"
        )

    if cp > pp:

        report.append(
            "🟢 CALL Premium 우세"
        )

    elif pp > cp:

        report.append(
            "🔴 PUT Premium 우세"
        )

    if np.isfinite(net_gex):

        if net_gex > 0:

            report.append(
                "📈 Net GEX Proxy: POSITIVE"
            )

        elif net_gex < 0:

            report.append(
                "📉 Net GEX Proxy: NEGATIVE"
            )

        else:

            report.append(
                "🟡 Net GEX Proxy: NEUTRAL"
            )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    report += [
        "",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "⚠️ DATA LIMITATIONS",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "• Yahoo Finance 무료 옵션 데이터",
        "• 전체 Yahoo 만기/행사가 수집",
        "• 분석 범위: 지정 Strike / DTE",
        "• DTE 0 제외",
        "• Premium = 거래대금 Proxy",
        "• 실제 Buy/Sell 방향 확인 불가",
        "• OI만으로 Long/Short 확정 불가",
        "• Volume/OI = 당일 Volume ÷ 기존 OI",
        "• GEX = OI 기반 Proxy",
        "• Yahoo gamma 부족 시 GEX 정확도 제한",
        "• Key Strike = OI 중심 자동 탐지",
        "• Price Scenario = 현재가와 OI 집중도 기반 추정",
        "",
        (
            "Generated: "
            +
            started.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )
    ]

    return "\n".join(report)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    if (
        not token
        or not chat_id
    ):

        raise RuntimeError(
            "Telegram credentials not configured."
        )

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    max_length = 3900

    chunks = []

    remaining = text

    while len(remaining) > max_length:

        split_at = remaining.rfind(
            "\n",
            0,
            max_length
        )

        if split_at <= 0:

            split_at = max_length

        chunks.append(
            remaining[:split_at]
        )

        remaining = remaining[
            split_at:
        ]

    if remaining:

        chunks.append(
            remaining
        )

    print()
    print("=" * 70)
    print("TELEGRAM")
    print("=" * 70)

    sent = 0

    for index, chunk in enumerate(
        chunks,
        start=1
    ):

        payload = (
            urllib.parse.urlencode(
                {
                    "chat_id":
                        chat_id,
                    "text":
                        chunk
                }
            )
            .encode("utf-8")
        )

        request = (
            urllib.request.Request(
                url,
                data=payload,
                method="POST"
            )
        )

        try:

            with urllib.request.urlopen(
                request,
                timeout=30
            ) as response:

                result = (
                    response
                    .read()
                    .decode("utf-8")
                )

            if '"ok":true' not in result.lower():

                raise RuntimeError(
                    f"Telegram API rejected message: "
                    f"{result[:500]}"
                )

            sent += 1

            print(
                f"✅ Telegram "
                f"{index}/{len(chunks)} sent"
            )

        except Exception as exc:

            print(
                f"❌ Telegram "
                f"{index}/{len(chunks)} failed:"
            )

            print(
                repr(exc)
            )

            raise RuntimeError(
                "Telegram message delivery failed."
            ) from exc

    if sent != len(chunks):

        raise RuntimeError(
            "Telegram delivery incomplete."
        )

    print(
        f"✅ Telegram sent: "
        f"{sent} message(s)"
    )


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    data,
    strike_table,
    expiration_structure,
    strike_expiration,
    key_strike_summary,
    key_strikes,
    top_contracts,
    summary,
    report,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    files = {
        "contracts.csv":
            data,

        "strike_structure.csv":
            strike_table,

        "expiration_structure.csv":
            expiration_structure,

        "strike_expiration_structure.csv":
            strike_expiration,

        "key_strike_summary.csv":
            key_strike_summary,

        "auto_key_strikes.csv":
            key_strikes,

        "top_contracts.csv":
            top_contracts,

        "summary.csv":
            summary
    }

    saved_files = []

    print()
    print("=" * 70)
    print("SAVE OUTPUTS")
    print("=" * 70)

    for filename, dataframe in (
        files.items()
    ):

        path = os.path.join(
            output_dir,
            filename
        )

        if dataframe is None:

            dataframe = pd.DataFrame()

        dataframe.to_csv(
            path,
            index=False
        )

        saved_files.append(
            path
        )

        print(
            f"💾 {filename:40s}"
            f" rows={len(dataframe):,}"
        )

    report_path = os.path.join(
        output_dir,
        "report.md"
    )

    with open(
        report_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    saved_files.append(
        report_path
    )

    print(
        f"💾 {'report.md':40s}"
        f" chars={len(report):,}"
    )

    manifest_path = os.path.join(
        output_dir,
        "save_manifest.txt"
    )

    with open(
        manifest_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "OPTION STRUCTURE SAVE MANIFEST\n"
        )

        file.write(
            "========================================\n"
        )

        file.write(
            f"Generated UTC: "
            f"{datetime.now(timezone.utc)}\n"
        )

        file.write(
            f"Output Directory: "
            f"{os.path.abspath(output_dir)}\n\n"
        )

        for path in saved_files:

            file.write(
                f"{os.path.basename(path)}\n"
            )

    print()
    print(
        "VERIFY SAVED FILES"
    )

    all_ok = True

    for path in (
        saved_files
        +
        [manifest_path]
    ):

        exists = os.path.isfile(
            path
        )

        size = (
            os.path.getsize(path)
            if exists
            else 0
        )

        if exists and size > 0:

            print(
                f"✅ "
                f"{os.path.basename(path)} "
                f"({size:,} bytes)"
            )

        else:

            print(
                f"❌ SAVE FAILED: "
                f"{path}"
            )

            all_ok = False

    print()

    if all_ok:

        print(
            "✅ ALL OUTPUT FILES SAVED"
        )

    else:

        raise RuntimeError(
            "One or more output files failed to save."
        )


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    symbol = (
        args.symbol
        .upper()
        .strip()
    )

    manual_price = (
        args.price
        if args.price
        else None
    )

    min_strike = args.min_strike
    max_strike = args.max_strike
    max_dte = args.max_dte
    output_dir = args.output

    started = datetime.now(
        timezone.utc
    )

    # ========================================================
    # HEADER
    # ========================================================

    print()
    print("=" * 70)
    print(
        "🔥 FULL OPTION STRUCTURE SCANNER"
    )
    print("=" * 70)

    print(
        f"SYMBOL       : {symbol}"
    )

    print(
        f"ANALYSIS     : "
        f"${min_strike:g}"
        f"~"
        f"${max_strike:g}"
    )

    print(
        f"DTE ANALYSIS : "
        f"1~{max_dte}"
    )

    print(
        f"AUTO KEY     : "
        f"TOP {KEY_STRIKE_COUNT}"
    )

    print(
        f"KEY DISTANCE : "
        f"{KEY_STRIKE_MAX_DISTANCE * 100:.0f}%"
    )

    print(
        f"BAR WIDTH    : "
        f"{BAR_WIDTH} MAX"
    )

    print(
        f"OUTPUT       : "
        f"{os.path.abspath(output_dir)}"
    )

    print("=" * 70)

    # ========================================================
    # 1. FULL FETCH
    # ========================================================

    raw, spot = fetch_options(
        symbol,
        manual_price
    )

    # ========================================================
    # 2. NORMALIZE
    # ========================================================

    data = normalize(
        raw
    )

    # ========================================================
    # 3. FILTER
    # ========================================================

    data = apply_filters(
        data,
        min_strike,
        max_strike,
        max_dte
    )

    if data.empty:

        raise RuntimeError(
            "No options remain after filtering."
        )

    # ========================================================
    # 4. METRICS
    # ========================================================

    data = calculate_metrics(
        data,
        spot
    )

    # ========================================================
    # 5. STRIKE STRUCTURE
    # ========================================================

    strike_table = (
        build_strike_table(
            data
        )
    )

    if strike_table.empty:

        raise RuntimeError(
            "Strike structure is empty."
        )

    # ========================================================
    # 6. ⭐ AUTO KEY STRIKE
    # ========================================================

    key_strikes = (
        detect_key_strikes(
            strike_table,
            spot,
            KEY_STRIKE_COUNT
        )
    )

    print()
    print("=" * 70)
    print("AUTO KEY STRIKE DETECTION")
    print("=" * 70)

    if key_strikes.empty:

        print(
            "⚠️ No automatic Key Strike detected."
        )

    else:

        for _, row in (
            key_strikes.iterrows()
        ):

            print(
                f"${row['strike']:g}"
                f" | {row['key_type']}"
                f" | OI "
                f"{row['total_oi']:,.0f}"
                f" | Score "
                f"{row['key_score']:.2f}"
            )

    # ========================================================
    # 7. PRICE SCENARIO
    # ========================================================

    price_scenarios = (
        build_price_scenario(
            key_strikes,
            spot
        )
    )

    # ========================================================
    # 8. EXPIRATION
    # ========================================================

    expiration_structure = (
        build_expiration_structure(
            data
        )
    )

    # ========================================================
    # 9. STRIKE × EXPIRATION
    # ========================================================

    if not key_strikes.empty:

        focus_strikes = (
            key_strikes[
                "strike"
            ]
            .tolist()
        )

    else:

        focus_strikes = []

    strike_expiration = (
        build_strike_expiration_structure(
            data,
            focus_strikes
        )
    )

    # ========================================================
    # 10. KEY STRIKE SUMMARY
    # ========================================================

    key_strike_summary = (
        build_key_strike_summary(
            strike_expiration
        )
    )

    # ========================================================
    # 11. TOP CONTRACTS
    # ========================================================

    top_contracts = (
        build_top_contracts(
            data
        )
    )

    # ========================================================
    # 12. SUMMARY
    # ========================================================

    calls = data[
        data["option_type"]
        == "CALL"
    ]

    puts = data[
        data["option_type"]
        == "PUT"
    ]

    cv = (
        calls["volume"]
        .fillna(0)
        .sum()
    )

    pv = (
        puts["volume"]
        .fillna(0)
        .sum()
    )

    coi = (
        calls["openInterest"]
        .fillna(0)
        .sum()
    )

    poi = (
        puts["openInterest"]
        .fillna(0)
        .sum()
    )

    cp = (
        calls["premium_proxy"]
        .fillna(0)
        .sum()
    )

    pp = (
        puts["premium_proxy"]
        .fillna(0)
        .sum()
    )

    tgex = calls["gex"].sum(
        min_count=1
    )

    pgex = puts["gex"].sum(
        min_count=1
    )

    if (
        np.isfinite(tgex)
        and np.isfinite(pgex)
    ):

        net_gex = tgex + pgex

    elif np.isfinite(tgex):

        net_gex = tgex

    elif np.isfinite(pgex):

        net_gex = pgex

    else:

        net_gex = np.nan

    summary = pd.DataFrame(
        [
            {
                "symbol": symbol,
                "spot": spot,
                "min_strike": min_strike,
                "max_strike": max_strike,
                "max_dte": max_dte,
                "rows": len(data),

                "call_volume": cv,
                "put_volume": pv,

                "call_oi": coi,
                "put_oi": poi,

                "call_premium": cp,
                "put_premium": pp,

                "call_volume_ratio":
                    cv / (cv + pv) * 100
                    if cv + pv > 0
                    else np.nan,

                "call_oi_ratio":
                    coi / (coi + poi) * 100
                    if coi + poi > 0
                    else np.nan,

                "call_volume_oi":
                    cv / coi
                    if coi > 0
                    else np.nan,

                "put_volume_oi":
                    pv / poi
                    if poi > 0
                    else np.nan,

                "call_gex": tgex,
                "put_gex": pgex,
                "net_gex": net_gex,

                "auto_key_strikes":
                    ",".join(
                        [
                            str(
                                int(x)
                                if float(x).is_integer()
                                else x
                            )
                            for x in focus_strikes
                        ]
                    )
            }
        ]
    )

    # ========================================================
    # 13. REPORT
    # ========================================================

    report = build_report(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        top_contracts=top_contracts,
        key_strikes=key_strikes,
        price_scenarios=price_scenarios,
        spot=spot,
        symbol=symbol,
        min_strike=min_strike,
        max_strike=max_strike,
        max_dte=max_dte,
        started=started
    )

    # ========================================================
    # 14. SAVE
    # ========================================================

    save_outputs(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        key_strikes=key_strikes,
        top_contracts=top_contracts,
        summary=summary,
        report=report,
        output_dir=output_dir
    )

    # ========================================================
    # 15. PRINT
    # ========================================================

    print()
    print(report)

    # ========================================================
    # 16. TELEGRAM
    #
    # 실패하면 RuntimeError 발생
    # → GitHub Actions workflow도 실패
    # ========================================================

    send_telegram(
        report
    )

    # ========================================================
    # COMPLETE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "✅ SCAN COMPLETE"
    )
    print("=" * 70)

    print(
        f"📁 Saved to: "
        f"{os.path.abspath(output_dir)}"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print(
            "Scanner interrupted."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 70)
        print(
            "❌ SCANNER FAILED"
        )
        print("=" * 70)

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: "
            f"{repr(exc)}"
        )

        raise
