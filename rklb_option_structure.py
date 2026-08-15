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
# FOCUS STRIKES
# ============================================================

FOCUS_STRIKES = [
    80,
    85,
    90,
    95,
    100
]


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

    value = safe_float(
        value
    )

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

    value = safe_float(
        value
    )

    if not np.isfinite(value):

        return "N/A"

    return f"{value:,.0f}"


def fmt_pct(value):

    value = safe_float(
        value
    )

    if not np.isfinite(value):

        return "N/A"

    return f"{value:.1f}%"


def fmt_iv(value):

    value = safe_float(
        value
    )

    if not np.isfinite(value):

        return "N/A"

    if value < 2:

        value *= 100

    return f"{value:.1f}%"


def safe_sum(series):

    if series is None:

        return 0.0

    try:

        return safe_float(
            series.fillna(0).sum()
        ) or 0.0

    except Exception:

        return 0.0


# ============================================================
# DYNAMIC CALL / PUT BAR
# ============================================================

def make_dynamic_dual_bar(
    call_value,
    put_value,
    reference_total,
    max_width=BAR_WIDTH
):

    call_value = safe_float(
        call_value
    )

    put_value = safe_float(
        put_value
    )

    reference_total = safe_float(
        reference_total
    )

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

    total = (
        call_value
        +
        put_value
    )

    if total <= 0:

        return "·"

    if reference_total > 0:

        scale_ratio = (
            total
            /
            reference_total
        )

    else:

        scale_ratio = 1.0

    scale_ratio = max(
        0.0,
        min(
            scale_ratio,
            1.0
        )
    )

    bar_length = int(
        round(
            scale_ratio
            *
            max_width
        )
    )

    if bar_length < BAR_MIN_WIDTH:

        bar_length = BAR_MIN_WIDTH

    bar_length = min(
        bar_length,
        max_width
    )

    call_ratio = (
        call_value
        /
        total
    )

    call_width = int(
        round(
            call_ratio
            *
            bar_length
        )
    )

    put_width = (
        bar_length
        -
        call_width
    )

    if (
        call_value > 0
        and put_value > 0
    ):

        if call_width <= 0:

            call_width = 1

            put_width = (
                bar_length
                -
                1
            )

        elif put_width <= 0:

            put_width = 1

            call_width = (
                bar_length
                -
                1
            )

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
            bar_length
            -
            call_width
        )
    )

    return (
        "🟩" * call_width
        +
        "🟥" * put_width
    )


# ============================================================
# BACKWARD COMPATIBILITY
# ============================================================

def make_dual_bar(
    call_value,
    put_value,
    width=BAR_WIDTH
):

    call_value = safe_float(
        call_value
    )

    put_value = safe_float(
        put_value
    )

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

    total = (
        call_value
        +
        put_value
    )

    return make_dynamic_dual_bar(
        call_value,
        put_value,
        total,
        max_width=width
    )


# ============================================================
# BAR LINE
# ============================================================

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

    call_value = safe_float(
        call_value
    )

    put_value = safe_float(
        put_value
    )

    if not np.isfinite(
        call_value
    ):

        call_value = 0

    if not np.isfinite(
        put_value
    ):

        put_value = 0

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
# PRICE
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

def calculate_dte(
    expiration
):

    try:

        expiry = pd.Timestamp(
            expiration
        ).date()

        return (
            expiry
            -
            market_today()
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

    volume = safe_float(
        volume
    )

    bid = safe_float(
        bid
    )

    ask = safe_float(
        ask
    )

    last_price = safe_float(
        last_price
    )

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
            bid
            +
            ask
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
        *
        mid
        *
        100
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

    gamma = safe_float(
        gamma
    )

    open_interest = safe_float(
        open_interest
    )

    spot = safe_float(
        spot
    )

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
        *
        open_interest
        *
        100
        *
        spot
        *
        spot
        *
        0.01
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

            frame["option_type"] = (
                "CALL"
            )

            frame["expiration"] = (
                expiration
            )

            rows.append(
                frame
            )

        if not puts.empty:

            frame = puts.copy()

            frame["option_type"] = (
                "PUT"
            )

            frame["expiration"] = (
                expiration
            )

            rows.append(
                frame
            )

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

def normalize(
    data
):

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

    raw_count = len(
        data
    )

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

def build_strike_table(
    data
):

    if data.empty:

        return pd.DataFrame(
            columns=[
                "strike",
                "call_volume",
                "put_volume",
                "total_volume",
                "call_oi",
                "put_oi",
                "total_oi",
                "call_premium",
                "put_premium",
                "total_premium",
                "call_volume_oi",
                "put_volume_oi",
                "call_volume_ratio",
                "call_oi_ratio",
                "call_premium_ratio",
                "call_gex",
                "put_gex",
                "net_gex"
            ]
        )

    rows = []

    for strike, frame in (
        data.groupby(
            "strike"
        )
    ):

        calls = frame[
            frame["option_type"]
            ==
            "CALL"
        ]

        puts = frame[
            frame["option_type"]
            ==
            "PUT"
        ]

        cv = safe_sum(
            calls["volume"]
        )

        pv = safe_sum(
            puts["volume"]
        )

        coi = safe_sum(
            calls["openInterest"]
        )

        poi = safe_sum(
            puts["openInterest"]
        )

        cp = safe_sum(
            calls["premium_proxy"]
        )

        pp = safe_sum(
            puts["premium_proxy"]
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

    return pd.DataFrame(
        rows
    )


# ============================================================
# DYNAMIC BAR STRUCTURE
# ============================================================

def build_bar_structure(
    strike_table
):

    if (
        strike_table is None
        or strike_table.empty
    ):

        return [
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
            "📊 CALL / PUT BAR STRUCTURE",
            "🟩 CALL   🟥 PUT",
            "⚠️ 표시할 Strike 데이터가 없습니다.",
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
        ]

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

    oi_totals = (
        sorted_table[
            "call_oi"
        ].fillna(0)
        +
        sorted_table[
            "put_oi"
        ].fillna(0)
    )

    volume_totals = (
        sorted_table[
            "call_volume"
        ].fillna(0)
        +
        sorted_table[
            "put_volume"
        ].fillna(0)
    )

    premium_totals = (
        sorted_table[
            "call_premium"
        ].fillna(0)
        +
        sorted_table[
            "put_premium"
        ].fillna(0)
    )

    oi_reference = (
        oi_totals.max()
        if not oi_totals.empty
        else 0
    )

    volume_reference = (
        volume_totals.max()
        if not volume_totals.empty
        else 0
    )

    premium_reference = (
        premium_totals.max()
        if not premium_totals.empty
        else 0
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 CALL / PUT BAR STRUCTURE"
    )

    lines.append(
        "🟩 CALL   🟥 PUT"
    )

    lines.append(
        f"📏 BAR MAX: {BAR_WIDTH}칸"
    )

    lines.append(
        "📐 규모가 작으면 BAR도 짧게 표시"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

    lines.append(
        "🟢 OI STRUCTURE"
    )

    lines.append("")

    for _, row in (
        sorted_table.iterrows()
    ):

        strike = row["strike"]

        call_oi = safe_float(
            row["call_oi"]
        )

        put_oi = safe_float(
            row["put_oi"]
        )

        lines.append(
            f"🎯 ${strike:g}   "
            +
            make_dual_bar_line(
                call_oi,
                put_oi,
                oi_reference
            )
        )

    lines.append("")

    # --------------------------------------------------------
    # VOLUME
    # --------------------------------------------------------

    lines.append(
        "🔥 VOLUME STRUCTURE"
    )

    lines.append("")

    for _, row in (
        sorted_table.iterrows()
    ):

        strike = row["strike"]

        call_volume = safe_float(
            row["call_volume"]
        )

        put_volume = safe_float(
            row["put_volume"]
        )

        lines.append(
            f"🎯 ${strike:g}   "
            +
            make_dual_bar_line(
                call_volume,
                put_volume,
                volume_reference
            )
        )

    lines.append("")

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    lines.append(
        "💰 PREMIUM STRUCTURE"
    )

    lines.append("")

    for _, row in (
        sorted_table.iterrows()
    ):

        strike = row["strike"]

        call_premium = safe_float(
            row["call_premium"]
        )

        put_premium = safe_float(
            row["put_premium"]
        )

        if not np.isfinite(
            call_premium
        ):

            call_premium = 0

        if not np.isfinite(
            put_premium
        ):

            put_premium = 0

        bar = make_dynamic_dual_bar(
            call_premium,
            put_premium,
            premium_reference
        )

        lines.append(
            f"🎯 ${strike:g}   "
            f"{bar} "
            f"C {fmt_money(call_premium)} "
            f"/ "
            f"P {fmt_money(put_premium)}"
        )

    return lines


# ============================================================
# EXPIRATION STRUCTURE
# ============================================================

def build_expiration_structure(
    data
):

    # ========================================================
    # IMPORTANT:
    # 빈 데이터 방어
    # ========================================================

    empty_columns = [
        "expiration",
        "DTE",
        "call_volume",
        "put_volume",
        "total_volume",
        "call_oi",
        "put_oi",
        "total_oi",
        "call_premium",
        "put_premium",
        "total_premium",
        "call_volume_ratio",
        "call_oi_ratio",
        "total_oi_concentration_pct"
    ]

    if (
        data is None
        or data.empty
    ):

        return pd.DataFrame(
            columns=empty_columns
        )

    if (
        "expiration" not in data.columns
        or "option_type" not in data.columns
    ):

        return pd.DataFrame(
            columns=empty_columns
        )

    rows = []

    for expiration, frame in (
        data.groupby(
            "expiration"
        )
    ):

        if frame.empty:

            continue

        calls = frame[
            frame["option_type"]
            ==
            "CALL"
        ]

        puts = frame[
            frame["option_type"]
            ==
            "PUT"
        ]

        cv = safe_sum(
            calls["volume"]
        )

        pv = safe_sum(
            puts["volume"]
        )

        coi = safe_sum(
            calls["openInterest"]
        )

        poi = safe_sum(
            puts["openInterest"]
        )

        cp = safe_sum(
            calls["premium_proxy"]
        )

        pp = safe_sum(
            puts["premium_proxy"]
        )

        rows.append(
            {
                "expiration":
                    expiration,

                "DTE":
                    calculate_dte(
                        expiration
                    ),

                "call_volume":
                    cv,

                "put_volume":
                    pv,

                "total_volume":
                    cv + pv,

                "call_oi":
                    coi,

                "put_oi":
                    poi,

                "total_oi":
                    coi + poi,

                "call_premium":
                    cp,

                "put_premium":
                    pp,

                "total_premium":
                    cp + pp,

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

    result = pd.DataFrame(
        rows
    )

    # ========================================================
    # IMPORTANT:
    # groupby 결과가 비어있는 경우 방어
    # ========================================================

    if result.empty:

        return pd.DataFrame(
            columns=empty_columns
        )

    total_oi = safe_float(
        result["total_oi"].sum()
    )

    if (
        np.isfinite(total_oi)
        and total_oi > 0
    ):

        result[
            "total_oi_concentration_pct"
        ] = (
            result["total_oi"]
            /
            total_oi
            *
            100
        )

    else:

        result[
            "total_oi_concentration_pct"
        ] = np.nan

    return (
        result
        .sort_values(
            "DTE"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# STRIKE × EXPIRATION
# ============================================================

def build_strike_expiration_structure(
    data,
    focus_strikes
):

    empty_columns = [
        "strike",
        "expiration",
        "DTE",
        "call_volume",
        "put_volume",
        "total_volume",
        "call_oi",
        "put_oi",
        "total_oi",
        "call_premium",
        "put_premium",
        "total_premium",
        "total_oi_pct",
        "call_oi_pct",
        "put_oi_pct"
    ]

    if (
        data is None
        or data.empty
    ):

        return pd.DataFrame(
            columns=empty_columns
        )

    rows = []

    for target in focus_strikes:

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

            if frame.empty:

                continue

            calls = frame[
                frame["option_type"]
                ==
                "CALL"
            ]

            puts = frame[
                frame["option_type"]
                ==
                "PUT"
            ]

            cv = safe_sum(
                calls["volume"]
            )

            pv = safe_sum(
                puts["volume"]
            )

            coi = safe_sum(
                calls["openInterest"]
            )

            poi = safe_sum(
                puts["openInterest"]
            )

            cp = safe_sum(
                calls["premium_proxy"]
            )

            pp = safe_sum(
                puts["premium_proxy"]
            )

            rows.append(
                {
                    "strike":
                        target,

                    "expiration":
                        expiration,

                    "DTE":
                        calculate_dte(
                            expiration
                        ),

                    "call_volume":
                        cv,

                    "put_volume":
                        pv,

                    "total_volume":
                        cv + pv,

                    "call_oi":
                        coi,

                    "put_oi":
                        poi,

                    "total_oi":
                        coi + poi,

                    "call_premium":
                        cp,

                    "put_premium":
                        pp,

                    "total_premium":
                        cp + pp
                }
            )

    result = pd.DataFrame(
        rows
    )

    if result.empty:

        return pd.DataFrame(
            columns=empty_columns
        )

    for strike in (
        result["strike"].dropna().unique()
    ):

        mask = (
            result["strike"]
            ==
            strike
        )

        total_oi = safe_float(
            result.loc[
                mask,
                "total_oi"
            ].sum()
        )

        call_oi = safe_float(
            result.loc[
                mask,
                "call_oi"
            ].sum()
        )

        put_oi = safe_float(
            result.loc[
                mask,
                "put_oi"
            ].sum()
        )

        if (
            np.isfinite(total_oi)
            and total_oi > 0
        ):

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

        if (
            np.isfinite(call_oi)
            and call_oi > 0
        ):

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

        if (
            np.isfinite(put_oi)
            and put_oi > 0
        ):

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
        .reset_index(
            drop=True
        )
    )


# ============================================================
# KEY STRIKE SUMMARY
# ============================================================

def build_key_strike_summary(
    strike_expiration
):

    empty_columns = [
        "strike",
        "total_oi",
        "call_oi",
        "put_oi",
        "top_expiration",
        "top_DTE",
        "top_expiration_total_oi",
        "top_expiration_call_oi",
        "top_expiration_put_oi",
        "top_expiration_oi_pct"
    ]

    if (
        strike_expiration is None
        or strike_expiration.empty
    ):

        return pd.DataFrame(
            columns=empty_columns
        )

    rows = []

    for strike, frame in (
        strike_expiration.groupby(
            "strike"
        )
    ):

        if frame.empty:

            continue

        frame = frame.sort_values(
            "total_oi",
            ascending=False
        )

        top = frame.iloc[0]

        rows.append(
            {
                "strike":
                    strike,

                "total_oi":
                    safe_sum(
                        frame["total_oi"]
                    ),

                "call_oi":
                    safe_sum(
                        frame["call_oi"]
                    ),

                "put_oi":
                    safe_sum(
                        frame["put_oi"]
                    ),

                "top_expiration":
                    top.get(
                        "expiration",
                        "N/A"
                    ),

                "top_DTE":
                    top.get(
                        "DTE",
                        np.nan
                    ),

                "top_expiration_total_oi":
                    top.get(
                        "total_oi",
                        0
                    ),

                "top_expiration_call_oi":
                    top.get(
                        "call_oi",
                        0
                    ),

                "top_expiration_put_oi":
                    top.get(
                        "put_oi",
                        0
                    ),

                "top_expiration_oi_pct":
                    top.get(
                        "total_oi_pct",
                        np.nan
                    )
            }
        )

    if not rows:

        return pd.DataFrame(
            columns=empty_columns
        )

    return (
        pd.DataFrame(
            rows
        )
        .sort_values(
            "strike"
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(
    data
):

    if (
        data is None
        or data.empty
    ):

        result = data.copy()

        result["importance"] = (
            pd.Series(
                dtype=float
            )
        )

        return result

    result = data.copy()

    result["importance"] = (
        np.log1p(
            result[
                "premium_proxy"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
        +
        np.log1p(
            result[
                "volume"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
        +
        np.log1p(
            result[
                "openInterest"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
        +
        np.log1p(
            result[
                "gex"
            ]
            .fillna(0)
            .abs()
            .clip(
                lower=0
            )
        )
    )

    return (
        result
        .sort_values(
            "importance",
            ascending=False
        )
        .head(50)
        .reset_index(
            drop=True
        )
    )


# ============================================================
# WALL
# ============================================================

def find_wall(
    strike_table,
    spot,
    option_type
):

    if (
        strike_table is None
        or strike_table.empty
    ):

        return None

    spot = safe_float(
        spot
    )

    if (
        not np.isfinite(spot)
        or spot <= 0
    ):

        return None

    if option_type == "CALL":

        candidates = strike_table[
            strike_table[
                "strike"
            ]
            >=
            spot
        ].copy()

        candidates["oi"] = (
            candidates[
                "call_oi"
            ]
        )

        candidates["gex_abs"] = (
            candidates[
                "call_gex"
            ].abs()
        )

        candidates["volume"] = (
            candidates[
                "call_volume"
            ]
        )

    else:

        candidates = strike_table[
            strike_table[
                "strike"
            ]
            <=
            spot
        ].copy()

        candidates["oi"] = (
            candidates[
                "put_oi"
            ]
        )

        candidates["gex_abs"] = (
            candidates[
                "put_gex"
            ].abs()
        )

        candidates["volume"] = (
            candidates[
                "put_volume"
            ]
        )

    if candidates.empty:

        return None

    candidates["distance"] = (
        (
            candidates[
                "strike"
            ]
            -
            spot
        ).abs()
        /
        spot
    )

    candidates = candidates[
        candidates[
            "distance"
        ]
        <=
        0.20
    ].copy()

    if candidates.empty:

        return None

    candidates["score"] = (
        np.log1p(
            candidates[
                "oi"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
        +
        np.log1p(
            candidates[
                "gex_abs"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
        +
        0.25
        *
        np.log1p(
            candidates[
                "volume"
            ]
            .fillna(0)
            .clip(
                lower=0
            )
        )
    )

    candidates["score"] += (
        3
        /
        (
            1
            +
            candidates[
                "distance"
            ]
            *
            20
        )
    )

    if candidates.empty:

        return None

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
    spot,
    symbol,
    min_strike,
    max_strike,
    max_dte,
    started
):

    # ========================================================
    # EMPTY DATA DEFENSE
    # ========================================================

    if data is None:

        data = pd.DataFrame()

    if strike_table is None:

        strike_table = pd.DataFrame()

    if expiration_structure is None:

        expiration_structure = pd.DataFrame()

    if strike_expiration is None:

        strike_expiration = pd.DataFrame()

    if key_strike_summary is None:

        key_strike_summary = pd.DataFrame()

    if top_contracts is None:

        top_contracts = pd.DataFrame()

    if data.empty:

        return "\n".join(
            [
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                f"🔥 {symbol} OPTION STRUCTURE",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "",
                f"💰 현재가: ${spot:.2f}",
                (
                    f"🎯 분석 Strike: "
                    f"${min_strike:g} ~ ${max_strike:g}"
                ),
                f"📅 분석 DTE: 1 ~ {max_dte}",
                "📊 분석 행수: 0",
                "",
                "❌ 분석할 옵션 데이터가 없습니다.",
                "",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "⚠️ DATA LIMITATIONS",
                "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
                "• Yahoo Finance 무료 옵션 데이터",
                "• 현재 필터 조건에서 데이터가 없음",
                "",
                (
                    "Generated: "
                    +
                    started.strftime(
                        "%Y-%m-%d %H:%M:%S UTC"
                    )
                )
            ]
        )

    calls = data[
        data["option_type"]
        ==
        "CALL"
    ]

    puts = data[
        data["option_type"]
        ==
        "PUT"
    ]

    cv = safe_sum(
        calls["volume"]
    )

    pv = safe_sum(
        puts["volume"]
    )

    coi = safe_sum(
        calls["openInterest"]
    )

    poi = safe_sum(
        puts["openInterest"]
    )

    cp = safe_sum(
        calls["premium_proxy"]
    )

    pp = safe_sum(
        puts["premium_proxy"]
    )

    total_volume = cv + pv
    total_oi = coi + poi
    total_premium = cp + pp

    cv_ratio = (
        cv
        /
        total_volume
        *
        100
        if total_volume > 0
        else np.nan
    )

    coi_ratio = (
        coi
        /
        total_oi
        *
        100
        if total_oi > 0
        else np.nan
    )

    cp_ratio = (
        cp
        /
        total_premium
        *
        100
        if total_premium > 0
        else np.nan
    )

    cv_oi = (
        cv
        /
        coi
        if coi > 0
        else np.nan
    )

    pv_oi = (
        pv
        /
        poi
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

        net_gex = (
            tgex + pgex
        )

    elif np.isfinite(tgex):

        net_gex = tgex

    elif np.isfinite(pgex):

        net_gex = pgex

    else:

        net_gex = np.nan

    temp = data.copy()

    temp["atm_distance"] = (
        temp["strike"]
        -
        spot
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
        (
            f"CALL Volume Ratio: "
            f"{fmt_pct(cv_ratio)}"
        ),
        "",
        f"CALL OI: {coi:,.0f}",
        f"PUT OI : {poi:,.0f}",
        (
            f"CALL OI Ratio: "
            f"{fmt_pct(coi_ratio)}"
        ),
        "",
        (
            f"CALL Premium Proxy: "
            f"{fmt_money(cp)}"
        ),
        (
            f"PUT Premium Proxy : "
            f"{fmt_money(pp)}"
        ),
        (
            f"CALL Premium Ratio: "
            f"{fmt_pct(cp_ratio)}"
        ),
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
            else
            "CALL Volume/OI: N/A"
        ),
        (
            f"PUT Volume/OI : "
            f"{pv_oi:.3f}"
            if np.isfinite(pv_oi)
            else
            "PUT Volume/OI : N/A"
        ),
        ""
    ]

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
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    ]

    if strike_table.empty:

        report.append(
            "N/A"
        )

    else:

        report += [
            (
                "STRIKE | C-VOL | P-VOL | "
                "C-OI | P-OI | C-PREM | P-PREM"
            ),
            "────────────────────────────────────────"
        ]

        for _, row in (
            strike_table
            .sort_values(
                "strike"
            )
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

    if strike_table.empty:

        report.append(
            "N/A"
        )

    else:

        high_oi = (
            strike_table
            .sort_values(
                "total_oi",
                ascending=False
            )
            .head(10)
        )

        if high_oi.empty:

            report.append(
                "N/A"
            )

        else:

            for _, row in (
                high_oi.iterrows()
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

    # ========================================================
    # IMPORTANT:
    # expiration_structure 빈 데이터 방어
    # ========================================================

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
            expiration_structure.iterrows()
        ):

            dte = safe_float(
                row.get(
                    "DTE",
                    np.nan
                )
            )

            report.append(
                f"{int(dte) if np.isfinite(dte) else 'N/A'} | "
                f"{row.get('expiration', 'N/A')} | "
                f"{safe_float(row.get('call_oi', 0)):,.0f} | "
                f"{safe_float(row.get('put_oi', 0)):,.0f} | "
                f"{safe_float(row.get('total_oi', 0)):,.0f} | "
                f"{fmt_pct(row.get('total_oi_concentration_pct', np.nan))}"
            )

    report.append("")

    # ========================================================
    # KEY STRIKE
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🔥 KEY STRIKE × EXPIRATION OI",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 $80 / $85 / $90 / $95 / $100",
        ""
    ]

    if key_strike_summary.empty:

        report.append(
            "⚠️ 지정 Focus Strike 데이터가 없습니다."
        )

    else:

        for _, row in (
            key_strike_summary.iterrows()
        ):

            dte = safe_float(
                row.get(
                    "top_DTE",
                    np.nan
                )
            )

            report += [
                f"💥 ${safe_float(row.get('strike', np.nan)):g}",
                (
                    f"   Total OI: "
                    f"{safe_float(row.get('total_oi', 0)):,.0f}"
                ),
                (
                    f"   CALL OI:  "
                    f"{safe_float(row.get('call_oi', 0)):,.0f}"
                ),
                (
                    f"   PUT OI :  "
                    f"{safe_float(row.get('put_oi', 0)):,.0f}"
                ),
                (
                    f"   🏆 최대 집중: "
                    f"{row.get('top_expiration', 'N/A')} "
                    f"| DTE "
                    f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                ),
                (
                    f"   OI: "
                    f"{safe_float(row.get('top_expiration_total_oi', 0)):,.0f}"
                    f" | "
                    f"{fmt_pct(row.get('top_expiration_oi_pct', np.nan))}"
                ),
                (
                    f"   C-OI: "
                    f"{safe_float(row.get('top_expiration_call_oi', 0)):,.0f}"
                    f" | P-OI: "
                    f"{safe_float(row.get('top_expiration_put_oi', 0)):,.0f}"
                ),
                ""
            ]

    # ========================================================
    # DETAILED EXPIRATION
    # ========================================================

    if (
        strike_expiration is not None
        and not strike_expiration.empty
    ):

        report += [
            "📌 상세 만기 분포",
            ""
        ]

        for strike in FOCUS_STRIKES:

            frame = strike_expiration[
                abs(
                    strike_expiration[
                        "strike"
                    ]
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
                    row.get(
                        "DTE",
                        np.nan
                    )
                )

                report.append(
                    f"DTE "
                    f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                    f" | "
                    f"{row.get('expiration', 'N/A')}"
                    f" | C-OI "
                    f"{safe_float(row.get('call_oi', 0)):,.0f}"
                    f" | P-OI "
                    f"{safe_float(row.get('put_oi', 0)):,.0f}"
                    f" | TOTAL "
                    f"{safe_float(row.get('total_oi', 0)):,.0f}"
                    f" | "
                    f"{fmt_pct(row.get('total_oi_pct', np.nan))}"
                )

            report.append("")

    else:

        report += [
            "📌 상세 만기 분포",
            "⚠️ Focus Strike의 만기별 데이터가 없습니다.",
            ""
        ]

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
                row.get(
                    "DTE",
                    np.nan
                )
            )

            volume = safe_float(
                row.get(
                    "volume",
                    np.nan
                )
            )

            oi = safe_float(
                row.get(
                    "openInterest",
                    np.nan
                )
            )

            premium = safe_float(
                row.get(
                    "premium_proxy",
                    np.nan
                )
            )

            volume_oi = safe_float(
                row.get(
                    "volume_oi",
                    np.nan
                )
            )

            option_type = (
                str(
                    row.get(
                        "option_type",
                        "N/A"
                    )
                )
            )

            strike = safe_float(
                row.get(
                    "strike",
                    np.nan
                )
            )

            base = (
                f"{option_type:4s} "
                f"${strike:g}"
                f" | DTE "
                f"{int(dte) if np.isfinite(dte) else 'N/A'}"
                f" | Vol "
                f"{fmt_number(volume)}"
                f" | OI "
                f"{fmt_number(oi)}"
                f" | Premium "
                f"{fmt_money(premium)}"
            )

            if np.isfinite(
                volume_oi
            ):

                report.append(
                    base
                    +
                    f" | V/OI {volume_oi:.2f}"
                )

            else:

                report.append(
                    base
                    +
                    " | V/OI N/A"
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

    else:

        report.append(
            "🟡 CALL/PUT Volume 동률"
        )

    if coi > poi:

        report.append(
            "🟢 CALL OI 우세"
        )

    elif poi > coi:

        report.append(
            "🔴 PUT OI 우세"
        )

    else:

        report.append(
            "🟡 CALL/PUT OI 동률"
        )

    if cp > pp:

        report.append(
            "🟢 CALL Premium 우세"
        )

    elif pp > cp:

        report.append(
            "🔴 PUT Premium 우세"
        )

    else:

        report.append(
            "🟡 CALL/PUT Premium 동률"
        )

    if np.isfinite(
        net_gex
    ):

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

    else:

        report.append(
            "⚪ Net GEX Proxy: N/A"
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
        "• 수집: 전체 Yahoo 만기/행사가",
        "• 분석: 지정 Strike/DTE 범위",
        "• DTE 0 제외",
        "• Premium = 거래대금 Proxy",
        "• 실제 Buy/Sell 방향 확인 불가",
        "• OI만으로 Long/Short 확정 불가",
        "• Volume/OI = 당일 Volume ÷ 기존 OI",
        "• GEX = OI 기반 Proxy",
        "• Yahoo gamma 부족 시 GEX 정확도 제한",
        "",
        (
            "Generated: "
            +
            started.strftime(
                "%Y-%m-%d %H:%M:%S UTC"
            )
        )
    ]

    return "\n".join(
        report
    )


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(
    text
):

    token = os.getenv(
        "TELEGRAM_BOT_TOKEN"
    )

    chat_id = os.getenv(
        "TELEGRAM_CHAT_ID"
    )

    # ========================================================
    # IMPORTANT:
    # Telegram 설정이 없으면 성공으로 처리하지 않는다.
    # GitHub Actions에서 확실히 실패시키기 위해
    # RuntimeError를 발생시킨다.
    # ========================================================

    if (
        not token
        or not chat_id
    ):

        error_message = (
            "Telegram credentials not configured. "
            "TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID "
            "must be configured."
        )

        print(
            f"❌ {error_message}"
        )

        raise RuntimeError(
            error_message
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
            remaining[
                :split_at
            ]
        )

        remaining = remaining[
            split_at:
        ]

    if remaining:

        chunks.append(
            remaining
        )

    if not chunks:

        raise RuntimeError(
            "Telegram message is empty."
        )

    print()
    print("=" * 70)
    print("SEND TELEGRAM")
    print("=" * 70)

    successful_chunks = 0

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
                        chunk,

                    "disable_web_page_preview":
                        "true"
                }
            )
            .encode(
                "utf-8"
            )
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

                status_code = (
                    response.status
                )

                result = (
                    response
                    .read()
                    .decode(
                        "utf-8"
                    )
                )

                print(
                    f"Telegram "
                    f"[{index}/{len(chunks)}] "
                    f"HTTP {status_code}"
                )

                if status_code != 200:

                    raise RuntimeError(
                        (
                            "Telegram HTTP "
                            f"status {status_code}: "
                            f"{result[:500]}"
                        )
                    )

                # Telegram API response 확인
                try:

                    result_json = (
                        __import__(
                            "json"
                        ).loads(
                            result
                        )
                    )

                except Exception:

                    result_json = {}

                if not result_json.get(
                    "ok",
                    False
                ):

                    raise RuntimeError(
                        (
                            "Telegram API returned "
                            f"ok=false: "
                            f"{result[:500]}"
                        )
                    )

                successful_chunks += 1

                print(
                    "✅ Telegram chunk sent"
                )

        except Exception as exc:

            print()
            print(
                "❌ TELEGRAM SEND FAILED"
            )

            print(
                f"Chunk: "
                f"{index}/{len(chunks)}"
            )

            print(
                f"Error type: "
                f"{type(exc).__name__}"
            )

            print(
                f"Error: "
                f"{repr(exc)}"
            )

            # =================================================
            # IMPORTANT:
            # Telegram 하나라도 실패하면 workflow 실패
            # =================================================

            raise RuntimeError(
                (
                    "Telegram delivery failed. "
                    f"Successful chunks: "
                    f"{successful_chunks}/"
                    f"{len(chunks)}"
                )
            ) from exc

    if successful_chunks != len(
        chunks
    ):

        raise RuntimeError(
            (
                "Telegram delivery incomplete: "
                f"{successful_chunks}/"
                f"{len(chunks)}"
            )
        )

    print()
    print(
        f"✅ Telegram sent successfully: "
        f"{successful_chunks} message(s)"
    )

    return True


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    data,
    strike_table,
    expiration_structure,
    strike_expiration,
    key_strike_summary,
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

        if dataframe is None:

            dataframe = pd.DataFrame()

        path = os.path.join(
            output_dir,
            filename
        )

        dataframe.to_csv(
            path,
            index=False
        )

        saved_files.append(
            path
        )

        print(
            f"💾 {filename:40s} "
            f"rows={len(dataframe):,}"
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
        f"💾 {'report.md':40s} "
        f"chars={len(report):,}"
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

    min_strike = (
        args.min_strike
    )

    max_strike = (
        args.max_strike
    )

    max_dte = (
        args.max_dte
    )

    output_dir = (
        args.output
    )

    # ========================================================
    # BASIC VALIDATION
    # ========================================================

    if not symbol:

        raise ValueError(
            "Symbol cannot be empty."
        )

    if min_strike < 0:

        raise ValueError(
            "Minimum strike cannot be negative."
        )

    if max_strike < min_strike:

        raise ValueError(
            "MAX_STRIKE must be >= MIN_STRIKE."
        )

    if max_dte <= 0:

        raise ValueError(
            "MAX_DTE must be greater than zero."
        )

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
        f" ~ "
        f"${max_strike:g}"
    )

    print(
        f"DTE ANALYSIS : "
        f"1 ~ {max_dte}"
    )

    print(
        f"FOCUS        : "
        f"{FOCUS_STRIKES}"
    )

    print(
        f"BAR WIDTH    : "
        f"{BAR_WIDTH} MAX"
    )

    print(
        "BAR SCALE    : "
        "MAX SIZE = 10 / SMALL SIZE = SHORTER"
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

    print(
        f"Normalized rows: "
        f"{len(data):,}"
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
    # 5. STRUCTURES
    # ========================================================

    strike_table = (
        build_strike_table(
            data
        )
    )

    expiration_structure = (
        build_expiration_structure(
            data
        )
    )

    strike_expiration = (
        build_strike_expiration_structure(
            data,
            FOCUS_STRIKES
        )
    )

    key_strike_summary = (
        build_key_strike_summary(
            strike_expiration
        )
    )

    top_contracts = (
        build_top_contracts(
            data
        )
    )

    # ========================================================
    # STRUCTURE DEBUG
    # ========================================================

    print()
    print("=" * 70)
    print("STRUCTURE CHECK")
    print("=" * 70)

    print(
        f"Strike rows: "
        f"{len(strike_table):,}"
    )

    print(
        f"Expiration rows: "
        f"{len(expiration_structure):,}"
    )

    print(
        f"Strike × Expiration rows: "
        f"{len(strike_expiration):,}"
    )

    print(
        f"Key Strike rows: "
        f"{len(key_strike_summary):,}"
    )

    print(
        f"Top Contracts rows: "
        f"{len(top_contracts):,}"
    )

    if expiration_structure.empty:

        print(
            "⚠️ WARNING: "
            "expiration_structure is EMPTY"
        )

    else:

        print(
            "✅ expiration_structure OK"
        )

    if strike_expiration.empty:

        print(
            "⚠️ WARNING: "
            "strike_expiration is EMPTY"
        )

    else:

        print(
            "✅ strike_expiration OK"
        )

    # ========================================================
    # 6. SUMMARY
    # ========================================================

    calls = data[
        data["option_type"]
        ==
        "CALL"
    ]

    puts = data[
        data["option_type"]
        ==
        "PUT"
    ]

    cv = safe_sum(
        calls["volume"]
    )

    pv = safe_sum(
        puts["volume"]
    )

    coi = safe_sum(
        calls["openInterest"]
    )

    poi = safe_sum(
        puts["openInterest"]
    )

    cp = safe_sum(
        calls["premium_proxy"]
    )

    pp = safe_sum(
        puts["premium_proxy"]
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

        net_gex = (
            tgex + pgex
        )

    elif np.isfinite(tgex):

        net_gex = tgex

    elif np.isfinite(pgex):

        net_gex = pgex

    else:

        net_gex = np.nan

    summary = pd.DataFrame(
        [
            {
                "symbol":
                    symbol,

                "spot":
                    spot,

                "min_strike":
                    min_strike,

                "max_strike":
                    max_strike,

                "max_dte":
                    max_dte,

                "rows":
                    len(data),

                "call_volume":
                    cv,

                "put_volume":
                    pv,

                "call_oi":
                    coi,

                "put_oi":
                    poi,

                "call_premium":
                    cp,

                "put_premium":
                    pp,

                "call_volume_ratio":
                    cv
                    /
                    (cv + pv)
                    *
                    100
                    if cv + pv > 0
                    else np.nan,

                "call_oi_ratio":
                    coi
                    /
                    (coi + poi)
                    *
                    100
                    if coi + poi > 0
                    else np.nan,

                "call_volume_oi":
                    cv
                    /
                    coi
                    if coi > 0
                    else np.nan,

                "put_volume_oi":
                    pv
                    /
                    poi
                    if poi > 0
                    else np.nan,

                "call_gex":
                    tgex,

                "put_gex":
                    pgex,

                "net_gex":
                    net_gex
            }
        ]
    )

    # ========================================================
    # 7. REPORT
    # ========================================================

    report = build_report(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        top_contracts=top_contracts,
        spot=spot,
        symbol=symbol,
        min_strike=min_strike,
        max_strike=max_strike,
        max_dte=max_dte,
        started=started
    )

    # ========================================================
    # 8. SAVE
    # ========================================================

    save_outputs(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        top_contracts=top_contracts,
        summary=summary,
        report=report,
        output_dir=output_dir
    )

    # ========================================================
    # 9. PRINT
    # ========================================================

    print()
    print(
        report
    )

    # ========================================================
    # 10. TELEGRAM
    #
    # 중요:
    # send_telegram()이 실패하면 RuntimeError가 발생하고
    # 아래 except에서 다시 raise되므로
    # GitHub Actions가 FAILED가 된다.
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

        print()
        print(
            "⚠️ Scanner interrupted."
        )

        sys.exit(
            130
        )

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

        # ====================================================
        # 중요:
        # GitHub Actions에서 반드시 실패 처리
        # ====================================================

        sys.exit(1)
