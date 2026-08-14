import os
import sys
import time
import argparse
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

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# US MARKET DATE
#
# Yahoo expiration date is US market date.
# Therefore DTE must use America/New_York,
# NOT UTC date.
# ============================================================

US_EASTERN = ZoneInfo(
    "America/New_York"
)


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

    sign = "-" if value < 0 else ""

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


def fmt_iv(value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    if value < 2:
        value *= 100

    return f"{value:.1f}%"


def fmt_number(value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:,.0f}"


# ============================================================
# COMMAND LINE
#
# Examples:
#
# python rklb_option_structure.py
#
# python rklb_option_structure.py RKLB
#
# python rklb_option_structure.py RKLB 80.15
#
# python rklb_option_structure.py NVTS 7.20
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Yahoo Finance Option Structure Scanner"
        )
    )

    parser.add_argument(
        "symbol",
        nargs="?",
        default=DEFAULT_SYMBOL,
        help="Ticker symbol"
    )

    parser.add_argument(
        "price",
        nargs="?",
        default=DEFAULT_PRICE,
        help="Manual option calculation price"
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

    # --------------------------------------------------------
    # MANUAL PRICE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 1 MINUTE
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # 5 DAY FALLBACK
    # --------------------------------------------------------

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

        expiry_date = pd.Timestamp(
            expiration
        ).date()

        today = market_today()

        return (
            expiry_date - today
        ).days

    except Exception as exc:

        print(
            f"DTE error "
            f"{expiration}: "
            f"{repr(exc)}"
        )

        return np.nan


# ============================================================
# PREMIUM PROXY
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

    # --------------------------------------------------------
    # BID / ASK MID
    # --------------------------------------------------------

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
#
# IMPORTANT:
#
# Yahoo gamma missing
# -> N/A
#
# Do NOT convert missing gamma to zero.
# ============================================================

def calculate_gex(
    gamma,
    open_interest,
    spot,
    option_type
):

    gamma = safe_float(gamma)
    open_interest = safe_float(
        open_interest
    )
    spot = safe_float(spot)

    if not np.isfinite(gamma):
        return np.nan

    if not np.isfinite(open_interest):
        return np.nan

    if not np.isfinite(spot):
        return np.nan

    if gamma <= 0:
        return 0.0

    if open_interest <= 0:
        return 0.0

    if spot <= 0:
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
#
# NO FILTER DURING COLLECTION
# ============================================================

def fetch_options(
    symbol,
    manual_price=None
):

    print()
    print("=" * 70)
    print(
        "FETCH YAHOO FINANCE OPTION DATA"
    )
    print("=" * 70)

    ticker = yf.Ticker(
        symbol
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    spot = get_current_price(
        ticker,
        manual_price
    )

    # --------------------------------------------------------
    # EXPIRATIONS
    # --------------------------------------------------------

    try:

        expirations = list(
            ticker.options
        )

    except Exception as exc:

        print()
        print(
            "YAHOO EXPIRATION ERROR"
        )

        print(
            repr(exc)
        )

        raise RuntimeError(
            "Unable to get Yahoo option expirations."
        )

    print()
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
            "-" * 70
        )

        print(
            f"[{index}/{len(expirations)}] "
            f"EXPIRATION: {expiration}"
        )

        try:

            chain = ticker.option_chain(
                expiration
            )

        except Exception as exc:

            failed += 1

            print(
                "❌ option_chain FAILED"
            )

            print(
                repr(exc)
            )

            continue

        # ----------------------------------------------------
        # CALL
        # ----------------------------------------------------

        try:

            calls = chain.calls

        except Exception:

            calls = pd.DataFrame()

        # ----------------------------------------------------
        # PUT
        # ----------------------------------------------------

        try:

            puts = chain.puts

        except Exception:

            puts = pd.DataFrame()

        call_count = (
            len(calls)
            if calls is not None
            else 0
        )

        put_count = (
            len(puts)
            if puts is not None
            else 0
        )

        print(
            f"CALL rows: {call_count}"
        )

        print(
            f"PUT rows : {put_count}"
        )

        # ----------------------------------------------------
        # CALL
        # ----------------------------------------------------

        if (
            calls is not None
            and not calls.empty
        ):

            frame = calls.copy()

            frame[
                "option_type"
            ] = "CALL"

            frame[
                "expiration"
            ] = expiration

            rows.append(
                frame
            )

        # ----------------------------------------------------
        # PUT
        # ----------------------------------------------------

        if (
            puts is not None
            and not puts.empty
        ):

            frame = puts.copy()

            frame[
                "option_type"
            ] = "PUT"

            frame[
                "expiration"
            ] = expiration

            rows.append(
                frame
            )

        if (
            call_count > 0
            or put_count > 0
        ):

            successful += 1

        time.sleep(
            0.25
        )

    print()
    print("=" * 70)
    print(
        "YAHOO COLLECTION COMPLETE"
    )
    print("=" * 70)

    print(
        f"Successful expirations: "
        f"{successful}"
    )

    print(
        f"Failed expirations: "
        f"{failed}"
    )

    if not rows:

        raise RuntimeError(
            "No option rows collected."
        )

    data = pd.concat(
        rows,
        ignore_index=True
    )

    print(
        f"RAW OPTION ROWS: "
        f"{len(data):,}"
    )

    return data, spot


# ============================================================
# NORMALIZE
# ============================================================

def normalize(
    data
):

    print()
    print("=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

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

    print(
        f"Rows after normalization: "
        f"{len(data):,}"
    )

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
    print("APPLY FILTERS")
    print("=" * 70)

    before = len(data)

    # --------------------------------------------------------
    # DTE
    # --------------------------------------------------------

    data = data[
        data["DTE"].notna()
    ].copy()

    data = data[
        (
            data["DTE"] >= 0
        )
        &
        (
            data["DTE"] <= max_dte
        )
    ].copy()

    print(
        f"After DTE 0~{max_dte}: "
        f"{len(data):,}"
    )

    # --------------------------------------------------------
    # STRIKE
    # --------------------------------------------------------

    data = data[
        data["strike"].between(
            min_strike,
            max_strike,
            inclusive="both"
        )
    ].copy()

    print(
        f"After Strike "
        f"${min_strike:g}~"
        f"${max_strike:g}: "
        f"{len(data):,}"
    )

    print(
        f"Rows removed: "
        f"{before - len(data):,}"
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

    # --------------------------------------------------------
    # PREMIUM
    # --------------------------------------------------------

    data[
        "premium_proxy"
    ] = data.apply(
        lambda row:
        calculate_premium(
            row["volume"],
            row["bid"],
            row["ask"],
            row["lastPrice"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    data[
        "gex"
    ] = data.apply(
        lambda row:
        calculate_gex(
            row["gamma"],
            row["openInterest"],
            spot,
            row["option_type"]
        ),
        axis=1
    )

    # --------------------------------------------------------
    # VOLUME / OI
    # --------------------------------------------------------

    data[
        "volume_oi"
    ] = np.where(
        data["openInterest"] > 0,
        data["volume"]
        /
        data["openInterest"],
        np.nan
    )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    data[
        "distance_pct"
    ] = (
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
# TODAY EXPIRATION
# ============================================================

def get_today_expiration(
    data
):

    today = market_today()

    today_rows = data[
        data["expiration"].apply(
            lambda x:
            pd.Timestamp(
                x
            ).date() == today
        )
    ].copy()

    return today_rows


# ============================================================
# STRIKE TABLE
# ============================================================

def build_strike_table(
    data
):

    rows = []

    strikes = sorted(
        data["strike"]
        .dropna()
        .unique()
    )

    for strike in strikes:

        frame = data[
            data["strike"] == strike
        ]

        calls = frame[
            frame["option_type"]
            == "CALL"
        ]

        puts = frame[
            frame["option_type"]
            == "PUT"
        ]

        call_volume = (
            calls["volume"]
            .fillna(0)
            .sum()
        )

        put_volume = (
            puts["volume"]
            .fillna(0)
            .sum()
        )

        call_oi = (
            calls["openInterest"]
            .fillna(0)
            .sum()
        )

        put_oi = (
            puts["openInterest"]
            .fillna(0)
            .sum()
        )

        call_premium = (
            calls[
                "premium_proxy"
            ]
            .fillna(0)
            .sum()
        )

        put_premium = (
            puts[
                "premium_proxy"
            ]
            .fillna(0)
            .sum()
        )

        # ----------------------------------------------------
        # GEX
        #
        # min_count=1 means:
        # if all GEX are NaN -> NaN
        # instead of 0
        # ----------------------------------------------------

        call_gex = (
            calls["gex"]
            .sum(
                min_count=1
            )
        )

        put_gex = (
            puts["gex"]
            .sum(
                min_count=1
            )
        )

        net_gex = (
            call_gex
            + put_gex
        )

        rows.append(
            {
                "strike":
                    strike,

                "call_volume":
                    call_volume,

                "put_volume":
                    put_volume,

                "total_volume":
                    call_volume
                    + put_volume,

                "call_oi":
                    call_oi,

                "put_oi":
                    put_oi,

                "total_oi":
                    call_oi
                    + put_oi,

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "total_premium":
                    call_premium
                    + put_premium,

                "call_gex":
                    call_gex,

                "put_gex":
                    put_gex,

                "net_gex":
                    net_gex
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TODAY STRIKE TABLE
# ============================================================

def build_today_strike_table(
    today_data
):

    if today_data.empty:

        return pd.DataFrame()

    return build_strike_table(
        today_data
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(
    data
):

    result = data.copy()

    result[
        "importance"
    ] = (
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
            >= spot
        ].copy()

        if candidates.empty:
            return None

        candidates["oi"] = (
            candidates["call_oi"]
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
            strike_table["strike"]
            <= spot
        ].copy()

        if candidates.empty:
            return None

        candidates["oi"] = (
            candidates["put_oi"]
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
        <= 0.20
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
# TODAY SECTION
# ============================================================

def build_today_report(
    today_data,
    spot
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔥 2. TODAY EXPIRATION"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if today_data.empty:

        lines.append(
            "📅 Today expiration: N/A"
        )

        lines.append(
            "현재 데이터에 오늘 만기 옵션이 없습니다."
        )

        return lines

    expiration = (
        pd.Timestamp(
            today_data[
                "expiration"
            ].iloc[0]
        ).strftime(
            "%Y-%m-%d"
        )
    )

    lines.append(
        f"📅 Expiration: "
        f"{expiration}"
    )

    lines.append(
        f"💰 현재가: "
        f"${spot:.2f}"
    )

    lines.append("")

    lines.append(
        "STRIKE | C-VOL | P-VOL | "
        "C-OI | P-OI | C-PREM | P-PREM"
    )

    lines.append(
        "────────────────────────────────────────"
    )

    table = build_today_strike_table(
        today_data
    )

    for _, row in (
        table
        .sort_values(
            "strike"
        )
        .iterrows()
    ):

        lines.append(
            f"${row['strike']:g} | "
            f"{row['call_volume']:,.0f} | "
            f"{row['put_volume']:,.0f} | "
            f"{row['call_oi']:,.0f} | "
            f"{row['put_oi']:,.0f} | "
            f"{fmt_money(row['call_premium'])} | "
            f"{fmt_money(row['put_premium'])}"
        )

    lines.append("")

    # --------------------------------------------------------
    # TOP OI
    # --------------------------------------------------------

    lines.append(
        "🔥 TODAY TOP OI"
    )

    top_oi = (
        table
        .sort_values(
            "total_oi",
            ascending=False
        )
        .head(5)
    )

    for _, row in top_oi.iterrows():

        lines.append(
            f"${row['strike']:g} | "
            f"Total OI "
            f"{row['total_oi']:,.0f}"
            f" | C "
            f"{row['call_oi']:,.0f}"
            f" / P "
            f"{row['put_oi']:,.0f}"
        )

    lines.append("")

    # --------------------------------------------------------
    # TOP VOLUME
    # --------------------------------------------------------

    lines.append(
        "🔥 TODAY TOP VOLUME"
    )

    top_volume = (
        table
        .sort_values(
            "total_volume",
            ascending=False
        )
        .head(5)
    )

    for _, row in top_volume.iterrows():

        lines.append(
            f"${row['strike']:g} | "
            f"Total Vol "
            f"{row['total_volume']:,.0f}"
            f" | C "
            f"{row['call_volume']:,.0f}"
            f" / P "
            f"{row['put_volume']:,.0f}"
        )

    return lines


# ============================================================
# REPORT
# ============================================================

def build_report(
    data,
    strike_table,
    top_contracts,
    today_data,
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

    # ========================================================
    # FLOW
    # ========================================================

    call_volume = (
        calls["volume"]
        .fillna(0)
        .sum()
    )

    put_volume = (
        puts["volume"]
        .fillna(0)
        .sum()
    )

    call_oi = (
        calls["openInterest"]
        .fillna(0)
        .sum()
    )

    put_oi = (
        puts["openInterest"]
        .fillna(0)
        .sum()
    )

    call_premium = (
        calls["premium_proxy"]
        .fillna(0)
        .sum()
    )

    put_premium = (
        puts["premium_proxy"]
        .fillna(0)
        .sum()
    )

    total_volume = (
        call_volume
        + put_volume
    )

    total_oi = (
        call_oi
        + put_oi
    )

    total_premium = (
        call_premium
        + put_premium
    )

    call_volume_ratio = (
        call_volume
        /
        total_volume
        *
        100
        if total_volume > 0
        else np.nan
    )

    call_oi_ratio = (
        call_oi
        /
        total_oi
        *
        100
        if total_oi > 0
        else np.nan
    )

    call_premium_ratio = (
        call_premium
        /
        total_premium
        *
        100
        if total_premium > 0
        else np.nan
    )

    # ========================================================
    # GEX
    #
    # min_count=1 is IMPORTANT
    # ========================================================

    call_gex = (
        calls["gex"]
        .sum(
            min_count=1
        )
    )

    put_gex = (
        puts["gex"]
        .sum(
            min_count=1
        )
    )

    if (
        np.isfinite(call_gex)
        and np.isfinite(put_gex)
    ):

        net_gex = (
            call_gex
            + put_gex
        )

    elif np.isfinite(call_gex):

        net_gex = call_gex

    elif np.isfinite(put_gex):

        net_gex = put_gex

    else:

        net_gex = np.nan

    # ========================================================
    # ATM IV
    # ========================================================

    temp = data.copy()

    temp["atm_distance"] = (
        (
            temp["strike"]
            - spot
        ).abs()
    )

    atm_rows = (
        temp
        .sort_values(
            "atm_distance"
        )
        .head(10)
    )

    atm_iv = (
        atm_rows[
            "impliedVolatility"
        ]
        .dropna()
        .mean()
    )

    # ========================================================
    # WALL
    # ========================================================

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

    # ========================================================
    # REPORT
    # ========================================================

    report = []

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"🔥 {symbol} OPTION STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append("")

    report.append(
        f"💰 현재가: ${spot:.2f}"
    )

    report.append(
        f"🎯 Strike: "
        f"${min_strike:g} ~ "
        f"${max_strike:g}"
    )

    report.append(
        f"📅 DTE: 0 ~ {max_dte}"
    )

    report.append(
        f"📊 옵션 행수: {len(data):,}"
    )

    report.append("")

    # ========================================================
    # 1 FLOW
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "📊 1. OPTION FLOW"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"CALL Volume: "
        f"{call_volume:,.0f}"
    )

    report.append(
        f"PUT Volume: "
        f"{put_volume:,.0f}"
    )

    report.append(
        f"CALL Volume Ratio: "
        f"{call_volume_ratio:.1f}%"
        if np.isfinite(
            call_volume_ratio
        )
        else
        "CALL Volume Ratio: N/A"
    )

    report.append(
        f"CALL OI: "
        f"{call_oi:,.0f}"
    )

    report.append(
        f"PUT OI: "
        f"{put_oi:,.0f}"
    )

    report.append(
        f"CALL OI Ratio: "
        f"{call_oi_ratio:.1f}%"
        if np.isfinite(
            call_oi_ratio
        )
        else
        "CALL OI Ratio: N/A"
    )

    report.append(
        f"CALL Premium Proxy: "
        f"{fmt_money(call_premium)}"
    )

    report.append(
        f"PUT Premium Proxy: "
        f"{fmt_money(put_premium)}"
    )

    report.append(
        f"CALL Premium Ratio: "
        f"{call_premium_ratio:.1f}%"
        if np.isfinite(
            call_premium_ratio
        )
        else
        "CALL Premium Ratio: N/A"
    )

    report.append("")

    # ========================================================
    # 2 TODAY
    # ========================================================

    report.extend(
        build_today_report(
            today_data,
            spot
        )
    )

    report.append("")

    # ========================================================
    # 3 WALL / GEX
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧱 3. WALL / GEX"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if call_wall is not None:

        report.append(
            f"📈 Call Wall: "
            f"${call_wall['strike']:g}"
            f" | OI "
            f"{call_wall['call_oi']:,.0f}"
            f" | GEX "
            f"{fmt_money(call_wall['call_gex'])}"
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
            f" | GEX "
            f"{fmt_money(put_wall['put_gex'])}"
        )

    else:

        report.append(
            "📉 Put Wall: N/A"
        )

    if not np.isfinite(
        call_gex
    ):

        report.append(
            "⚠️ Yahoo gamma 데이터 부족"
        )

    report.append("")

    report.append(
        f"CALL GEX: "
        f"{fmt_money(call_gex)}"
    )

    report.append(
        f"PUT GEX: "
        f"{fmt_money(put_gex)}"
    )

    report.append(
        f"NET GEX: "
        f"{fmt_money(net_gex)}"
    )

    report.append(
        f"ATM IV: "
        f"{fmt_iv(atm_iv)}"
    )

    report.append("")

    # ========================================================
    # 4 STRIKE STRUCTURE
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"🎯 4. "
        f"${min_strike:g}~"
        f"${max_strike:g} "
        f"STRIKE STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "STRIKE | C-VOL | P-VOL | "
        "C-OI | P-OI | "
        "C-PREM | P-PREM | NET-GEX"
    )

    report.append(
        "────────────────────────────────────────"
    )

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
            f"{fmt_money(row['put_premium'])} | "
            f"{fmt_money(row['net_gex'])}"
        )

    report.append("")

    # ========================================================
    # 5 HIGH OI
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 5. HIGH OI STRIKES"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    high_oi = (
        strike_table
        .sort_values(
            "total_oi",
            ascending=False
        )
        .head(10)
    )

    for _, row in high_oi.iterrows():

        report.append(
            f"${row['strike']:g}"
            f" | Total OI "
            f"{row['total_oi']:,.0f}"
            f" | C "
            f"{row['call_oi']:,.0f}"
            f" / P "
            f"{row['put_oi']:,.0f}"
            f" | NET GEX "
            f"{fmt_money(row['net_gex'])}"
        )

    report.append("")

    # ========================================================
    # 6 TOP CONTRACTS
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 6. TOP OPTION CONTRACTS"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for _, row in (
        top_contracts
        .head(20)
        .iterrows()
    ):

        dte = safe_float(
            row["DTE"]
        )

        volume = safe_float(
            row["volume"]
        )

        oi = safe_float(
            row["openInterest"]
        )

        report.append(
            f"{row['option_type']:4s} "
            f"${row['strike']:g}"
            f" | DTE "
            f"{int(dte) if np.isfinite(dte) else 'N/A'}"
            f" | Vol "
            f"{fmt_number(volume)}"
            f" | OI "
            f"{fmt_number(oi)}"
            f" | Premium "
            f"{fmt_money(row['premium_proxy'])}"
            f" | GEX "
            f"{fmt_money(row['gex'])}"
        )

    report.append("")

    # ========================================================
    # 7 STRUCTURE
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧠 7. STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if (
        call_wall is not None
        and put_wall is not None
    ):

        cw = safe_float(
            call_wall["strike"]
        )

        pw = safe_float(
            put_wall["strike"]
        )

        if spot > cw:

            report.append(
                "🟢 가격 위치: "
                "ABOVE_CALL_WALL"
            )

        elif spot < pw:

            report.append(
                "🔴 가격 위치: "
                "BELOW_PUT_WALL"
            )

        else:

            report.append(
                "🟡 가격 위치: "
                "BETWEEN_WALLS"
            )

        report.append(
            f"Put Wall: ${pw:g}"
        )

        report.append(
            f"Call Wall: ${cw:g}"
        )

    if (
        np.isfinite(net_gex)
    ):

        if net_gex > 0:

            report.append(
                "📈 Net GEX: POSITIVE"
            )

        elif net_gex < 0:

            report.append(
                "📉 Net GEX: NEGATIVE"
            )

        else:

            report.append(
                "🟡 Net GEX: NEUTRAL"
            )

    if call_volume > put_volume:

        report.append(
            "🟢 CALL Volume 우세"
        )

    elif put_volume > call_volume:

        report.append(
            "🔴 PUT Volume 우세"
        )

    if call_oi > put_oi:

        report.append(
            "🟢 CALL OI 우세"
        )

    elif put_oi > call_oi:

        report.append(
            "🔴 PUT OI 우세"
        )

    report.append("")

    # ========================================================
    # LIMITATIONS
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "⚠️ DATA LIMITATIONS"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "• Yahoo Finance 무료 옵션 데이터"
    )

    report.append(
        "• Premium = 거래대금 Proxy"
    )

    report.append(
        "• 실제 Buy/Sell 방향 확인 불가"
    )

    report.append(
        "• OI만으로 Long/Short 확정 불가"
    )

    report.append(
        "• GEX = OI 기반 모델링 Proxy"
    )

    report.append(
        "• Yahoo gamma 부족 시 GEX = N/A"
    )

    report.append("")

    report.append(
        "Generated: "
        +
        started.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

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

    if not token:

        print(
            "TELEGRAM_BOT_TOKEN not configured."
        )

        return

    if not chat_id:

        print(
            "TELEGRAM_CHAT_ID not configured."
        )

        return

    import urllib.parse
    import urllib.request

    url = (
        "https://api.telegram.org/"
        f"bot{token}/sendMessage"
    )

    max_length = 3900

    chunks = []

    while len(text) > max_length:

        split_at = text.rfind(
            "\n",
            0,
            max_length
        )

        if split_at <= 0:

            split_at = max_length

        chunks.append(
            text[:split_at]
        )

        text = text[split_at:]

    if text:

        chunks.append(
            text
        )

    for chunk in chunks:

        payload = urllib.parse.urlencode(
            {
                "chat_id":
                    chat_id,

                "text":
                    chunk
            }
        ).encode(
            "utf-8"
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
                    .decode(
                        "utf-8"
                    )
                )

                print(
                    "Telegram:",
                    result[:500]
                )

        except Exception as exc:

            print(
                "Telegram error:",
                repr(exc)
            )

    print(
        f"Telegram sent: "
        f"{len(chunks)} message(s)"
    )


# ============================================================
# SAVE CSV
# ============================================================

def save_outputs(
    data,
    strike_table,
    today_data,
    top_contracts,
    summary,
    report,
    output_dir
):

    os.makedirs(
        output_dir,
        exist_ok=True
    )

    data.to_csv(
        os.path.join(
            output_dir,
            "contracts.csv"
        ),
        index=False
    )

    strike_table.to_csv(
        os.path.join(
            output_dir,
            "strike_structure.csv"
        ),
        index=False
    )

    today_data.to_csv(
        os.path.join(
            output_dir,
            "today_expiration.csv"
        ),
        index=False
    )

    top_contracts.to_csv(
        os.path.join(
            output_dir,
            "top_contracts.csv"
        ),
        index=False
    )

    summary.to_csv(
        os.path.join(
            output_dir,
            "summary.csv"
        ),
        index=False
    )

    report_file = os.path.join(
        output_dir,
        "report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    print()
    print(
        "OUTPUT FILES"
    )

    print(
        f"contracts.csv"
    )

    print(
        f"strike_structure.csv"
    )

    print(
        f"today_expiration.csv"
    )

    print(
        f"top_contracts.csv"
    )

    print(
        f"summary.csv"
    )

    print(
        f"report.md"
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

    started = datetime.now(
        timezone.utc
    )

    print()
    print("=" * 70)
    print(
        "🔥 OPTION STRUCTURE SCANNER"
    )
    print("=" * 70)

    print(
        f"SYMBOL       : {symbol}"
    )

    print(
        f"PRICE INPUT  : "
        f"{manual_price if manual_price else 'AUTO'}"
    )

    print(
        f"STRIKE RANGE : "
        f"${min_strike:g} ~ "
        f"${max_strike:g}"
    )

    print(
        f"DTE RANGE    : "
        f"0 ~ {max_dte}"
    )

    print(
        f"OUTPUT DIR   : "
        f"{output_dir}"
    )

    print("=" * 70)

    # ========================================================
    # 1 FETCH
    # ========================================================

    raw, spot = fetch_options(
        symbol,
        manual_price
    )

    # ========================================================
    # 2 NORMALIZE
    # ========================================================

    data = normalize(
        raw
    )

    # ========================================================
    # 3 FILTER
    # ========================================================

    data = apply_filters(
        data,
        min_strike,
        max_strike,
        max_dte
    )

    if data.empty:

        raise RuntimeError(
            "No option rows remain "
            "after filtering."
        )

    # ========================================================
    # 4 METRICS
    # ========================================================

    data = calculate_metrics(
        data,
        spot
    )

    print()
    print(
        f"FINAL OPTION ROWS: "
        f"{len(data):,}"
    )

    # ========================================================
    # 5 TODAY
    # ========================================================

    today_data = (
        get_today_expiration(
            data
        )
    )

    print(
        f"TODAY OPTION ROWS: "
        f"{len(today_data):,}"
    )

    # ========================================================
    # 6 STRIKE TABLE
    # ========================================================

    strike_table = (
        build_strike_table(
            data
        )
    )

    # ========================================================
    # 7 TOP CONTRACTS
    # ========================================================

    top_contracts = (
        build_top_contracts(
            data
        )
    )

    # ========================================================
    # 8 SUMMARY
    # ========================================================

    calls = data[
        data["option_type"]
        == "CALL"
    ]

    puts = data[
        data["option_type"]
        == "PUT"
    ]

    call_volume = (
        calls["volume"]
        .fillna(0)
        .sum()
    )

    put_volume = (
        puts["volume"]
        .fillna(0)
        .sum()
    )

    call_oi = (
        calls["openInterest"]
        .fillna(0)
        .sum()
    )

    put_oi = (
        puts["openInterest"]
        .fillna(0)
        .sum()
    )

    call_premium = (
        calls["premium_proxy"]
        .fillna(0)
        .sum()
    )

    put_premium = (
        puts["premium_proxy"]
        .fillna(0)
        .sum()
    )

    total_volume = (
        call_volume
        + put_volume
    )

    total_oi = (
        call_oi
        + put_oi
    )

    total_premium = (
        call_premium
        + put_premium
    )

    call_volume_ratio = (
        call_volume
        /
        total_volume
        *
        100
        if total_volume > 0
        else np.nan
    )

    call_oi_ratio = (
        call_oi
        /
        total_oi
        *
        100
        if total_oi > 0
        else np.nan
    )

    call_premium_ratio = (
        call_premium
        /
        total_premium
        *
        100
        if total_premium > 0
        else np.nan
    )

    call_gex = (
        calls["gex"]
        .sum(
            min_count=1
        )
    )

    put_gex = (
        puts["gex"]
        .sum(
            min_count=1
        )
    )

    if (
        np.isfinite(call_gex)
        and np.isfinite(put_gex)
    ):

        net_gex = (
            call_gex
            + put_gex
        )

    elif np.isfinite(call_gex):

        net_gex = call_gex

    elif np.isfinite(put_gex):

        net_gex = put_gex

    else:

        net_gex = np.nan

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

    # ========================================================
    # ATM IV
    # ========================================================

    data["atm_distance"] = (
        (
            data["strike"]
            - spot
        ).abs()
    )

    atm_rows = (
        data
        .sort_values(
            "atm_distance"
        )
        .head(10)
    )

    atm_iv = (
        atm_rows[
            "impliedVolatility"
        ]
        .dropna()
        .mean()
    )

    summary = pd.DataFrame(
        [
            {
                "symbol":
                    symbol,

                "spot":
                    spot,

                "manual_price":
                    (
                        safe_float(
                            manual_price
                        )
                        if manual_price
                        else np.nan
                    ),

                "min_strike":
                    min_strike,

                "max_strike":
                    max_strike,

                "max_dte":
                    max_dte,

                "rows":
                    len(data),

                "today_rows":
                    len(today_data),

                "call_volume":
                    call_volume,

                "put_volume":
                    put_volume,

                "call_volume_ratio":
                    call_volume_ratio,

                "call_oi":
                    call_oi,

                "put_oi":
                    put_oi,

                "call_oi_ratio":
                    call_oi_ratio,

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "call_premium_ratio":
                    call_premium_ratio,

                "call_gex":
                    call_gex,

                "put_gex":
                    put_gex,

                "net_gex":
                    net_gex,

                "call_wall":
                    (
                        call_wall["strike"]
                        if call_wall
                        is not None
                        else np.nan
                    ),

                "put_wall":
                    (
                        put_wall["strike"]
                        if put_wall
                        is not None
                        else np.nan
                    ),

                "atm_iv":
                    atm_iv
            }
        ]
    )

    # ========================================================
    # 9 REPORT
    # ========================================================

    report = build_report(
        data=data,
        strike_table=strike_table,
        top_contracts=top_contracts,
        today_data=today_data,
        spot=spot,
        symbol=symbol,
        min_strike=min_strike,
        max_strike=max_strike,
        max_dte=max_dte,
        started=started
    )

    # ========================================================
    # 10 SAVE
    # ========================================================

    save_outputs(
        data=data,
        strike_table=strike_table,
        today_data=today_data,
        top_contracts=top_contracts,
        summary=summary,
        report=report,
        output_dir=output_dir
    )

    # ========================================================
    # 11 PRINT
    # ========================================================

    print()
    print(report)

    # ========================================================
    # 12 TELEGRAM
    # ========================================================

    print()
    print("=" * 70)
    print("TELEGRAM")
    print("=" * 70)

    send_telegram(
        report
    )

    # ========================================================
    # DONE
    # ========================================================

    print()
    print("=" * 70)
    print(
        "✅ SCAN COMPLETE"
    )
    print("=" * 70)


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print()
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
