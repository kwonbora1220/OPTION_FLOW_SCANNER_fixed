
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

# ------------------------------------------------------------
# KEY STRIKES
#
# These are the strikes we specifically want to inspect
# by expiration.
# ------------------------------------------------------------

FOCUS_STRIKES = [
    80,
    85,
    90,
    95,
    100
]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# US MARKET DATE
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


def fmt_pct(value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:.1f}%"


# ============================================================
# ARGUMENTS
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
    # MANUAL
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
    # FALLBACK
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

    open_interest = safe_float(
        open_interest
    )

    spot = safe_float(
        spot
    )

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
# IMPORTANT:
# NO DTE / STRIKE FILTER DURING COLLECTION
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

    spot = get_current_price(
        ticker,
        manual_price
    )

    try:

        expirations = list(
            ticker.options
        )

    except Exception as exc:

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

        try:

            calls = chain.calls

        except Exception:

            calls = pd.DataFrame()

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

    return data


# ============================================================
# FILTER
#
# DTE 0 EXCLUDED
# FINAL DTE = 1 ~ MAX_DTE
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
        f"After DTE 1~{max_dte}: "
        f"{len(data):,}"
    )

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
#
# DTE 0 has already been removed.
# Therefore this is normally EMPTY.
# ============================================================

def get_today_expiration(
    data
):

    today = market_today()

    return data[
        data["expiration"].apply(
            lambda x:
            pd.Timestamp(
                x
            ).date() == today
        )
    ].copy()


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
