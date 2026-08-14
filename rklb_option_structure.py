# ============================================================
# OPTION STRUCTURE SCANNER
# Yahoo Finance FREE DATA
#
# FEATURES
# ------------------------------------------------------------
# 1. Full Yahoo option collection
# 2. DTE 1~180
# 3. Configurable strike range
# 4. Current-price distance analysis
# 5. Key Strike analysis
# 6. Strike × Expiration OI
# 7. Expiration structure
# 8. Call / Put Wall
# 9. GEX Proxy
# 10. PRICE ZONE CALL / PUT OI
# 11. PRICE ZONE CALL / PUT VOLUME
# 12. PRICE ZONE CALL / PUT PREMIUM
# 13. Telegram report
# ============================================================

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

# ------------------------------------------------------------
# IMPORTANT
#
# Expanded strike range
# ------------------------------------------------------------

MIN_STRIKE = float(
    os.getenv(
        "MIN_STRIKE",
        "60"
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
# KEY STRIKES
# ============================================================

FOCUS_STRIKES = [
    80,
    85,
    90,
    95,
    100
]


# ============================================================
# PRICE ZONES
#
# These are absolute strike zones.
# ============================================================

PRICE_ZONES = [
    (60, 70),
    (70, 80),
    (80, 90),
    (90, 100)
]


# ------------------------------------------------------------
# Alternative automatic zones
#
# Current price based:
# -25%
# -10%
# ATM
# +10%
# +25%
#
# These are additionally calculated.
# ------------------------------------------------------------

DISTANCE_ZONES = [
    ("BELOW -25%", -np.inf, -25),
    ("-25% ~ -10%", -25, -10),
    ("-10% ~ ATM", -10, 0),
    ("ATM ~ +10%", 0, 10),
    ("+10% ~ +25%", 10, 25),
    ("ABOVE +25%", 25, np.inf)
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

        expiry_date = pd.Timestamp(
            expiration
        ).date()

        today = market_today()

        return (
            expiry_date - today
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
# GEX
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

        calls = getattr(
            chain,
            "calls",
            pd.DataFrame()
        )

        puts = getattr(
            chain,
            "puts",
            pd.DataFrame()
        )

        call_count = len(calls)
        put_count = len(puts)

        print(
            f"CALL rows: {call_count}"
        )

        print(
            f"PUT rows : {put_count}"
        )

        if not calls.empty:

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

        if not puts.empty:

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

def normalize(data):

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

    data[
        "volume_oi"
    ] = np.where(
        data["openInterest"] > 0,
        data["volume"]
        /
        data["openInterest"],
        np.nan
    )

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
# TODAY
# ============================================================

def get_today_expiration(data):

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

def build_strike_table(data):

    rows = []

    for strike in sorted(
        data["strike"]
        .dropna()
        .unique()
    ):

        frame = data[
            data["strike"] == strike
        ]

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
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

        call_gex = calls["gex"].sum(
            min_count=1
        )

        put_gex = puts["gex"].sum(
            min_count=1
        )

        total_volume = (
            call_volume
            + put_volume
        )

        total_oi = (
            call_oi
            + put_oi
        )

        rows.append(
            {
                "strike": strike,
                "call_volume": call_volume,
                "put_volume": put_volume,
                "total_volume": total_volume,
                "call_oi": call_oi,
                "put_oi": put_oi,
                "total_oi": total_oi,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "total_premium":
                    call_premium
                    + put_premium,
                "call_volume_oi":
                    call_volume / call_oi
                    if call_oi > 0
                    else np.nan,
                "put_volume_oi":
                    put_volume / put_oi
                    if put_oi > 0
                    else np.nan,
                "call_gex": call_gex,
                "put_gex": put_gex,
                "net_gex":
                    call_gex + put_gex
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# PRICE ZONE STRUCTURE
#
# THIS IS THE NEW CORE
# ============================================================

def build_price_zone_structure(
    data,
    zones
):

    rows = []

    for zone_low, zone_high in zones:

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # lower bound INCLUDED
        # upper bound EXCLUDED
        #
        # except final zone where high is included.
        # ----------------------------------------------------

        if zone_high == zones[-1][1]:

            mask = (
                (data["strike"] >= zone_low)
                &
                (data["strike"] <= zone_high)
            )

        else:

            mask = (
                (data["strike"] >= zone_low)
                &
                (data["strike"] < zone_high)
            )

        frame = data[
            mask
        ].copy()

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

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

        total_oi = (
            call_oi
            + put_oi
        )

        total_volume = (
            call_volume
            + put_volume
        )

        total_premium = (
            call_premium
            + put_premium
        )

        rows.append(
            {
                "zone":
                    f"${zone_low:g}~${zone_high:g}",

                "low":
                    zone_low,

                "high":
                    zone_high,

                "rows":
                    len(frame),

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

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "total_premium":
                    total_premium
            }
        )

    result = pd.DataFrame(
        rows
    )

    if result.empty:
        return result

    # --------------------------------------------------------
    # OI PERCENT
    # --------------------------------------------------------

    total_oi_all = (
        result["total_oi"].sum()
    )

    if total_oi_all > 0:

        result[
            "oi_concentration_pct"
        ] = (
            result["total_oi"]
            /
            total_oi_all
            *
            100
        )

    else:

        result[
            "oi_concentration_pct"
        ] = np.nan

    # --------------------------------------------------------
    # CALL / PUT RATIO
    # --------------------------------------------------------

    result[
        "call_oi_ratio"
    ] = np.where(
        result["total_oi"] > 0,
        result["call_oi"]
        /
        result["total_oi"]
        *
        100,
        np.nan
    )

    result[
        "put_oi_ratio"
    ] = np.where(
        result["total_oi"] > 0,
        result["put_oi"]
        /
        result["total_oi"]
        *
        100,
        np.nan
    )

    return result


# ============================================================
# VISUAL BAR
#
# CALL = 🟩
# PUT  = 🟥
# ============================================================

def make_call_put_bar(
    call_value,
    put_value,
    width=24
):

    call_value = safe_float(
        call_value
    )

    put_value = safe_float(
        put_value
    )

    if not np.isfinite(call_value):
        call_value = 0

    if not np.isfinite(put_value):
        put_value = 0

    total = (
        call_value
        + put_value
    )

    if total <= 0:

        return "⬜" * width

    call_blocks = round(
        call_value
        /
        total
        *
        width
    )

    call_blocks = max(
        0,
        min(
            width,
            call_blocks
        )
    )

    put_blocks = (
        width
        -
        call_blocks
    )

    return (
        "🟩" * call_blocks
        +
        "🟥" * put_blocks
    )


# ============================================================
# DISTANCE ZONE
#
# Current-price-relative zones
# ============================================================

def build_distance_zone_structure(
    data
):

    rows = []

    for name, low_pct, high_pct in DISTANCE_ZONES:

        if np.isneginf(low_pct):

            mask = (
                data["distance_pct"]
                < high_pct
            )

        elif np.isposinf(high_pct):

            mask = (
                data["distance_pct"]
                >= low_pct
            )

        else:

            mask = (
                (data["distance_pct"] >= low_pct)
                &
                (data["distance_pct"] < high_pct)
            )

        frame = data[
            mask
        ].copy()

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

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

        rows.append(
            {
                "zone":
                    name,

                "low_pct":
                    low_pct,

                "high_pct":
                    high_pct,

                "rows":
                    len(frame),

                "call_oi":
                    call_oi,

                "put_oi":
                    put_oi,

                "total_oi":
                    call_oi + put_oi,

                "call_volume":
                    call_volume,

                "put_volume":
                    put_volume,

                "total_volume":
                    call_volume + put_volume
            }
        )

    return pd.DataFrame(rows)


# ============================================================
# STRIKE × EXPIRATION
# ============================================================

def build_strike_expiration_structure(
    data,
    focus_strikes
):

    rows = []

    available_strikes = set(
        data["strike"]
        .dropna()
        .tolist()
    )

    for target_strike in focus_strikes:

        matching = [
            x
            for x in available_strikes
            if abs(
                float(x)
                -
                float(target_strike)
            ) < 0.001
        ]

        if not matching:
            continue

        actual_strike = matching[0]

        strike_data = data[
            abs(
                data["strike"]
                -
                actual_strike
            ) < 0.001
        ]

        for expiration, frame in (
            strike_data
            .groupby("expiration")
        ):

            calls = frame[
                frame["option_type"] == "CALL"
            ]

            puts = frame[
                frame["option_type"] == "PUT"
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

            rows.append(
                {
                    "strike":
                        actual_strike,

                    "expiration":
                        expiration,

                    "DTE":
                        calculate_dte(
                            expiration
                        ),

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
                        calls[
                            "premium_proxy"
                        ]
                        .fillna(0)
                        .sum(),

                    "put_premium":
                        puts[
                            "premium_proxy"
                        ]
                        .fillna(0)
                        .sum()
                }
            )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    for strike in result["strike"].unique():

        mask = (
            result["strike"] == strike
        )

        total = (
            result.loc[
                mask,
                "total_oi"
            ].sum()
        )

        if total > 0:

            result.loc[
                mask,
                "total_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "total_oi"
                ]
                /
                total
                *
                100
            )

        else:

            result.loc[
                mask,
                "total_oi_pct"
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

    if strike_expiration.empty:

        return pd.DataFrame()

    rows = []

    for strike, frame in (
        strike_expiration
        .groupby("strike")
    ):

        frame = (
            frame
            .sort_values(
                "total_oi",
                ascending=False
            )
        )

        top = frame.iloc[0]

        rows.append(
            {
                "strike":
                    strike,

                "total_oi":
                    frame["total_oi"].sum(),

                "call_oi":
                    frame["call_oi"].sum(),

                "put_oi":
                    frame["put_oi"].sum(),

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
                    top["total_oi_pct"]
            }
        )

    return (
        pd.DataFrame(rows)
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================
# EXPIRATION STRUCTURE
# ============================================================

def build_expiration_structure(data):

    rows = []

    for expiration, frame in (
        data.groupby("expiration")
    ):

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

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

        rows.append(
            {
                "expiration":
                    expiration,

                "DTE":
                    calculate_dte(
                        expiration
                    ),

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
                    + put_premium
            }
        )

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    total_oi = (
        result["total_oi"].sum()
    )

    result[
        "total_oi_concentration_pct"
    ] = np.where(
        total_oi > 0,
        result["total_oi"]
        /
        total_oi
        *
        100,
        np.nan
    )

    return (
        result
        .sort_values(
            "DTE"
        )
        .reset_index(drop=True)
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(data):

    result = data.copy()

    result[
        "importance"
    ] = (
        np.log1p(
            result["premium_proxy"]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result["volume"]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result["openInterest"]
            .fillna(0)
            .clip(lower=0)
        )
        +
        np.log1p(
            result["gex"]
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
            strike_table["strike"] >= spot
        ].copy()

        if candidates.empty:
            return None

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
            strike_table["strike"] <= spot
        ].copy()

        if candidates.empty:
            return None

        candidates["oi"] = (
            candidates["put_oi"]
        )

        candidates["gex_abs"] = (
            candidates["put_gex"].abs()
        )

        candidates["volume"] = (
            candidates["put_volume"]
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
        candidates["distance"] <= 0.20
    ]

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
    today_data,
    zone_structure,
    distance_structure,
    spot,
    symbol,
    min_strike,
    max_strike,
    max_dte,
    started
):

    calls = data[
        data["option_type"] == "CALL"
    ]

    puts = data[
        data["option_type"] == "PUT"
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

    call_volume_oi = (
        call_volume
        /
        call_oi
        if call_oi > 0
        else np.nan
    )

    put_volume_oi = (
        put_volume
        /
        put_oi
        if put_oi > 0
        else np.nan
    )

    call_gex = (
        calls["gex"]
        .sum(min_count=1)
    )

    put_gex = (
        puts["gex"]
        .sum(min_count=1)
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
        .sort_values("atm_distance")
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

    report = []

    # ========================================================
    # HEADER
    # ========================================================

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
        f"📅 DTE: 1 ~ {max_dte}"
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
        f"{fmt_pct(call_volume_ratio)}"
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
        f"{fmt_pct(call_oi_ratio)}"
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
        f"{fmt_pct(call_premium_ratio)}"
    )

    report.append("")

    # ========================================================
    # 2 VOLUME / OI
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 2. VOLUME / OI"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"CALL Volume/OI: "
        f"{call_volume_oi:.3f}"
        if np.isfinite(call_volume_oi)
        else
        "CALL Volume/OI: N/A"
    )

    report.append(
        f"PUT Volume/OI: "
        f"{put_volume_oi:.3f}"
        if np.isfinite(put_volume_oi)
        else
        "PUT Volume/OI: N/A"
    )

    # ========================================================
    # 3 ZONE OI
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 3. PRICE ZONE CALL / PUT OI"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🟩 CALL   🟥 PUT"
    )

    report.append("")

    if zone_structure.empty:

        report.append(
            "N/A"
        )

    else:

        max_zone_total = max(
            zone_structure["total_oi"].max(),
            1
        )

        for _, row in (
            zone_structure.iterrows()
        ):

            bar = make_call_put_bar(
                row["call_oi"],
                row["put_oi"],
                width=24
            )

            report.append(
                f"{row['zone']:>11}"
                f" | {bar}"
            )

            report.append(
                f"   C-OI "
                f"{row['call_oi']:,.0f}"
                f" | P-OI "
                f"{row['put_oi']:,.0f}"
            )

            report.append(
                f"   CALL "
                f"{fmt_pct(row['call_oi_ratio'])}"
                f" | PUT "
                f"{fmt_pct(row['put_oi_ratio'])}"
                f" | 집중도 "
                f"{fmt_pct(row['oi_concentration_pct'])}"
            )

            report.append("")

    # ========================================================
    # 4 DISTANCE ZONE
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "📐 4. CURRENT PRICE DISTANCE ZONE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"현재가 ${spot:.2f} 기준"
    )

    report.append("")

    for _, row in (
        distance_structure.iterrows()
    ):

        bar = make_call_put_bar(
            row["call_oi"],
            row["put_oi"],
            width=20
        )

        report.append(
            f"{row['zone']:>13}"
            f" | {bar}"
        )

        report.append(
            f"   C-OI "
            f"{row['call_oi']:,.0f}"
            f" | P-OI "
            f"{row['put_oi']:,.0f}"
        )

    # ========================================================
    # 5 TODAY
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "📅 5. TODAY EXPIRATION"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if today_data.empty:

        report.append(
            "DTE 0 제외 → 오늘 만기 데이터 없음"
        )

    else:

        report.append(
            f"Today rows: "
            f"{len(today_data):,}"
        )

    # ========================================================
    # 6 WALL / GEX
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧱 6. WALL / GEX"
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

    # ========================================================
    # 7 STRIKE STRUCTURE
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🎯 7. STRIKE STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for _, row in (
        strike_table
        .sort_values("strike")
        .iterrows()
    ):

        report.append(
            f"${row['strike']:g}"
            f" | C-OI "
            f"{row['call_oi']:,.0f}"
            f" | P-OI "
            f"{row['put_oi']:,.0f}"
            f" | C-VOL "
            f"{row['call_volume']:,.0f}"
            f" | P-VOL "
            f"{row['put_volume']:,.0f}"
        )

    # ========================================================
    # 8 HIGH OI
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 8. HIGH OI STRIKES"
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

    # ========================================================
    # 9 EXPIRATION
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "📅 9. EXPIRATION STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for _, row in (
        expiration_structure.iterrows()
    ):

        dte = safe_float(
            row["DTE"]
        )

        report.append(
            f"DTE "
            f"{int(dte) if np.isfinite(dte) else 'N/A'}"
            f" | {row['expiration']}"
            f" | C-OI "
            f"{row['call_oi']:,.0f}"
            f" | P-OI "
            f"{row['put_oi']:,.0f}"
            f" | TOTAL "
            f"{row['total_oi']:,.0f}"
            f" | 집중도 "
            f"{fmt_pct(row['total_oi_concentration_pct'])}"
        )

    # ========================================================
    # 10 KEY STRIKE
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 10. KEY STRIKE × EXPIRATION OI"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🎯 $80 / $85 / $90 / $95 / $100"
    )

    report.append("")

    if key_strike_summary.empty:

        report.append(
            "N/A"
        )

    else:

        for _, row in (
            key_strike_summary.iterrows()
        ):

            top_dte = safe_float(
                row["top_DTE"]
            )

            report.append(
                f"💥 ${row['strike']:g}"
                f" | Total OI "
                f"{row['total_oi']:,.0}"
            )

            report.append(
                f"   C-OI "
                f"{row['call_oi']:,.0}"
                f" | P-OI "
                f"{row['put_oi']:,.0}"
            )

            report.append(
                f"   🏆 최대 집중: "
                f"{row['top_expiration']}"
                f" | DTE "
                f"{int(top_dte) if np.isfinite(top_dte) else 'N/A'}"
            )

            report.append(
                f"   해당 만기 OI "
                f"{row['top_expiration_total_oi']:,.0}"
                f" | "
                f"{fmt_pct(row['top_expiration_oi_pct'])}"
            )

            report.append("")

    # ========================================================
    # 11 TOP CONTRACTS
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 11. TOP OPTION CONTRACTS"
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
            f"{row['volume_oi']:.2f}"
            if np.isfinite(row["volume_oi"])
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
    # 12 STRUCTURE
    # ========================================================

    report.append("")

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧠 12. STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if call_wall is not None:

        report.append(
            f"📈 Call Wall "
            f"${call_wall['strike']:g}"
        )

    if put_wall is not None:

        report.append(
            f"📉 Put Wall "
            f"${put_wall['strike']:g}"
        )

    if np.isfinite(net_gex):

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

    if call_oi > put_oi:

        report.append(
            "🟢 전체 CALL OI 우세"
        )

    elif put_oi > call_oi:

        report.append(
            "🔴 전체 PUT OI 우세"
        )

    else:

        report.append(
            "🟡 CALL / PUT OI 균형"
        )

    if call_volume > put_volume:

        report.append(
            "🟢 CALL Volume 우세"
        )

    elif put_volume > call_volume:

        report.append(
            "🔴 PUT Volume 우세"
        )

    # ========================================================
    # LIMITATIONS
    # ========================================================

    report.append("")

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
        "• DTE 0 제외"
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
        "• GEX = OI 기반 Proxy"
    )

    report.append(
        "• Zone OI = 현재 수집 범위 내 OI 합산"
    )

    report.append(
        "• 🟩 CALL / 🟥 PUT은 비율 시각화"
    )

    report.append("")

    report.append(
        "Generated: "
        +
        started.strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        )
    )

    return "\n".join(report)


# ============================================================
# SAVE
# ============================================================

def save_outputs(
    data,
    strike_table,
    expiration_structure,
    strike_expiration,
    key_strike_summary,
    today_data,
    top_contracts,
    zone_structure,
    distance_structure,
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

    expiration_structure.to_csv(
        os.path.join(
            output_dir,
            "expiration_structure.csv"
        ),
        index=False
    )

    strike_expiration.to_csv(
        os.path.join(
            output_dir,
            "strike_expiration_structure.csv"
        ),
        index=False
    )

    key_strike_summary.to_csv(
        os.path.join(
            output_dir,
            "key_strike_summary.csv"
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

    zone_structure.to_csv(
        os.path.join(
            output_dir,
            "price_zone_structure.csv"
        ),
        index=False
    )

    distance_structure.to_csv(
        os.path.join(
            output_dir,
            "distance_zone_structure.csv"
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

    with open(
        os.path.join(
            output_dir,
            "report.md"
        ),
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
        "contracts.csv"
    )

    print(
        "strike_structure.csv"
    )

    print(
        "expiration_structure.csv"
    )

    print(
        "strike_expiration_structure.csv"
    )

    print(
        "key_strike_summary.csv"
    )

    print(
        "today_expiration.csv"
    )

    print(
        "top_contracts.csv"
    )

    print(
        "price_zone_structure.csv"
    )

    print(
        "distance_zone_structure.csv"
    )

    print(
        "summary.csv"
    )

    print(
        "report.md"
    )


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
        ).encode("utf-8")

        request = urllib.request.Request(
            url,
            data=payload,
            method="POST"
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
        f"STRIKE RANGE : "
        f"${min_strike:g} ~ ${max_strike:g}"
    )

    print(
        f"DTE RANGE    : 1 ~ {max_dte}"
    )

    print(
        f"FOCUS STRIKE : "
        f"{FOCUS_STRIKES}"
    )

    print(
        f"PRICE ZONES  : "
        f"{PRICE_ZONES}"
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
            "No option rows remain."
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

    # ========================================================
    # 6 STRIKE
    # ========================================================

    strike_table = (
        build_strike_table(
            data
        )
    )

    # ========================================================
    # 7 EXPIRATION
    # ========================================================

    expiration_structure = (
        build_expiration_structure(
            data
        )
    )

    # ========================================================
    # 8 STRIKE × EXPIRATION
    # ========================================================

    strike_expiration = (
        build_strike_expiration_structure(
            data,
            FOCUS_STRIKES
        )
    )

    # ========================================================
    # 9 KEY STRIKE
    # ========================================================

    key_strike_summary = (
        build_key_strike_summary(
            strike_expiration
        )
    )

    # ========================================================
    # 10 TOP
    # ========================================================

    top_contracts = (
        build_top_contracts(
            data
        )
    )

    # ========================================================
    # 11 PRICE ZONE
    # ========================================================

    print()
    print(
        "BUILD PRICE ZONE STRUCTURE"
    )

    zone_structure = (
        build_price_zone_structure(
            data,
            PRICE_ZONES
        )
    )

    print(
        zone_structure[
            [
                "zone",
                "call_oi",
                "put_oi",
                "total_oi"
            ]
        ]
        .to_string(
            index=False
        )
    )

    # ========================================================
    # 12 DISTANCE ZONE
    # ========================================================

    print()
    print(
        "BUILD DISTANCE ZONE STRUCTURE"
    )

    distance_structure = (
        build_distance_zone_structure(
            data
        )
    )

    # ========================================================
    # SUMMARY
    # ========================================================

    calls = data[
        data["option_type"] == "CALL"
    ]

    puts = data[
        data["option_type"] == "PUT"
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

    call_volume_oi = (
        call_volume
        /
        call_oi
        if call_oi > 0
        else np.nan
    )

    put_volume_oi = (
        put_volume
        /
        put_oi
        if put_oi > 0
        else np.nan
    )

    call_gex = (
        calls["gex"]
        .sum(min_count=1)
    )

    put_gex = (
        puts["gex"]
        .sum(min_count=1)
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

    # ========================================================
    # SUMMARY CSV
    # ========================================================

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

                "call_volume_oi":
                    call_volume_oi,

                "put_volume_oi":
                    put_volume_oi,

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
                        if call_wall is not None
                        else np.nan
                    ),

                "put_wall":
                    (
                        put_wall["strike"]
                        if put_wall is not None
                        else np.nan
                    ),

                "atm_iv":
                    atm_iv
            }
        ]
    )

    # ========================================================
    # REPORT
    # ========================================================

    report = build_report(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        top_contracts=top_contracts,
        today_data=today_data,
        zone_structure=zone_structure,
        distance_structure=distance_structure,
        spot=spot,
        symbol=symbol,
        min_strike=min_strike,
        max_strike=max_strike,
        max_dte=max_dte,
        started=started
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_outputs(
        data=data,
        strike_table=strike_table,
        expiration_structure=expiration_structure,
        strike_expiration=strike_expiration,
        key_strike_summary=key_strike_summary,
        today_data=today_data,
        top_contracts=top_contracts,
        zone_structure=zone_structure,
        distance_structure=distance_structure,
        summary=summary,
        report=report,
        output_dir=output_dir
    )

    # ========================================================
    # PRINT
    # ========================================================

    print()
    print(report)

    # ========================================================
    # TELEGRAM
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

        print(
            f"Error type: "
            f"{type(exc).__name__}"
        )

        print(
            f"Error: "
            f"{repr(exc)}"
        )

        raise
