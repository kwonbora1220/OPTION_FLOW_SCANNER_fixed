import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

SYMBOL = os.getenv("SYMBOL", "RKLB").upper()

MIN_STRIKE = float(os.getenv("MIN_STRIKE", "80"))
MAX_STRIKE = float(os.getenv("MAX_STRIKE", "100"))
MAX_DTE = int(os.getenv("MAX_DTE", "180"))

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "rklb_option_structure"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


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
        return f"{sign}${value / 1_000_000_000:.2f}B"

    if value >= 1_000_000:
        return f"{sign}${value / 1_000_000:.2f}M"

    if value >= 1_000:
        return f"{sign}${value / 1_000:.1f}K"

    return f"{sign}${value:,.0f}"


def fmt_iv(value):

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    if value < 2:
        value *= 100

    return f"{value:.1f}%"


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(ticker):

    print()
    print("=" * 70)
    print("FETCH CURRENT PRICE")
    print("=" * 70)

    # --------------------------------------------------------
    # 1. 1-minute
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

            close = history["Close"].dropna()

            if not close.empty:

                price = float(close.iloc[-1])

                print(
                    f"CURRENT PRICE: ${price:.2f}"
                )

                return price

    except Exception as exc:

        print(
            f"1m price error: {repr(exc)}"
        )

    # --------------------------------------------------------
    # 2. 5-day fallback
    # --------------------------------------------------------

    try:

        history = ticker.history(
            period="5d"
        )

        if (
            not history.empty
            and "Close" in history.columns
        ):

            close = history["Close"].dropna()

            if not close.empty:

                price = float(close.iloc[-1])

                print(
                    f"CURRENT PRICE: ${price:.2f}"
                )

                return price

    except Exception as exc:

        print(
            f"5d price error: {repr(exc)}"
        )

    raise RuntimeError(
        "Unable to determine current price."
    )


# ============================================================
# GEX PROXY
#
# IMPORTANT:
# If Yahoo gamma is missing/zero,
# return NaN instead of fake $0.
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

    if not all(
        np.isfinite(x)
        for x in [
            gamma,
            open_interest,
            spot
        ]
    ):
        return np.nan

    if gamma <= 0:
        return np.nan

    if open_interest <= 0:
        return np.nan

    if spot <= 0:
        return np.nan

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

    if not np.isfinite(volume):
        return 0.0

    if volume <= 0:
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
# DTE
# ============================================================

def calculate_dte(expiration):

    try:

        expiry_date = pd.Timestamp(
            expiration
        ).date()

        today_date = datetime.now(
            timezone.utc
        ).date()

        return (
            expiry_date
            - today_date
        ).days

    except Exception as exc:

        print(
            f"DTE error {expiration}: "
            f"{repr(exc)}"
        )

        return np.nan


# ============================================================
# TODAY EXPIRATION
# ============================================================

def get_today_expiration(expirations):

    today = datetime.now(
        timezone.utc
    ).date()

    today_string = today.isoformat()

    for expiration in expirations:

        if str(expiration) == today_string:

            return str(expiration)

    return None


# ============================================================
# FETCH ALL YAHOO OPTIONS
# ============================================================

def fetch_options():

    print()
    print("=" * 70)
    print("FETCH YAHOO FINANCE OPTION DATA")
    print("=" * 70)

    ticker = yf.Ticker(SYMBOL)

    spot = get_current_price(
        ticker
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

    today_expiration = get_today_expiration(
        expirations
    )

    if today_expiration:

        print(
            f"TODAY EXPIRATION: "
            f"{today_expiration}"
        )

    else:

        print(
            "TODAY EXPIRATION: NOT FOUND"
        )

    rows = []

    successful_expirations = 0
    failed_expirations = 0

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

            failed_expirations += 1

            print(
                "❌ option_chain() FAILED"
            )

            print(
                f"Error: {repr(exc)}"
            )

            continue

        # ----------------------------------------------------
        # CALL
        # ----------------------------------------------------

        try:

            calls = chain.calls

       
