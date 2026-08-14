# ============================================================
# RKLB OPTION STRUCTURE SCANNER
# ============================================================
#
# Yahoo Finance FREE OPTIONS
#
# FLOW:
#
# Yahoo expirations
#        ↓
# ALL CALL / PUT DATA COLLECTION
#        ↓
# NORMALIZATION
#        ↓
# DTE FILTER
#        ↓
# STRIKE FILTER $80 ~ $100
#        ↓
# STRIKE AGGREGATION
#        ↓
# CALL / PUT VOLUME
# CALL / PUT OI
# CALL / PUT PREMIUM
# CALL / PUT GEX
# NET GEX
#        ↓
# REPORT / CSV
#
# IMPORTANT:
# Premium = volume * midpoint * 100
# GEX = OI based proxy
# Trade direction cannot be confirmed from Yahoo free data.
# ============================================================

import os
import traceback
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

SYMBOL = os.getenv(
    "SYMBOL",
    "RKLB"
).upper()

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
# HELPERS
# ============================================================

def safe_float(value):
    """
    Safely convert value to float.
    """

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


def numeric(series):
    """
    Convert pandas series to numeric.
    """

    return pd.to_numeric(
        series,
        errors="coerce"
    )


def fmt_money(value):
    """
    Human readable money.
    """

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
    """
    Format IV.
    """

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    if value < 2:
        value *= 100

    return f"{value:.1f}%"


def fmt_number(value):
    """
    Format integer-like numbers.
    """

    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return f"{value:,.0f}"


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(ticker):
    """
    Try several Yahoo price sources.

    We intentionally do NOT use Timestamp.utcnow().
    """

    # --------------------------------------------------------
    # 1. 1 minute
    # --------------------------------------------------------

    try:

        history = ticker.history(
            period="1d",
            interval="1m",
            prepost=True
        )

        if history is not None and not history.empty:

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                price = safe_float(
                    close.iloc[-1]
                )

                if np.isfinite(price):
                    return price

    except Exception as exc:

        print(
            f"[PRICE] 1m failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # 2. 5 day
    # --------------------------------------------------------

    try:

        history = ticker.history(
            period="5d",
            interval="1d",
            prepost=True
        )

        if history is not None and not history.empty:

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                price = safe_float(
                    close.iloc[-1]
                )

                if np.isfinite(price):
                    return price

    except Exception as exc:

        print(
            f"[PRICE] 5d failed: "
            f"{type(exc).__name__}: {exc}"
        )

    # --------------------------------------------------------
    # 3. fast_info
    # --------------------------------------------------------

    try:

        price = safe_float(
            ticker.fast_info.get(
                "last_price"
            )
        )

        if np.isfinite(price):
            return price

    except Exception as exc:

        print(
            f"[PRICE] fast_info failed: "
            f"{type(exc).__name__}: {exc}"
        )

    raise RuntimeError(
        "Unable to determine current price."
    )


# ============================================================
# DTE CALCULATION
# ============================================================

def calculate_dte(expiration):
    """
    Simple expiration date calculation.

    Yahoo expiration is YYYY-MM-DD.

    DTE is calculated against current UTC date.
    """

    try:

        today = datetime.now(
            timezone.utc
        ).date()

        expiry = datetime.strptime(
            expiration,
            "%Y-%m-%d"
        ).date()

        return (
            expiry - today
        ).days

    except Exception as exc:

        print(
            f"[DTE] failed for "
            f"{expiration}: "
            f"{type(exc).__name__}: {exc}"
        )

        return None


# ============================================================
# GEX PROXY
# ============================================================

def calculate_gex(
    gamma,
    open_interest,
    spot,
    option_type
):
    """
    GEX proxy.

    CALL = positive
    PUT  = negative

    This is NOT dealer-confirmed GEX.
    """

    gamma = safe_float(gamma)

    open_interest = safe_float(
        open_interest
    )

    spot = safe_float(
        spot
    )

    if not all(
        np.isfinite(x)
        for x in [
            gamma,
            open_interest,
            spot
        ]
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
# PREMIUM PROXY
# ============================================================

def calculate_premium(
    volume,
    bid,
    ask,
    last_price
):
    """
    Premium proxy:

        volume * midpoint * 100

    If bid/ask unavailable,
    use lastPrice.
    """

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

    # --------------------------------------------------------
    # Mid price
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
# FETCH ALL YAHOO OPTIONS
# ============================================================

def fetch_options():
    """
    IMPORTANT CHANGE:

    We do NOT filter DTE or strike during collection.

    FLOW:

        Yahoo expirations
                ↓
        CALL / PUT FULL COLLECTION
                ↓
        raw dataframe
                ↓
        later normalization/filtering
    """

    print()
    print("=" * 72)
    print("🔥 FETCH YAHOO FINANCE OPTION DATA")
    print("=" * 72)

    ticker = yf.Ticker(
        SYMBOL
    )

    # --------------------------------------------------------
    # CURRENT PRICE
    # --------------------------------------------------------

    spot = get_current_price(
        ticker
    )

    print(
        f"CURRENT PRICE : {spot:.2f}"
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
            "❌ Yahoo expiration request failed"
        )

        print(
            f"ERROR TYPE : "
            f"{type(exc).__name__}"
        )

        print(
            f"ERROR      : {exc}"
        )

        traceback.print_exc()

        raise RuntimeError(
            "Unable to get Yahoo expirations."
        ) from exc

    print(
        f"TOTAL EXPIRATIONS : "
        f"{len(expirations)}"
    )

    if not expirations:

        raise RuntimeError(
            "Yahoo returned zero expirations."
        )

    print()

    # --------------------------------------------------------
    # RAW COLLECTION
    # --------------------------------------------------------

    rows = []

    successful_expirations = 0

    failed_expirations = 0

    total_call_rows = 0

    total_put_rows = 0

    # --------------------------------------------------------
    # LOOP ALL EXPIRATIONS
    # --------------------------------------------------------

    for index, expiration in enumerate(
        expirations,
        start=1
    ):

        print(
            "-" * 72
        )

        print(
            f"[{index}/{len(expirations)}] "
            f"EXPIRATION : {expiration}"
        )

        # ----------------------------------------------------
        # DTE
        #
        # Used ONLY for information here.
        #
        # NO DTE FILTER.
        # ----------------------------------------------------

        dte = calculate_dte(
            expiration
        )

        if dte is None:

            print(
                "DTE : UNKNOWN"
            )

        else:

            print(
                f"DTE : {dte}"
            )

        # ----------------------------------------------------
        # OPTION CHAIN
        # ----------------------------------------------------

        try:

            chain = ticker
