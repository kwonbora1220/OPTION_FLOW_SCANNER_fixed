# ============================================================
# RKLB OPTION STRUCTURE SCANNER
# Yahoo Finance FREE OPTION DATA
#
# FEATURES
# ------------------------------------------------------------
# 1. Yahoo Finance option chain collection
# 2. DTE 1~180
# 3. Full price-zone OI structure
#      $60~$70
#      $70~$80
#      $80~$90
#      $90~$100
# 4. CALL / PUT OI bars
# 5. Detailed $80~$100 structure
# 6. Expiration structure
# 7. Distance structure
# 8. CALL WALL
# 9. PUT WALL
# 10. CSV outputs
#
# IMPORTANT
# ------------------------------------------------------------
# Zone analysis is done BEFORE the detailed $80~$100 filter.
# Therefore $60~$70 and $70~$80 will NOT incorrectly become 0.
# ============================================================

from __future__ import annotations

from pathlib import Path
from datetime import datetime, date
import math
import sys
import traceback

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIGURATION
# ============================================================

SYMBOL = "RKLB"

# DTE collection range
MIN_DTE = 1
MAX_DTE = 180

# Full PRICE ZONE analysis
ZONE_MIN_STRIKE = 60.0
ZONE_MAX_STRIKE = 100.0

# Detailed option structure
DETAIL_MIN_STRIKE = 80.0
DETAIL_MAX_STRIKE = 100.0

# Output
BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = (
    BASE_DIR
    / "data"
    / "analysis"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RAW_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_raw.csv"
)

ZONE_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_zones.csv"
)

DETAIL_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_detail.csv"
)

EXPIRATION_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_expiration.csv"
)

DISTANCE_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_distance.csv"
)

SUMMARY_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_summary.csv"
)

TEXT_OUTPUT = (
    OUTPUT_DIR
    / "option_structure_report.txt"
)


# ============================================================
# DISPLAY SETTINGS
# ============================================================

BAR_WIDTH = 30

# Maximum number of detailed rows printed
DETAIL_PRINT_ROWS = 100


# ============================================================
# SAFE FORMAT HELPERS
#
# IMPORTANT:
# This fixes:
#
# ValueError:
# Precision not allowed in integer format specifier
#
# Never directly use:
#
# f"{integer:,.2f}"
#
# Instead everything passes through these helpers.
# ============================================================

def fmt_int(value) -> str:

    try:
        if pd.isna(value):
            return "0"

        return f"{int(round(float(value))):,}"

    except Exception:
        return "0"


def fmt_float(
    value,
    decimals: int = 2,
) -> str:

    try:

        if pd.isna(value):
            return "N/A"

        value = float(value)

        if not np.isfinite(value):
            return "N/A"

        return f"{value:,.{decimals}f}"

    except Exception:
        return "N/A"


def fmt_price(value) -> str:

    return fmt_float(
        value,
        2,
    )


def fmt_pct(value) -> str:

    try:

        if pd.isna(value):
            return "N/A"

        return f"{float(value):.2f}%"

    except Exception:
        return "N/A"


def safe_float(value):

    try:

        value = float(value)

        if np.isfinite(value):
            return value

    except Exception:
        pass

    return np.nan


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:

    result = df.copy()

    result.columns = [
        str(c)
        .strip()
        .lower()
        .replace(" ", "_")
        for c in result.columns
    ]

    return result


def normalize_option_type(
    series: pd.Series,
) -> pd.Series:

    return (
        series
        .astype(str)
        .str.upper()
        .str.strip()
        .replace(
            {
                "C": "CALL",
                "CALLS": "CALL",
                "P": "PUT",
                "PUTS": "PUT",
            }
        )
    )


# ============================================================
# GET CURRENT PRICE
# ============================================================

def get_current_price(
    ticker: yf.Ticker,
) -> float:

    print()
    print("=" * 70)
    print("GET CURRENT PRICE")
    print("=" * 70)

    price = np.nan

    # --------------------------------------------------------
    # Fast info
    # --------------------------------------------------------

    try:

        fast_info = ticker.fast_info

        if fast_info is not None:

            for key in [
                "last_price",
                "lastPrice",
            ]:

                try:

                    value = fast_info.get(key)

                    value = safe_float(value)

                    if np.isfinite(value) and value > 0:

                        price = value
                        break

                except Exception:
                    pass

    except Exception:
        pass

    # --------------------------------------------------------
    # History fallback
    # --------------------------------------------------------

    if not np.isfinite(price):

        try:

            hist = ticker.history(
                period="5d",
                auto_adjust=False,
            )

            if not hist.empty:

                close = pd.to_numeric(
                    hist["Close"],
                    errors="coerce",
                ).dropna()

                if not close.empty:

                    price = float(
                        close.iloc[-1]
                    )

        except Exception as exc:

            print(
                "PRICE HISTORY ERROR:",
                repr(exc),
            )

    if not np.isfinite(price):

        raise RuntimeError(
            "Unable to determine current price"
        )

    print(
        "CURRENT PRICE:",
        fmt_price(price),
    )

    return price


# ============================================================
# DTE
# ============================================================

def calculate_dte(
    expiration: str,
) -> int:

    try:

        exp = datetime.strptime(
            str(expiration),
            "%Y-%m-%d",
        ).date()

        today = date.today()

        return (
            exp - today
        ).days

    except Exception:

        return -999


# ============================================================
# COLLECT ONE EXPIRATION
# ============================================================

def collect_expiration(
    ticker: yf.Ticker,
    expiration: str,
    dte: int,
) -> pd.DataFrame:

    print()
    print("-" * 70)
    print(
        f"EXPIRATION: {expiration}"
    )

    print(
        f"DTE       : {dte}"
    )

    try:

        chain = ticker.option_chain(
            expiration
        )

    except Exception as exc:

        print(
            "FAILED:",
            repr(exc),
        )

        return pd.DataFrame()

    frames = []

    # --------------------------------------------------------
    # CALL
    # --------------------------------------------------------

    if chain.calls is not None:

        calls = chain.calls.copy()

        if not calls.empty:

            calls["option_type"] = "CALL"

            calls["expiration"] = expiration

            calls["DTE"] = dte

            frames.append(calls)

    # --------------------------------------------------------
    # PUT
    # --------------------------------------------------------

    if chain.puts is not None:

        puts = chain.puts.copy()

        if not puts.empty:

            puts["option_type"] = "PUT"

            puts["expiration"] = expiration

            puts["DTE"] = dte

            frames.append(puts)

    if not frames:

        print(
            "CALL rows: 0"
        )

        print(
            "PUT rows : 0"
        )

        return pd.DataFrame()

    result = pd.concat(
        frames,
        ignore_index=True,
    )

    calls_count = (
        result["option_type"]
        == "CALL"
    ).sum()

    puts_count = (
        result["option_type"]
        == "PUT"
    ).sum()

    print(
        "CALL rows:",
        calls_count,
    )

    print(
        "PUT rows :",
        puts_count,
    )

    return result


# ============================================================
# COLLECT ALL YAHOO OPTIONS
# ============================================================

def collect_all_options(
    ticker: yf.Ticker,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("YAHOO OPTION COLLECTION")
    print("=" * 70)

    try:

        expirations = list(
            ticker.options
        )

    except Exception as exc:

        raise RuntimeError(
            "Unable to retrieve Yahoo expirations"
        ) from exc

    print(
        "Yahoo expirations:",
        len(expirations),
    )

    selected = []

    for expiration in expirations:

        dte = calculate_dte(
            expiration
        )

        if (
            MIN_DTE
            <= dte
            <= MAX_DTE
        ):

            selected.append(
                (
                    expiration,
                    dte,
                )
            )

    print(
        "Selected expirations:",
        len(selected),
    )

    if not selected:

        raise RuntimeError(
            "No expiration inside DTE range"
        )

    all_frames = []

    successful = 0
    failed = 0

    for index, (
        expiration,
        dte,
    ) in enumerate(
        selected,
        start=1,
    ):

        print()
        print(
            f"[{index}/{len(selected)}]"
        )

        frame = collect_expiration(
            ticker,
            expiration,
            dte,
        )

        if frame.empty:

            failed += 1

        else:

            successful += 1

            all_frames.append(
                frame
            )

    if not all_frames:

        raise RuntimeError(
            "All Yahoo expiration collections failed"
        )

    raw = pd.concat(
        all_frames,
        ignore_index=True,
    )

    print()
    print("=" * 70)
    print("YAHOO COLLECTION COMPLETE")
    print("=" * 70)

    print(
        "Successful expirations:",
        successful,
    )

    print(
        "Failed expirations:",
        failed,
    )

    print(
        "RAW OPTION ROWS:",
        fmt_int(len(raw)),
    )

    return raw


# ============================================================
# NORMALIZE YAHOO DATA
# ============================================================

def normalize_option_data(
    raw: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

    df = normalize_columns(
        raw
    )

    required = [
        "strike",
        "openinterest",
        "volume",
        "option_type",
        "expiration",
        "dte",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        raise RuntimeError(
            "Missing Yahoo columns: "
            + ", ".join(missing)
        )

    df["strike"] = pd.to_numeric(
        df["strike"],
        errors="coerce",
    )

    df["openinterest"] = pd.to_numeric(
        df["openinterest"],
        errors="coerce",
    ).fillna(0)

    df["volume"] = pd.to_numeric(
        df["volume"],
        errors="coerce",
    ).fillna(0)

    df["dte"] = pd.to_numeric(
        df["dte"],
        errors="coerce",
    )

    df["option_type"] = (
        normalize_option_type(
            df["option_type"]
        )
    )

    df
