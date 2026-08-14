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

    df["expiration"] = (
        df["expiration"]
        .astype(str)
    )

    df = df[
        df["strike"].notna()
        &
        df["dte"].notna()
        &
        df["option_type"].isin(
            [
                "CALL",
                "PUT",
            ]
        )
    ].copy()

    df["openinterest"] = (
        df["openinterest"]
        .clip(lower=0)
    )

    df["volume"] = (
        df["volume"]
        .clip(lower=0)
    )

    print(
        "Normalized rows:",
        fmt_int(len(df)),
    )

    return df


# ============================================================
# BUILD PRICE ZONES
#
# IMPORTANT:
# We do NOT filter to $80~$100 here.
#
# Zone analysis must see $60~$100.
# ============================================================

def build_price_zones(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("BUILD PRICE ZONE STRUCTURE")
    print("=" * 70)

    zone_definitions = [
        (
            "$60~$70",
            60.0,
            70.0,
        ),
        (
            "$70~$80",
            70.0,
            80.0,
        ),
        (
            "$80~$90",
            80.0,
            90.0,
        ),
        (
            "$90~$100",
            90.0,
            100.0,
        ),
    ]

    rows = []

    for label, low, high in zone_definitions:

        # Lower bound inclusive
        # Upper bound exclusive
        #
        # Last zone is inclusive at $100.

        if high == ZONE_MAX_STRIKE:

            zone = df[
                (df["strike"] >= low)
                &
                (df["strike"] <= high)
            ].copy()

        else:

            zone = df[
                (df["strike"] >= low)
                &
                (df["strike"] < high)
            ].copy()

        call_oi = zone.loc[
            zone["option_type"] == "CALL",
            "openinterest",
        ].sum()

        put_oi = zone.loc[
            zone["option_type"] == "PUT",
            "openinterest",
        ].sum()

        call_volume = zone.loc[
            zone["option_type"] == "CALL",
            "volume",
        ].sum()

        put_volume = zone.loc[
            zone["option_type"] == "PUT",
            "volume",
        ].sum()

        total_oi = (
            call_oi
            + put_oi
        )

        total_volume = (
            call_volume
            + put_volume
        )

        rows.append(
            {
                "zone": label,
                "lower": low,
                "upper": high,
                "call_oi": int(
                    round(call_oi)
                ),
                "put_oi": int(
                    round(put_oi)
                ),
                "total_oi": int(
                    round(total_oi)
                ),
                "call_volume": int(
                    round(call_volume)
                ),
                "put_volume": int(
                    round(put_volume)
                ),
                "total_volume": int(
                    round(total_volume)
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print()

    print(
        f"{'ZONE':<12}"
        f"{'CALL OI':>15}"
        f"{'PUT OI':>15}"
        f"{'TOTAL OI':>15}"
    )

    print("-" * 57)

    for _, row in result.iterrows():

        print(
            f"{row['zone']:<12}"
            f"{fmt_int(row['call_oi']):>15}"
            f"{fmt_int(row['put_oi']):>15}"
            f"{fmt_int(row['total_oi']):>15}"
        )

    result.to_csv(
        ZONE_OUTPUT,
        index=False,
    )

    return result


# ============================================================
# ASCII BAR
# ============================================================

def make_bar(
    value,
    maximum,
    width=BAR_WIDTH,
) -> str:

    try:

        value = float(value)
        maximum = float(maximum)

        if (
            not np.isfinite(value)
            or
            not np.isfinite(maximum)
            or
            maximum <= 0
        ):

            return ""

        count = int(
            round(
                value
                / maximum
                * width
            )
        )

        count = max(
            0,
            min(
                width,
                count,
            ),
        )

        return "█" * count

    except Exception:

        return ""


# ============================================================
# PRINT OI BAR STRUCTURE
# ============================================================

def print_oi_bars(
    zones: pd.DataFrame,
) -> list[str]:

    print()
    print("=" * 70)
    print("PRICE ZONE OI BARS")
    print("=" * 70)

    max_oi = max(
        zones["call_oi"].max(),
        zones["put_oi"].max(),
        1,
    )

    lines = []

    for _, row in zones.iterrows():

        zone = row["zone"]

        call_oi = row["call_oi"]
        put_oi = row["put_oi"]

        call_bar = make_bar(
            call_oi,
            max_oi,
        )

        put_bar = make_bar(
            put_oi,
            max_oi,
        )

        call_line = (
            f"CALL {zone:<8} "
            f"{call_bar:<{BAR_WIDTH}} "
            f"{fmt_int(call_oi)}"
        )

        put_line = (
            f"PUT  {zone:<8} "
            f"{put_bar:<{BAR_WIDTH}} "
            f"{fmt_int(put_oi)}"
        )

        print(call_line)
        print(put_line)
        print()

        lines.append(call_line)
        lines.append(put_line)

    return lines


# ============================================================
# DETAIL $80~$100
# ============================================================

def build_detail_structure(
    df: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print(
        f"DETAILED STRUCTURE "
        f"${DETAIL_MIN_STRIKE:.0f}"
        f"~"
        f"${DETAIL_MAX_STRIKE:.0f}"
    )
    print("=" * 70)

    detail = df[
        (df["strike"] >= DETAIL_MIN_STRIKE)
        &
        (df["strike"] <= DETAIL_MAX_STRIKE)
    ].copy()

    print(
        "Detail rows:",
        fmt_int(len(detail)),
    )

    detail.to_csv(
        DETAIL_OUTPUT,
        index=False,
    )

    return detail


# ============================================================
# EXPIRATION STRUCTURE
# ============================================================

def build_expiration_structure(
    detail: pd.DataFrame,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("BUILD EXPIRATION STRUCTURE")
    print("=" * 70)

    rows = []

    grouped = detail.groupby(
        [
            "expiration",
            "dte",
        ],
        dropna=False,
    )

    for (
        expiration,
        dte,
    ), group in grouped:

        call_oi = group.loc[
            group["option_type"] == "CALL",
            "openinterest",
        ].sum()

        put_oi = group.loc[
            group["option_type"] == "PUT",
            "openinterest",
        ].sum()

        rows.append(
            {
                "expiration": expiration,
                "DTE": int(dte),
                "call_oi": int(
                    round(call_oi)
                ),
                "put_oi": int(
                    round(put_oi)
                ),
                "total_oi": int(
                    round(
                        call_oi
                        + put_oi
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    if not result.empty:

        result = result.sort_values(
            [
                "DTE",
                "expiration",
            ]
        ).reset_index(
            drop=True
        )

    result.to_csv(
        EXPIRATION_OUTPUT,
        index=False,
    )

    return result


# ============================================================
# DISTANCE ZONES
#
# Based on current underlying price.
# ============================================================

def build_distance_structure(
    detail: pd.DataFrame,
    current_price: float,
) -> pd.DataFrame:

    print()
    print("=" * 70)
    print("BUILD DISTANCE ZONE STRUCTURE")
    print("=" * 70)

    if (
        not np.isfinite(current_price)
        or current_price <= 0
    ):

        return pd.DataFrame()

    work = detail.copy()

    work["distance_pct"] = (
        (
            work["strike"]
            - current_price
        )
        /
        current_price
        * 100.0
    )

    bins = [
        -1000,
        -15,
        -10,
        -5,
        0,
        5,
        10,
        15,
        1000,
    ]

    labels = [
        "< -15%",
        "-15%~-10%",
        "-10%~-5%",
        "-5%~0%",
        "0%~5%",
        "5%~10%",
        "10%~15%",
        "> 15%",
    ]

    work["distance_zone"] = pd.cut(
        work["distance_pct"],
        bins=bins,
        labels=labels,
        right=False,
    )

    rows = []

    for label in labels:

        group = work[
            work["distance_zone"]
            == label
        ]

        call_oi = group.loc[
            group["option_type"] == "CALL",
            "openinterest",
        ].sum()

        put_oi = group.loc[
            group["option_type"] == "PUT",
            "openinterest",
        ].sum()

        rows.append(
            {
                "distance_zone": label,
                "call_oi": int(
                    round(call_oi)
                ),
                "put_oi": int(
                    round(put_oi)
                ),
                "total_oi": int(
                    round(
                        call_oi
                        + put_oi
                    )
                ),
            }
        )

    result = pd.DataFrame(
        rows
    )

    result.to_csv(
        DISTANCE_OUTPUT,
        index=False,
    )

    return result


# ============================================================
# CALL WALL / PUT WALL
#
# Here OI is used as the primary structural signal.
# ============================================================

def find_walls(
    detail: pd.DataFrame,
    current_price: float,
) -> tuple[float, float]:

    print()
    print("=" * 70)
    print("CALL WALL / PUT WALL")
    print("=" * 70)

    calls = detail[
        (
            detail["option_type"]
            == "CALL"
        )
        &
        (
            detail["strike"]
            >= current_price
        )
    ].copy()

    puts = detail[
        (
            detail["option_type"]
            == "PUT"
        )
        &
        (
            detail["strike"]
            <= current_price
        )
    ].copy()

    call_wall = np.nan
    put_wall = np.nan

    if not calls.empty:

        calls = (
            calls
            .groupby("strike")[
                "openinterest"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not calls.empty:

            call_wall = safe_float(
                calls.index[0]
            )

    if not puts.empty:

        puts = (
            puts
            .groupby("strike")[
                "openinterest"
            ]
            .sum()
            .sort_values(
                ascending=False
            )
        )

        if not puts.empty:

            put_wall = safe_float(
                puts.index[0]
            )

    print(
        "CALL WALL:",
        fmt_price(call_wall),
    )

    print(
        "PUT WALL :",
        fmt_price(put_wall),
    )

    return (
        call_wall,
        put_wall,
    )


# ============================================================
# STRUCTURE CLASSIFICATION
# ============================================================

def classify_structure(
    current_price: float,
    call_wall: float,
    put_wall: float,
    zones: pd.DataFrame,
) -> str:

    # --------------------------------------------------------
    # Wall break
    # --------------------------------------------------------

    if np.isfinite(call_wall):

        if current_price > call_wall:

            return "BULLISH BREAKOUT"

    if np.isfinite(put_wall):

        if current_price < put_wall:

            return "BEARISH BREAKDOWN"

    # --------------------------------------------------------
    # Zone OI
    # --------------------------------------------------------

    call_total = (
        zones["call_oi"].sum()
    )

    put_total = (
        zones["put_oi"].sum()
    )

    if call_total > put_total * 1.20:

        return "CALL OI DOMINANT"

    if put_total > call_total * 1.20:

        return "PUT OI DOMINANT"

    return "BALANCED OI"


# ============================================================
# SUMMARY
# ============================================================

def build_summary(
    current_price: float,
    raw: pd.DataFrame,
    detail: pd.DataFrame,
    zones: pd.DataFrame,
    expiration: pd.DataFrame,
    distance: pd.DataFrame,
    call_wall: float,
    put_wall: float,
) -> pd.DataFrame:

    call_oi = detail.loc[
        detail["option_type"] == "CALL",
        "openinterest",
    ].sum()

    put_oi = detail.loc[
        detail["option_type"] == "PUT",
        "openinterest",
    ].sum()

    call_volume = detail.loc[
        detail["option_type"] == "CALL",
        "volume",
    ].sum()

    put_volume = detail.loc[
        detail["option_type"] == "PUT",
        "volume",
    ].sum()

    total_oi = (
        call_oi
        + put_oi
    )

    total_volume = (
        call_volume
        + put_volume
    )

    if put_oi > 0:

        call_put_oi_ratio = (
            call_oi
            /
            put_oi
        )

    else:

        call_put_oi_ratio = np.nan

    if put_volume > 0:

        call_put_volume_ratio = (
            call_volume
            /
            put_volume
        )

    else:

        call_put_volume_ratio = np.nan

    structure = classify_structure(
        current_price,
        call_wall,
        put_wall,
        zones,
    )

    result = pd.DataFrame(
        [
            {
                "symbol": SYMBOL,
                "current_price": current_price,
                "raw_rows": len(raw),
                "detail_rows": len(detail),
                "zone_call_oi": int(
                    round(
                        zones["call_oi"].sum()
                    )
                ),
                "zone_put_oi": int(
                    round(
                        zones["put_oi"].sum()
                    )
                ),
                "detail_call_oi": int(
                    round(call_oi)
                ),
                "detail_put_oi": int(
                    round(put_oi)
                ),
                "detail_total_oi": int(
                    round(total_oi)
                ),
                "call_volume": int(
                    round(call_volume)
                ),
                "put_volume": int(
                    round(put_volume)
                ),
                "total_volume": int(
                    round(total_volume)
                ),
                "call_put_oi_ratio": call_put_oi_ratio,
                "call_put_volume_ratio": call_put_volume_ratio,
                "call_wall": call_wall,
                "put_wall": put_wall,
                "structure": structure,
            }
        ]
    )

    result.to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    return result


# ============================================================
# REPORT
# ============================================================

def build_text_report(
    current_price: float,
    raw: pd.DataFrame,
    zones: pd.DataFrame,
    detail: pd.DataFrame,
    expiration: pd.DataFrame,
    distance: pd.DataFrame,
    call_wall: float,
    put_wall: float,
    summary: pd.DataFrame,
    bar_lines: list[str],
) -> str:

    row = summary.iloc[0]

    lines = []

    lines.append(
        "=" * 70
    )

    lines.append(
        f"🔥 {SYMBOL} OPTION STRUCTURE"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"Current Price : ${fmt_price(current_price)}"
    )

    lines.append(
        f"Raw Rows      : {fmt_int(len(raw))}"
    )

    lines.append(
        f"DTE Range     : {MIN_DTE}~{MAX_DTE}"
    )

    lines.append(
        f"Zone Range    : "
        f"${ZONE_MIN_STRIKE:.0f}"
        f"~"
        f"${ZONE_MAX_STRIKE:.0f}"
    )

    lines.append(
        f"Detail Range  : "
        f"${DETAIL_MIN_STRIKE:.0f}"
        f"~"
        f"${DETAIL_MAX_STRIKE:.0f}"
    )

    lines.append("")

    lines.append(
        "CALL WALL : "
        + fmt_price(call_wall)
    )

    lines.append(
        "PUT WALL  : "
        + fmt_price(put_wall)
    )

    lines.append(
        "STRUCTURE : "
        + str(row["structure"])
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "PRICE ZONE OI"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"{'ZONE':<12}"
        f"{'CALL OI':>15}"
        f"{'PUT OI':>15}"
        f"{'TOTAL OI':>15}"
    )

    lines.append(
        "-" * 57
    )

    for _, zone in zones.iterrows():

        lines.append(
            f"{zone['zone']:<12}"
            f"{fmt_int(zone['call_oi']):>15}"
            f"{fmt_int(zone['put_oi']):>15}"
            f"{fmt_int(zone['total_oi']):>15}"
        )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "CALL / PUT OI BARS"
    )

    lines.append(
        "=" * 70
    )

    lines.extend(
        bar_lines
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "DETAIL $80~$100"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        f"CALL OI : "
        f"{fmt_int(row['detail_call_oi'])}"
    )

    lines.append(
        f"PUT OI  : "
        f"{fmt_int(row['detail_put_oi'])}"
    )

    lines.append(
        f"TOTAL OI: "
        f"{fmt_int(row['detail_total_oi'])}"
    )

    lines.append(
        f"CALL/PUT OI: "
        f"{fmt_float(row['call_put_oi_ratio'], 2)}"
    )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "DISTANCE STRUCTURE"
    )

    lines.append(
        "=" * 70
    )

    if not distance.empty:

        for _, d in distance.iterrows():

            lines.append(
                f"{str(d['distance_zone']):<15}"
                f" CALL {fmt_int(d['call_oi']):>12}"
                f" PUT {fmt_int(d['put_oi']):>12}"
            )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "EXPIRATION STRUCTURE"
    )

    lines.append(
        "=" * 70
    )

    if not expiration.empty:

        for _, e in expiration.iterrows():

            lines.append(
                f"{e['expiration']} "
                f"DTE={fmt_int(e['DTE']):>3} "
                f"CALL={fmt_int(e['call_oi']):>10} "
                f"PUT={fmt_int(e['put_oi']):>10}"
            )

    lines.append("")

    lines.append(
        "=" * 70
    )

    lines.append(
        "DATA NOTE"
    )

    lines.append(
        "=" * 70
    )

    lines.append(
        "OI = Yahoo Finance reported Open Interest"
    )

    lines.append(
        "Volume = Yahoo Finance reported Volume"
    )

    lines.append(
        "CALL/PUT side does not prove dealer positioning."
    )

    lines.append(
        "CALL/PUT OI bars show structural OI concentration."
    )

    lines.append(
        "This is not exchange-supplied dealer GEX."
    )

    report = "\n".join(
        lines
    )

    TEXT_OUTPUT.write_text(
        report,
        encoding="utf-8",
    )

    return report


# ============================================================
# SAVE RAW
# ============================================================

def save_raw(
    raw: pd.DataFrame,
) -> None:

    raw.to_csv(
        RAW_OUTPUT,
        index=False,
    )

    print()
    print(
        "RAW SAVED:",
        RAW_OUTPUT,
    )


# ============================================================
# PRINT DETAIL TOP OI
# ============================================================

def print_detail_top(
    detail: pd.DataFrame,
) -> None:

    print()
    print("=" * 70)
    print(
        "TOP OI STRIKES "
        f"${DETAIL_MIN_STRIKE:.0f}"
        f"~"
        f"${DETAIL_MAX_STRIKE:.0f}"
    )
    print("=" * 70)

    if detail.empty:

        print(
            "NO DETAIL DATA"
        )

        return

    grouped = (
        detail
        .groupby(
            [
                "option_type",
                "strike",
            ]
        )[
            "openinterest"
        ]
        .sum()
        .reset_index()
    )

    grouped = (
        grouped
        .sort_values(
            "openinterest",
            ascending=False,
        )
        .head(
            DETAIL_PRINT_ROWS
        )
    )

    for _, row in grouped.iterrows():

        print(
            f"{row['option_type']:<5} "
            f"${fmt_price(row['strike']):>8} "
            f"OI={fmt_int(row['openinterest'])}"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    started = datetime.now()

    print()
    print("=" * 70)
    print(
        f"🔥 {SYMBOL} OPTION STRUCTURE SCANNER"
    )
    print("=" * 70)

    print(
        "DTE:",
        MIN_DTE,
        "~",
        MAX_DTE,
    )

    print(
        "ZONE:",
        f"${ZONE_MIN_STRIKE:.0f}",
        "~",
        f"${ZONE_MAX_STRIKE:.0f}",
    )

    print(
        "DETAIL:",
        f"${DETAIL_MIN_STRIKE:.0f}",
        "~",
        f"${DETAIL_MAX_STRIKE:.0f}",
    )

    # --------------------------------------------------------
    # Yahoo ticker
    # --------------------------------------------------------

    ticker = yf.Ticker(
        SYMBOL
    )

    # --------------------------------------------------------
    # Current price
    # --------------------------------------------------------

    current_price = get_current_price(
        ticker
    )

    # --------------------------------------------------------
    # Collect
    # --------------------------------------------------------

    raw = collect_all_options(
        ticker
    )

    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    df = normalize_option_data(
        raw
    )

    # --------------------------------------------------------
    # Save RAW
    # --------------------------------------------------------

    save_raw(
        df
    )

    # --------------------------------------------------------
    # IMPORTANT
    #
    # Zone analysis first.
    #
    # This prevents:
    #
    # $60~$70 = 0
    # $70~$80 = 0
    #
    # simply because detail filter is $80~$100.
    # --------------------------------------------------------

    zone_source = df[
        (df["strike"] >= ZONE_MIN_STRIKE)
        &
        (df["strike"] <= ZONE_MAX_STRIKE)
    ].copy()

    print()
    print("=" * 70)
    print("APPLY ZONE RANGE")
    print("=" * 70)

    print(
        "After DTE:",
        fmt_int(len(df)),
    )

    print(
        "After Zone "
        f"${ZONE_MIN_STRIKE:.0f}"
        f"~"
        f"${ZONE_MAX_STRIKE:.0f}:",
        fmt_int(len(zone_source)),
    )

    # --------------------------------------------------------
    # Zone
    # --------------------------------------------------------

    zones = build_price_zones(
        zone_source
    )

    # --------------------------------------------------------
    # OI Bars
    # --------------------------------------------------------

    bar_lines = print_oi_bars(
        zones
    )

    # --------------------------------------------------------
    # Detail
    # --------------------------------------------------------

    detail = build_detail_structure(
        zone_source
    )

    # --------------------------------------------------------
    # Expiration
    # --------------------------------------------------------

    expiration = (
        build_expiration_structure(
            detail
        )
    )

    # --------------------------------------------------------
    # Distance
    # --------------------------------------------------------

    distance = (
        build_distance_structure(
            detail,
            current_price,
        )
    )

    # --------------------------------------------------------
    # Walls
    # --------------------------------------------------------

    (
        call_wall,
        put_wall,
    ) = find_walls(
        detail,
        current_price,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = build_summary(
        current_price=current_price,
        raw=df,
        detail=detail,
        zones=zones,
        expiration=expiration,
        distance=distance,
        call_wall=call_wall,
        put_wall=put_wall,
    )

    # --------------------------------------------------------
    # Detail top
    # --------------------------------------------------------

    print_detail_top(
        detail
    )

    # --------------------------------------------------------
    # Text report
    # --------------------------------------------------------

    report = build_text_report(
        current_price=current_price,
        raw=df,
        zones=zones,
        detail=detail,
        expiration=expiration,
        distance=distance,
        call_wall=call_wall,
        put_wall=put_wall,
        summary=summary,
        bar_lines=bar_lines,
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    elapsed = (
        datetime.now()
        - started
    ).total_seconds()

    print()
    print("=" * 70)
    print("✅ SCANNER COMPLETE")
    print("=" * 70)

    print(
        "SYMBOL:",
        SYMBOL,
    )

    print(
        "CURRENT PRICE:",
        fmt_price(current_price),
    )

    print(
        "RAW ROWS:",
        fmt_int(len(df)),
    )

    print(
        "ZONE ROWS:",
        fmt_int(len(zone_source)),
    )

    print(
        "DETAIL ROWS:",
        fmt_int(len(detail)),
    )

    print(
        "CALL WALL:",
        fmt_price(call_wall),
    )

    print(
        "PUT WALL:",
        fmt_price(put_wall),
    )

    print(
        "STRUCTURE:",
        summary.iloc[0]["structure"],
    )

    print(
        "ELAPSED:",
        fmt_float(elapsed, 2),
        "sec",
    )

    print()
    print(
        "OUTPUT FILES:"
    )

    print(
        " -",
        RAW_OUTPUT,
    )

    print(
        " -",
        ZONE_OUTPUT,
    )

    print(
        " -",
        DETAIL_OUTPUT,
    )

    print(
        " -",
        EXPIRATION_OUTPUT,
    )

    print(
        " -",
        DISTANCE_OUTPUT,
    )

    print(
        " -",
        SUMMARY_OUTPUT,
    )

    print(
        " -",
        TEXT_OUTPUT,
    )

    print()
    print("=" * 70)

    return 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print()
        print(
            "❌ INTERRUPTED"
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 70)
        print("❌ SCANNER FAILED")
        print("=" * 70)

        print(
            "Error type:",
            type(exc).__name__,
        )

        print(
            "Error:",
            repr(exc),
        )

        print()
        traceback.print_exc()

        sys.exit(1)
