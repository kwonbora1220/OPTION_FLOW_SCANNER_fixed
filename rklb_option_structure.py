# ============================================================
# RKLB OPTION STRUCTURE
# Yahoo Finance FREE DATA
# $80 -> $85 -> $90 -> $95 -> $100 자동 구조 분석
# ============================================================

import os
import sys
import time
import argparse
import json
import urllib.parse
import urllib.request
import math

from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

DEFAULT_SYMBOL = os.getenv("SYMBOL", "RKLB").upper()

DEFAULT_PRICE = os.getenv("PRICE", "")

MIN_STRIKE = float(os.getenv("MIN_STRIKE", "80"))
MAX_STRIKE = float(os.getenv("MAX_STRIKE", "100"))

MIN_DTE = int(os.getenv("MIN_DTE", "1"))
MAX_DTE = int(os.getenv("MAX_DTE", "180"))

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TZ = ZoneInfo("America/New_York")


# ============================================================
# UTIL
# ============================================================

def safe_float(value, default=np.nan):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").replace("$", "").strip()

        if value == "":
            return default

        return float(value)

    except Exception:
        return default


def safe_int(value, default=0):
    try:
        if value is None:
            return default

        if isinstance(value, str):
            value = value.replace(",", "").strip()

        if value == "":
            return default

        return int(float(value))

    except Exception:
        return default


def fmt_num(value, decimals=0):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        return f"{float(value):,.{decimals}f}"

    except Exception:
        return "N/A"


def fmt_money(value):
    if value is None:
        return "N/A"

    try:
        if pd.isna(value):
            return "N/A"

        value = float(value)

        if abs(value) >= 1_000_000:
            return f"${value / 1_000_000:.2f}M"

        if abs(value) >= 1_000:
            return f"${value / 1_000:.1f}K"

        return f"${value:,.0f}"

    except Exception:
        return "N/A"


def fmt_price(value):
    try:
        if pd.isna(value):
            return "N/A"

        return f"${float(value):,.2f}"

    except Exception:
        return "N/A"


def normalize_columns(df):
    df = df.copy()

    df.columns = [
        str(c).strip()
        for c in df.columns
    ]

    return df


# ============================================================
# PRICE
# ============================================================

def get_current_price(symbol):
    """
    Yahoo Finance current/last price.
    """

    try:
        ticker = yf.Ticker(symbol)

        fast_info = getattr(ticker, "fast_info", None)

        if fast_info:
            for key in [
                "last_price",
                "regular_market_price",
            ]:
                try:
                    value = fast_info.get(key)

                    if value is not None:
                        value = safe_float(value)

                        if not pd.isna(value) and value > 0:
                            return value

                except Exception:
                    pass

    except Exception:
        pass

    # fallback
    try:
        ticker = yf.Ticker(symbol)

        hist = ticker.history(
            period="5d",
            interval="1d",
            auto_adjust=False,
        )

        if hist is not None and not hist.empty:
            close = safe_float(hist["Close"].dropna().iloc[-1])

            if not pd.isna(close):
                return close

    except Exception:
        pass

    return np.nan


# ============================================================
# EXPIRATIONS
# ============================================================

def get_expirations(symbol):
    ticker = yf.Ticker(symbol)

    try:
        expirations = list(ticker.options)

        if not expirations:
            return []

        return expirations

    except Exception as e:
        print(f"[ERROR] expiration load failed: {e}")
        return []


# ============================================================
# DTE
# ============================================================

def calculate_dte(expiration):
    try:
        today = datetime.now(TZ).date()

        exp_date = datetime.strptime(
            expiration,
            "%Y-%m-%d",
        ).date()

        return (exp_date - today).days

    except Exception:
        return None


# ============================================================
# OPTION CHAIN
# ============================================================

def load_option_chain(symbol, expiration):
    ticker = yf.Ticker(symbol)

    try:
        chain = ticker.option_chain(expiration)

        calls = chain.calls.copy()
        puts = chain.puts.copy()

        calls["option_type"] = "CALL"
        puts["option_type"] = "PUT"

        calls["expiration"] = expiration
        puts["expiration"] = expiration

        return calls, puts

    except Exception as e:
        print(
            f"[WARNING] chain failed "
            f"{symbol} {expiration}: {e}"
        )

        return (
            pd.DataFrame(),
            pd.DataFrame(),
        )


# ============================================================
# NORMALIZE OPTION DATA
# ============================================================

def normalize_option_df(df):
    if df is None or df.empty:
        return pd.DataFrame()

    df = normalize_columns(df)

    required_defaults = {
        "strike": np.nan,
        "volume": 0,
        "openInterest": 0,
        "bid": np.nan,
        "ask": np.nan,
        "lastPrice": np.nan,
        "impliedVolatility": np.nan,
    }

    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    numeric_cols = [
        "strike",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
    ]

    for col in numeric_cols:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        )

    if "expiration" not in df.columns:
        df["expiration"] = ""

    if "option_type" not in df.columns:
        df["option_type"] = ""

    df["expiration"] = df["expiration"].astype(str)

    df["option_type"] = (
        df["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    # --------------------------------------------------------
    # Mid price
    # --------------------------------------------------------

    df["mid_price"] = (
        (df["bid"] + df["ask"]) / 2
    )

    df["mid_price"] = df["mid_price"].where(
        df["mid_price"] > 0,
        df["lastPrice"],
    )

    df["mid_price"] = df["mid_price"].fillna(0)

    # --------------------------------------------------------
    # Estimated traded premium
    #
    # IMPORTANT:
    # This is NOT confirmed buy/sell premium.
    # It is a proxy.
    # --------------------------------------------------------

    df["estimated_traded_premium"] = (
        df["volume"].fillna(0)
        * df["mid_price"].fillna(0)
        * 100
    )

    # --------------------------------------------------------
    # Volume / OI
    # --------------------------------------------------------

    df["volume_oi_ratio"] = np.where(
        df["openInterest"] > 0,
        df["volume"] / df["openInterest"],
        np.nan,
    )

    return df


# ============================================================
# COLLECT ALL OPTIONS
# ============================================================

def collect_options(
    symbol,
    min_dte=1,
    max_dte=180,
):
    print("=" * 70)
    print("OPTION DATA COLLECTION")
    print("=" * 70)

    expirations = get_expirations(symbol)

    print(
        f"[INFO] total expirations: "
        f"{len(expirations)}"
    )

    all_frames = []

    for i, expiration in enumerate(expirations, 1):

        dte = calculate_dte(expiration)

        if dte is None:
            continue

        if dte < min_dte or dte > max_dte:
            continue

        print(
            f"[{i}/{len(expirations)}] "
            f"{expiration} "
            f"DTE={dte}"
        )

        calls, puts = load_option_chain(
            symbol,
            expiration,
        )

        for df in [calls, puts]:

            if df is None or df.empty:
                continue

            df = normalize_option_df(df)

            if df.empty:
                continue

            df["DTE"] = dte

            all_frames.append(df)

        # small delay to avoid Yahoo throttling
        time.sleep(0.15)

    if not all_frames:
        raise RuntimeError(
            "No option data collected from Yahoo Finance."
        )

    result = pd.concat(
        all_frames,
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Strike filter
    # --------------------------------------------------------

    result = result[
        result["strike"].between(
            MIN_STRIKE,
            MAX_STRIKE,
        )
    ].copy()

    result.reset_index(drop=True, inplace=True)

    print(
        f"[INFO] filtered rows: "
        f"{len(result):,}"
    )

    return result


# ============================================================
# SUMMARY
# ============================================================

def calculate_summary(df):
    if df is None or df.empty:
        return {}

    calls = df[
        df["option_type"] == "CALL"
    ].copy()

    puts = df[
        df["option_type"] == "PUT"
    ].copy()

    call_volume = calls["volume"].sum()
    put_volume = puts["volume"].sum()

    call_oi = calls["openInterest"].sum()
    put_oi = puts["openInterest"].sum()

    call_premium = calls[
        "estimated_traded_premium"
    ].sum()

    put_premium = puts[
        "estimated_traded_premium"
    ].sum()

    total_volume = call_volume + put_volume

    total_oi = call_oi + put_oi

    total_premium = (
        call_premium + put_premium
    )

    call_volume_ratio = (
        call_volume / total_volume * 100
        if total_volume > 0
        else np.nan
    )

    call_oi_ratio = (
        call_oi / total_oi * 100
        if total_oi > 0
        else np.nan
    )

    call_premium_ratio = (
        call_premium / total_premium * 100
        if total_premium > 0
        else np.nan
    )

    return {
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "total_volume": total_volume,
        "total_oi": total_oi,
        "total_premium": total_premium,
        "call_volume_ratio": call_volume_ratio,
        "call_oi_ratio": call_oi_ratio,
        "call_premium_ratio": call_premium_ratio,
    }


# ============================================================
# STRIKE STRUCTURE
# ============================================================

def calculate_strike_structure(df):
    """
    Aggregate CALL / PUT data by strike.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(
            ["strike", "option_type"],
            dropna=False,
        )
        .agg(
            volume=("volume", "sum"),
            openInterest=("openInterest", "sum"),
            premium=(
                "estimated_traded_premium",
                "sum",
            ),
        )
        .reset_index()
    )

    if grouped.empty:
        return pd.DataFrame()

    pivot = grouped.pivot_table(
        index="strike",
        columns="option_type",
        values=[
            "volume",
            "openInterest",
            "premium",
        ],
        aggfunc="sum",
        fill_value=0,
    )

    pivot.columns = [
        "_".join(
            [
                str(x)
                for x in col
                if str(x) != ""
            ]
        )
        for col in pivot.columns
    ]

    pivot = pivot.reset_index()

    required = [
        "volume_CALL",
        "volume_PUT",
        "openInterest_CALL",
        "openInterest_PUT",
        "premium_CALL",
        "premium_PUT",
    ]

    for col in required:
        if col not in pivot.columns:
            pivot[col] = 0

    pivot["total_oi"] = (
        pivot["openInterest_CALL"]
        + pivot["openInterest_PUT"]
    )

    pivot["total_volume"] = (
        pivot["volume_CALL"]
        + pivot["volume_PUT"]
    )

    pivot["call_oi_ratio"] = np.where(
        pivot["total_oi"] > 0,
        pivot["openInterest_CALL"]
        / pivot["total_oi"]
        * 100,
        np.nan,
    )

    pivot["call_volume_ratio"] = np.where(
        pivot["total_volume"] > 0,
        pivot["volume_CALL"]
        / pivot["total_volume"]
        * 100,
        np.nan,
    )

    pivot["premium_total"] = (
        pivot["premium_CALL"]
        + pivot["premium_PUT"]
    )

    pivot["call_premium_ratio"] = np.where(
        pivot["premium_total"] > 0,
        pivot["premium_CALL"]
        / pivot["premium_total"]
        * 100,
        np.nan,
    )

    return pivot.sort_values(
        "strike"
    ).reset_index(drop=True)


# ============================================================
# TARGET STRIKES
# ============================================================

def get_target_strikes(
    price,
    structure,
):
    """
    Automatically identify important strikes.

    Priority:
    1. $80 / current-area
    2. $85
    3. $90
    4. $95
    5. $100

    If current price differs significantly,
    dynamically generate nearby $5 levels.
    """

    if structure is None or structure.empty:
        return []

    available = sorted(
        structure["strike"]
        .dropna()
        .unique()
        .tolist()
    )

    if not available:
        return []

    # Standard $5 ladder around current price
    lower = math.floor(price / 5) * 5

    candidates = [
        lower,
        lower + 5,
        lower + 10,
        lower + 15,
        lower + 20,
    ]

    # Always include requested RKLB levels if available
    standard = [
        80,
        85,
        90,
        95,
        100,
    ]

    candidates.extend(standard)

    selected = []

    for target in candidates:

        # exact match
        exact = [
            x
            for x in available
            if abs(x - target) < 0.001
        ]

        if exact:
            strike = exact[0]

        else:
            # nearest strike within $1.50
            nearest = min(
                available,
                key=lambda x: abs(x - target),
            )

            if abs(nearest - target) <= 1.5:
                strike = nearest
            else:
                continue

        if strike not in selected:
            selected.append(strike)

    return sorted(selected)


# ============================================================
# STRIKE IMPORTANCE
# ============================================================

def strike_importance(row, price):
    """
    Score structural importance.

    OI + Volume + Premium + Call dominance
    """

    call_oi = safe_float(
        row.get("openInterest_CALL", 0),
        0,
    )

    put_oi = safe_float(
        row.get("openInterest_PUT", 0),
        0,
    )

    call_volume = safe_float(
        row.get("volume_CALL", 0),
        0,
    )

    put_volume = safe_float(
        row.get("volume_PUT", 0),
        0,
    )

    call_premium = safe_float(
        row.get("premium_CALL", 0),
        0,
    )

    put_premium = safe_float(
        row.get("premium_PUT", 0),
        0,
    )

    total_oi = call_oi + put_oi
    total_volume = call_volume + put_volume
    total_premium = (
        call_premium + put_premium
    )

    call_oi_ratio = (
        call_oi / total_oi * 100
        if total_oi > 0
        else 50
    )

    call_volume_ratio = (
        call_volume / total_volume * 100
        if total_volume > 0
        else 50
    )

    call_premium_ratio = (
        call_premium / total_premium * 100
        if total_premium > 0
        else 50
    )

    # log scaling prevents huge OI from dominating everything
    oi_component = math.log1p(total_oi)

    volume_component = math.log1p(
        total_volume
    )

    premium_component = math.log1p(
        total_premium
    )

    call_bias = (
        call_oi_ratio * 0.35
        + call_volume_ratio * 0.35
        + call_premium_ratio * 0.30
    )

    # distance penalty
    strike = safe_float(
        row.get("strike"),
        price,
    )

    distance = abs(strike - price)

    distance_factor = 1 / (
        1 + distance / max(price, 1)
    )

    raw = (
        oi_component * 0.35
        + volume_component * 0.25
        + premium_component * 0.20
        + call_bias / 10 * 0.20
    )

    raw *= (
        0.75
        + 0.25 * distance_factor
    )

    return raw


# ============================================================
# TARGET LABEL
# ============================================================

def target_label(
    strike,
    price,
    ordered_targets,
):
    """
    Assign human-readable structure label.
    """

    strike = float(strike)

    if not ordered_targets:
        return "STRUCTURE"

    # Exact RKLB-style labels
    if abs(strike - 80) <= 0.01:
        return "🛡 $80 방어선"

    if abs(strike - 85) <= 0.01:
        return "🟢 $85 1차 돌파"

    if abs(strike - 90) <= 0.01:
        return "🔥 $90 핵심"

    if abs(strike - 95) <= 0.01:
        return "🚀 $95 상승 확인"

    if abs(strike - 100) <= 0.01:
        return "🎯 $100 최대 집중"

    if strike < price:
        return f"🛡 ${strike:.0f} 지지"

    return f"🎯 ${strike:.0f} 저항/목표"


# ============================================================
# TARGET STRUCTURE
# ============================================================

def build_target_structure(
    structure,
    price,
):
    if structure is None or structure.empty:
        return []

    targets = get_target_strikes(
        price,
        structure,
    )

    if not targets:
        return []

    results = []

    for strike in targets:

        rows = structure[
            np.isclose(
                structure["strike"],
                strike,
                atol=0.001,
            )
        ]

        if rows.empty:
            continue

        row = rows.iloc[0]

        importance = strike_importance(
            row,
            price,
        )

        call_oi = safe_int(
            row.get(
                "openInterest_CALL",
                0,
            )
        )

        put_oi = safe_int(
            row.get(
                "openInterest_PUT",
                0,
            )
        )

        call_volume = safe_int(
            row.get(
                "volume_CALL",
                0,
            )
        )

        put_volume = safe_int(
            row.get(
                "volume_PUT",
                0,
            )
        )

        call_premium = safe_float(
            row.get(
                "premium_CALL",
                0,
            ),
            0,
        )

        put_premium = safe_float(
            row.get(
                "premium_PUT",
                0,
            ),
            0,
        )

        total_oi = call_oi + put_oi

        total_volume = (
            call_volume + put_volume
        )

        total_premium = (
            call_premium + put_premium
        )

        call_oi_ratio = (
            call_oi / total_oi * 100
            if total_oi > 0
            else 0
        )

        call_volume_ratio = (
            call_volume / total_volume * 100
            if total_volume > 0
            else 0
        )

        call_premium_ratio = (
            call_premium
            / total_premium
            * 100
            if total_premium > 0
            else 0
        )

        # ----------------------------------------------------
        # Strength
        # ----------------------------------------------------

        bullish_count = 0

        if call_oi_ratio >= 60:
            bullish_count += 1

        if call_volume_ratio >= 60:
            bullish_count += 1

        if call_premium_ratio >= 60:
            bullish_count += 1

        if call_volume > put_volume * 2:
            bullish_count += 1

        if call_oi > put_oi * 2:
            bullish_count += 1

        if bullish_count >= 4:
            strength = "🟢🟢 VERY STRONG"

        elif bullish_count >= 3:
            strength = "🟢 STRONG"

        elif bullish_count >= 2:
            strength = "🟡 MIXED"

        else:
            strength = "🔴 WEAK"

        results.append(
            {
                "strike": strike,
                "label": target_label(
                    strike,
                    price,
                    targets,
                ),
                "call_oi": call_oi,
                "put_oi": put_oi,
                "call_volume": call_volume,
                "put_volume": put_volume,
                "call_premium": call_premium,
                "put_premium": put_premium,
                "total_oi": total_oi,
                "total_volume": total_volume,
                "call_oi_ratio": call_oi_ratio,
                "call_volume_ratio": call_volume_ratio,
                "call_premium_ratio": call_premium_ratio,
                "importance": importance,
                "strength": strength,
            }
        )

    return sorted(
        results,
        key=lambda x: x["strike"],
    )


# ============================================================
# EXPIRATION STRUCTURE
# ============================================================

def calculate_expiration_structure(df):
    """
    Protect against empty expiration data.
    """

    if df is None or df.empty:
        return pd.DataFrame()

    required = [
        "expiration",
        "option_type",
        "volume",
        "openInterest",
        "estimated_traded_premium",
        "DTE",
    ]

    for col in required:
        if col not in df.columns:
            return pd.DataFrame()

    try:

        result = (
            df.groupby(
                [
                    "expiration",
                    "DTE",
                    "option_type",
                ],
                dropna=False,
            )
            .agg(
                volume=("volume", "sum"),
                openInterest=(
                    "openInterest",
                    "sum",
                ),
                premium=(
                    "estimated_traded_premium",
                    "sum",
                ),
            )
            .reset_index()
        )

        if result.empty:
            return pd.DataFrame()

        return result

    except Exception as e:
        print(
            f"[WARNING] expiration structure failed: {e}"
        )

        return pd.DataFrame()


# ============================================================
# SHORT-TERM EXPIRATION
# ============================================================

def get_short_term_structure(
    expiration_structure,
):
    """
    Find nearest DTE <= 7.
    """

    if (
        expiration_structure is None
        or expiration_structure.empty
    ):
        return []

    result = []

    try:

        expirations = (
            expiration_structure[
                expiration_structure["DTE"]
                <= 7
            ]
            .sort_values(
                ["DTE", "expiration"]
            )
        )

        if expirations.empty:
            return []

        for expiration, group in expirations.groupby(
            "expiration"
        ):

            call = group[
                group["option_type"] == "CALL"
            ]

            put = group[
                group["option_type"] == "PUT"
            ]

            call_oi = (
                call["openInterest"].sum()
                if not call.empty
                else 0
            )

            put_oi = (
                put["openInterest"].sum()
                if not put.empty
                else 0
            )

            call_volume = (
                call["volume"].sum()
                if not call.empty
                else 0
            )

            put_volume = (
                put["volume"].sum()
                if not put.empty
                else 0
            )

            result.append(
                {
                    "expiration": expiration,
                    "DTE": int(
                        group["DTE"].iloc[0]
                    ),
                    "call_oi": call_oi,
                    "put_oi": put_oi,
                    "call_volume": call_volume,
                    "put_volume": put_volume,
                }
            )

            # Only nearest few expirations
            if len(result) >= 3:
                break

    except Exception as e:
        print(
            f"[WARNING] short term structure failed: {e}"
        )

    return result


# ============================================================
# BULLISH SCORE
# ============================================================

def calculate_bullish_score(
    summary,
    target_structures,
):
    """
    0-100 composite structural score.

    This is an estimate based only on free Yahoo data.
    """

    if not summary:
        return 0

    score = 0.0

    # --------------------------------------------------------
    # Volume
    # --------------------------------------------------------

    call_volume_ratio = safe_float(
        summary.get(
            "call_volume_ratio",
            50,
        ),
        50,
    )

    if call_volume_ratio >= 80:
        score += 25

    elif call_volume_ratio >= 70:
        score += 20

    elif call_volume_ratio >= 60:
        score += 15

    elif call_volume_ratio >= 50:
        score += 8

    # --------------------------------------------------------
    # OI
    # --------------------------------------------------------

    call_oi_ratio = safe_float(
        summary.get(
            "call_oi_ratio",
            50,
        ),
        50,
    )

    if call_oi_ratio >= 75:
        score += 25

    elif call_oi_ratio >= 65:
        score += 20

    elif call_oi_ratio >= 55:
        score += 12

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    call_premium_ratio = safe_float(
        summary.get(
            "call_premium_ratio",
            50,
        ),
        50,
    )

    if call_premium_ratio >= 75:
        score += 25

    elif call_premium_ratio >= 65:
        score += 20

    elif call_premium_ratio >= 55:
        score += 12

    # --------------------------------------------------------
    # Key strike structure
    # --------------------------------------------------------

    key_bonus = 0

    for item in target_structures:

        strike = item["strike"]

        if strike in [
            85,
            90,
            95,
            100,
        ]:

            if (
                item["call_oi_ratio"] >= 65
                and item["call_volume_ratio"] >= 65
            ):
                key_bonus += 6

    score += min(key_bonus, 25)

    return min(
        round(score),
        100,
    )


# ============================================================
# BULLISH LABEL
# ============================================================

def score_label(score):

    if score >= 80:
        return "🟢 BULLISH — 매우 강함"

    if score >= 70:
        return "🟢 BULLISH"

    if score >= 60:
        return "🟡 BULLISH — 보통"

    if score >= 45:
        return "🟡 MIXED"

    return "🔴 BEARISH / WEAK"


# ============================================================
# SCENARIO
# ============================================================

def build_price_scenario(
    price,
    target_structures,
):
    """
    Generate:

    $80 support
    $85 breakout
    $90 key
    $95 confirmation
    $100 target/resistance
    """

    lookup = {
        round(x["strike"], 2): x
        for x in target_structures
    }

    lines = []

    # --------------------------------------------------------
    # $80
    # --------------------------------------------------------

    row = lookup.get(80)

    if row:

        if row["call_oi"] > row["put_oi"]:
            lines.append(
                "🛡 $80 방어선 → CALL OI 우위"
            )
        else:
            lines.append(
                "🛡 $80 방어선 → 양방향 OI 집중"
            )

    # --------------------------------------------------------
    # $85
    # --------------------------------------------------------

    row = lookup.get(85)

    if row:

        if (
            row["call_volume"]
            > row["put_volume"] * 2
        ):
            lines.append(
                "🟢 $85 → 1차 돌파 확인 구간"
            )
        else:
            lines.append(
                "🟡 $85 → 거래량 확인 필요"
            )

    # --------------------------------------------------------
    # $90
    # --------------------------------------------------------

    row = lookup.get(90)

    if row:

        if (
            row["call_oi"]
            > row["put_oi"] * 2
            and row["call_volume"]
            > row["put_volume"] * 2
        ):
            lines.append(
                "🔥 $90 → 핵심 저항 / "
                "돌파 시 상승 가속 가능"
            )
        else:
            lines.append(
                "🔥 $90 → 핵심 판단 구간"
            )

    # --------------------------------------------------------
    # $95
    # --------------------------------------------------------

    row = lookup.get(95)

    if row:

        if (
            row["call_oi"]
            > row["put_oi"] * 2
        ):
            lines.append(
                "🚀 $95 → 상승 확인 / "
                "다음 목표 구간"
            )
        else:
            lines.append(
                "🟡 $95 → 추가 확인 필요"
            )

    # --------------------------------------------------------
    # $100
    # --------------------------------------------------------

    row = lookup.get(100)

    if row:

        if row["total_oi"] >= 10000:
            lines.append(
                "🎯 $100 → 최대 옵션 집중 "
                "목표/저항"
            )
        else:
            lines.append(
                "🎯 $100 → 상방 목표 후보"
            )

    # --------------------------------------------------------
    # Dynamic fallback
    # --------------------------------------------------------

    if not lines:

        above = [
            x
            for x in target_structures
            if x["strike"] > price
        ]

        below = [
            x
            for x in target_structures
            if x["strike"] < price
        ]

        if below:
            nearest_support = max(
                below,
                key=lambda x: x["strike"],
            )

            lines.append(
                f"🛡 ${nearest_support['strike']:.0f} "
                "지지 후보"
            )

        if above:
            nearest_resistance = min(
                above,
                key=lambda x: x["strike"],
            )

            lines.append(
                f"🎯 ${nearest_resistance['strike']:.0f} "
                "저항 후보"
            )

    return lines


# ============================================================
# TELEGRAM ESCAPE
# ============================================================

def escape_telegram(text):
    """
    Telegram HTML mode.
    """

    if text is None:
        return ""

    text = str(text)

    text = text.replace("&", "&amp;")
    text = text.replace("<", "&lt;")
    text = text.replace(">", "&gt;")

    return text


# ============================================================
# TELEGRAM SEND
# ============================================================

def send_telegram(
    message,
    token=None,
    chat_id=None,
):
    token = token or TELEGRAM_BOT_TOKEN
    chat_id = chat_id or TELEGRAM_CHAT_ID

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is missing."
        )

    if not chat_id:
        raise RuntimeError(
            "TELEGRAM_CHAT_ID is missing."
        )

    url = (
        f"https://api.telegram.org/bot"
        f"{token}/sendMessage"
    )

    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    data = urllib.parse.urlencode(
        payload
    ).encode("utf-8")

    request = urllib.request.Request(
        url,
        data=data,
        method="POST",
    )

    try:

        with urllib.request.urlopen(
            request,
            timeout=30,
        ) as response:

            raw = response.read().decode(
                "utf-8"
            )

            result = json.loads(raw)

            if not result.get("ok"):
                raise RuntimeError(
                    f"Telegram API error: {result}"
                )

            return True

    except Exception as e:

        print(
            f"[ERROR] Telegram send failed: {e}"
        )

        # IMPORTANT:
        # Workflow must fail if Telegram fails.
        raise


# ============================================================
# TELEGRAM MESSAGE
# ============================================================

def build_telegram_message(
    symbol,
    price,
    summary,
    target_structures,
    short_term,
    score,
):
    msg = []

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        f"🔥 <b>{escape_telegram(symbol)} "
        "OPTION STRUCTURE</b>"
    )

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        f"💰 현재가: <b>{fmt_price(price)}</b>"
    )

    msg.append(
        f"🎯 분석 Strike: "
        f"${MIN_STRIKE:.0f} ~ "
        f"${MAX_STRIKE:.0f}"
    )

    msg.append(
        f"📅 분석 DTE: "
        f"{MIN_DTE} ~ {MAX_DTE}"
    )

    msg.append("")

    # ========================================================
    # OVERALL
    # ========================================================

    msg.append(
        "📊 <b>1. OPTION FLOW</b>"
    )

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        f"CALL Volume: "
        f"{fmt_num(summary['call_volume'])}"
    )

    msg.append(
        f"PUT Volume : "
        f"{fmt_num(summary['put_volume'])}"
    )

    msg.append(
        f"CALL Volume Ratio: "
        f"{summary['call_volume_ratio']:.1f}%"
    )

    msg.append(
        f"CALL OI: "
        f"{fmt_num(summary['call_oi'])}"
    )

    msg.append(
        f"PUT OI : "
        f"{fmt_num(summary['put_oi'])}"
    )

    msg.append(
        f"CALL OI Ratio: "
        f"{summary['call_oi_ratio']:.1f}%"
    )

    msg.append(
        f"CALL Premium: "
        f"{fmt_money(summary['call_premium'])}"
    )

    msg.append(
        f"PUT Premium : "
        f"{fmt_money(summary['put_premium'])}"
    )

    msg.append(
        f"CALL Premium Ratio: "
        f"{summary['call_premium_ratio']:.1f}%"
    )

    msg.append("")

    # ========================================================
    # SCORE
    # ========================================================

    msg.append(
        "⭐ <b>2. COMPOSITE STRUCTURE</b>"
    )

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        f"🧮 Bullish Score: "
        f"<b>{score}/100</b>"
    )

    msg.append(
        f"📌 {score_label(score)}"
    )

    msg.append("")

    # ========================================================
    # KEY STRIKES
    # ========================================================

    msg.append(
        "🎯 <b>3. KEY PRICE STRUCTURE</b>"
    )

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for item in target_structures:

        strike = item["strike"]

        msg.append(
            f"\n<b>{escape_telegram(item['label'])}</b>"
        )

        msg.append(
            f"CALL OI: "
            f"{fmt_num(item['call_oi'])}"
            f" / PUT OI: "
            f"{fmt_num(item['put_oi'])}"
        )

        msg.append(
            f"CALL Vol: "
            f"{fmt_num(item['call_volume'])}"
            f" / PUT Vol: "
            f"{fmt_num(item['put_volume'])}"
        )

        msg.append(
            f"CALL Premium: "
            f"{fmt_money(item['call_premium'])}"
            f" / PUT: "
            f"{fmt_money(item['put_premium'])}"
        )

        msg.append(
            f"CALL OI Ratio: "
            f"{item['call_oi_ratio']:.1f}%"
        )

        msg.append(
            f"CALL Volume Ratio: "
            f"{item['call_volume_ratio']:.1f}%"
        )

        msg.append(
            f"Strength: "
            f"{item['strength']}"
        )

    # ========================================================
    # SHORT TERM
    # ========================================================

    if short_term:

        msg.append("")

        msg.append(
            "📅 <b>4. SHORT TERM DTE</b>"
        )

        msg.append(
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        for item in short_term:

            msg.append(
                f"{item['expiration']} "
                f"(DTE {item['DTE']})"
            )

            msg.append(
                f"CALL OI "
                f"{fmt_num(item['call_oi'])}"
                f" / PUT OI "
                f"{fmt_num(item['put_oi'])}"
            )

            msg.append(
                f"CALL Vol "
                f"{fmt_num(item['call_volume'])}"
                f" / PUT Vol "
                f"{fmt_num(item['put_volume'])}"
            )

    # ========================================================
    # SCENARIO
    # ========================================================

    scenario = build_price_scenario(
        price,
        target_structures,
    )

    if scenario:

        msg.append("")

        msg.append(
            "📈 <b>5. PRICE SCENARIO</b>"
        )

        msg.append(
            "━━━━━━━━━━━━━━━━━━━━━━━━"
        )

        for line in scenario:
            msg.append(line)

    # ========================================================
    # WARNING
    # ========================================================

    msg.append("")

    msg.append(
        "⚠️ <b>IMPORTANT</b>"
    )

    msg.append(
        "CALL OI/Volume은 CALL-side 구조를 "
        "의미하지만 실제 BTO/STO를 확정하지 않습니다."
    )

    msg.append(
        "Yahoo 무료 데이터만으로 "
        "Call Buy / Call Sell / Covered Call / "
        "Spread를 완전히 구분할 수 없습니다."
    )

    msg.append(
        "Premium은 실제 매수대금이 아닌 "
        "Volume × Mid Price × 100 기반 Proxy입니다."
    )

    msg.append("")

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        f"🧠 <b>FINAL VIEW</b>: "
        f"{score_label(score)}"
    )

    msg.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    msg.append(
        "🛡 $80 → 방어선"
    )

    msg.append(
        "🟢 $85 → 1차 돌파"
    )

    msg.append(
        "🔥 $90 → 핵심"
    )

    msg.append(
        "🚀 $95 → 상승 확인"
    )

    msg.append(
        "🎯 $100 → 최대 집중 목표/저항"
    )

    return "\n".join(msg)


# ============================================================
# SAVE CSV
# ============================================================

def save_outputs(
    df,
    structure,
    target_structures,
    output_dir="data/analysis",
):
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    df.to_csv(
        os.path.join(
            output_dir,
            "rklb_option_raw.csv",
        ),
        index=False,
    )

    structure.to_csv(
        os.path.join(
            output_dir,
            "rklb_strike_structure.csv",
        ),
        index=False,
    )

    if target_structures:

        pd.DataFrame(
            target_structures
        ).to_csv(
            os.path.join(
                output_dir,
                "rklb_key_levels.csv",
            ),
            index=False,
        )

    print(
        f"[INFO] outputs saved to "
        f"{output_dir}"
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--symbol",
        default=DEFAULT_SYMBOL,
    )

    parser.add_argument(
        "--price",
        default=DEFAULT_PRICE,
    )

    parser.add_argument(
        "--min-strike",
        type=float,
        default=MIN_STRIKE,
    )

    parser.add_argument(
        "--max-strike",
        type=float,
        default=MAX_STRIKE,
    )

    parser.add_argument(
        "--min-dte",
        type=int,
        default=MIN_DTE,
    )

    parser.add_argument(
        "--max-dte",
        type=int,
        default=MAX_DTE,
    )

    parser.add_argument(
        "--no-telegram",
        action="store_true",
    )

    args = parser.parse_args()

    symbol = args.symbol.upper()

    global MIN_STRIKE
    global MAX_STRIKE
    global MIN_DTE
    global MAX_DTE

    MIN_STRIKE = args.min_strike
    MAX_STRIKE = args.max_strike

    MIN_DTE = args.min_dte
    MAX_DTE = args.max_dte

    print("=" * 70)
    print(
        f"🔥 {symbol} OPTION STRUCTURE"
    )
    print("=" * 70)

    # ========================================================
    # PRICE
    # ========================================================

    if args.price:
        price = safe_float(args.price)

    else:
        price = get_current_price(
            symbol
        )

    if pd.isna(price) or price <= 0:
        raise RuntimeError(
            "Unable to determine current price."
        )

    print(
        f"[INFO] current price: "
        f"${price:.2f}"
    )

    # ========================================================
    # COLLECT
    # ========================================================

    df = collect_options(
        symbol,
        MIN_DTE,
        MAX_DTE,
    )

    if df.empty:
        raise RuntimeError(
            "Option dataframe is empty."
        )

    # ========================================================
    # SUMMARY
    # ========================================================

    summary = calculate_summary(df)

    if not summary:
        raise RuntimeError(
            "Unable to calculate option summary."
        )

    # ========================================================
    # STRIKE STRUCTURE
    # ========================================================

    structure = calculate_strike_structure(
        df
    )

    if structure.empty:
        raise RuntimeError(
            "Strike structure is empty."
        )

    # ========================================================
    # KEY LEVELS
    # ========================================================

    target_structures = (
        build_target_structure(
            structure,
            price,
        )
    )

    # ========================================================
    # EXPIRATION STRUCTURE
    # ========================================================

    expiration_structure = (
        calculate_expiration_structure(
            df
        )
    )

    # ========================================================
    # SHORT TERM
    # ========================================================

    short_term = get_short_term_structure(
        expiration_structure
    )

    # ========================================================
    # SCORE
    # ========================================================

    score = calculate_bullish_score(
        summary,
        target_structures,
    )

    # ========================================================
    # SAVE
    # ========================================================

    save_outputs(
        df,
        structure,
        target_structures,
    )

    # ========================================================
    # BUILD TELEGRAM
    # ========================================================

    message = build_telegram_message(
        symbol=symbol,
        price=price,
        summary=summary,
        target_structures=target_structures,
        short_term=short_term,
        score=score,
    )

    print("")
    print(message)
    print("")

    # ========================================================
    # TELEGRAM
    # ========================================================

    if args.no_telegram:

        print(
            "[INFO] Telegram disabled."
        )

        return 0

    # IMPORTANT:
    # Telegram failure raises exception.
    # Therefore GitHub Actions workflow fails.
    send_telegram(
        message
    )

    print(
        "[SUCCESS] Telegram sent."
    )

    return 0


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    try:

        sys.exit(
            main()
        )

    except KeyboardInterrupt:

        print(
            "\n[ERROR] interrupted"
        )

        sys.exit(130)

    except Exception as e:

        print(
            "\n[ERROR]"
        )

        print(
            str(e)
        )

        sys.exit(1)
