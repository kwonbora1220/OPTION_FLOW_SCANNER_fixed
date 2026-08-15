
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

DEFAULT_SYMBOL = os.getenv("SYMBOL", "RKLB").upper()
DEFAULT_PRICE = os.getenv("PRICE", "")

MIN_STRIKE = float(os.getenv("MIN_STRIKE", "80"))
MAX_STRIKE = float(os.getenv("MAX_STRIKE", "100"))
MAX_DTE = int(os.getenv("MAX_DTE", "180"))

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    "rklb_option_structure"
)

FOCUS_STRIKES = [80, 85, 90, 95, 100]

BAR_WIDTH = 10
BAR_MIN_WIDTH = 1

US_EASTERN = ZoneInfo("America/New_York")

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# BASIC HELPERS
# ============================================================

def market_today():
    return datetime.now(US_EASTERN).date()


def safe_float(value):
    try:
        value = float(value)
        if np.isfinite(value):
            return value
    except Exception:
        pass
    return np.nan


def numeric(series):
    return pd.to_numeric(series, errors="coerce")


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


def fmt_dte(value):
    value = safe_float(value)

    if not np.isfinite(value):
        return "N/A"

    return str(int(value))


def clean_expiration(value):
    if value is None:
        return None

    try:
        return pd.Timestamp(value).strftime("%Y-%m-%d")
    except Exception:
        return str(value)


# ============================================================
# BAR
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

    if not np.isfinite(call_value) or call_value < 0:
        call_value = 0.0

    if not np.isfinite(put_value) or put_value < 0:
        put_value = 0.0

    if not np.isfinite(reference_total) or reference_total <= 0:
        reference_total = 0.0

    total = call_value + put_value

    if total <= 0:
        return "·"

    if reference_total > 0:
        scale_ratio = total / reference_total
    else:
        scale_ratio = 1.0

    scale_ratio = max(0.0, min(scale_ratio, 1.0))

    bar_length = int(round(scale_ratio * max_width))
    bar_length = max(BAR_MIN_WIDTH, bar_length)
    bar_length = min(bar_length, max_width)

    call_ratio = call_value / total if total > 0 else 0

    call_width = int(round(call_ratio * bar_length))
    put_width = bar_length - call_width

    if call_value > 0 and put_value > 0:
        if call_width <= 0:
            call_width = 1
            put_width = bar_length - 1
        elif put_width <= 0:
            put_width = 1
            call_width = bar_length - 1

    call_width = max(0, min(call_width, bar_length))
    put_width = max(0, min(put_width, bar_length - call_width))

    return "🟩" * call_width + "🟥" * put_width


def make_dual_bar(call_value, put_value, width=BAR_WIDTH):
    call_value = safe_float(call_value)
    put_value = safe_float(put_value)

    if not np.isfinite(call_value) or call_value < 0:
        call_value = 0

    if not np.isfinite(put_value) or put_value < 0:
        put_value = 0

    return make_dynamic_dual_bar(
        call_value,
        put_value,
        call_value + put_value,
        width
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
        f"/ P {fmt_number(put_value)}"
    )


# ============================================================
# ARGUMENTS
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Yahoo Finance Independent Full Option Structure Scanner"
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

def get_current_price(ticker, manual_price=None):
    print()
    print("=" * 70)
    print("FETCH CURRENT PRICE")
    print("=" * 70)

    if manual_price not in (None, ""):
        manual = safe_float(manual_price)

        if np.isfinite(manual) and manual > 0:
            print(f"MANUAL PRICE: ${manual:.2f}")
            return manual

    try:
        history = ticker.history(
            period="1d",
            interval="1m",
            prepost=True
        )

        if not history.empty and "Close" in history.columns:
            close = history["Close"].dropna()

            if not close.empty:
                price = safe_float(close.iloc[-1])

                if np.isfinite(price) and price > 0:
                    print(f"YAHOO PRICE: ${price:.2f}")
                    return price

    except Exception as exc:
        print("1m price error:", repr(exc))

    try:
        history = ticker.history(period="5d")

        if not history.empty and "Close" in history.columns:
            close = history["Close"].dropna()

            if not close.empty:
                price = safe_float(close.iloc[-1])

                if np.isfinite(price) and price > 0:
                    print(f"YAHOO PRICE: ${price:.2f}")
                    return price

    except Exception as exc:
        print("5d price error:", repr(exc))

    raise RuntimeError(
        "Unable to determine current price."
    )


# ============================================================
# DTE
# ============================================================

def calculate_dte(expiration):
    try:
        expiry = pd.Timestamp(expiration).date()
        return (expiry - market_today()).days
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

    if not np.isfinite(volume) or volume <= 0:
        return 0.0

    if (
        np.isfinite(bid)
        and np.isfinite(ask)
        and bid >= 0
        and ask >= bid
        and ask > 0
    ):
        mid = (bid + ask) / 2

    elif np.isfinite(last_price) and last_price > 0:
        mid = last_price

    else:
        return 0.0

    return volume * mid * 100


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

    if gamma <= 0 or open_interest <= 0 or spot <= 0:
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
# FETCH ALL YAHOO EXPIRATIONS
# ============================================================

def fetch_options(symbol, manual_price=None):
    print()
    print("=" * 70)
    print("FETCH YAHOO FINANCE FULL OPTION DATA")
    print("=" * 70)

    ticker = yf.Ticker(symbol)

    spot = get_current_price(
        ticker,
        manual_price
    )

    try:
        expirations = list(ticker.options)
    except Exception as exc:
        raise RuntimeError(
            "Unable to get Yahoo option expirations."
        ) from exc

    print(
        f"TOTAL EXPIRATIONS FOUND: {len(expirations)}"
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
            f"[{index}/{len(expirations)}] {expiration}"
        )

        try:
            chain = ticker.option_chain(
                expiration
            )

        except Exception as exc:
            failed += 1
            print("❌ FAILED:", repr(exc))
            continue

        calls = (
            chain.calls.copy()
            if chain.calls is not None
            else pd.DataFrame()
        )

        puts = (
            chain.puts.copy()
            if chain.puts is not None
            else pd.DataFrame()
        )

        print(
            f"CALL: {len(calls):,} | "
            f"PUT: {len(puts):,}"
        )

        if not calls.empty:
            calls["option_type"] = "CALL"
            calls["expiration"] = expiration
            rows.append(calls)

        if not puts.empty:
            puts["option_type"] = "PUT"
            puts["expiration"] = expiration
            rows.append(puts)

        if not calls.empty or not puts.empty:
            successful += 1

        time.sleep(0.25)

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

    print(f"Successful expirations: {successful}")
    print(f"Failed expirations: {failed}")
    print(f"RAW ROWS: {len(data):,}")

    return data, spot


# ============================================================
# NORMALIZE
# ============================================================

def normalize(data):
    data = data.copy()

    required_numeric = [
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

    for column in required_numeric:
        if column in data.columns:
            data[column] = numeric(data[column])
        else:
            data[column] = np.nan

    if "expiration" not in data.columns:
        raise RuntimeError(
            "Missing expiration column."
        )

    data["expiration"] = (
        data["expiration"]
        .apply(clean_expiration)
    )

    data["DTE"] = (
        data["expiration"]
        .apply(calculate_dte)
    )

    data = data[
        data["strike"].notna()
    ].copy()

    data = data[
        data["option_type"].isin(
            ["CALL", "PUT"]
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
        (data["DTE"] > 0)
        &
        (data["DTE"] <= max_dte)
    ].copy()

    print(
        f"DTE 1~{max_dte}: {len(data):,}"
    )

    data = data[
        data["strike"].between(
            min_strike,
            max_strike
        )
    ].copy()

    print(
        f"Strike ${min_strike:g}~"
        f"${max_strike:g}: "
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

def calculate_metrics(data, spot):
    data = data.copy()

    data["premium_proxy"] = data.apply(
        lambda row: calculate_premium(
            row.get("volume", np.nan),
            row.get("bid", np.nan),
            row.get("ask", np.nan),
            row.get("lastPrice", np.nan)
        ),
        axis=1
    )

    data["gex"] = data.apply(
        lambda row: calculate_gex(
            row.get("gamma", np.nan),
            row.get("openInterest", np.nan),
            spot,
            row.get("option_type", "")
        ),
        axis=1
    )

    data["volume_oi"] = np.where(
        data["openInterest"] > 0,
        data["volume"] / data["openInterest"],
        np.nan
    )

    data["distance_pct"] = (
        (data["strike"] - spot)
        / spot
        * 100
    )

    return data


# ============================================================
# STRIKE STRUCTURE
# ============================================================

def build_strike_table(data):
    if data.empty:
        return pd.DataFrame()

    rows = []

    for strike, frame in data.groupby("strike"):
        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

        cv = calls["volume"].fillna(0).sum()
        pv = puts["volume"].fillna(0).sum()

        coi = calls["openInterest"].fillna(0).sum()
        poi = puts["openInterest"].fillna(0).sum()

        cp = calls["premium_proxy"].fillna(0).sum()
        pp = puts["premium_proxy"].fillna(0).sum()

        cg = calls["gex"].sum(min_count=1)
        pg = puts["gex"].sum(min_count=1)

        total_volume = cv + pv
        total_oi = coi + poi
        total_premium = cp + pp

        net_gex = np.nan

        if np.isfinite(cg) and np.isfinite(pg):
            net_gex = cg + pg
        elif np.isfinite(cg):
            net_gex = cg
        elif np.isfinite(pg):
            net_gex = pg

        rows.append({
            "strike": strike,
            "call_volume": cv,
            "put_volume": pv,
            "total_volume": total_volume,
            "call_oi": coi,
            "put_oi": poi,
            "total_oi": total_oi,
            "call_premium": cp,
            "put_premium": pp,
            "total_premium": total_premium,

            "call_volume_oi":
                cv / coi if coi > 0 else np.nan,

            "put_volume_oi":
                pv / poi if poi > 0 else np.nan,

            "call_volume_ratio":
                cv / total_volume * 100
                if total_volume > 0 else np.nan,

            "call_oi_ratio":
                coi / total_oi * 100
                if total_oi > 0 else np.nan,

            "call_premium_ratio":
                cp / total_premium * 100
                if total_premium > 0 else np.nan,

            "call_gex": cg,
            "put_gex": pg,
            "net_gex": net_gex
        })

    return (
        pd.DataFrame(rows)
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================
# KEY PRICE STRUCTURE
# ============================================================

def build_key_price_structure(data, spot):
    rows = []

    for target in FOCUS_STRIKES:
        frame = data[
            np.isclose(
                data["strike"],
                target,
                atol=0.001
            )
        ]

        if frame.empty:
            rows.append({
                "strike": target,
                "call_oi": 0,
                "put_oi": 0,
                "total_oi": 0,
                "call_volume": 0,
                "put_volume": 0,
                "total_volume": 0,
                "call_premium": 0,
                "put_premium": 0,
                "total_premium": 0,
                "distance_pct":
                    (
                        (target - spot) / spot * 100
                        if spot > 0 else np.nan
                    )
            })
            continue

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

        call_oi = calls[
            "openInterest"
        ].fillna(0).sum()

        put_oi = puts[
            "openInterest"
        ].fillna(0).sum()

        call_volume = calls[
            "volume"
        ].fillna(0).sum()

        put_volume = puts[
            "volume"
        ].fillna(0).sum()

        call_premium = calls[
            "premium_proxy"
        ].fillna(0).sum()

        put_premium = puts[
            "premium_proxy"
        ].fillna(0).sum()

        rows.append({
            "strike": target,

            "call_oi": call_oi,
            "put_oi": put_oi,
            "total_oi": call_oi + put_oi,

            "call_volume": call_volume,
            "put_volume": put_volume,
            "total_volume":
                call_volume + put_volume,

            "call_premium": call_premium,
            "put_premium": put_premium,
            "total_premium":
                call_premium + put_premium,

            "distance_pct":
                (
                    (target - spot) / spot * 100
                    if spot > 0 else np.nan
                )
        })

    return pd.DataFrame(rows)


# ============================================================
# PRICE ROLE
# ============================================================

def classify_price_role(
    strike,
    spot,
    key_table
):
    strike = safe_float(strike)
    spot = safe_float(spot)

    if not np.isfinite(strike):
        return "N/A"

    if strike <= spot:
        if strike == min(FOCUS_STRIKES):
            return "🛡 방어선"
        return "🛡 지지 / 방어"

    above = [
        x for x in FOCUS_STRIKES
        if x > spot
    ]

    if not above:
        return "🎯 저항"

    first_above = min(above)

    if strike == first_above:
        return "🟢 1차 돌파"

    if not key_table.empty:
        positive = key_table[
            key_table["strike"] > spot
        ]

        if not positive.empty:
            max_oi_strike = positive.loc[
                positive["total_oi"].idxmax(),
                "strike"
            ]

            if strike == max_oi_strike:
                return "🔥 핵심 저항"

    if strike == max(FOCUS_STRIKES):
        return "🎯 최대 집중"

    return "🚀 상승 확인"


# ============================================================
# PRICE SCENARIO
# ============================================================

def build_price_scenario(
    key_table,
    spot
):
    if key_table.empty:
        return [
            "⚠️ 가격 시나리오 데이터 없음"
        ]

    lines = []

    table = (
        key_table
        .sort_values("strike")
        .reset_index(drop=True)
    )

    max_oi = table["total_oi"].max()

    for _, row in table.iterrows():
        strike = safe_float(row["strike"])

        call_oi = safe_float(row["call_oi"])
        put_oi = safe_float(row["put_oi"])

        call_volume = safe_float(row["call_volume"])
        put_volume = safe_float(row["put_volume"])

        role = classify_price_role(
            strike,
            spot,
            key_table
        )

        if strike <= spot:
            structure = (
                "CALL OI 우위"
                if call_oi >= put_oi
                else "PUT OI 우위"
            )

            prefix = (
                "🛡"
                if strike == min(FOCUS_STRIKES)
                else "🟡"
            )

        else:
            if (
                call_volume > put_volume
                and call_oi > put_oi
            ):
                structure = "CALL OI + Volume 우위"

            elif call_oi > put_oi:
                structure = "CALL OI 우위"

            elif put_oi > call_oi:
                structure = "PUT OI 우위"

            else:
                structure = "혼조"

            if (
                np.isfinite(max_oi)
                and max_oi > 0
                and row["total_oi"] == max_oi
            ):
                prefix = "🎯"
            elif strike == 85:
                prefix = "🟢"
            elif strike == 90:
                prefix = "🔥"
            elif strike == 95:
                prefix = "🚀"
            elif strike == 100:
                prefix = "🎯"
            else:
                prefix = "📍"

        lines.append(
            f"{prefix} ${strike:g} "
            f"→ {role} | {structure}"
        )

    return lines


# ============================================================
# BAR STRUCTURE
# ============================================================

def build_bar_structure(strike_table):
    if strike_table.empty:
        return [
            "⚠️ BAR STRUCTURE DATA 없음"
        ]

    table = (
        strike_table
        .sort_values("strike")
        .reset_index(drop=True)
    )

    oi_reference = (
        table["call_oi"].fillna(0)
        + table["put_oi"].fillna(0)
    ).max()

    volume_reference = (
        table["call_volume"].fillna(0)
        + table["put_volume"].fillna(0)
    ).max()

    premium_reference = (
        table["call_premium"].fillna(0)
        + table["put_premium"].fillna(0)
    ).max()

    lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "📊 4. CALL / PUT BAR STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🟩 CALL   🟥 PUT",
        f"📏 BAR MAX: {BAR_WIDTH}칸",
        "📐 규모가 작으면 BAR도 짧게 표시",
        "",
        "🟢 OI STRUCTURE",
        ""
    ]

    for _, row in table.iterrows():
        lines.append(
            f"🎯 ${row['strike']:g} "
            + make_dual_bar_line(
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

    for _, row in table.iterrows():
        lines.append(
            f"🎯 ${row['strike']:g} "
            + make_dual_bar_line(
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

    for _, row in table.iterrows():
        bar = make_dynamic_dual_bar(
            row["call_premium"],
            row["put_premium"],
            premium_reference
        )

        lines.append(
            f"🎯 ${row['strike']:g} "
            f"{bar} "
            f"C {fmt_money(row['call_premium'])} "
            f"/ P {fmt_money(row['put_premium'])}"
        )

    return lines


# ============================================================
# EXPIRATION STRUCTURE
# IMPORTANT: EMPTY DATA SAFE
# ============================================================

def build_expiration_structure(data):
    if data is None:
        return pd.DataFrame()

    if data.empty:
        return pd.DataFrame()

    if "expiration" not in data.columns:
        return pd.DataFrame()

    rows = []

    for expiration, frame in data.groupby(
        "expiration",
        dropna=True
    ):
        if frame.empty:
            continue

        calls = frame[
            frame["option_type"] == "CALL"
        ]

        puts = frame[
            frame["option_type"] == "PUT"
        ]

        cv = calls["volume"].fillna(0).sum()
        pv = puts["volume"].fillna(0).sum()

        coi = calls[
            "openInterest"
        ].fillna(0).sum()

        poi = puts[
            "openInterest"
        ].fillna(0).sum()

        cp = calls[
            "premium_proxy"
        ].fillna(0).sum()

        pp = puts[
            "premium_proxy"
        ].fillna(0).sum()

        total_oi = coi + poi

        rows.append({
            "expiration":
                clean_expiration(expiration),

            "DTE":
                calculate_dte(expiration),

            "call_volume": cv,
            "put_volume": pv,
            "total_volume": cv + pv,

            "call_oi": coi,
            "put_oi": poi,
            "total_oi": total_oi,

            "call_premium": cp,
            "put_premium": pp,
            "total_premium": cp + pp,

            "call_volume_ratio":
                cv / (cv + pv) * 100
                if cv + pv > 0
                else np.nan,

            "call_oi_ratio":
                coi / total_oi * 100
                if total_oi > 0
                else np.nan
        })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame()

    total_oi = result["total_oi"].sum()

    if total_oi > 0:
        result[
            "total_oi_concentration_pct"
        ] = (
            result["total_oi"]
            / total_oi
            * 100
        )
    else:
        result[
            "total_oi_concentration_pct"
        ] = np.nan

    return (
        result
        .sort_values(
            ["DTE", "expiration"],
            na_position="last"
        )
        .reset_index(drop=True)
    )


# ============================================================
# STRIKE × EXPIRATION
# ============================================================

def build_strike_expiration_structure(
    data,
    focus_strikes
):
    if data is None or data.empty:
        return pd.DataFrame()

    rows = []

    for target in focus_strikes:
        strike_data = data[
            np.isclose(
                data["strike"],
                target,
                atol=0.001
            )
        ]

        if strike_data.empty:
            continue

        for expiration, frame in strike_data.groupby(
            "expiration",
            dropna=True
        ):
            if frame.empty:
                continue

            calls = frame[
                frame["option_type"] == "CALL"
            ]

            puts = frame[
                frame["option_type"] == "PUT"
            ]

            cv = calls[
                "volume"
            ].fillna(0).sum()

            pv = puts[
                "volume"
            ].fillna(0).sum()

            coi = calls[
                "openInterest"
            ].fillna(0).sum()

            poi = puts[
                "openInterest"
            ].fillna(0).sum()

            cp = calls[
                "premium_proxy"
            ].fillna(0).sum()

            pp = puts[
                "premium_proxy"
            ].fillna(0).sum()

            rows.append({
                "strike": target,
                "expiration":
                    clean_expiration(expiration),

                "DTE":
                    calculate_dte(expiration),

                "call_volume": cv,
                "put_volume": pv,
                "total_volume": cv + pv,

                "call_oi": coi,
                "put_oi": poi,
                "total_oi": coi + poi,

                "call_premium": cp,
                "put_premium": pp,
                "total_premium": cp + pp
            })

    if not rows:
        return pd.DataFrame()

    result = pd.DataFrame(rows)

    if result.empty:
        return pd.DataFrame()

    result["total_oi_pct"] = np.nan
    result["call_oi_pct"] = np.nan
    result["put_oi_pct"] = np.nan

    for strike in result["strike"].unique():
        mask = result["strike"] == strike

        total_oi = result.loc[
            mask,
            "total_oi"
        ].sum()

        call_oi = result.loc[
            mask,
            "call_oi"
        ].sum()

        put_oi = result.loc[
            mask,
            "put_oi"
        ].sum()

        if total_oi > 0:
            result.loc[
                mask,
                "total_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "total_oi"
                ]
                / total_oi
                * 100
            )

        if call_oi > 0:
            result.loc[
                mask,
                "call_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "call_oi"
                ]
                / call_oi
                * 100
            )

        if put_oi > 0:
            result.loc[
                mask,
                "put_oi_pct"
            ] = (
                result.loc[
                    mask,
                    "put_oi"
                ]
                / put_oi
                * 100
            )

    return (
        result
        .sort_values(
            ["strike", "total_oi"],
            ascending=[True, False]
        )
        .reset_index(drop=True)
    )


# ============================================================
# KEY STRIKE SUMMARY
# ============================================================

def build_key_strike_summary(
    strike_expiration
):
    if (
        strike_expiration is None
        or strike_expiration.empty
    ):
        return pd.DataFrame()

    rows = []

    for strike, frame in (
        strike_expiration.groupby("strike")
    ):
        if frame.empty:
            continue

        frame = frame.sort_values(
            "total_oi",
            ascending=False
        )

        top = frame.iloc[0]

        rows.append({
            "strike": strike,

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
                top.get(
                    "total_oi_pct",
                    np.nan
                )
        })

    if not rows:
        return pd.DataFrame()

    return (
        pd.DataFrame(rows)
        .sort_values("strike")
        .reset_index(drop=True)
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(data):
    if data is None or data.empty:
        return pd.DataFrame()

    result = data.copy()

    result["importance"] = (
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
        .reset_index(drop=True)
    )


# ============================================================
# WALL
# ============================================================

def find_wall(
    strike_table,
    spot,
    option_type
):
    if strike_table is None or strike_table.empty:
        return None

    if option_type == "CALL":
        candidates = strike_table[
            strike_table["strike"] >= spot
        ].copy()

        candidates["oi"] = candidates["call_oi"]
        candidates["gex_abs"] = candidates["call_gex"].abs()
        candidates["volume"] = candidates["call_volume"]

    else:
        candidates = strike_table[
            strike_table["strike"] <= spot
        ].copy()

        candidates["oi"] = candidates["put_oi"]
        candidates["gex_abs"] = candidates["put_gex"].abs()
        candidates["volume"] = candidates["put_volume"]

    if candidates.empty:
        return None

    candidates["distance"] = (
        (candidates["strike"] - spot).abs()
        / spot
    )

    candidates = candidates[
        candidates["distance"] <= 0.20
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
            + candidates["distance"] * 20
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
    key_price_structure,
    top_contracts,
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

    cv = calls["volume"].fillna(0).sum()
    pv = puts["volume"].fillna(0).sum()

    coi = calls[
        "openInterest"
    ].fillna(0).sum()

    poi = puts[
        "openInterest"
    ].fillna(0).sum()

    cp = calls[
        "premium_proxy"
    ].fillna(0).sum()

    pp = puts[
        "premium_proxy"
    ].fillna(0).sum()

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

    tgex = calls["gex"].sum(min_count=1)
    pgex = puts["gex"].sum(min_count=1)

    if np.isfinite(tgex) and np.isfinite(pgex):
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
        .sort_values("atm_distance")
        .head(10)["impliedVolatility"]
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
        f"🎯 분석 Strike: ${min_strike:g} ~ ${max_strike:g}",
        f"📅 분석 DTE: 1 ~ {max_dte}",
        f"📊 분석 행수: {len(data):,}",
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
        f"CALL Premium: {fmt_money(cp)}",
        f"PUT Premium : {fmt_money(pp)}",
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
            f"CALL Volume/OI: {cv_oi:.3f}"
            if np.isfinite(cv_oi)
            else "CALL Volume/OI: N/A"
        ),
        (
            f"PUT Volume/OI : {pv_oi:.3f}"
            if np.isfinite(pv_oi)
            else "PUT Volume/OI : N/A"
        ),
        ""
    ]

    # ========================================================
    # KEY PRICE
    # ========================================================

    report += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        "🎯 3. KEY PRICE STRUCTURE",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        ""
    ]

    if key_price_structure.empty:
        report.append(
            "⚠️ KEY PRICE STRUCTURE DATA 없음"
        )
    else:
        for _, row in (
            key_price_structure
            .sort_values("strike")
            .iterrows()
        ):
            strike = safe_float(row["strike"])

            call_oi = safe_float(row["call_oi"])
            put_oi = safe_float(row["put_oi"])

            call_volume = safe_float(
                row["call_volume"]
            )

            put_volume = safe_float(
                row["put_volume"]
            )

            call_premium = safe_float(
                row["call_premium"]
            )

            put_premium = safe_float(
                row["put_premium"]
            )

            if strike <= spot:
                if strike == 80:
                    title = "🛡 $80 방어선"
                else:
                    title = f"🛡 ${strike:g} 지지"

            elif strike == 85:
                title = "🟢 $85 1차 돌파"
            elif strike == 90:
                title = "🔥 $90 핵심"
            elif strike == 95:
                title = "🚀 $95 상승 확인"
            elif strike == 100:
                title = "🎯 $100 최대 집중"
            else:
                title = f"🎯 ${strike:g}"

            report += [
                title,
                (
                    f"CALL OI: {call_oi:,.0f}"
                    f" / PUT OI: {put_oi:,.0f}"
                ),
                (
                    f"CALL Vol: {call_volume:,.0f}"
                    f" / PUT Vol: {put_volume:,.0f}"
                ),
                (
                    f"CALL Premium: {fmt_money(call_premium)}"
                    f" / PUT Premium: {fmt_money(put_premium)}"
                )
            ]

            if call_oi > put_oi:
                report.append("🟢 OI: CALL 우위")
            elif put_oi > call_oi:
                report.append("🔴 OI: PUT 우위")
            else:
                report.append("🟡 OI: 균형")

            if call_volume > put_volume:
                report.append("🟢 Volume: CALL 우위")
            elif put_volume > call_volume:
                report.append("🔴 Volume: PUT 우위")
else:
