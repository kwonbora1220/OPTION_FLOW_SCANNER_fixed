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

MIN_STRIKE = float(
    os.getenv("MIN_STRIKE", "80")
)

MAX_STRIKE = float(
    os.getenv("MAX_STRIKE", "100")
)

# ============================================================
# IMPORTANT
#
# 0DTE 제외
# 최소 DTE = 1
# ============================================================

MIN_DTE = int(
    os.getenv("MIN_DTE", "1")
)

MAX_DTE = int(
    os.getenv("MAX_DTE", "180")
)

OUTPUT_DIR = os.getenv(
    "OUTPUT_DIR",
    f"{SYMBOL.lower()}_option_structure"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


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

            close = (
                history["Close"]
                .dropna()
            )

            if not close.empty:

                price = float(
                    close.iloc[-1]
                )

                print(
                    f"CURRENT PRICE: ${price:.2f}"
                )

                return price

    except Exception as exc:

        print(
            f"1m price error: {repr(exc)}"
        )

    # --------------------------------------------------------
    # 2. fallback
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
    # Mid
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
            f"DTE error "
            f"{expiration}: {repr(exc)}"
        )

        return np.nan


# ============================================================
# FETCH YAHOO OPTIONS
#
# IMPORTANT
#
# 여기서는 필터링하지 않는다.
# Yahoo가 주는 모든 만기를 먼저 수집.
# ============================================================

def fetch_options():

    print()
    print("=" * 70)
    print("FETCH YAHOO FINANCE OPTION DATA")
    print("=" * 70)

    ticker = yf.Ticker(
        SYMBOL
    )

    # --------------------------------------------------------
    # PRICE
    # --------------------------------------------------------

    spot = get_current_price(
        ticker
    )

    # --------------------------------------------------------
    # EXPIRATIONS
    # --------------------------------------------------------

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
            "Unable to get Yahoo expirations."
        )

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
            f"{expiration}"
        )

        try:

            chain = ticker.option_chain(
                expiration
            )

        except Exception as exc:

            failed += 1

            print(
                f"❌ FAILED: {repr(exc)}"
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

        print(
            f"CALL rows: {len(calls):,}"
        )

        print(
            f"PUT rows : {len(puts):,}"
        )

        if (
            calls is not None
            and not calls.empty
        ):

            frame = calls.copy()

            frame["option_type"] = "CALL"

            frame["expiration"] = (
                expiration
            )

            rows.append(frame)

        if (
            puts is not None
            and not puts.empty
        ):

            frame = puts.copy()

            frame["option_type"] = "PUT"

            frame["expiration"] = (
                expiration
            )

            rows.append(frame)

        if (
            not calls.empty
            or not puts.empty
        ):

            successful += 1

        time.sleep(
            0.25
        )

    print()
    print("=" * 70)
    print("YAHOO COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Successful: {successful}"
    )

    print(
        f"Failed: {failed}"
    )

    if not rows:

        raise RuntimeError(
            "No Yahoo option rows collected."
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
    data,
    spot
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
        .apply(calculate_dte)
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
#
# ★★★★★ IMPORTANT ★★★★★
#
# DTE 0 제거
#
# MIN_DTE = 1
#
# 따라서 오늘 만기 옵션은
# 전체 분석에서 완전히 제외된다.
# ============================================================

def apply_filters(data):

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
            data["DTE"]
            >= MIN_DTE
        )
        &
        (
            data["DTE"]
            <= MAX_DTE
        )
    ].copy()

    after_dte = len(data)

    print(
        f"After DTE "
        f"{MIN_DTE}~{MAX_DTE}: "
        f"{after_dte:,}"
    )

    # --------------------------------------------------------
    # Strike
    # --------------------------------------------------------

    data = data[
        data["strike"].between(
            MIN_STRIKE,
            MAX_STRIKE,
            inclusive="both"
        )
    ].copy()

    after_strike = len(data)

    print(
        f"After Strike "
        f"${MIN_STRIKE:g}"
        f"~"
        f"${MAX_STRIKE:g}: "
        f"{after_strike:,}"
    )

    # --------------------------------------------------------
    # EXPLICIT 0DTE CHECK
    # --------------------------------------------------------

    zero_dte = (
        data["DTE"] == 0
    ).sum()

    if zero_dte > 0:

        print(
            f"⚠️ WARNING: "
            f"{zero_dte} zero-DTE rows found"
        )

        data = data[
            data["DTE"] >= 1
        ].copy()

    print(
        f"0DTE rows in final data: "
        f"{(data['DTE'] == 0).sum()}"
    )

    print(
        f"Rows removed total: "
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

    data["premium_proxy"] = data.apply(
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

    data["gex"] = data.apply(
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

    data["volume_oi"] = np.where(
        data["openInterest"] > 0,
        data["volume"]
        /
        data["openInterest"],
        np.nan
    )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

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
            data["strike"]
            == strike
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
            calls["premium_proxy"]
            .fillna(0)
            .sum()
        )

        put_premium = (
            puts["premium_proxy"]
            .fillna(0)
            .sum()
        )

        call_gex = (
            calls["gex"]
            .fillna(0)
            .sum()
        )

        put_gex = (
            puts["gex"]
            .fillna(0)
            .sum()
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
                    +
                    put_volume,

                "call_oi":
                    call_oi,

                "put_oi":
                    put_oi,

                "total_oi":
                    call_oi
                    +
                    put_oi,

                "call_premium":
                    call_premium,

                "put_premium":
                    put_premium,

                "total_premium":
                    call_premium
                    +
                    put_premium,

                "call_gex":
                    call_gex,

                "put_gex":
                    put_gex,

                "net_gex":
                    call_gex
                    +
                    put_gex
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# TOP CONTRACTS
# ============================================================

def build_top_contracts(data):

    result = data.copy()

    result["importance"] = (

        np.log1p(
            result[
                "premium_proxy"
            ]
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
            candidates["call_gex"]
            .abs()
        )

        candidates["volume"] = (
            candidates["call_volume"]
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
            candidates["put_gex"]
            .abs()
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
        candidates["distance"]
        <= 0.20
    ].copy()

    if candidates.empty:

        return None

    candidates["score"] = (

        np.log1p(
            candidates["oi"]
            .clip(lower=0)
        )

        +

        np.log1p(
            candidates["gex_abs"]
            .clip(lower=0)
        )

        +

        0.25
        *
        np.log1p(
            candidates["volume"]
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
    top_contracts,
    spot,
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

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

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

    call_gex = (
        calls["gex"]
        .fillna(0)
        .sum()
    )

    put_gex = (
        puts["gex"]
        .fillna(0)
        .sum()
    )

    net_gex = (
        call_gex
        +
        put_gex
    )

    total_volume = (
        call_volume
        +
        put_volume
    )

    total_oi = (
        call_oi
        +
        put_oi
    )

    total_premium = (
        call_premium
        +
        put_premium
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

    # --------------------------------------------------------
    # ATM IV
    # --------------------------------------------------------

    temp = data.copy()

    temp["atm_distance"] = (
        (
            temp["strike"]
            -
            spot
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

    # --------------------------------------------------------
    # WALL
    # --------------------------------------------------------

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
        f"🔥 {SYMBOL} OPTION STRUCTURE"
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
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )

    # ★ 0DTE 제외 표시

    report.append(
        f"📅 DTE: "
        f"{MIN_DTE} ~ "
        f"{MAX_DTE}"
        f"  (0DTE 제외)"
    )

    report.append(
        f"📊 옵션 행수: "
        f"{len(data):,}"
    )

    report.append("")

    # ========================================================
    # FLOW
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
    # WALL
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧱 2. WALL / GEX"
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

    report.append("")

    if (
        np.isfinite(call_gex)
        and np.isfinite(put_gex)
    ):

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

    else:

        report.append(
            "CALL GEX: N/A"
        )

        report.append(
            "PUT GEX: N/A"
        )

        report.append(
            "NET GEX: N/A"
        )

        report.append(
            "⚠️ Yahoo gamma 데이터 부족"
        )

    report.append(
        f"ATM IV: "
        f"{fmt_iv(atm_iv)}"
    )

    report.append("")

    # ========================================================
    # STRIKE STRUCTURE
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        f"🎯 3. "
        f"${MIN_STRIKE:g}"
        f"~"
        f"${MAX_STRIKE:g} "
        f"STRIKE STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "STRIKE | C-VOL | P-VOL | "
        "C-OI | P-OI | C-PREM | P-PREM | NET-GEX"
    )

    report.append(
        "────────────────────────────────────────"
    )

    for _, row in (
        strike_table
        .sort_values("strike")
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
    # HIGH OI
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 4. HIGH OI STRIKES"
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
    # TOP CONTRACTS
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🔥 5. TOP OPTION CONTRACTS"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    for _, row in (
        top_contracts
        .head(20)
        .iterrows()
    ):

        volume = safe_float(
            row["volume"]
        )

        oi = safe_float(
            row["openInterest"]
        )

        dte = safe_float(
            row["DTE"]
        )

        report.append(
            f"{row['option_type']:4s} "
            f"${row['strike']:g}"
            f" | DTE "
            f"{int(dte) if np.isfinite(dte) else 'N/A'}"
            f" | Vol "
            f"{volume:,.0f}"
            f" | OI "
            f"{oi:,.0f}"
            f" | Premium "
            f"{fmt_money(row['premium_proxy'])}"
            f" | GEX "
            f"{fmt_money(row['gex'])}"
        )

    report.append("")

    # ========================================================
    # STRUCTURE
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🧠 6. STRUCTURE"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    if (
        call_wall is not None
        and put_wall is not None
    ):

        cw = call_wall["strike"]
        pw = put_wall["strike"]

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

    # --------------------------------------------------------
    # FLOW SIGNAL
    # --------------------------------------------------------

    if (
        call_volume
        >
        put_volume
    ):

        report.append(
            "🟢 CALL Volume 우세"
        )

    elif (
        put_volume
        >
        call_volume
    ):

        report.append(
            "🔴 PUT Volume 우세"
        )

    if (
        call_oi
        >
        put_oi
    ):

        report.append(
            "🟢 CALL OI 우세"
        )

    elif (
        put_oi
        >
        call_oi
    ):

        report.append(
            "🔴 PUT OI 우세"
        )

    if (
        call_premium
        >
        put_premium
    ):

        report.append(
            "🟢 CALL Premium 우세"
        )

    elif (
        put_premium
        >
        call_premium
    ):

        report.append(
            "🔴 PUT Premium 우세"
        )

    report.append("")

    # ========================================================
    # FINAL SIGNAL
    # ========================================================

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    report.append(
        "🎯 7. FINAL OPTION SIGNAL"
    )

    report.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    bullish_score = 0

    # CALL volume
    if call_volume > put_volume:
        bullish_score += 1

    # CALL OI
    if call_oi > put_oi:
        bullish_score += 1

    # CALL premium
    if call_premium > put_premium:
        bullish_score += 1

    # Wall relationship
    if (
        call_wall is not None
        and put_wall is not None
    ):

        if spot >= put_wall["strike"]:

            bullish_score += 1

    # Ratio
    if (
        np.isfinite(
            call_volume_ratio
        )
        and call_volume_ratio >= 60
    ):

        bullish_score += 1

    if bullish_score >= 4:

        signal = "🟢 BULLISH"

    elif bullish_score >= 3:

        signal = "🟡 NEUTRAL-BULLISH"

    elif bullish_score <= 1:

        signal = "🔴 BEARISH"

    else:

        signal = "🟡 NEUTRAL"

    report.append(
        f"FINAL: {signal}"
    )

    report.append(
        f"Score: "
        f"{bullish_score}/5"
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
        "• 0DTE 옵션은 전체 분석에서 제외"
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
        "• Dealer 실제 포지션 데이터 아님"
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
        ).encode(
            "utf-8"
        )

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
                f"Telegram error: "
                f"{repr(exc)}"
            )

    print(
        f"Telegram sent: "
        f"{len(chunks)} message(s)"
    )


# ============================================================
# SAVE SUMMARY
# ============================================================

def save_summary(
    data,
    strike_table,
    spot
):

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

    call_gex = (
        calls["gex"]
        .fillna(0)
        .sum()
    )

    put_gex = (
        puts["gex"]
        .fillna(0)
        .sum()
    )

    total_volume = (
        call_volume
        +
        put_volume
    )

    total_oi = (
        call_oi
        +
        put_oi
    )

    total_premium = (
        call_premium
        +
        put_premium
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

    summary = pd.DataFrame(
        [
            {
                "symbol":
                    SYMBOL,

                "spot":
                    spot,

                "min_strike":
                    MIN_STRIKE,

                "max_strike":
                    MAX_STRIKE,

                "min_dte":
                    MIN_DTE,

                "max_dte":
                    MAX_DTE,

                "zero_dte_excluded":
                    True,

                "rows":
                    len(data),

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
                    call_gex
                    +
                    put_gex,

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
                    )
            }
        ]
    )

    summary.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "summary.csv"
        ),
        index=False
    )


# ============================================================
# MAIN
# ============================================================

def main():

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
        f"SYMBOL       : {SYMBOL}"
    )

    print(
        f"STRIKE RANGE : "
        f"${MIN_STRIKE:g}"
        f" ~ "
        f"${MAX_STRIKE:g}"
    )

    print(
        f"DTE RANGE    : "
        f"{MIN_DTE}"
        f" ~ "
        f"{MAX_DTE}"
        f" (0DTE 제외)"
    )

    print("=" * 70)

    # ========================================================
    # 1. FETCH
    # ========================================================

    raw, spot = fetch_options()

    # ========================================================
    # 2. NORMALIZE
    # ========================================================

    data = normalize(
        raw,
        spot
    )

    # ========================================================
    # 3. FILTER
    #
    # ★ 0DTE 제거
    # ========================================================

    data = apply_filters(
        data
    )

    if data.empty:

        raise RuntimeError(
            "No option rows remain "
            "after filtering."
        )

    # ========================================================
    # FINAL SAFETY
    #
    # 여기서 다시 한번 0DTE 제거
    # ========================================================

    data = data[
        data["DTE"] >= 1
    ].copy()

    if (
        data["DTE"] == 0
    ).any():

        raise RuntimeError(
            "CRITICAL: 0DTE detected "
            "in final dataset."
        )

    # ========================================================
    # 4. METRICS
    # ========================================================

    data = calculate_metrics(
        data,
        spot
    )

    print()
    print(
        f"FINAL OPTION ROWS "
        f"(0DTE EXCLUDED): "
        f"{len(data):,}"
    )

    print(
        f"FINAL 0DTE ROWS: "
        f"{(data['DTE'] == 0).sum()}"
    )

    # ========================================================
    # 5. TABLES
    # ========================================================

    strike_table = (
        build_strike_table(
            data
        )
    )

    top_contracts = (
        build_top_contracts(
            data
        )
    )

    # ========================================================
    # 6. SAVE
    # ========================================================

    data.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "contracts.csv"
        ),
        index=False
    )

    strike_table.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "strike_structure.csv"
        ),
        index=False
    )

    top_contracts.to_csv(
        os.path.join(
            OUTPUT_DIR,
            "top_contracts.csv"
        ),
        index=False
    )

    save_summary(
        data,
        strike_table,
        spot
    )

    # ========================================================
    # 7. REPORT
    # ========================================================

    report = build_report(
        data,
        strike_table,
        top_contracts,
        spot,
        started
    )

    report_file = os.path.join(
        OUTPUT_DIR,
        "report.md"
    )

    with open(
        report_file,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            report
        )

    print()
    print(
        report
    )

    # ========================================================
    # 8. TELEGRAM
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
   
