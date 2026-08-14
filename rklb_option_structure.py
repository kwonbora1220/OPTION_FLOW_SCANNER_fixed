# ============================================================
# RKLB OPTION STRUCTURE SCANNER
# ============================================================
# 목적
# 1. Yahoo Finance에서 RKLB 전체 옵션 수집
# 2. DTE 1~180 필터
# 3. $60~$100 PRICE ZONE별 CALL/PUT OI 집계
# 4. $80~$100 상세 옵션 구조 분석
# 5. Zone별 CALL/PUT OI 막대 출력
# 6. Distance Zone 분석
# 7. 절대값/정수/실수 포맷 오류 방지
# ============================================================

import os
import sys
import math
import time
from datetime import datetime, timezone

import pandas as pd
import yfinance as yf


# ============================================================
# CONFIG
# ============================================================

SYMBOL = os.getenv("INPUT_SYMBOL", "RKLB").upper()

MIN_DTE = 1
MAX_DTE = 180

# 전체 PRICE ZONE
ZONE_MIN_STRIKE = 60
ZONE_MAX_STRIKE = 100

# 상세 분석 구간
DETAIL_MIN_STRIKE = 80
DETAIL_MAX_STRIKE = 100

# Yahoo 요청 간격
REQUEST_DELAY = 0.25

# 출력
TOP_N = 20


# ============================================================
# SAFE FORMAT FUNCTIONS
# ============================================================

def safe_int(value):
    """
    숫자를 안전하게 int로 변환.
    NaN / None / 잘못된 값은 0.
    """
    try:
        if pd.isna(value):
            return 0
        return int(float(value))
    except Exception:
        return 0


def safe_float(value):
    """
    숫자를 안전하게 float로 변환.
    """
    try:
        if pd.isna(value):
            return 0.0
        return float(value)
    except Exception:
        return 0.0


def fmt_int(value):
    """
    정수 출력.
    절대로 :.1f 같은 float precision을 사용하지 않는다.
    """
    return f"{safe_int(value):,}"


def fmt_float(value, digits=2):
    """
    실수 출력.
    """
    return f"{safe_float(value):,.{digits}f}"


def fmt_price(value):
    return f"${safe_float(value):,.2f}"


def fmt_pct(value):
    return f"{safe_float(value):.1f}%"


# ============================================================
# HEADER
# ============================================================

print()
print("=" * 70)
print("🔥 RKLB OPTION STRUCTURE SCANNER")
print("=" * 70)
print(f"DTE: {MIN_DTE} ~ {MAX_DTE}")
print(
    f"ZONE: ${ZONE_MIN_STRIKE} ~ ${ZONE_MAX_STRIKE}"
)
print(
    f"DETAIL: ${DETAIL_MIN_STRIKE} ~ ${DETAIL_MAX_STRIKE}"
)
print("=" * 70)


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(symbol):
    print()
    print("=" * 70)
    print("GET CURRENT PRICE")
    print("=" * 70)

    ticker = yf.Ticker(symbol)

    price = None

    try:
        fast_info = ticker.fast_info

        if fast_info:
            price = fast_info.get("last_price")

    except Exception:
        pass

    if price is None:
        try:
            hist = ticker.history(period="5d", auto_adjust=False)

            if not hist.empty:
                price = hist["Close"].dropna().iloc[-1]

        except Exception:
            pass

    if price is None:
        raise RuntimeError("현재 가격을 가져오지 못했습니다.")

    price = safe_float(price)

    print(f"CURRENT PRICE: {fmt_price(price)}")

    return price


# ============================================================
# DTE
# ============================================================

def calculate_dte(expiration):
    try:
        exp_date = datetime.strptime(
            str(expiration), "%Y-%m-%d"
        ).date()

        today = datetime.now(timezone.utc).date()

        return (exp_date - today).days

    except Exception:
        return -999


# ============================================================
# OPTION COLLECTION
# ============================================================

def collect_options(symbol):

    print()
    print("=" * 70)
    print("YAHOO OPTION COLLECTION")
    print("=" * 70)

    ticker = yf.Ticker(symbol)

    try:
        expirations = list(ticker.options)
    except Exception as e:
        raise RuntimeError(
            f"Yahoo expiration 조회 실패: {e}"
        )

    print(f"Yahoo expirations: {len(expirations)}")

    selected = []

    for exp in expirations:

        dte = calculate_dte(exp)

        if MIN_DTE <= dte <= MAX_DTE:
            selected.append((exp, dte))

    print(f"Selected expirations: {len(selected)}")

    if not selected:
        raise RuntimeError(
            "DTE 조건에 맞는 expiration이 없습니다."
        )

    all_rows = []

    for idx, (expiration, dte) in enumerate(
        selected,
        start=1
    ):

        print()
        print("-" * 70)
        print(f"EXPIRATION: {expiration}")
        print(f"DTE       : {dte}")

        try:

            chain = ticker.option_chain(expiration)

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["option_type"] = "CALL"
            puts["option_type"] = "PUT"

            calls["expiration"] = expiration
            puts["expiration"] = expiration

            calls["DTE"] = dte
            puts["DTE"] = dte

            calls["symbol"] = symbol
            puts["symbol"] = symbol

            print(
                f"CALL rows: {len(calls)}"
            )

            print(
                f"PUT rows : {len(puts)}"
            )

            all_rows.append(calls)
            all_rows.append(puts)

        except Exception as e:

            print(
                f"⚠️ EXPIRATION FAILED: {expiration}"
            )

            print(
                f"ERROR: {repr(e)}"
            )

        time.sleep(REQUEST_DELAY)

    if not all_rows:
        raise RuntimeError(
            "옵션 데이터를 하나도 수집하지 못했습니다."
        )

    df = pd.concat(
        all_rows,
        ignore_index=True
    )

    print()
    print("=" * 70)
    print("YAHOO COLLECTION COMPLETE")
    print("=" * 70)

    print(
        f"Successful expirations: {len(selected)}"
    )

    print(
        f"RAW OPTION ROWS: {len(df):,}"
    )

    return df


# ============================================================
# NORMALIZATION
# ============================================================

def normalize_options(df):

    print()
    print("=" * 70)
    print("NORMALIZATION")
    print("=" * 70)

    df = df.copy()

    numeric_columns = [
        "strike",
        "volume",
        "openInterest",
        "bid",
        "ask",
        "lastPrice",
        "impliedVolatility",
        "DTE"
    ]

    for col in numeric_columns:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    df["strike"] = df["strike"].astype(float)

    df["volume"] = df["volume"].astype(float)

    df["openInterest"] = (
        df["openInterest"]
        .astype(float)
    )

    df["DTE"] = df["DTE"].astype(int)

    return df


# ============================================================
# FILTER
# ============================================================

def apply_filters(df):

    print()
    print("=" * 70)
    print("APPLY FILTERS")
    print("=" * 70)

    original_count = len(df)

    # DTE
    df = df[
        (df["DTE"] >= MIN_DTE)
        &
        (df["DTE"] <= MAX_DTE)
    ].copy()

    print(
        f"After DTE {MIN_DTE}~{MAX_DTE}: "
        f"{len(df):,}"
    )

    # IMPORTANT:
    # 여기서는 $60~$100 전체를 남긴다.
    # 그래야 $60~70 / $70~80 Zone도 계산 가능하다.
    df = df[
        (df["strike"] >= ZONE_MIN_STRIKE)
        &
        (df["strike"] <= ZONE_MAX_STRIKE)
    ].copy()

    print(
        f"After Strike "
        f"${ZONE_MIN_STRIKE}~${ZONE_MAX_STRIKE}: "
        f"{len(df):,}"
    )

    print(
        f"Rows removed: "
        f"{original_count - len(df):,}"
    )

    print()
    print(
        f"ZONE OPTION ROWS: {len(df):,}"
    )

    return df


# ============================================================
# PRICE ZONE
# ============================================================

def get_price_zone(strike):

    strike = safe_float(strike)

    if (
        ZONE_MIN_STRIKE
        <= strike
        < 70
    ):
        return "$60~$70"

    if (
        70
        <= strike
        < 80
    ):
        return "$70~$80"

    if (
        80
        <= strike
        < 90
    ):
        return "$80~$90"

    if (
        90
        <= strike
        <= 100
    ):
        return "$90~$100"

    return None


# ============================================================
# BUILD PRICE ZONE STRUCTURE
# ============================================================

def build_price_zone_structure(df):

    print()
    print("=" * 70)
    print("BUILD PRICE ZONE STRUCTURE")
    print("=" * 70)

    zones = [
        "$60~$70",
        "$70~$80",
        "$80~$90",
        "$90~$100"
    ]

    result = []

    for zone in zones:

        zone_df = df[
            df["strike"].apply(
                get_price_zone
            ) == zone
        ].copy()

        call_oi = safe_int(
            zone_df.loc[
                zone_df["option_type"] == "CALL",
                "openInterest"
            ].sum()
        )

        put_oi = safe_int(
            zone_df.loc[
                zone_df["option_type"] == "PUT",
                "openInterest"
            ].sum()
        )

        total_oi = call_oi + put_oi

        result.append({
            "zone": zone,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "total_oi": total_oi
        })

    result_df = pd.DataFrame(result)

    print()

    print(
        f"{'ZONE':>10}"
        f"{'CALL OI':>14}"
        f"{'PUT OI':>14}"
        f"{'TOTAL OI':>14}"
    )

    print("-" * 55)

    for _, row in result_df.iterrows():

        print(
            f"{row['zone']:>10}"
            f"{fmt_int(row['call_oi']):>14}"
            f"{fmt_int(row['put_oi']):>14}"
            f"{fmt_int(row['total_oi']):>14}"
        )

    return result_df


# ============================================================
# BAR
# ============================================================

def make_bar(value, max_value, width=30):

    value = safe_int(value)
    max_value = safe_int(max_value)

    if max_value <= 0:
        return ""

    ratio = value / max_value

    count = int(
        round(
            ratio * width
        )
    )

    count = max(
        0,
        min(
            width,
            count
        )
    )

    return "█" * count


# ============================================================
# PRICE ZONE OI BAR
# ============================================================

def print_price_zone_bars(zone_df):

    print()
    print("=" * 70)
    print("📊 PRICE ZONE CALL / PUT OI")
    print("=" * 70)

    max_oi = max(
        zone_df["call_oi"].max(),
        zone_df["put_oi"].max()
    )

    max_oi = safe_int(max_oi)

    for _, row in zone_df.iterrows():

        zone = row["zone"]

        call_oi = safe_int(
            row["call_oi"]
        )

        put_oi = safe_int(
            row["put_oi"]
        )

        print()
        print(zone)

        call_bar = make_bar(
            call_oi,
            max_oi
        )

        put_bar = make_bar(
            put_oi,
            max_oi
        )

        print(
            f"CALL "
            f"{call_bar:<30} "
            f"{fmt_int(call_oi)}"
        )

        print(
            f"PUT  "
            f"{put_bar:<30} "
            f"{fmt_int(put_oi)}"
        )


# ============================================================
# ZONE ANALYSIS
# ============================================================

def analyze_zone_direction(zone_df):

    print()
    print("=" * 70)
    print("📈 PRICE ZONE ANALYSIS")
    print("=" * 70)

    for _, row in zone_df.iterrows():

        zone = row["zone"]

        call_oi = safe_int(
            row["call_oi"]
        )

        put_oi = safe_int(
            row["put_oi"]
        )

        total = call_oi + put_oi

        if total <= 0:
            call_ratio = 0.0
        else:
            call_ratio = (
                call_oi / total * 100
            )

        if call_oi > put_oi:
            bias = "CALL OI DOMINANT"
        elif put_oi > call_oi:
            bias = "PUT OI DOMINANT"
        else:
            bias = "BALANCED"

        print()
        print(zone)

        print(
            f"CALL OI : {fmt_int(call_oi)}"
        )

        print(
            f"PUT OI  : {fmt_int(put_oi)}"
        )

        print(
            f"CALL %  : {fmt_pct(call_ratio)}"
        )

        print(
            f"BIAS    : {bias}"
        )


# ============================================================
# DETAIL FILTER
# ============================================================

def get_detail_options(df):

    detail = df[
        (df["strike"] >= DETAIL_MIN_STRIKE)
        &
        (df["strike"] <= DETAIL_MAX_STRIKE)
    ].copy()

    return detail


# ============================================================
# DISTANCE ZONE
# ============================================================

def get_distance_zone(strike, current_price):

    strike = safe_float(strike)
    current_price = safe_float(
        current_price
    )

    distance = strike - current_price

    if distance <= -10:
        return "< -$10"

    if distance < -5:
        return "-$10 ~ -$5"

    if distance < 0:
        return "-$5 ~ ATM"

    if distance == 0:
        return "ATM"

    if distance <= 5:
        return "ATM ~ +$5"

    if distance <= 10:
        return "+$5 ~ +$10"

    return "> +$10"


# ============================================================
# BUILD DISTANCE STRUCTURE
# ============================================================

def build_distance_structure(
    df,
    current_price
):

    print()
    print("=" * 70)
    print("BUILD DISTANCE ZONE STRUCTURE")
    print("=" * 70)

    detail = get_detail_options(df)

    detail["distance_zone"] = detail[
        "strike"
    ].apply(
        lambda x:
        get_distance_zone(
            x,
            current_price
        )
    )

    zones = [
        "< -$10",
        "-$10 ~ -$5",
        "-$5 ~ ATM",
        "ATM",
        "ATM ~ +$5",
        "+$5 ~ +$10",
        "> +$10"
    ]

    result = []

    for zone in zones:

        z = detail[
            detail["distance_zone"] == zone
        ]

        call_oi = safe_int(
            z.loc[
                z["option_type"] == "CALL",
                "openInterest"
            ].sum()
        )

        put_oi = safe_int(
            z.loc[
                z["option_type"] == "PUT",
                "openInterest"
            ].sum()
        )

        result.append({
            "zone": zone,
            "call_oi": call_oi,
            "put_oi": put_oi,
            "total_oi": call_oi + put_oi
        })

    result_df = pd.DataFrame(result)

    print()

    print(
        f"{'ZONE':>16}"
        f"{'CALL OI':>14}"
        f"{'PUT OI':>14}"
        f"{'TOTAL':>14}"
    )

    print("-" * 60)

    for _, row in result_df.iterrows():

        print(
            f"{row['zone']:>16}"
            f"{fmt_int(row['call_oi']):>14}"
            f"{fmt_int(row['put_oi']):>14}"
            f"{fmt_int(row['total_oi']):>14}"
        )

    return result_df


# ============================================================
# TOP OI
# ============================================================

def print_top_oi(df):

    print()
    print("=" * 70)
    print(
        f"🔥 TOP {TOP_N} OPEN INTEREST "
        f"${DETAIL_MIN_STRIKE}~${DETAIL_MAX_STRIKE}"
    )
    print("=" * 70)

    detail = get_detail_options(df)

    detail = detail.sort_values(
        "openInterest",
        ascending=False
    ).head(TOP_N)

    if detail.empty:

        print("No detail options found.")

        return

    print()

    for idx, (_, row) in enumerate(
        detail.iterrows(),
        start=1
    ):

        option_type = str(
            row["option_type"]
        )

        strike = safe_float(
            row["strike"]
        )

        oi = safe_int(
            row["openInterest"]
        )

        volume = safe_int(
            row["volume"]
        )

        expiration = row[
            "expiration"
        ]

        dte = safe_int(
            row["DTE"]
        )

        print(
            f"{idx:02d}. "
            f"{option_type:<4} "
            f"${strike:,.2f} "
            f"EXP {expiration} "
            f"DTE {dte:<3} "
            f"OI {fmt_int(oi):>10} "
            f"VOL {fmt_int(volume):>10}"
        )


# ============================================================
# SUMMARY
# ============================================================

def print_summary(
    current_price,
    zone_df,
    detail_df,
    raw_df
):

    print()
    print("=" * 70)
    print("🔥 FINAL SUMMARY")
    print("=" * 70)

    print(
        f"SYMBOL          : {SYMBOL}"
    )

    print(
        f"CURRENT PRICE   : {fmt_price(current_price)}"
    )

    print(
        f"RAW ROWS        : {len(raw_df):,}"
    )

    print(
        f"ZONE ROWS       : {len(zone_df):,}"
    )

    print(
        f"DETAIL ROWS     : {len(detail_df):,}"
    )

    # 전체 Call / Put OI
    call_oi = safe_int(
        detail_df.loc[
            detail_df["option_type"] == "CALL",
            "openInterest"
        ].sum()
    )

    put_oi = safe_int(
        detail_df.loc[
            detail_df["option_type"] == "PUT",
            "openInterest"
        ].sum()
    )

    total_oi = call_oi + put_oi

    print()
    print(
        f"DETAIL CALL OI  : {fmt_int(call_oi)}"
    )

    print(
        f"DETAIL PUT OI   : {fmt_int(put_oi)}"
    )

    print(
        f"DETAIL TOTAL OI : {fmt_int(total_oi)}"
    )

    if total_oi > 0:

        call_ratio = (
            call_oi /
            total_oi *
            100
        )

        put_ratio = (
            put_oi /
            total_oi *
            100
        )

        print(
            f"CALL OI RATIO   : "
            f"{fmt_pct(call_ratio)}"
        )

        print(
            f"PUT OI RATIO    : "
            f"{fmt_pct(put_ratio)}"
        )

    if call_oi > put_oi:

        print(
            "STRUCTURE       : "
            "CALL OI DOMINANT 🟢"
        )

    elif put_oi > call_oi:

        print(
            "STRUCTURE       : "
            "PUT OI DOMINANT 🔴"
        )

    else:

        print(
            "STRUCTURE       : "
            "BALANCED 🟡"
        )


# ============================================================
# MAIN
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # 1. CURRENT PRICE
        # ----------------------------------------------------

        current_price = get_current_price(
            SYMBOL
        )

        # ----------------------------------------------------
        # 2. YAHOO COLLECTION
        # ----------------------------------------------------

        raw_df = collect_options(
            SYMBOL
        )

        # ----------------------------------------------------
        # 3. NORMALIZATION
        # ----------------------------------------------------

        normalized_df = normalize_options(
            raw_df
        )

        # ----------------------------------------------------
        # 4. FILTER
        # ----------------------------------------------------

        zone_df_raw = apply_filters(
            normalized_df
        )

        # ----------------------------------------------------
        # 5. PRICE ZONE
        # ----------------------------------------------------

        zone_structure = (
            build_price_zone_structure(
                zone_df_raw
            )
        )

        # ----------------------------------------------------
        # 6. PRICE ZONE BARS
        # ----------------------------------------------------

        print_price_zone_bars(
            zone_structure
        )

        # ----------------------------------------------------
        # 7. ZONE ANALYSIS
        # ----------------------------------------------------

        analyze_zone_direction(
            zone_structure
        )

        # ----------------------------------------------------
        # 8. DETAIL OPTIONS
        # ----------------------------------------------------

        detail_df = get_detail_options(
            zone_df_raw
        )

        # ----------------------------------------------------
        # 9. DISTANCE ZONE
        # ----------------------------------------------------

        build_distance_structure(
            zone_df_raw,
            current_price
        )

        # ----------------------------------------------------
        # 10. TOP OI
        # ----------------------------------------------------

        print_top_oi(
            zone_df_raw
        )

        # ----------------------------------------------------
        # 11. FINAL SUMMARY
        # ----------------------------------------------------

        print_summary(
            current_price,
            zone_df_raw,
            detail_df,
            normalized_df
        )

        print()
        print("=" * 70)
        print("✅ SCANNER COMPLETE")
        print("=" * 70)

        return 0

    except Exception as e:

        print()
        print("=" * 70)
        print("❌ SCANNER FAILED")
        print("=" * 70)

        print(
            f"Error type: {type(e).__name__}"
        )

        print(
            f"Error: {repr(e)}"
        )

        return 1


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":

    sys.exit(
        main()
  )
