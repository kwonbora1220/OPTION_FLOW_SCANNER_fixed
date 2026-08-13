import os
import sys
import time
import math
import re
import glob
import requests
import yfinance as yf
import pandas as pd

from datetime import datetime, date


# ============================================================
# CONFIG
# ============================================================

BOT_TOKEN = os.environ.get(
    "TELEGRAM_BOT_TOKEN",
    ""
)

CHAT_ID = os.environ.get(
    "TELEGRAM_CHAT_ID",
    ""
)

MAX_DTE = 180

RISK_FREE_RATE = 0.04

CONTRACT_MULTIPLIER = 100

WALL_RANGE_PCT = 30

ATM_RANGE_PCT = 5

ENTRY_RESISTANCE_PCT = 3.0

HOLDING_SUPPORT_BUFFER_PCT = 2.0


# ============================================================
# PATH
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "03_RESULTS",
    "daily"
)

if os.path.basename(BASE_DIR).upper() == "02_PROGRAM":

    RESULT_DIR = os.path.abspath(
        os.path.join(
            BASE_DIR,
            "..",
            "03_RESULTS",
            "daily"
        )
    )

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# OI HISTORY
# ============================================================

OI_HISTORY_DIR = os.path.join(
    RESULT_DIR,
    "oi_history"
)

os.makedirs(
    OI_HISTORY_DIR,
    exist_ok=True
)


# ============================================================
# FORMAT
# ============================================================

def format_money(x):

    try:
        x = float(x)
    except Exception:
        return "$0"

    sign = ""

    if x < 0:
        sign = "-"
        x = abs(x)

    if x >= 1_000_000_000:
        return f"{sign}${x / 1_000_000_000:.2f}B"

    if x >= 1_000_000:
        return f"{sign}${x / 1_000_000:.2f}M"

    if x >= 1_000:
        return f"{sign}${x / 1_000:.1f}K"

    return f"{sign}${x:.0f}"


def format_signed_money(x):

    try:
        x = float(x)
    except Exception:
        return "$0"

    if x > 0:
        return f"+{format_money(x)}"

    return format_money(x)


# ============================================================
# MARKET PRICE
# ============================================================

def get_market_price_context(ticker):

    for attempt in range(1, 4):

        try:

            print(
                f"💰 {ticker} 가격 조회 ({attempt}/3)"
            )

            t = yf.Ticker(ticker)

            hist = t.history(
                period="5d",
                interval="1d",
                auto_adjust=False
            )

            regular_close = None

            if not hist.empty:

                close = hist["Close"].dropna()

                if not close.empty:

                    regular_close = float(
                        close.iloc[-1]
                    )

            if regular_close is None:

                raise ValueError(
                    "정규장 종가를 가져오지 못했습니다."
                )

            market_state = None

            try:

                info = t.get_info()

                market_state = info.get(
                    "marketState"
                )

            except Exception:

                market_state = None

            after_hours_price = None

            try:

                intraday = t.history(
                    period="1d",
                    interval="1m",
                    prepost=True,
                    auto_adjust=False
                )

                if not intraday.empty:

                    close_1m = (
                        intraday["Close"]
                        .dropna()
                    )

                    if not close_1m.empty:

                        latest_price = float(
                            close_1m.iloc[-1]
                        )

                        if market_state in {
                            "POST",
                            "POSTPOST"
                        }:

                            if abs(
                                latest_price
                                - regular_close
                            ) > 0.000001:

                                after_hours_price = (
                                    latest_price
                                )

            except Exception as e:

                print(
                    f"⚠️ 시간외 가격 조회 실패: {e}"
                )

            after_hours_change_pct = None

            if (
                after_hours_price is not None
                and regular_close != 0
            ):

                after_hours_change_pct = (
                    (
                        after_hours_price
                        / regular_close
                    ) - 1
                ) * 100

            print(
                f"💰 {ticker} 정규장 종가: "
                f"${regular_close:.2f}"
            )

            if after_hours_price is not None:

                print(
                    f"🌙 {ticker} 시간외 현재가: "
                    f"${after_hours_price:.2f}"
                    + (
                        f" ({after_hours_change_pct:+.2f}%)"
                        if after_hours_change_pct is not None
                        else ""
                    )
                )

            else:

                print(
                    f"🌙 {ticker} 시간외 현재가: N/A"
                )

            return {
                "regular_close": regular_close,
                "after_hours_price": after_hours_price,
                "after_hours_change_pct": after_hours_change_pct,
                "market_state": market_state,
                "option_analysis_price": regular_close
            }

        except Exception as e:

            print(
                f"⚠️ 가격 조회 실패: {e}"
            )

        if attempt < 3:

            time.sleep(2)

    return None


def get_current_price(ticker):

    context = get_market_price_context(
        ticker
    )

    if context is None:
        return None

    return context[
        "option_analysis_price"
    ]


# ============================================================
# OI SNAPSHOT PATH
# ============================================================

def get_oi_snapshot_path(
    ticker,
    snapshot_date=None
):

    if snapshot_date is None:

        snapshot_date = date.today()

    return os.path.join(
        OI_HISTORY_DIR,
        f"{ticker.upper().strip()}_OI_"
        f"{snapshot_date.strftime('%Y%m%d')}.csv"
    )


# ============================================================
# BUILD OI SNAPSHOT
# ============================================================

def build_oi_snapshot(df):

    required = [
        "expiration",
        "DTE",
        "strike",
        "option_type",
        "openInterest"
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:

        raise ValueError(
            f"OI snapshot 필수 컬럼 누락: {missing}"
        )

    snapshot = df[
        required
    ].copy()

    snapshot["expiration"] = (
        snapshot["expiration"]
        .astype(str)
        .str.strip()
    )

    snapshot["option_type"] = (
        snapshot["option_type"]
        .astype(str)
        .str.upper()
        .str.strip()
    )

    snapshot["strike"] = pd.to_numeric(
        snapshot["strike"],
        errors="coerce"
    )

    snapshot["DTE"] = pd.to_numeric(
        snapshot["DTE"],
        errors="coerce"
    )

    snapshot["openInterest"] = pd.to_numeric(
        snapshot["openInterest"],
        errors="coerce"
    ).fillna(0)

    snapshot["openInterest"] = (
        snapshot["openInterest"]
        .clip(lower=0)
    )

    snapshot = snapshot.dropna(
        subset=[
            "expiration",
            "strike"
        ]
    )

    snapshot = snapshot[
        snapshot["option_type"].isin(
            ["CALL", "PUT"]
        )
    ]

    keys = [
        "expiration",
        "strike",
        "option_type"
    ]

    snapshot = (
        snapshot
        .groupby(
            keys,
            as_index=False
        )
        .agg(
            DTE=("DTE", "max"),
            openInterest=(
                "openInterest",
                "max"
            )
        )
    )

    return snapshot


# ============================================================
# FIND PREVIOUS OI SNAPSHOT
# ============================================================

def find_previous_oi_snapshot(
    ticker,
    current_date=None
):

    if current_date is None:

        current_date = date.today()

    pattern = os.path.join(
        OI_HISTORY_DIR,
        f"{ticker.upper().strip()}_OI_*.csv"
    )

    candidates = []

    for path in glob.glob(pattern):

        filename = os.path.basename(path)

        match = re.search(
            r"_OI_(\d{8})\.csv$",
            filename
        )

        if not match:
            continue

        try:

            snapshot_date = datetime.strptime(
                match.group(1),
                "%Y%m%d"
            ).date()

        except Exception:

            continue

        if snapshot_date < current_date:

            candidates.append(
                (
                    snapshot_date,
                    path
                )
            )

    if not candidates:

        return None, None

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    return candidates[0]


# ============================================================
# CALCULATE OI DELTA
# ============================================================

def calculate_oi_delta(
    current_df,
    ticker
):

    df = current_df.copy()

    df["previous_openInterest"] = 0.0

    df["oi_delta"] = 0.0

    df["oi_delta_available"] = False

    df["oi_delta_status"] = (
        "NO_PREVIOUS_SNAPSHOT"
    )

    previous_date, previous_path = (
        find_previous_oi_snapshot(
            ticker
        )
    )

    if previous_path is None:

        print(
            "⚠️ 이전 OI snapshot 없음"
        )

        return df, None

    print(
        f"📂 이전 OI snapshot: "
        f"{previous_path}"
    )

    try:

        previous = pd.read_csv(
            previous_path,
            encoding="utf-8-sig"
        )

        previous = build_oi_snapshot(
            previous
        )

        current = build_oi_snapshot(
            df
        )

        keys = [
            "expiration",
            "strike",
            "option_type"
        ]

        previous = previous[
            keys + ["openInterest"]
        ].rename(
            columns={
                "openInterest":
                "previous_openInterest"
            }
        )

        current = current[
            keys + ["openInterest"]
        ]

        # ----------------------------------------------------
        # UNION MERGE
        # ----------------------------------------------------
        # 이전에만 존재하는 계약,
        # 현재에만 존재하는 계약,
        # 양쪽 모두 존재하는 계약을 모두 비교한다.

        merged = pd.merge(
            current,
            previous,
            on=keys,
            how="outer"
        )

        merged["openInterest"] = pd.to_numeric(
            merged["openInterest"],
            errors="coerce"
        ).fillna(0)

        merged["previous_openInterest"] = pd.to_numeric(
            merged["previous_openInterest"],
            errors="coerce"
        ).fillna(0)

        merged["oi_delta"] = (
            merged["openInterest"]
            - merged["previous_openInterest"]
        )

        # ----------------------------------------------------
        # CURRENT DATA에 PREVIOUS OI 연결
        # ----------------------------------------------------

        current_df = df.copy()

        current_df["expiration"] = (
            current_df["expiration"]
            .astype(str)
            .str.strip()
        )

        current_df["option_type"] = (
            current_df["option_type"]
            .astype(str)
            .str.upper()
            .str.strip()
        )

        current_df["strike"] = pd.to_numeric(
            current_df["strike"],
            errors="coerce"
        )

        current_df = current_df.merge(
            merged[
                keys
                + [
                    "previous_openInterest",
                    "oi_delta"
                ]
            ],
            on=keys,
            how="left"
        )

        current_df["previous_openInterest"] = (
            pd.to_numeric(
                current_df[
                    "previous_openInterest"
                ],
                errors="coerce"
            )
            .fillna(0)
        )

        current_df["oi_delta"] = (
            pd.to_numeric(
                current_df["oi_delta"],
                errors="coerce"
            )
            .fillna(0)
        )

        current_df[
            "oi_delta_available"
        ] = True

        current_df[
            "oi_delta_status"
        ] = (
            f"COMPARED_WITH_"
            f"{previous_date.strftime('%Y%m%d')}"
        )

        # ----------------------------------------------------
        # STATISTICS
        # ----------------------------------------------------

        total_delta = float(
            current_df["oi_delta"].sum()
        )

        positive = int(
            (
                current_df["oi_delta"] > 0
            ).sum()
        )

        negative = int(
            (
                current_df["oi_delta"] < 0
            ).sum()
        )

        print(
            f"📈 OI 비교 완료: "
            f"{previous_date.strftime('%Y-%m-%d')}"
        )

        print(
            f"📈 OI 증가 계약: "
            f"{positive:,}"
        )

        print(
