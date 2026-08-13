import os
import sys
import time
import math
import re
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
        return f"{sign}${x / 1_0_000_000_000:.2f}B"

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


def format_pct(x):

    try:
        return f"{float(x):.1f}%"
    except Exception:
        return "0.0%"


# ============================================================
# MARKET PRICE CONTEXT
# ============================================================

def get_market_price_context(ticker):

    for attempt in range(1, 4):

        try:

            print(
                f"💰 {ticker} 가격 조회 ({attempt}/3)"
            )

            t = yf.Ticker(ticker)

            # ------------------------------------------------
            # REGULAR CLOSE
            # ------------------------------------------------

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

            # ------------------------------------------------
            # MARKET STATE
            # ------------------------------------------------

            market_state = None

            try:

                info = t.get_info()

                market_state = info.get(
                    "marketState"
                )

            except Exception:

                market_state = None

            # ------------------------------------------------
            # AFTER HOURS
            # ------------------------------------------------

            after_hours_price = None

            try:

                intraday = t.history(
                    period="1d",
                    interval="1m",
                    prepost=True,
                    auto_adjust=False
                )

                if not intraday.empty:

                    intraday_close = (
                        intraday["Close"]
                        .dropna()
                    )

                    if not intraday_close.empty:

                        latest_price = float(
                            intraday_close.iloc[-1]
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
                               
