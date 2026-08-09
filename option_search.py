import os
import sys
import time
import math
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

# Wall 계산 범위
WALL_RANGE_PCT = 30

# ATM IV 범위
ATM_RANGE_PCT = 5


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
# CURRENT PRICE
# ============================================================

def get_current_price(ticker):

    for attempt in range(1, 4):

        try:

            print(
                f"💰 {ticker} 현재가 조회 ({attempt}/3)"
            )

            t = yf.Ticker(ticker)

            hist = t.history(
                period="5d",
                interval="1d",
                auto_adjust=False
            )

            if not hist.empty:

                close = (
                    hist["Close"]
                    .dropna()
                )

                if not close.empty:

                    price = float(
                        close.iloc[-1]
                    )

                    print(
                        f"💰 {ticker} 현재가: "
                        f"${price:.2f}"
                    )

                    return price

        except Exception as e:

            print(
                f"⚠️ 현재가 조회 실패: {e}"
            )

        if attempt < 3:
            time.sleep(2)

    print(
        f"❌ {ticker} 현재가 조회 실패"
    )

    return None


# ============================================================
# OPTION DATA
# ============================================================

def get_option_data(ticker):

    t = yf.Ticker(ticker)

    expirations = t.options

    if not expirations:

        raise Exception(
            "옵션 만기 데이터가 없습니다."
        )

    today = date.today()

    valid_expirations = []

    for exp in expirations:

        try:

            exp_date = datetime.strptime(
                exp,
                "%Y-%m-%d"
            ).date()

            dte = (
                exp_date - today
            ).days

            if 0 <= dte <= MAX_DTE:

                valid_expirations.append(
                    (exp, dte)
                )

        except Exception:

            continue

    print(
        f"📅 전체 만기: "
        f"{len(expirations)}개"
    )

    print(
        f"📅 0~180 DTE 만기: "
        f"{len(valid_expirations)}개"
    )

    if not valid_expirations:

        raise Exception(
            "0~180 DTE 옵션 만기가 없습니다."
        )

    rows = []

    for exp, dte in valid_expirations:

        print(
            f"   수집: {exp} | DTE {dte}"
        )

        try:

            chain = t.option_chain(exp)

            calls = chain.calls.copy()
            puts = chain.puts.copy()

            calls["option_type"] = "CALL"
            puts["option_type"] = "PUT"

            calls["expiration"] = exp
            puts["expiration"] = exp

            calls["DTE"] = dte
            puts["DTE"] = dte

            rows.append(calls)
            rows.append(puts)

        except Exception as e:

            print(
                f"⚠️ {exp} 오류: {e}"
            )

    if not rows:

        raise Exception(
            "옵션 데이터를 가져오지 못했습니다."
        )

    return pd.concat(
        rows,
        ignore_index=True
    )


# ============================================================
# NORMALIZE
# ============================================================

def normalize_option_data(df):

    df = df.copy()

    numeric_cols = [
        "strike",
        "lastPrice",
        "bid",
        "ask",
        "volume",
        "openInterest",
        "impliedVolatility",
        "DTE"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

    df["volume"] = (
        df["volume"]
        .clip(lower=0)
    )

    df["openInterest"] = (
        df["openInterest"]
        .clip(lower=0)
    )

    return df


# ============================================================
# BLACK-SCHOLES HELPERS
# ============================================================

def norm_pdf(x):

    return (
        math.exp(-0.5 * x * x)
        / math.sqrt(2 * math.pi)
    )


def norm_cdf(x):

    return (
        0.5
        * (
            1
            + math.erf(
                x / math.sqrt(2)
            )
        )
    )


def safe_iv(iv):

    try:

        iv = float(iv)

        if iv <= 0:
            return 0.0001

        if iv > 5:
            iv = iv / 100

        return max(
            0.0001,
            min(iv, 5.0)
        )

    except Exception:

        return 0.0001


# ============================================================
# BLACK-SCHOLES GREEKS
# ============================================================

def calculate_greeks(
    spot,
    strike,
    iv,
    dte,
    option_type
):

    try:

        spot = float(spot)
        strike = float(strike)
        iv = safe_iv(iv)
        dte = float(dte)

        if spot <= 0:
            raise ValueError("invalid spot")

        if strike <= 0:
            raise ValueError("invalid strike")

        # 만기 당일은 1일 근사.
        # 무료 데이터의 특성상 장중 정확한 초단기 T를
        # 알 수 없기 때문에 안정성을 우선한다.
        T = max(
            dte / 365.0,
            1 / 365.0
        )

        r = RISK_FREE_RATE

        sqrt_T = math.sqrt(T)

        d1 = (
            math.log(
                spot / strike
            )
            + (
                r
                + 0.5 * iv * iv
            ) * T
        ) / (
            iv * sqrt_T
        )

        d2 = (
            d1
            - iv * sqrt_T
        )

        pdf = norm_pdf(d1)

        # ----------------------------------------------------
        # DELTA
        # ----------------------------------------------------

        if option_type == "CALL":

            delta = norm_cdf(d1)

        else:

            delta = (
                norm_cdf(d1) - 1
            )

        # ----------------------------------------------------
        # GAMMA
        # 동일한 공식
        # ----------------------------------------------------

        gamma = (
            pdf
            / (
                spot
                * iv
                * sqrt_T
            )
        )

        # ----------------------------------------------------
        # VEGA
        # 1.00 = volatility 100% 변화 기준
        # ----------------------------------------------------

        vega = (
            spot
            * pdf
            * sqrt_T
        )

        # ----------------------------------------------------
        # VANNA
        # dDelta / dVol
        # ----------------------------------------------------

        vanna = (
            -pdf
            * d2
            / iv
        )

        # ----------------------------------------------------
        # CHARM
        # Delta decay proxy
        # ----------------------------------------------------

        charm_common = (
            pdf
            * (
                2 * r * T
                - d2 * iv * sqrt_T
            )
            / (
                2 * T
            )
        )

        charm = (
            -charm_common
            / (
                spot
                * iv
                * sqrt_T
            )
        )

        # ----------------------------------------------------
        # THETA
        # ----------------------------------------------------

        if option_type == "CALL":

            theta = (
                -(
                    spot
                    * pdf
                    * iv
                    / (
                        2 * sqrt_T
                    )
                )
                - (
                    r
                    * strike
                    * math.exp(
                        -r * T
                    )
                    * norm_cdf(d2)
                )
            )

        else:

            theta = (
                -(
                    spot
                    * pdf
                    * iv
                    / (
                        2 * sqrt_T
                    )
                )
                + (
                    r
                    * strike
                    * math.exp(
                        -r * T
                    )
                    * norm_cdf(-d2)
                )
            )

        return {
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "vanna": vanna,
            "charm": charm,
            "theta": theta,
            "iv": iv
        }

    except Exception:

        return {
            "delta": 0.0,
            "gamma": 0.0,
            "vega": 0.0,
            "vanna": 0.0,
            "charm": 0.0,
            "theta": 0.0,
            "iv": safe_iv(iv)
        }


# ============================================================
# OPTION METRICS
# ============================================================

def calculate_option_metrics(
    df,
    current_price
):

    df = normalize_option_data(df)

    delta_list = []
    gamma_list = []
    vega_list = []
    vanna_list = []
    charm_list = []
    theta_list = []
    iv_list = []

    for _, row in df.iterrows():

        greeks = calculate_greeks(
            current_price,
            row["strike"],
            row["impliedVolatility"],
            row["DTE"],
            row["option_type"]
        )

        delta_list.append(
            greeks["delta"]
        )

        gamma_list.append(
            greeks["gamma"]
        )

        vega_list.append(
            greeks["vega"]
        )

        vanna_list.append(
            greeks["vanna"]
        )

        charm_list.append(
            greeks["charm"]
        )

        theta_list.append(
            greeks["theta"]
        )

        iv_list.append(
            greeks["iv"]
        )

    df["delta"] = delta_list
    df["gamma"] = gamma_list
    df["vega"] = vega_list
    df["vanna"] = vanna_list
    df["charm"] = charm_list
    df["theta"] = theta_list
    df["IV"] = iv_list

    # ========================================================
    # TRADED PREMIUM PROXY
    # ========================================================
    #
    # 중요:
    # 실제 BUY/SELL Premium Flow가 아니다.
    # 거래량 × 마지막 가격으로 계산한 거래대금 Proxy.
    #

    df["traded_premium_proxy"] = (
        df["lastPrice"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
    )

    # 기존 컬럼 호환
    df["premium_flow"] = (
        df["traded_premium_proxy"]
    )

    # ========================================================
    # BID / ASK SPREAD
    # ========================================================

    df["mid_price"] = (
        (
            df["bid"]
            + df["ask"]
        )
        / 2
    )

    df["spread_pct"] = 0.0

    valid_mid = (
        df["mid_price"] > 0
    )

    df.loc[
        valid_mid,
        "spread_pct"
    ] = (
        (
            df.loc[
                valid_mid,
                "ask"
            ]
            -
            df.loc[
                valid_mid,
                "bid"
            ]
        )
        /
        df.loc[
            valid_mid,
            "mid_price"
        ]
        * 100
    )

    # ========================================================
    # DEALER GEX PROXY
    # ========================================================
    #
    # 1% underlying move 기준.
    #
    # Call = +
    # Put  = -
    #
    # 이것은 실제 dealer positioning이 아니라
    # OI 기반 positioning 가정이다.
    #

    df["GEX"] = (
        df["gamma"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
        * current_price
        * 0.01
    )

    df.loc[
        df["option_type"] == "PUT",
        "GEX"
    ] *= -1

    # ========================================================
    # NET OI DELTA EXPOSURE PROXY
    # ========================================================
    #
    # OI가 실제 long/short 방향을 알려주지는 않는다.
    # 따라서 "OI Delta Exposure Proxy"로 사용한다.
    #

    df["delta_exposure"] = (
        df["delta"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # ========================================================
    # VANNA EXPOSURE
    # ========================================================

    df["vanna_exposure"] = (
        df["vanna"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # ========================================================
    # CHARM EXPOSURE
    # ========================================================

    df["charm_exposure"] = (
        df["charm"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # ========================================================
    # VEGA EXPOSURE
    # ========================================================

    df["vega_exposure"] = (
        df["vega"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
    )

    # ========================================================
    # TRADE SIDE PROXY
    # ========================================================
    #
    # 실제 체결 방향 데이터가 없기 때문에 Proxy.
    #

    def estimate_trade_side(row):

        try:

            last = float(
                row["lastPrice"]
            )

            bid = float(
                row["bid"]
            )

            ask = float(
                row["ask"]
            )

            if (
                ask > 0
                and bid > 0
                and ask >= bid
            ):

                if (
                    last
                    >= ask - (
                        ask - bid
                    ) * 0.20
                ):

                    return 1

                if (
                    last
                    <= bid + (
                        ask - bid
                    ) * 0.20
                ):

                    return -1

            return 0

        except Exception:

            return 0

    df["trade_side_proxy"] = df.apply(
        estimate_trade_side,
        axis=1
    )

    # 기존 호환
    df["trade_side"] = (
        df["trade_side_proxy"]
    )

    # ========================================================
    # HIRO-LIKE FLOW PROXY
    # ========================================================
    #
    # 실제 HIRO가 아니다.
    # delta × volume × spot × estimated trade side
    #

    df["HIRO_proxy"] = (
        df["delta"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
        * current_price
        * df["trade_side_proxy"]
    )

    return df


# ============================================================
# AGGREGATED GREEKS
# ============================================================

def calculate_aggregate_greeks(df):

    result = {}

    # --------------------------------------------------------
    # IV
    # --------------------------------------------------------

    valid_iv = df[
        (df["IV"] > 0)
        & (df["IV"] < 5)
        & (df["volume"] > 0)
    ].copy()

    if not valid_iv.empty:

        volume_sum = (
            valid_iv["volume"]
            .sum()
        )

        if volume_sum > 0:

            result["IV"] = (
                (
                    valid_iv["IV"]
                    * valid_iv["volume"]
                ).sum()
                / volume_sum
            )

        else:

            result["IV"] = (
                valid_iv["IV"].mean()
            )

    else:

        result["IV"] = 0.0

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    result["Delta"] = (
        df["delta_exposure"]
        .sum()
    )

    # --------------------------------------------------------
    # Gamma / GEX
    # --------------------------------------------------------

    result["Gamma"] = (
        df["GEX"]
        .sum()
    )

    result["GEX"] = (
        df["GEX"]
        .sum()
    )

    # --------------------------------------------------------
    # Vanna
    # --------------------------------------------------------

    result["Vanna"] = (
        df["vanna_exposure"]
        .sum()
    )

    # --------------------------------------------------------
    # Charm
    # --------------------------------------------------------

    result["Charm"] = (
        df["charm_exposure"]
        .sum()
    )

    # --------------------------------------------------------
    # Vega
    # --------------------------------------------------------

    result["Vega"] = (
        df["vega_exposure"]
        .sum()
    )

    # --------------------------------------------------------
    # HIRO Proxy
    # --------------------------------------------------------

    result["HIRO"] = (
        df["HIRO_proxy"]
        .sum()
    )

    return result


# ============================================================
# MARKET / FLOW SUMMARY
# ============================================================

def calculate_flow_summary(
    df,
    current_price
):

    calls = df[
        df["option_type"] == "CALL"
    ]

    puts = df[
        df["option_type"] == "PUT"
    ]

    call_volume = float(
        calls["volume"].sum()
    )

    put_volume = float(
        puts["volume"].sum()
    )

    call_oi = float(
        calls["openInterest"].sum()
    )

    put_oi = float(
        puts["openInterest"].sum()
    )

    call_premium = float(
        calls[
            "traded_premium_proxy"
        ].sum()
    )

    put_premium = float(
        puts[
            "traded_premium_proxy"
        ].sum()
    )

    total_volume = (
        call_volume
        + put_volume
    )

    total_oi = (
        call_oi
        + put_oi
    )

    total_premium = (
        call_premium
        + put_premium
    )

    if total_volume > 0:

        call_volume_ratio = (
            call_volume
            / total_volume
        )

    else:

        call_volume_ratio = 0.5

    if total_oi > 0:

        call_oi_ratio = (
            call_oi
            / total_oi
        )

    else:

        call_oi_ratio = 0.5

    if total_premium > 0:

        call_premium_ratio = (
            call_premium
            / total_premium
        )

    else:

        call_premium_ratio = 0.5

    # --------------------------------------------------------
    # ATM IV
    # --------------------------------------------------------

    atm = df[
        (
            abs(
                df["strike"]
                - current_price
            )
            / current_price
            * 100
        )
        <= ATM_RANGE_PCT
    ].copy()

    atm = atm[
        (atm["IV"] > 0)
        & (atm["volume"] > 0)
    ]

    if not atm.empty:

        volume_sum = (
            atm["volume"].sum()
        )

        if volume_sum > 0:

            atm_iv = (
                (
                    atm["IV"]
                    * atm["volume"]
                ).sum()
                / volume_sum
            )

        else:

            atm_iv = atm["IV"].mean()

    else:

        atm_iv = 0.0

    # --------------------------------------------------------
    # DTE distribution
    # --------------------------------------------------------

    dte_buckets = {
        "0_7": int(
            (
                (df["DTE"] >= 0)
                &
                (df["DTE"] <= 7)
            ).sum()
        ),

        "8_30": int(
            (
                (df["DTE"] >= 8)
                &
                (df["DTE"] <= 30)
            ).sum()
        ),

        "31_60": int(
            (
                (df["DTE"] >= 31)
                &
                (df["DTE"] <= 60)
            ).sum()
        ),

        "61_180": int(
            (
                (df["DTE"] >= 61)
                &
                (df["DTE"] <= 180)
            ).sum()
        )
    }

    return {
        "call_volume": call_volume,
        "put_volume": put_volume,
        "call_oi": call_oi,
        "put_oi": put_oi,
        "call_premium": call_premium,
        "put_premium": put_premium,
        "call_volume_ratio": call_volume_ratio,
        "call_oi_ratio": call_oi_ratio,
        "call_premium_ratio": call_premium_ratio,
        "atm_iv": atm_iv,
        "dte_buckets": dte_buckets
    }


# ============================================================
# WALLS
# ============================================================

def find_walls(
    df,
    current_price
):

    active = df[
        df["openInterest"] > 0
    ].copy()

    if active.empty:

        return {
            "call_wall": None,
            "put_wall": None,
            "call_wall_gex": 0.0,
            "put_wall_gex": 0.0,
            "call_wall_oi": 0.0,
            "put_wall_oi": 0.0
        }

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"]
        <= WALL_RANGE_PCT
    ].copy()

    if active.empty:

        return {
            "call_wall": None,
            "put_wall": None,
            "call_wall_gex": 0.0,
            "put_wall_gex": 0.0,
            "call_wall_oi": 0.0,
            "put_wall_oi": 0.0
        }

    # --------------------------------------------------------
    # CALL WALL
    # --------------------------------------------------------

    calls = active[
        (
            active["option_type"]
            == "CALL"
        )
        &
        (
            active["strike"]
            > current_price
        )
    ]

    call_wall = None
    call_wall_gex = 0.0
    call_wall_oi = 0.0

    if not calls.empty:

        grouped = (
            calls
            .groupby("strike")
            .agg(
                gex=("GEX", "sum"),
                oi=("openInterest", "sum"),
                volume=("volume", "sum")
            )
        )

        if not grouped.empty:

            # GEX가 가장 큰 Strike
            idx = grouped[
                "gex"
            ].idxmax()

            call_wall = float(idx)

            call_wall_gex = float(
                grouped.loc[
                    idx,
                    "gex"
                ]
            )

            call_wall_oi = float(
                grouped.loc[
                    idx,
                    "oi"
                ]
            )

    # --------------------------------------------------------
    # PUT WALL
    # --------------------------------------------------------

    puts = active[
        (
            active["option_type"]
            == "PUT"
        )
        &
        (
            active["strike"]
            < current_price
        )
    ]

    put_wall = None
    put_wall_gex = 0.0
    put_wall_oi = 0.0

    if not puts.empty:

        grouped = (
            puts
            .groupby("strike")
            .agg(
                gex=("GEX", "sum"),
                oi=("openInterest", "sum"),
                volume=("volume", "sum")
            )
        )

        if not grouped.empty:

            idx = grouped[
                "gex"
            ].idxmin()

            put_wall = float(idx)

            put_wall_gex = float(
                grouped.loc[
                    idx,
                    "gex"
                ]
            )

            put_wall_oi = float(
                grouped.loc[
                    idx,
                    "oi"
                ]
            )

    return {
        "call_wall": call_wall,
        "put_wall": put_wall,
        "call_wall_gex": call_wall_gex,
        "put_wall_gex": put_wall_gex,
        "call_wall_oi": call_wall_oi,
        "put_wall_oi": put_wall_oi
    }


# ============================================================
# TOP FLOW
# ============================================================

def find_top_flow(
    df,
    current_price
):

    active = df[
        df["volume"] > 0
    ].copy()

    if active.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"]
        <= WALL_RANGE_PCT
    ].copy()

    if active.empty:

        return (
            pd.DataFrame(),
            pd.DataFrame()
        )

    calls = (
        active[
            active["option_type"]
            == "CALL"
        ]
        .sort_values(
            [
                "traded_premium_proxy",
                "volume"
            ],
            ascending=False
        )
        .head(5)
    )

    puts = (
        active[
            active["option_type"]
            == "PUT"
        ]
        .sort_values(
            [
                "traded_premium_proxy",
                "volume"
            ],
            ascending=False
        )
        .head(5)
    )

    return calls, puts


# ============================================================
# DATA QUALITY
# ============================================================

def calculate_data_quality(df):

    total = len(df)

    if total == 0:

        return {
            "score": 0,
            "label": "LOW"
        }

    valid_bidask = (
        (
            df["bid"] > 0
        )
        &
        (
            df["ask"] > 0
        )
        &
        (
            df["ask"] >= df["bid"]
        )
    ).mean()

    active_volume = (
        df["volume"] > 0
    ).mean()

    score = (
        valid_bidask * 60
        + active_volume * 40
    )

    if score >= 75:

        label = "HIGH"

    elif score >= 45:

        label = "MEDIUM"

    else:

        label = "LOW"

    return {
        "score": score,
        "label": label
    }


# ============================================================
# REPORT
# ============================================================

def build_report(
    ticker,
    current_price,
    df,
    greeks,
    flow,
    walls,
    quality
):

    lines = []

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        f"🔥 <b>{ticker} OPTION SEARCH</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    lines.append(
        f"💰 현재가: "
        f"<b>${current_price:.2f}</b>"
    )

    lines.append(
        f"📊 옵션 행수: "
        f"{len(df):,}"
    )

    lines.append(
        f"📅 DTE: "
        f"{int(df['DTE'].min())}"
        f" ~ "
        f"{int(df['DTE'].max())}"
    )

    lines.append(
        f"🧪 데이터 품질: "
        f"{quality['label']} "
        f"({quality['score']:.0f}/100)"
    )

    lines.append("")

    # --------------------------------------------------------
    # FLOW
    # --------------------------------------------------------

    lines.append(
        "📊 <b>CALL / PUT 구조</b>"
    )

    lines.append(
        f"CALL Volume: "
        f"{flow['call_volume']:,.0f}"
    )

    lines.append(
        f"PUT Volume: "
        f"{flow['put_volume']:,.0f}"
    )

    lines.append(
        f"CALL OI: "
        f"{flow['call_oi']:,.0f}"
    )

    lines.append(
        f"PUT OI: "
        f"{flow['put_oi']:,.0f}"
    )

    lines.append(
        f"CALL 거래대금 Proxy: "
        f"{format_money(flow['call_premium'])}"
    )

    lines.append(
        f"PUT 거래대금 Proxy: "
        f"{format_money(flow['put_premium'])}"
    )

    lines.append("")

    # --------------------------------------------------------
    # GREEKS
    # --------------------------------------------------------

    lines.append(
        "🧮 <b>AGGREGATE GREEKS</b>"
    )

    lines.append(
        f"Delta Exposure Proxy: "
        f"{format_money(greeks['Delta'])}"
    )

    lines.append(
        f"GEX Proxy / 1%: "
        f"{format_money(greeks['GEX'])}"
    )

    lines.append(
        f"Vanna Exposure: "
        f"{format_money(greeks['Vanna'])}"
    )

    lines.append(
        f"Charm Exposure: "
        f"{format_money(greeks['Charm'])}"
    )

    lines.append(
        f"Vega Exposure: "
        f"{format_money(greeks['Vega'])}"
    )

    lines.append(
        f"HIRO-like Flow Proxy: "
        f"{format_money(greeks['HIRO'])}"
    )

    lines.append(
        f"ATM IV: "
        f"{flow['atm_iv'] * 100:.1f}%"
    )

    lines.append("")

    # --------------------------------------------------------
    # WALL
    # --------------------------------------------------------

    lines.append(
        "🧱 <b>OPTION WALL</b>"
    )

    if walls["call_wall"] is not None:

        distance = (
            (
                walls["call_wall"]
                - current_price
            )
            / current_price
            * 100
        )

        lines.append(
            f"📈 Call Wall: "
            f"${walls['call_wall']:g} "
            f"(+{distance:.1f}%)"
        )

    else:

        lines.append(
            "📈 Call Wall: N/A"
        )

    if walls["put_wall"] is not None:

        distance = (
            (
                current_price
                - walls["put_wall"]
            )
            / current_price
            * 100
        )

        lines.append(
            f"📉 Put Wall: "
            f"${walls['put_wall']:g} "
            f"(-{distance:.1f}%)"
        )

    else:

        lines.append(
            "📉 Put Wall: N/A"
        )

    lines.append("")

    # --------------------------------------------------------
    # TOP FLOW
    # --------------------------------------------------------

    top_calls, top_puts = find_top_flow(
        df,
        current_price
    )

    lines.append(
        "🔥 <b>TOP CALL FLOW PROXY</b>"
    )

    if top_calls.empty:

        lines.append(
            "없음"
        )

    else:

        for _, row in top_calls.iterrows():

            lines.append(
                f"• ${row['strike']:g} "
                f"| DTE {int(row['DTE'])} "
                f"| Vol {int(row['volume']):,} "
                f"| {format_money(row['traded_premium_proxy'])}"
            )

    lines.append("")

    lines.append(
        "🔻 <b>TOP PUT FLOW PROXY</b>"
    )

    if top_puts.empty:

        lines.append(
            "없음"
        )

    else:

        for _, row in top_puts.iterrows():

            lines.append(
                f"• ${row['strike']:g} "
                f"| DTE {int(row['DTE'])} "
                f"| Vol {int(row['volume']):,} "
                f"| {format_money(row['traded_premium_proxy'])}"
            )

    lines.append("")

    lines.append(
        "⚠️ 거래대금 Proxy는 "
        "실제 Buy/Sell 체결 방향이 아닙니다."
    )

    lines.append(
        "⚠️ GEX는 OI 기반 Dealer Positioning Proxy입니다."
    )

    lines.append(
        "⚠️ HIRO-like Flow는 실제 HIRO가 아닙니다."
    )

    lines.append(
        "⚠️ OI만으로 실제 Long/Short 포지션을 "
        "확정할 수 없습니다."
    )

    return "\n".join(lines)


# ============================================================
# SAVE OPTION CSV
# ============================================================

def save_option_csv(
    ticker,
    df
):

    filename = os.path.join(
        RESULT_DIR,
        f"{ticker}_OPTION_SEARCH.csv"
    )

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print(
        f"💾 CSV 저장: {filename}"
    )

    return filename


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

    if not BOT_TOKEN:

        print(
            "⚠️ TELEGRAM_BOT_TOKEN이 없습니다."
        )

        return False

    if not CHAT_ID:

        print(
            "⚠️ TELEGRAM_CHAT_ID가 없습니다."
        )

        return False

    url = (
        "https://api.telegram.org/"
        f"bot{BOT_TOKEN}/sendMessage"
    )

    data = {
        "chat_id": CHAT_ID,
        "text": text,
        "parse_mode": "HTML"
    }

    try:

        response = requests.post(
            url,
            data=data,
            timeout=30
        )

        print(
            f"📨 Telegram: "
            f"{response.status_code}"
        )

        if not response.ok:

            print(response.text)

        return response.ok

    except Exception as e:

        print(
            f"❌ Telegram 오류: {e}"
        )

        return False


# ============================================================
# ONE TICKER ANALYSIS
# ============================================================

def analyze_ticker(ticker):

    ticker = (
        ticker
        .upper()
        .strip()
    )

    print("")
    print("=" * 70)

    print(
        f"🔥 OPTION SEARCH : {ticker}"
    )

    print("=" * 70)

    current_price = get_current_price(
        ticker
    )

    if current_price is None:

        return None

    print("")

    df = get_option_data(
        ticker
    )

    print("")

    print(
        f"📊 옵션 행수: {len(df):,}"
    )

    print(
        f"📅 DTE: "
        f"{df['DTE'].min()} ~ "
        f"{df['DTE'].max()}"
    )

    print("")

    print(
        "📊 Greeks / GEX 계산..."
    )

    df = calculate_option_metrics(
        df,
        current_price
    )

    greeks = (
        calculate_aggregate_greeks(
            df
        )
    )

    flow = (
        calculate_flow_summary(
            df,
            current_price
        )
    )

    walls = (
        find_walls(
            df,
            current_price
        )
    )

    quality = (
        calculate_data_quality(
            df
        )
    )

    report = build_report(
        ticker,
        current_price,
        df,
        greeks,
        flow,
        walls,
        quality
    )

    print("")
    print(report)

    csv_file = save_option_csv(
        ticker,
        df
    )

    # 개별 종목 Telegram
    telegram_ok = send_telegram(
        report
    )

    return {
        "ticker": ticker,
        "current_price": current_price,
        "df": df,
        "greeks": greeks,
        "flow": flow,
        "walls": walls,
        "quality": quality,
        "report": report,
        "csv_file": csv_file,
        "telegram_ok": telegram_ok
    }


# ============================================================
# COMPATIBILITY
# ============================================================

def run(ticker):

    result = analyze_ticker(
        ticker
    )

    if result is None:

        return False

    return result["telegram_ok"]


# ============================================================
# STANDALONE
# ============================================================

def main():

    if len(sys.argv) < 2:

        print(
            "사용법:"
        )

        print(
            "python option_search.py TICKER"
        )

        sys.exit(1)

    result = analyze_ticker(
        sys.argv[1]
    )

    if result is None:

        sys.exit(1)

    print("")
    print(
        "🔥 OPTION SEARCH 완료"
    )


if __name__ == "__main__":

    main()
