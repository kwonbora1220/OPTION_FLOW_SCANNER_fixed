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
    "7729872113"
)

MAX_DTE = 180

RISK_FREE_RATE = 0.04

CONTRACT_MULTIPLIER = 100


# ============================================================
# CURRENT PRICE
# ============================================================

def get_current_price(ticker):

    for attempt in range(1, 4):

        try:

            print(
                f"💰 현재가 조회 ({attempt}/3)"
            )

            t = yf.Ticker(ticker)

            hist = t.history(
                period="5d",
                interval="1d",
                auto_adjust=False,
                timeout=30
            )

            if not hist.empty:

                close = hist["Close"].dropna()

                if not close.empty:

                    price = float(
                        close.iloc[-1]
                    )

                    print(
                        f"💰 현재가: ${price:.2f}"
                    )

                    return price

            print(
                f"⚠️ {ticker} 현재가 데이터 없음"
            )

        except Exception as e:

            print(
                f"⚠️ {ticker} 현재가 조회 실패 "
                f"({attempt}/3)"
            )

            print(f"   {e}")

        if attempt < 3:

            print(
                "⏳ 3초 후 재시도..."
            )

            time.sleep(3)

    print("")
    print(
        f"❌ {ticker} 현재가 조회 3회 실패"
    )

    print(
        f"⏭️ {ticker} 건너뜀"
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

            pass

    print(
        f"📅 전체 만기: "
        f"{len(expirations)}개"
    )

    print(
        f"📅 0~180 DTE 만기: "
        f"{len(valid_expirations)}개"
    )

    rows = []

    for exp, dte in valid_expirations:

        print(
            f"수집: {exp} | DTE {dte}"
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
# NORMALIZE DATA
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
        "impliedVolatility"
    ]

    for col in numeric_cols:

        if col in df.columns:

            df[col] = pd.to_numeric(
                df[col],
                errors="coerce"
            ).fillna(0)

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

        if spot <= 0 or strike <= 0:

            return (
                0.0,
                0.0,
                0.0,
                0.0,
                iv
            )

        T = max(
            dte / 365.0,
            1 / 365.0
        )

        r = RISK_FREE_RATE

        sqrt_T = math.sqrt(T)

        d1 = (
            math.log(spot / strike)
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

        if option_type == "CALL":

            delta = norm_cdf(d1)

        else:

            delta = norm_cdf(d1) - 1

        gamma = (
            pdf
            / (
                spot
                * iv
                * sqrt_T
            )
        )

        vanna = (
            -pdf
            * d2
            / iv
        )

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

        if option_type == "CALL":

            charm = (
                -charm_common
                / (
                    spot
                    * iv
                    * sqrt_T
                )
            )

        else:

            charm = (
                charm_common
                / (
                    spot
                    * iv
                    * sqrt_T
                )
            )

        return (
            delta,
            gamma,
            vanna,
            charm,
            iv
        )

    except Exception:

        return (
            0.0,
            0.0,
            0.0,
            0.0,
            safe_iv(iv)
        )


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
    vanna_list = []
    charm_list = []
    iv_list = []

    for _, row in df.iterrows():

        (
            delta,
            gamma,
            vanna,
            charm,
            iv
        ) = calculate_greeks(
            current_price,
            row["strike"],
            row["impliedVolatility"],
            row["DTE"],
            row["option_type"]
        )

        delta_list.append(delta)
        gamma_list.append(gamma)
        vanna_list.append(vanna)
        charm_list.append(charm)
        iv_list.append(iv)

    df["delta"] = delta_list
    df["gamma"] = gamma_list
    df["vanna"] = vanna_list
    df["charm"] = charm_list
    df["IV"] = iv_list

    # --------------------------------------------------------
    # Premium
    # --------------------------------------------------------

    df["premium_flow"] = (
        df["lastPrice"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
    )

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    df["GEX"] = (
        df["gamma"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
        * current_price
    )

    # Put GEX는 음수
    df.loc[
        df["option_type"] == "PUT",
        "GEX"
    ] *= -1

    # --------------------------------------------------------
    # Delta Exposure
    # --------------------------------------------------------

    df["delta_exposure"] = (
        df["delta"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # --------------------------------------------------------
    # Vanna Exposure
    # --------------------------------------------------------

    df["vanna_exposure"] = (
        df["vanna"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # --------------------------------------------------------
    # Charm Exposure
    # --------------------------------------------------------

    df["charm_exposure"] = (
        df["charm"]
        * df["openInterest"]
        * CONTRACT_MULTIPLIER
        * current_price
    )

    # --------------------------------------------------------
    # HIRO Proxy
    # --------------------------------------------------------

    def estimate_trade_side(row):

        try:

            last = float(row["lastPrice"])
            bid = float(row["bid"])
            ask = float(row["ask"])

            if ask > 0 and last >= ask * 0.98:

                return 1

            if bid > 0 and last <= bid * 1.02:

                return -1

            return 0

        except Exception:

            return 0

    df["trade_side"] = df.apply(
        estimate_trade_side,
        axis=1
    )

    df["HIRO_proxy"] = (
        df["delta"]
        * df["volume"]
        * CONTRACT_MULTIPLIER
        * current_price
        * df["trade_side"]
    )

    return df


# ============================================================
# AGGREGATED GREEKS
# ============================================================

def calculate_aggregate_greeks(df):

    result = {}

    result["IV"] = (
        df["IV"]
        .replace(
            [float("inf"), -float("inf")],
            0
        )
        .mean()
    )

    result["Delta"] = (
        df["delta_exposure"]
        .sum()
    )

    result["Gamma"] = (
        df["GEX"]
        .sum()
    )

    result["Vanna"] = (
        df["vanna_exposure"]
        .sum()
    )

    result["Charm"] = (
        df["charm_exposure"]
        .sum()
    )

    result["GEX"] = (
        df["GEX"]
        .sum()
    )

    result["HIRO"] = (
        df["HIRO_proxy"]
        .sum()
    )

    return result


# ============================================================
# CALL WALL / PUT WALL
#
# 핵심 수정:
#
# CALL WALL
# → 현재가보다 위에 있는 Call 중
#   가장 강한 GEX 집중 행사가
#
# PUT WALL
# → 현재가보다 아래에 있는 Put 중
#   가장 강한 하방 GEX 집중 행사가
#
# 따라서 Call Wall과 Put Wall이
# 같은 가격으로 나오는 문제를 방지
# ============================================================

def find_walls(
    df,
    current_price
):

    active = df[
        df["volume"] > 0
    ].copy()

    if active.empty:

        return None, None

    # --------------------------------------------------------
    # ±30% 범위
    # --------------------------------------------------------

    active["distance_pct"] = (
        abs(
            active["strike"]
            - current_price
        )
        / current_price
        * 100
    )

    active = active[
        active["distance_pct"] <= 30
    ].copy()

    if active.empty:

        return None, None

    # ========================================================
    # CALL WALL
    #
    # 현재가 위쪽 Call만 사용
    # ========================================================

    calls = active[
        (
            active["option_type"] == "CALL"
        )
        &
        (
            active["strike"] > current_price
        )
    ].copy()

    call_wall = None

    if not calls.empty:

        call_group = (
            calls
            .groupby("strike")["GEX"]
            .sum()
        )

        if not call_group.empty:

            # GEX가 가장 큰 상방 저항
            call_wall = float(
                call_group.idxmax()
            )

    # ========================================================
    # PUT WALL
    #
    # 현재가 아래쪽 Put만 사용
    # ========================================================

    puts = active[
        (
            active["option_type"] == "PUT"
        )
        &
        (
            active["strike"] < current_price
        )
    ].copy()

    put_wall = None

    if not puts.empty:

        put_group = (
            puts
            .groupby("strike")["GEX"]
            .sum()
        )

        if not put_group.empty:

            # Put GEX는 이미 음수이므로
            # 가장 작은 값이 가장 강한 하방 집중
            put_wall = float(
                put_group.idxmin()
            )

    return (
        call_wall,
        put_wall
    )


# ============================================================
# TOP CALL / PUT FLOW
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
        active["distance_pct"] <= 30
    ].copy()

    calls = active[
        active["option_type"] == "CALL"
    ].sort_values(
        "volume",
        ascending=False
    ).head(5)

    puts = active[
        active["option_type"] == "PUT"
    ].sort_values(
        "volume",
        ascending=False
    ).head(5)

    return (
        calls,
        puts
    )


# ============================================================
# MONEY FORMAT
# ============================================================

def format_money(x):

    x = float(x)

    sign = ""

    if x < 0:

        sign = "-"

        x = abs(x)

    if x >= 1_000_000:

        return (
            f"{sign}${x / 1_000_000:.2f}M"
        )

    if x >= 1_000:

        return (
            f"{sign}${x / 1_000:.1f}K"
        )

    return (
        f"{sign}${x:.0f}"
    )


# ============================================================
# GREEK FORMAT
# ============================================================

def format_greek(x):

    try:

        return f"{float(x):+.3f}"

    except Exception:

        return "0.000"


def format_iv(x):

    try:

        return (
            f"{float(x) * 100:.1f}%"
        )

    except Exception:

        return "0.0%"


# ============================================================
# STRUCTURE JUDGEMENT
# ============================================================

def build_structure_judgement(
    current_price,
    call_wall,
    put_wall,
    greeks
):

    delta = greeks["Delta"]
    gex = greeks["GEX"]
    hiro = greeks["HIRO"]

    # --------------------------------------------------------
    # Delta
    # --------------------------------------------------------

    if delta > 0:

        delta_label = "🟢 BULLISH"

    elif delta < 0:

        delta_label = "🔴 BEARISH"

    else:

        delta_label = "🟡 NEUTRAL"

    # --------------------------------------------------------
    # GEX
    # --------------------------------------------------------

    if gex > 0:

        gex_label = "🟢 POSITIVE"

    elif gex < 0:

        gex_label = "🔴 NEGATIVE"

    else:

        gex_label = "🟡 NEUTRAL"

    # --------------------------------------------------------
    # HIRO
    # --------------------------------------------------------

    if hiro > 0:

        hiro_label = "🟢 POSITIVE"

    elif hiro < 0:

        hiro_label = "🔴 NEGATIVE"

    else:

        hiro_label = "🟡 NEUTRAL"

    # --------------------------------------------------------
    # 종합
    # --------------------------------------------------------

    score = 0

    if delta > 0:
        score += 1

    elif delta < 0:
        score -= 1

    if hiro > 0:
        score += 1

    elif hiro < 0:
        score -= 1

    if gex > 0:
        score += 1

    elif gex < 0:
        score -= 1

    if score >= 2:

        overall = "🟢 BULLISH"

    elif score <= -2:

        overall = "🔴 BEARISH"

    else:

        overall = "🟡 NEUTRAL"

    return {
        "delta_label": delta_label,
        "gex_label": gex_label,
        "hiro_label": hiro_label,
        "overall": overall
    }


# ============================================================
# REPORT
# ============================================================

def build_report(
    ticker,
    current_price,
    df
):

    greeks = calculate_aggregate_greeks(
        df
    )

    (
        call_wall,
        put_wall
    ) = find_walls(
        df,
        current_price
    )

    (
        calls,
        puts
    ) = find_top_flow(
        df,
        current_price
    )

    judgement = build_structure_judgement(
        current_price,
        call_wall,
        put_wall,
        greeks
    )

    today_str = datetime.now().strftime(
        "%Y-%m-%d"
    )

    lines = []

    # ========================================================
    # HEADER
    # ========================================================

    lines.append(
        f"📅 <b>{today_str}</b>"
    )

    lines.append("")

    lines.append(
        "🔥 <b>OPTION SEARCH</b>"
    )

    lines.append("")

    lines.append(
        f"<b>{ticker}</b>"
    )

    lines.append(
        f"현재가: ${current_price:.2f}"
    )

    lines.append("")

    lines.append(
        "📅 DTE 0~180 전체 만기 분석"
    )

    lines.append("")

    # ========================================================
    # OPTION STRUCTURE
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🎯 <b>OPTION STRUCTURE</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    if call_wall is not None:

        lines.append(
            f"📈 CALL WALL "
            f"<b>${call_wall:g}</b>"
        )

    else:

        lines.append(
            "📈 CALL WALL N/A"
        )

    if put_wall is not None:

        lines.append(
            f"📉 PUT WALL "
            f"<b>${put_wall:g}</b>"
        )

    else:

        lines.append(
            "📉 PUT WALL N/A"
        )

    lines.append("")

    # ========================================================
    # GREEKS
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "📊 <b>GREEKS</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    lines.append(
        f"IV       "
        f"{format_iv(greeks['IV'])}"
    )

    lines.append(
        f"Delta    "
        f"{format_money(greeks['Delta'])}"
    )

    lines.append(
        f"Gamma    "
        f"{format_money(greeks['Gamma'])}"
    )

    lines.append(
        f"Vanna    "
        f"{format_money(greeks['Vanna'])}"
    )

    lines.append(
        f"Charm    "
        f"{format_money(greeks['Charm'])}"
    )

    lines.append(
        f"GEX      "
        f"{format_money(greeks['GEX'])}"
    )

    lines.append(
        f"HIRO*    "
        f"{format_money(greeks['HIRO'])}"
    )

    lines.append("")

    # ========================================================
    # TOP CALL
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🟢 <b>TOP CALL FLOW</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    for _, r in calls.iterrows():

        lines.append(
            f"💚 ${r['strike']:g}C | "
            f"DTE {int(r['DTE'])}"
        )

        lines.append(
            f"Vol {int(r['volume']):,} | "
            f"OI {int(r['openInterest']):,}"
        )

        lines.append(
            f"IV {format_iv(r['IV'])} | "
            f"Delta {format_greek(r['delta'])}"
        )

        lines.append(
            f"Gamma {format_greek(r['gamma'])} | "
            f"Vanna {format_greek(r['vanna'])}"
        )

        lines.append(
            f"Charm {format_greek(r['charm'])}"
        )

        lines.append(
            f"GEX {format_money(r['GEX'])} | "
            f"Premium "
            f"{format_money(r['premium_flow'])}"
        )

        lines.append("")

    # ========================================================
    # TOP PUT
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🔴 <b>TOP PUT FLOW</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    for _, r in puts.iterrows():

        lines.append(
            f"❤️ ${r['strike']:g}P | "
            f"DTE {int(r['DTE'])}"
        )

        lines.append(
            f"Vol {int(r['volume']):,} | "
            f"OI {int(r['openInterest']):,}"
        )

        lines.append(
            f"IV {format_iv(r['IV'])} | "
            f"Delta {format_greek(r['delta'])}"
        )

        lines.append(
            f"Gamma {format_greek(r['gamma'])} | "
            f"Vanna {format_greek(r['vanna'])}"
        )

        lines.append(
            f"Charm {format_greek(r['charm'])}"
        )

        lines.append(
            f"GEX {format_money(r['GEX'])} | "
            f"Premium "
            f"{format_money(r['premium_flow'])}"
        )

        lines.append("")

    # ========================================================
    # STRUCTURE
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧭 <b>STRUCTURE</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    lines.append(
        f"📍 현재가: "
        f"<b>${current_price:.2f}</b>"
    )

    if call_wall is not None:

        lines.append(
            f"🟢 상방 핵심: "
            f"<b>${call_wall:g}</b>"
        )

    if put_wall is not None:

        lines.append(
            f"🔴 하방 핵심: "
            f"<b>${put_wall:g}</b>"
        )

    lines.append(
        f"GEX: {judgement['gex_label']}"
    )

    lines.append(
        f"Delta: {judgement['delta_label']}"
    )

    lines.append(
        f"HIRO: {judgement['hiro_label']}"
    )

    lines.append("")

    # ========================================================
    # SUMMARY
    # ========================================================

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append(
        "🧠 <b>나의 정리</b>"
    )

    lines.append(
        "━━━━━━━━━━━━━━━━━━"
    )

    lines.append("")

    if (
        put_wall is not None
        and current_price > put_wall
    ):

        lines.append(
            f"현재가는 Put Wall "
            f"${put_wall:g} 위에 있습니다."
        )

        lines.append(
            f"→ ${put_wall:g} 부근이 "
            f"하방 핵심 구간입니다."
        )

    if (
        call_wall is not None
        and current_price < call_wall
    ):

        lines.append(
            f"${call_wall:g} 부근은 "
            f"상방 핵심 구간입니다."
        )

        lines.append(
            f"→ 해당 가격 돌파 여부가 "
            f"단기 방향성의 핵심입니다."
        )

    gex = greeks["GEX"]

    if gex > 0:

        lines.append(
            "GEX는 양수로 계산되어 "
            "상대적으로 변동성이 억제될 "
            "가능성이 있습니다."
        )

    elif gex < 0:

        lines.append(
            "GEX는 음수로 계산되어 "
            "가격 움직임과 변동성이 "
            "확대될 가능성에 주의해야 합니다."
        )

    if greeks["Delta"] > 0:

        lines.append(
            "Delta Exposure는 "
            "상방 우세로 계산됩니다."
        )

    elif greeks["Delta"] < 0:

        lines.append(
            "Delta Exposure는 "
            "하방 우세로 계산됩니다."
        )

    if greeks["HIRO"] > 0:

        lines.append(
            "HIRO Proxy는 양수로 "
            "상방 거래 흐름이 우세합니다."
        )

    elif greeks["HIRO"] < 0:

        lines.append(
            "HIRO Proxy는 음수로 "
            "하방 거래 흐름이 우세합니다."
        )

    lines.append("")

    lines.append(
        f"📌 종합 판단: "
        f"<b>{judgement['overall']}</b>"
    )

    lines.append("")

    lines.append(
        "⚠️ Greeks/GEX/Wall은 옵션 데이터 기반 "
        "계산값입니다."
    )

    lines.append(
        "⚠️ *HIRO는 실제 체결 방향 데이터가 없는 "
        "yfinance 환경에서 계산한 Proxy입니다."
    )

    lines.append(
        "⚠️ 옵션 거래량만으로 실제 BUY/SELL을 "
        "확정할 수 없습니다."
    )

    return "\n".join(lines)


# ============================================================
# TELEGRAM
# ============================================================

def send_telegram(text):

    if not BOT_TOKEN:

        print("")

        print(
            "❌ TELEGRAM_BOT_TOKEN이 설정되지 않았습니다."
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

        print("")

        print(
            "📨 Telegram 응답:"
        )

        print(
            response.text
        )

        return response.ok

    except Exception as e:

        print(
            f"❌ Telegram 전송 오류: {e}"
        )

        return False


# ============================================================
# RUN
# ============================================================

def run(ticker):

    ticker = ticker.upper().strip()

    print(
        "=" * 70
    )

    print(
        "🔥 OPTION SEARCH"
    )

    print(
        "=" * 70
    )

    print(
        f"🔎 종목: {ticker}"
    )

    print(
        "📌 DTE: 0~180"
    )

    print("")

    current_price = get_current_price(
        ticker
    )

    if current_price is None:

        return False

    print("")

    df = get_option_data(
        ticker
    )

    print("")

    print(
        f"📊 옵션 행수: "
        f"{len(df):,}"
    )

    print(
        f"📅 DTE: "
        f"{df['DTE'].min()} "
        f"~ "
        f"{df['DTE'].max()}"
    )

    print("")

    print(
        "📊 Greeks / GEX 계산 중..."
    )

    df = calculate_option_metrics(
        df,
        current_price
    )

    report = build_report(
        ticker,
        current_price,
        df
    )

    print("")

    print(
        "=" * 70
    )

    print(report)

    print(
        "=" * 70
    )

    telegram_ok = send_telegram(
        report
    )

    # ========================================================
    # CSV
    # ========================================================

    out_dir = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)
        ),
        "..",
        "03_RESULTS",
        "daily"
    )

    out_dir = os.path.abspath(
        out_dir
    )

    os.makedirs(
        out_dir,
        exist_ok=True
    )

    filename = os.path.join(
        out_dir,
        f"{ticker}_OPTION_SEARCH.csv"
    )

    df.to_csv(
        filename,
        index=False,
        encoding="utf-8-sig"
    )

    print("")

    print(
        f"💾 저장 완료: {filename}"
    )

    print(
        "🔥 OPTION SEARCH 완료"
    )

    return telegram_ok


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) < 2:

        print("")

        print(
            "사용법:"
        )

        print(
            "python option_search.py TICKER"
        )

        print("")

        sys.exit()

    run(
        sys.argv[1]
    )
