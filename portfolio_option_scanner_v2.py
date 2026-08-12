from __future__ import annotations

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

import batch_option_search as scanner

try:
    from signal_performance import (
        archive_ranking,
        build_performance_message,
    )
except Exception as exc:
    print(
        f"⚠️ signal_performance import failed: {exc}"
    )

    archive_ranking = None
    build_performance_message = None


# ============================================================
# CONFIG
# ============================================================

ET = ZoneInfo(
    "America/New_York"
)

ROOT_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

RESULT_DIR = os.path.join(
    ROOT_DIR,
    "03_RESULTS",
    "daily"
)


# ============================================================
# MARKET DATE
# ============================================================

def market_date_et():

    return datetime.now(
        ET
    ).strftime(
        "%Y%m%d"
    )


# ============================================================
# LOAD AM MARKET CONTEXT
# ============================================================

def load_am_context():

    path = os.path.join(
        RESULT_DIR,
        (
            "AM_MARKET_CONTEXT_"
            f"{market_date_et()}.json"
        )
    )

    if not os.path.exists(path):

        print(
            f"⚠️ AM Context not found: {path}"
        )

        return {
            "score": 50.0,
            "direction": "NEUTRAL",
            "regime": "🟡 NEUTRAL",
            "reasons": [],
            "available": False,
        }


    try:

        with open(
            path,
            "r",
            encoding="utf-8"
        ) as f:

            payload = json.load(f)


        context = payload.get(
            "market_context",
            {}
        )


        return {

            "score":
                float(
                    context.get(
                        "score",
                        50.0
                    )
                ),

            "direction":
                str(
                    context.get(
                        "direction",
                        "NEUTRAL"
                    )
                ),

            "regime":
                str(
                    context.get(
                        "regime",
                        "🟡 NEUTRAL"
                    )
                ),

            "reasons":
                context.get(
                    "reasons",
                    []
                ),

            "available":
                True,

        }


    except Exception as exc:

        print(
            f"⚠️ AM Context read failed: {exc}"
        )

        return {

            "score":
                50.0,

            "direction":
                "NEUTRAL",

            "regime":
                "🟡 NEUTRAL",

            "reasons":
                [],

            "available":
                False,

        }


# ============================================================
# AM CONTEXT
# ============================================================

AM_CONTEXT = load_am_context()


# ============================================================
# AM SCORE ADJUSTMENT
# ============================================================

def calculate_am_adjustment(
    option_direction
):

    market_direction = (
        AM_CONTEXT.get(
            "direction",
            "NEUTRAL"
        )
    )

    market_score = float(
        AM_CONTEXT.get(
            "score",
            50.0
        )
    )


    if (
        not AM_CONTEXT.get(
            "available",
            False
        )
        or
        market_direction == "NEUTRAL"
    ):

        return (
            0.0,
            "AM 중립 → 조정 없음"
        )


    # --------------------------------------------------------
    # MARKET STRENGTH
    #
    # AM Score 50 = 0
    # 최대 ±5
    # --------------------------------------------------------

    score_component = max(
        -5.0,
        min(
            5.0,
            (
                market_score
                - 50.0
            )
            * 0.20
        )
    )


    # --------------------------------------------------------
    # DIRECTION ALIGNMENT
    #
    # 일치 +5
    # 중립 0
    # 충돌 -5
    # --------------------------------------------------------

    if (
        option_direction
        == market_direction
    ):

        direction_component = 5.0

        reason = (
            f"AM {market_direction}"
            " ↔ 옵션 방향 일치"
        )


    elif (
        option_direction
        == "NEUTRAL"
    ):

        direction_component = 0.0

        reason = (
            f"AM {market_direction}"
            " ↔ 옵션 방향 중립"
        )


    else:

        direction_component = -5.0

        reason = (
            f"AM {market_direction}"
            " ↔ 옵션 방향 충돌"
        )


    adjustment = (
        score_component
        + direction_component
    )


    adjustment = max(
        -10.0,
        min(
            10.0,
            adjustment
        )
    )


    return (
        round(
            adjustment,
            2
        ),
        reason
    )


# ============================================================
# PATCH EXISTING SCORE
# ============================================================

_original_calculate_score = (
    scanner.calculate_score
)


def calculate_score_with_am(
    analysis
):

    result = (
        _original_calculate_score(
            analysis
        )
    )


    option_direction = (
        result.get(
            "direction",
            "NEUTRAL"
        )
    )


    adjustment, reason = (
        calculate_am_adjustment(
            option_direction
        )
    )


    # --------------------------------------------------------
    # ORIGINAL SCORE
    # --------------------------------------------------------

    result[
        "base_score"
    ] = result[
        "score"
    ]


    # --------------------------------------------------------
    # AM DATA
    # --------------------------------------------------------

    result[
        "am_market_score"
    ] = AM_CONTEXT[
        "score"
    ]

    result[
        "am_direction"
    ] = AM_CONTEXT[
        "direction"
    ]

    result[
        "am_regime"
    ] = AM_CONTEXT[
        "regime"
    ]

    result[
        "am_score_adjustment"
    ] = adjustment

    result[
        "am_alignment"
    ] = reason


    # --------------------------------------------------------
    # APPLY AM
    # --------------------------------------------------------

    result[
        "score"
    ] = max(
        0.0,
        min(
            100.0,
            result[
                "score"
            ]
            + adjustment
        )
    )


    # --------------------------------------------------------
    # REASONS
    # --------------------------------------------------------

    if adjustment > 0:

        result[
            "reasons"
        ].append(
            (
                f"AM Market 가점 "
                f"+{adjustment:.1f}"
            )
        )


    elif adjustment < 0:

        result[
            "reasons"
        ].append(
            (
                f"AM Market 감점 "
                f"{adjustment:.1f}"
            )
        )


    else:

        result[
            "reasons"
        ].append(
            "AM Market 조정 0.0"
        )


    # --------------------------------------------------------
    # FINAL CATEGORY RECHECK
    # --------------------------------------------------------

    quality_score = float(
        analysis
        .get(
            "quality",
            {}
        )
        .get(
            "score",
            0
        )
        or 0
    )


    if (
        result["score"]
        >= scanner.ENTRY_SCORE

        and
        result["direction"]
        == "BULLISH"

        and
        result["iv_pct"]
        < 150

        and
        quality_score
        >= 40
    ):

        result[
            "category"
        ] = (
            "🟢 오늘 진입 후보"
        )


    elif (
        result["score"]
        <= 35

        or
        result["direction"]
        == "BEARISH"
    ):

        result[
            "category"
        ] = "🔴 회피"


    else:

        result[
            "category"
        ] = "🟡 관망"


    return result


# ============================================================
# ACTIVATE PATCH
# ============================================================

scanner.calculate_score = (
    calculate_score_with_am
)


# ============================================================
# FINAL RESULT PATCH
# ============================================================

_original_make_final_result = (
    scanner.make_final_result
)


def make_final_result_with_am(
    analysis
):

    result = (
        _original_make_final_result(
            analysis
        )
    )


    base = (
        _original_calculate_score(
            analysis
        )
    )


    result[
        "base_score"
    ] = float(
        base.get(
            "score",
            result["score"]
        )
    )


    result[
        "am_market_score"
    ] = AM_CONTEXT[
        "score"
    ]

    result[
        "am_direction"
    ] = AM_CONTEXT[
        "direction"
    ]

    result[
        "am_regime"
    ] = AM_CONTEXT[
        "regime"
    ]


    result[
        "am_score_adjustment"
    ] = (
        float(
            result["score"]
        )
        -
        float(
            result["base_score"]
        )
    )


    result[
        "am_alignment"
    ] = (
        f"AM "
        f"{AM_CONTEXT['direction']}"
        f" ↔ 옵션 "
        f"{result['direction']}"
    )


    return result


scanner.make_final_result = (
    make_final_result_with_am
)


# ============================================================
# FINAL CSV PATCH
# ============================================================

_original_save_ranking = (
    scanner.save_ranking
)


def save_ranking_with_am(
    results
):

    _original_save_ranking(
        results
    )


    path = (
        scanner.RANKING_FILE
    )


    try:

        df = pd.read_csv(
            path
        )


        if (
            "am_score_adjustment"
            not in df.columns
        ):

            df[
                "am_score_adjustment"
            ] = 0.0


        df[
            "am_market_score"
        ] = AM_CONTEXT[
            "score"
        ]


        df[
            "am_direction"
        ] = AM_CONTEXT[
            "direction"
        ]


        df[
            "am_regime"
        ] = AM_CONTEXT[
            "regime"
        ]


        df[
            "am_alignment"
        ] = df[
            "direction"
        ].apply(

            lambda x:
                (
                    f"AM "
                    f"{AM_CONTEXT['direction']}"
                    f" ↔ 옵션 {x}"
                )

        )


        df.to_csv(
            path,
            index=False,
            encoding="utf-8-sig"
        )


        print(
            "✅ AM Context "
            "→ FINAL RANKING 저장 완료"
        )


    except Exception as exc:

        print(
            f"⚠️ AM CSV 저장 실패: {exc}"
        )


scanner.save_ranking = (
    save_ranking_with_am
)


# ============================================================
# TELEGRAM PATCH
# ============================================================

_original_send_telegram = (
    scanner.send_telegram
)


def send_telegram_with_performance(
    message
):

    performance = ""


    try:

        if (
            archive_ranking
            is not None
            and
            os.path.exists(
                scanner.RANKING_FILE
            )
        ):

            archive_ranking(
                scanner.RANKING_FILE
            )


        if (
            build_performance_message
            is not None
        ):

            performance = (
                build_performance_message(
                    limit=30
                )
            )


    except Exception as exc:

        print(
            "⚠️ Performance summary "
            f"실패: {exc}"
        )


    # --------------------------------------------------------
    # AM HEADER
    # --------------------------------------------------------

    header = (

        "━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "🌅 <b>AM MARKET FILTER</b>\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━\n"

        f"시장: "
        f"<b>{AM_CONTEXT['regime']}</b>\n"

        f"Market Score: "
        f"<b>{AM_CONTEXT['score']:.1f}</b>\n"

        f"Direction: "
        f"<b>{AM_CONTEXT['direction']}</b>\n"

        "→ AM Market Context가 "
        "FINAL SCORE에 반영되었습니다.\n\n"

    )


    enhanced = (
        header
        + message
    )


    if performance:

        enhanced += (
            "\n\n"
            + performance
        )


    return (
        _original_send_telegram(
            enhanced
        )
    )


scanner.send_telegram = (
    send_telegram_with_performance
)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )

    print(
        "🚀 PORTFOLIO OPTION SCANNER V2"
    )

    print(
        "🇺🇸 AM MARKET CONTEXT CONNECTED"
    )

    print(
        f"Market date ET: "
        f"{market_date_et()}"
    )

    print(
        f"AM direction: "
        f"{AM_CONTEXT['direction']}"
    )

    print(
        f"AM score: "
        f"{AM_CONTEXT['score']:.1f}"
    )

    print(
        "━━━━━━━━━━━━━━━━━━━━━━━━"
    )


    scanner.main()
