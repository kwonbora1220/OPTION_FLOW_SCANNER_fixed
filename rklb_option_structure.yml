name: RKLB - Option Structure $80-$100

on:
  workflow_dispatch:
    inputs:
      symbol:
        description: "Ticker"
        required: true
        default: "RKLB"

      min_strike:
        description: "Minimum strike"
        required: true
        default: "80"

      max_strike:
        description: "Maximum strike"
        required: true
        default: "100"

      max_dte:
        description: "Maximum DTE"
        required: true
        default: "180"

  schedule:
    - cron: "30 22 * * 1-5"

permissions:
  contents: read

concurrency:
  group: rklb-option-structure
  cancel-in-progress: false

jobs:

  option-structure:

    runs-on: ubuntu-latest

    steps:

      # ============================================================
      # 1. CHECKOUT
      # ============================================================

      - name: Checkout repository
        uses: actions/checkout@v4
        with:
          fetch-depth: 1


      # ============================================================
      # 2. PYTHON
      # ============================================================

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.12"


      # ============================================================
      # 3. DEPENDENCIES
      # ============================================================

      - name: Install dependencies
        run: |

          set -e

          python -m pip install --upgrade pip

          if [ -f requirements.txt ]; then
            pip install -r requirements.txt
          fi

          pip install yfinance pandas numpy


      # ============================================================
      # 4. CONFIG
      # ============================================================

      - name: Set configuration
        shell: bash
        env:
          INPUT_SYMBOL: ${{ github.event.inputs.symbol }}
          INPUT_MIN_STRIKE: ${{ github.event.inputs.min_strike }}
          INPUT_MAX_STRIKE: ${{ github.event.inputs.max_strike }}
          INPUT_MAX_DTE: ${{ github.event.inputs.max_dte }}
        run: |

          set -e

          SYMBOL="${INPUT_SYMBOL:-RKLB}"
          MIN_STRIKE="${INPUT_MIN_STRIKE:-80}"
          MAX_STRIKE="${INPUT_MAX_STRIKE:-100}"
          MAX_DTE="${INPUT_MAX_DTE:-180}"

          echo "SYMBOL=$SYMBOL" >> "$GITHUB_ENV"
          echo "MIN_STRIKE=$MIN_STRIKE" >> "$GITHUB_ENV"
          echo "MAX_STRIKE=$MAX_STRIKE" >> "$GITHUB_ENV"
          echo "MAX_DTE=$MAX_DTE" >> "$GITHUB_ENV"

          echo "=========================================="
          echo "RKLB OPTION STRUCTURE"
          echo "=========================================="
          echo "SYMBOL       : $SYMBOL"
          echo "STRIKE RANGE : $MIN_STRIKE ~ $MAX_STRIKE"
          echo "MAX DTE      : $MAX_DTE"
          echo "=========================================="


      # ============================================================
      # 5. CREATE ANALYZER
      # ============================================================

      - name: Create option structure analyzer
        shell: bash
        run: |

          set -e

          cat > rklb_option_structure.py <<'PY'

          from __future__ import annotations

          import os
          import math
          from datetime import datetime, timezone

          import numpy as np
          import pandas as pd
          import yfinance as yf


          # ============================================================
          # CONFIG
          # ============================================================

          SYMBOL = os.getenv(
              "SYMBOL",
              "RKLB",
          ).upper()

          MIN_STRIKE = float(
              os.getenv(
                  "MIN_STRIKE",
                  "80",
              )
          )

          MAX_STRIKE = float(
              os.getenv(
                  "MAX_STRIKE",
                  "100",
              )
          )

          MAX_DTE = int(
              os.getenv(
                  "MAX_DTE",
                  "180",
              )
          )

          OUTPUT_DIR = "rklb_option_structure"

          os.makedirs(
              OUTPUT_DIR,
              exist_ok=True,
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
                  errors="coerce",
              )


          def fmt_money(value):

              value = safe_float(value)

              if not np.isfinite(value):
                  return "N/A"

              value = abs(value)

              if value >= 1_000_000_000:
                  return f"${value / 1_000_000_000:.2f}B"

              if value >= 1_000_000:
                  return f"${value / 1_000_000:.2f}M"

              if value >= 1_000:
                  return f"${value / 1_000:.1f}K"

              return f"${value:,.0f}"


          def fmt_num(value):

              value = safe_float(value)

              if not np.isfinite(value):
                  return "N/A"

              if abs(value) >= 1000:
                  return f"{value:,.0f}"

              if abs(value) >= 100:
                  return f"{value:.0f}"

              if abs(value) >= 10:
                  return f"{value:.1f}"

              return f"{value:.2f}"


          def fmt_pct(value):

              value = safe_float(value)

              if not np.isfinite(value):
                  return "N/A"

              return f"{value:.1f}%"


          # ============================================================
          # GEX PROXY
          #
          # Same model used by STEP 8:
          #
          # gamma
          # × OI
          # × 100
          # × spot²
          # × 0.01
          #
          # CALL = positive
          # PUT  = negative
          #
          # IMPORTANT:
          # This is NOT dealer-supplied GEX.
          # It is a modelled proxy.
          # ============================================================

          def calculate_gex(
              gamma,
              open_interest,
              spot,
              option_type,
          ):

              gamma = safe_float(gamma)
              open_interest = safe_float(
                  open_interest
              )
              spot = safe_float(spot)

              if not all(
                  np.isfinite(x)
                  for x in [
                      gamma,
                      open_interest,
                      spot,
                  ]
              ):
                  return np.nan

              if (
                  gamma <= 0
                  or open_interest <= 0
                  or spot <= 0
              ):
                  return 0.0

              gex = (
                  gamma
                  * open_interest
                  * 100.0
                  * (spot ** 2)
                  * 0.01
              )

              if option_type == "PUT":
                  gex *= -1.0

              return gex


          # ============================================================
          # PREMIUM PROXY
          # ============================================================

          def calculate_premium(
              volume,
              bid,
              ask,
              last_price,
          ):

              volume = safe_float(volume)
              bid = safe_float(bid)
              ask = safe_float(ask)
              last_price = safe_float(last_price)

              if not np.isfinite(volume):
                  return 0.0

              if volume <= 0:
                  return 0.0

              if (
                  np.isfinite(bid)
                  and np.isfinite(ask)
                  and bid >= 0
                  and ask >= bid
                  and ask > 0
              ):

                  mid = (
                      bid + ask
                  ) / 2.0

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
                  * 100.0
              )


          # ============================================================
          # FETCH OPTION DATA
          # ============================================================

          def fetch_options():

              print()
              print("=" * 72)
              print("FETCH YAHOO FINANCE OPTION DATA")
              print("=" * 72)

              ticker = yf.Ticker(
                  SYMBOL
              )

              try:

                  spot = safe_float(
                      ticker.history(
                          period="1d",
                          interval="1m",
                          prepost=True,
                      )["Close"].dropna().iloc[-1]
                  )

              except Exception:

                  try:

                      spot = safe_float(
                          ticker.history(
                              period="5d"
                          )["Close"].dropna().iloc[-1]
                      )

                  except Exception:

                      spot = np.nan

              print(
                  f"CURRENT PRICE : {spot}"
              )

              try:

                  expirations = list(
                      ticker.options
                  )

              except Exception as exc:

                  raise RuntimeError(
                      f"Unable to obtain option expirations: {exc}"
                  )

              print(
                  f"TOTAL EXPIRATIONS : {len(expirations)}"
              )

              if not expirations:
                  raise RuntimeError(
                      "Yahoo Finance returned no option expirations."
                  )

              rows = []

              today = pd.Timestamp.utcnow().normalize()

              for expiration in expirations:

                  try:

                      expiry_date = pd.Timestamp(
                          expiration
                      )

                      dte = int(
                          (
                              expiry_date
                              - today
                          ).days
                      )

                  except Exception:

                      continue

                  if dte < 0:
                      continue

                  if dte > MAX_DTE:
                      continue

                  print(
                      f"FETCH {expiration} | DTE {dte}"
                  )

                  try:

                      chain = ticker.option_chain(
                          expiration
                      )

                  except Exception as exc:

                      print(
                          f"SKIP {expiration}: {exc}"
                      )

                      continue

                  for option_type, frame in [
                      ("CALL", chain.calls),
                      ("PUT", chain.puts),
                  ]:

                      if frame is None:
                          continue

                      if frame.empty:
                          continue

                      frame = frame.copy()

                      frame["option_type"] = (
                          option_type
                      )

                      frame["expiration"] = (
                          expiration
                      )

                      frame["DTE"] = (
                          dte
                      )

                      rows.append(
                          frame
                      )

              if not rows:

                  raise RuntimeError(
                      "No option rows were collected."
                  )

              data = pd.concat(
                  rows,
                  ignore_index=True,
              )

              return data, spot


          # ============================================================
          # NORMALIZE
          # ============================================================

          def normalize(
              data,
              spot,
          ):

              print()
              print("=" * 72)
              print("NORMALIZE OPTION DATA")
              print("=" * 72)

              required = [
                  "strike",
                  "volume",
                  "openInterest",
                  "bid",
                  "ask",
                  "lastPrice",
                  "impliedVolatility",
                  "gamma",
                  "delta",
                  "vega",
                  "option_type",
                  "expiration",
                  "DTE",
              ]

              missing = [
                  column
                  for column in required
                  if column not in data.columns
              ]

              if missing:

                  raise RuntimeError(
                      "Missing Yahoo columns: "
                      + ", ".join(missing)
                  )

              for column in [
                  "strike",
                  "volume",
                  "openInterest",
                  "bid",
                  "ask",
                  "lastPrice",
                  "impliedVolatility",
                  "gamma",
                  "delta",
                  "vega",
                  "DTE",
              ]:

                  data[column] = numeric(
                      data[column]
                  )

              data["option_type"] = (
                  data["option_type"]
                  .astype(str)
                  .str.upper()
                  .str.strip()
              )

              data["expiration"] = (
                  data["expiration"]
                  .astype(str)
              )

              data = data[
                  data["strike"].notna()
                  &
                  data["option_type"].isin(
                      [
                          "CALL",
                          "PUT",
                      ]
                  )
                  &
                  data["DTE"].notna()
                  &
                  (
                      data["DTE"] >= 0
                  )
                  &
                  (
                      data["DTE"] <= MAX_DTE
                  )
              ].copy()

              # --------------------------------------------------------
              # STRIKE FILTER
              # --------------------------------------------------------

              data = data[
                  (
                      data["strike"]
                      >= MIN_STRIKE
                  )
                  &
                  (
                      data["strike"]
                      <= MAX_STRIKE
                  )
              ].copy()

              # --------------------------------------------------------
              # GEX
              # --------------------------------------------------------

              data["gex"] = data.apply(
                  lambda row:
                      calculate_gex(
                          row["gamma"],
                          row["openInterest"],
                          spot,
                          row["option_type"],
                      ),
                  axis=1,
              )

              # --------------------------------------------------------
              # PREMIUM
              # --------------------------------------------------------

              data["premium_proxy"] = data.apply(
                  lambda row:
                      calculate_premium(
                          row["volume"],
                          row["bid"],
                          row["ask"],
                          row["lastPrice"],
                      ),
                  axis=1,
              )

              # --------------------------------------------------------
              # VOLUME/OI
              # --------------------------------------------------------

              data["volume_oi"] = np.where(
                  data["openInterest"] > 0,
                  data["volume"]
                  /
                  data["openInterest"],
                  np.nan,
              )

              # --------------------------------------------------------
              # DISTANCE FROM SPOT
              # --------------------------------------------------------

              if (
                  np.isfinite(spot)
                  and spot > 0
              ):

                  data["distance_pct"] = (
                      (
                          data["strike"]
                          -
                          spot
                      )
                      /
                      spot
                      *
                      100.0
                  )

              else:

                  data["distance_pct"] = np.nan

              return data


          # ============================================================
          # STRIKE AGGREGATION
          # ============================================================

          def build_strike_table(
              data
          ):

              print()
              print("=" * 72)
              print("BUILD STRIKE STRUCTURE")
              print("=" * 72)

              grouped = []

              for strike in sorted(
                  data["strike"].dropna().unique()
              ):

                  strike_df = data[
                      data["strike"]
                      == strike
                  ]

                  calls = strike_df[
                      strike_df[
                          "option_type"
                      ]
                      == "CALL"
                  ]

                  puts = strike_df[
                      strike_df[
                          "option_type"
                      ]
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

                  grouped.append(
                      {
                          "strike": strike,

                          "call_volume":
                              call_volume,

                          "put_volume":
                              put_volume,

                          "total_volume":
                              call_volume
                              + put_volume,

                          "call_oi":
                              call_oi,

                          "put_oi":
                              put_oi,

                          "total_oi":
                              call_oi
                              + put_oi,

                          "call_gex":
                              call_gex,

                          "put_gex":
                              put_gex,

                          "net_gex":
                              call_gex
                              + put_gex,

                          "call_premium":
                              call_premium,

                          "put_premium":
                              put_premium,

                          "total_premium":
                              call_premium
                              + put_premium,
                      }
                  )

              result = pd.DataFrame(
                  grouped
              )

              if result.empty:
                  raise RuntimeError(
                      "No strike structure available."
                  )

              return result


          # ============================================================
          # EXPIRATION STRUCTURE
          # ============================================================

          def build_expiration_table(
              data
          ):

              grouped = []

              for (
                  expiration,
                  dte,
              ), frame in data.groupby(
                  [
                      "expiration",
                      "DTE",
                  ]
              ):

                  calls = frame[
                      frame[
                          "option_type"
                      ]
                      == "CALL"
                  ]

                  puts = frame[
                      frame[
                          "option_type"
                      ]
                      == "PUT"
                  ]

                  grouped.append(
                      {
                          "expiration":
                              expiration,

                          "DTE":
                              dte,

                          "call_volume":
                              calls[
                                  "volume"
                              ]
                              .fillna(0)
                              .sum(),

                          "put_volume":
                              puts[
                                  "volume"
                              ]
                              .fillna(0)
                              .sum(),

                          "call_oi":
                              calls[
                                  "openInterest"
                              ]
                              .fillna(0)
                              .sum(),

                          "put_oi":
                              puts[
                                  "openInterest"
                              ]
                              .fillna(0)
                              .sum(),

                          "call_gex":
                              calls[
                                  "gex"
                              ]
                              .fillna(0)
                              .sum(),

                          "put_gex":
                              puts[
                                  "gex"
                              ]
                              .fillna(0)
                              .sum(),
                      }
                  )

              return pd.DataFrame(
                  grouped
              ).sort_values(
                  [
                      "DTE",
                      "expiration",
                  ]
              )


          # ============================================================
          # TOP CONTRACTS
          # ============================================================

          def build_top_contracts(
              data
          ):

              result = data.copy()

              result["importance"] = (
                  np.log1p(
                      result[
                          "premium_proxy"
                      ].clip(
                          lower=0
                      )
                  )
                  +
                  np.log1p(
                      result[
                          "volume"
                      ].clip(
                          lower=0
                      )
                  )
                  +
                  np.log1p(
                      result[
                          "openInterest"
                      ].clip(
                          lower=0
                      )
                  )
                  +
                  np.log1p(
                      result[
                          "gex"
                      ].abs().clip(
                          lower=0
                      )
                  )
              )

              return (
                  result
                  .sort_values(
                      "importance",
                      ascending=False,
                  )
                  .head(50)
              )


          # ============================================================
          # WALL DETECTION
          # ============================================================

          def find_wall(
              strike_table,
              spot,
              option_type,
          ):

              if not np.isfinite(spot):
                  return None

              if option_type == "CALL":

                  candidates = strike_table[
                      strike_table[
                          "strike"
                      ] >= spot
                  ].copy()

                  if candidates.empty:
                      return None

                  candidates["score"] = (
                      np.log1p(
                          candidates[
                              "call_oi"
                          ].clip(
                              lower=0
                          )
                      )
                      +
                      np.log1p(
                          candidates[
                              "call_gex"
                          ].abs().clip(
                              lower=0
                          )
                      )
                      +
                      0.25
                      *
                      np.log1p(
                          candidates[
                              "call_volume"
                          ].clip(
                              lower=0
                          )
                      )
                  )

              else:

                  candidates = strike_table[
                      strike_table[
                          "strike"
                      ] <= spot
                  ].copy()

                  if candidates.empty:
                      return None

                  candidates["score"] = (
                      np.log1p(
                          candidates[
                              "put_oi"
                          ].clip(
                              lower=0
                          )
                      )
                      +
                      np.log1p(
                          candidates[
                              "put_gex"
                          ].abs().clip(
                              lower=0
                          )
                      )
                      +
                      0.25
                      *
                      np.log1p(
                          candidates[
                              "put_volume"
                          ].clip(
                              lower=0
                          )
                      )
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

              candidates["selection"] = (
                  candidates["score"]
                  +
                  3.0
                  /
                  (
                      1.0
                      +
                      candidates["distance"]
                      * 20.0
                  )
              )

              best = (
                  candidates
                  .sort_values(
                      [
                          "selection",
                          "score",
                      ],
                      ascending=False,
                  )
                  .iloc[0]
              )

              return best


          # ============================================================
          # MAIN
          # ============================================================

          def main():

              started = datetime.now(
                  timezone.utc
              )

              print()
              print("=" * 72)
              print("🔥 RKLB OPTION STRUCTURE SCANNER")
              print("=" * 72)
              print(
                  f"SYMBOL       : {SYMBOL}"
              )
              print(
                  f"STRIKE RANGE : ${MIN_STRIKE:g} ~ ${MAX_STRIKE:g}"
              )
              print(
                  f"DTE RANGE    : 0 ~ {MAX_DTE}"
              )
              print("=" * 72)

              # --------------------------------------------------------
              # FETCH
              # --------------------------------------------------------

              raw, spot = fetch_options()

              print()
              print(
                  f"RAW OPTION ROWS : {len(raw):,}"
              )

              if not np.isfinite(spot):

                  raise RuntimeError(
                      "Unable to determine current underlying price."
                  )

              # --------------------------------------------------------
              # NORMALIZE
              # --------------------------------------------------------

              data = normalize(
                  raw,
                  spot,
              )

              print(
                  f"FILTERED ROWS   : {len(data):,}"
              )

              if data.empty:

                  raise RuntimeError(
                      "No options remain after strike/DTE filtering."
                  )

              # --------------------------------------------------------
              # STRIKE TABLE
              # --------------------------------------------------------

              strike_table = (
                  build_strike_table(
                      data
                  )
              )

              # --------------------------------------------------------
              # EXPIRATION TABLE
              # --------------------------------------------------------

              expiration_table = (
                  build_expiration_table(
                      data
                  )
              )

              # --------------------------------------------------------
              # TOP CONTRACTS
              # --------------------------------------------------------

              top_contracts = (
                  build_top_contracts(
                      data
                  )
              )

              # --------------------------------------------------------
              # WALLS
              # --------------------------------------------------------

              call_wall = find_wall(
                  strike_table,
                  spot,
                  "CALL",
              )

              put_wall = find_wall(
                  strike_table,
                  spot,
                  "PUT",
              )

              # --------------------------------------------------------
              # NET GEX
              # --------------------------------------------------------

              total_call_gex = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "CALL"
                  ]["gex"]
                  .fillna(0)
                  .sum()
              )

              total_put_gex = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "PUT"
                  ]["gex"]
                  .fillna(0)
                  .sum()
              )

              net_gex = (
                  total_call_gex
                  +
                  total_put_gex
              )

              # --------------------------------------------------------
              # TOTAL OI
              # --------------------------------------------------------

              total_call_oi = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "CALL"
                  ]["openInterest"]
                  .fillna(0)
                  .sum()
              )

              total_put_oi = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "PUT"
                  ]["openInterest"]
                  .fillna(0)
                  .sum()
              )

              # --------------------------------------------------------
              # TOTAL VOLUME
              # --------------------------------------------------------

              total_call_volume = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "CALL"
                  ]["volume"]
                  .fillna(0)
                  .sum()
              )

              total_put_volume = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "PUT"
                  ]["volume"]
                  .fillna(0)
                  .sum()
              )

              # --------------------------------------------------------
              # TOTAL PREMIUM
              # --------------------------------------------------------

              total_call_premium = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "CALL"
                  ]["premium_proxy"]
                  .fillna(0)
                  .sum()
              )

              total_put_premium = (
                  data[
                      data[
                          "option_type"
                      ]
                      == "PUT"
                  ]["premium_proxy"]
                  .fillna(0)
                  .sum()
              )

              # --------------------------------------------------------
              # ATM IV
              # --------------------------------------------------------

              data["atm_distance"] = (
                  (
                      data["strike"]
                      -
                      spot
                  ).abs()
              )

              atm_rows = (
                  data
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
              # CALL / PUT RATIOS
              # --------------------------------------------------------

              total_volume = (
                  total_call_volume
                  +
                  total_put_volume
              )

              total_oi = (
                  total_call_oi
                  +
                  total_put_oi
              )

              total_premium = (
                  total_call_premium
                  +
                  total_put_premium
              )

              call_volume_ratio = (
                  total_call_volume
                  /
                  total_volume
                  *
                  100.0
                  if total_volume > 0
                  else np.nan
              )

              call_oi_ratio = (
                  total_call_oi
                  /
                  total_oi
                  *
                  100.0
                  if total_oi > 0
                  else np.nan
              )

              call_premium_ratio = (
                  total_call_premium
                  /
                  total_premium
                  *
                  100.0
                  if total_premium > 0
                  else np.nan
              )

              # --------------------------------------------------------
              # SAVE CSV
              # --------------------------------------------------------

              raw_output = data.copy()

              raw_output.to_csv(
                  os.path.join(
                      OUTPUT_DIR,
                      "contracts.csv",
                  ),
                  index=False,
              )

              strike_table.to_csv(
                  os.path.join(
                      OUTPUT_DIR,
                      "strike_structure.csv",
                  ),
                  index=False,
              )

              expiration_table.to_csv(
                  os.path.join(
                      OUTPUT_DIR,
                      "expiration_structure.csv",
                  ),
                  index=False,
              )

              top_contracts.to_csv(
                  os.path.join(
                      OUTPUT_DIR,
                      "top_contracts.csv",
                  ),
                  index=False,
              )

              # --------------------------------------------------------
              # BUILD MARKDOWN REPORT
              # --------------------------------------------------------

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
                  f"🎯 분석 범위: ${MIN_STRIKE:g} ~ ${MAX_STRIKE:g}"
              )

              report.append(
                  f"📅 DTE: 0 ~ {MAX_DTE}"
              )

              report.append(
                  f"📊 옵션 행수: {len(data):,}"
              )

              report.append("")

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
                  f"CALL Volume: {total_call_volume:,.0f}"
              )

              report.append(
                  f"PUT Volume: {total_put_volume:,.0f}"
              )

              report.append(
                  f"CALL Volume Ratio: {fmt_pct(call_volume_ratio)}"
              )

              report.append(
                  f"CALL OI: {total_call_oi:,.0f}"
              )

              report.append(
                  f"PUT OI: {total_put_oi:,.0f}"
              )

              report.append(
                  f"CALL OI Ratio: {fmt_pct(call_oi_ratio)}"
              )

              report.append(
                  f"CALL Premium Proxy: {fmt_money(total_call_premium)}"
              )

              report.append(
                  f"PUT Premium Proxy: {fmt_money(total_put_premium)}"
              )

              report.append(
                  f"CALL Premium Ratio: {fmt_pct(call_premium_ratio)}"
              )

              report.append("")

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

                  cw_strike = call_wall[
                      "strike"
                  ]

                  report.append(
                      f"📈 Call Wall: ${cw_strike:g}"
                      f" | OI {cw_strike and call_wall['call_oi']:,.0f}"
                      f" | GEX {fmt_money(call_wall['call_gex'])}"
                  )

              else:

                  report.append(
                      "📈 Call Wall: N/A"
                  )

              if put_wall is not None:

                  pw_strike = put_wall[
                      "strike"
                  ]

                  report.append(
                      f"📉 Put Wall: ${pw_strike:g}"
                      f" | OI {pw_strike and put_wall['put_oi']:,.0f}"
                      f" | GEX {fmt_money(put_wall['put_gex'])}"
                  )

              else:

                  report.append(
                      "📉 Put Wall: N/A"
                  )

              report.append("")

              report.append(
                  f"CALL GEX: {fmt_money(total_call_gex)}"
              )

              report.append(
                  f"PUT GEX: {fmt_money(total_put_gex)}"
              )

              report.append(
                  f"NET GEX: {fmt_money(net_gex)}"
              )

              report.append(
                  f"ATM IV: {fmt_pct(atm_iv * 100 if np.isfinite(atm_iv) and atm_iv < 2 else atm_iv)}"
              )

              report.append("")

              # --------------------------------------------------------
              # STRIKE STRUCTURE
              # --------------------------------------------------------

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "🎯 3. $80~$100 STRIKE STRUCTURE"
              )

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "STRIKE | CALL VOL | PUT VOL | CALL OI | PUT OI | NET GEX"
              )

              report.append(
                  "────────────────────────────────────────"
              )

              for _, row in strike_table.iterrows():

                  strike = row["strike"]

                  report.append(
                      f"${strike:g} | "
                      f"{row['call_volume']:,.0f} | "
                      f"{row['put_volume']:,.0f} | "
                      f"{row['call_oi']:,.0f} | "
                      f"{row['put_oi']:,.0f} | "
                      f"{fmt_money(row['net_gex'])}"
                  )

              report.append("")

              # --------------------------------------------------------
              # IMPORTANT STRIKES
              # --------------------------------------------------------

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "🔥 4. HIGH OI / GEX STRIKES"
              )

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              high_oi = (
                  strike_table
                  .assign(
                      total_oi_rank=
                          strike_table[
                              "total_oi"
                          ]
                  )
                  .sort_values(
                      "total_oi",
                      ascending=False,
                  )
                  .head(10)
              )

              for _, row in high_oi.iterrows():

                  report.append(
                      f"${row['strike']:g}"
                      f" | OI {row['total_oi']:,.0f}"
                      f" | C {row['call_oi']:,.0f}"
                      f" / P {row['put_oi']:,.0f}"
                      f" | GEX {fmt_money(row['net_gex'])}"
                  )

              report.append("")

              # --------------------------------------------------------
              # TOP CONTRACTS
              # --------------------------------------------------------

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "🔥 5. TOP OPTION CONTRACTS"
              )

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              for _, row in top_contracts.head(20).iterrows():

                  report.append(
                      f"{row['option_type']:4s} "
                      f"${row['strike']:g} "
                      f"| DTE {int(row['DTE'])}"
                      f" | Vol {row['volume']:,.0f}"
                      f" | OI {row['openInterest']:,.0f}"
                      f" | Premium {fmt_money(row['premium_proxy'])}"
                      f" | GEX {fmt_money(row['gex'])}"
                  )

              report.append("")

              # --------------------------------------------------------
              # EXPIRATION
              # --------------------------------------------------------

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "📅 6. EXPIRATION STRUCTURE"
              )

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "DTE | EXPIRATION | CALL VOL | PUT VOL | CALL OI | PUT OI"
              )

              report.append(
                  "────────────────────────────────────────"
              )

              for _, row in expiration_table.iterrows():

                  report.append(
                      f"{int(row['DTE']):3d} | "
                      f"{row['expiration']} | "
                      f"{row['call_volume']:,.0f} | "
                      f"{row['put_volume']:,.0f} | "
                      f"{row['call_oi']:,.0f} | "
                      f"{row['put_oi']:,.0f}"
                  )

              report.append("")

              # --------------------------------------------------------
              # INTERPRETATION
              # --------------------------------------------------------

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              report.append(
                  "🧠 7. STRUCTURE INTERPRETATION"
              )

              report.append(
                  "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
              )

              if (
                  call_wall is not None
                  and put_wall is not None
              ):

                  cw = call_wall[
                      "strike"
                  ]

                  pw = put_wall[
                      "strike"
                  ]

                  if (
                      spot >= cw
                  ):

                      report.append(
                          f"🟢 현재가가 Call Wall ${cw:g} 위"
                      )

                  elif (
                      spot <= pw
                  ):

                      report.append(
                          f"🔴 현재가가 Put Wall ${pw:g} 아래"
                      )

                  else:

                      report.append(
                          f"🟡 현재가는 ${pw:g} ~ ${cw:g} 사이"
                      )

              if net_gex > 0:

                  report.append(
                      "📈 Net GEX: POSITIVE"
                  )

                  report.append(
                      "   → 모델상 가격 안정화 성격"
                  )

              elif net_gex < 0:

                  report.append(
                      "📉 Net GEX: NEGATIVE"
                  )

                  report.append(
                      "   → 모델상 가격 변동성 확대 가능성"
                  )

              if (
                  total_call_volume
                  >
                  total_put_volume
              ):

                  report.append(
                      "🟢 거래량: CALL 우세"
                  )

              elif (
                  total_put_volume
                  >
                  total_call_volume
              ):

                  report.append(
                      "🔴 거래량: PUT 우세"
                  )

              if (
                  total_call_oi
                  >
                  total_put_oi
              ):

                  report.append(
                      "🟢 OI: CALL 우세"
                  )

              else:

                  report.append(
                      "🔴 OI: PUT 우세"
                  )

              report.append("")

              # --------------------------------------------------------
              # LIMITATIONS
              # --------------------------------------------------------

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
                  "• Yahoo Finance 무료 옵션 데이터 기반"
              )

              report.append(
                  "• Premium은 실제 체결금액이 아닌 Proxy"
              )

              report.append(
                  "• 실제 Buy/Sell 체결 방향은 확인 불가"
              )

              report.append(
                  "• OI만으로 Long/Short 포지션 확정 불가"
              )

              report.append(
                  "• GEX는 Dealer Positioning의 모델링 Proxy"
              )

              report.append(
                  "• Call/Put GEX 부호는 모델 가정에 기반"
              )

              report.append("")

              report.append(
                  f"Generated: {started.strftime('%Y-%m-%d %H:%M:%S UTC')}"
              )

              report_text = "\n".join(
                  report
              )

              # --------------------------------------------------------
              # SAVE REPORT
              # --------------------------------------------------------

              with open(
                  os.path.join(
                      OUTPUT_DIR,
                      "report.md",
                  ),
                  "w",
                  encoding="utf-8",
              ) as file:

                  file.write(
                      report_text
                  )

              # --------------------------------------------------------
              # SAVE SUMMARY
              # --------------------------------------------------------

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

                          "max_dte":
                              MAX_DTE,

                          "option_rows":
                              len(data),

                          "call_volume":
                              total_call_volume,

                          "put_volume":
                              total_put_volume,

                          "call_oi":
                              total_call_oi,

                          "put_oi":
                              total_put_oi,

                          "call_gex":
                              total_call_gex,

                          "put_gex":
                              total_put_gex,

                          "net_gex":
                              net_gex,

                          "call_premium":
                              total_call_premium,

                          "put_premium":
                              total_put_premium,

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
                              ),

                          "atm_iv":
                              atm_iv,
                      }
                  ]
              )

              summary.to_csv(
                  os.path.join(
                      OUTPUT_DIR,
                      "summary.csv",
                  ),
                  index=False,
              )

              # --------------------------------------------------------
              # PRINT REPORT
              # --------------------------------------------------------

              print()
              print(report_text)

              print()
              print(
                  "=" * 72
              )

              print(
                  "FILES CREATED"
              )

              print(
                  "=" * 72
              )

              for filename in sorted(
                  os.listdir(
                      OUTPUT_DIR
                  )
              ):

                  path = os.path.join(
                      OUTPUT_DIR,
                      filename,
                  )

                  print(
                      f"- {path}"
                  )

              print()
              print(
                  "OPTION STRUCTURE SCAN : COMPLETE"
              )


          if __name__ == "__main__":

              main()

          PY


      # ============================================================
      # 6. RUN ANALYZER
      # ============================================================

      - name: Run RKLB option structure scan
        run: |

          set -e

          python rklb_option_structure.py


      # ============================================================
      # 7. VALIDATE OUTPUT
      # ============================================================

      - name: Validate option structure output
        run: |

          set -e

          echo "=========================================="
          echo "OUTPUT VALIDATION"
          echo "=========================================="

          test -f rklb_option_structure/report.md
          test -f rklb_option_structure/summary.csv
          test -f rklb_option_structure/contracts.csv
          test -f rklb_option_structure/strike_structure.csv
          test -f rklb_option_structure/expiration_structure.csv
          test -f rklb_option_structure/top_contracts.csv

          python - <<'PY'

          import pandas as pd

          summary = pd.read_csv(
              "rklb_option_structure/summary.csv"
          )

          strikes = pd.read_csv(
              "rklb_option_structure/strike_structure.csv"
          )

          contracts = pd.read_csv(
              "rklb_option_structure/contracts.csv"
          )

          print()
          print("SUMMARY")
          print(summary.to_string(index=False))

          print()
          print(
              "STRIKE ROWS:",
              len(strikes)
          )

          print(
              "CONTRACT ROWS:",
              len(contracts)
          )

          if len(strikes) == 0:
              raise RuntimeError(
                  "Strike structure is empty."
              )

          if len(contracts) == 0:
              raise RuntimeError(
                  "Contract data is empty."
              )

          print()
          print("VALIDATION : OK")

          PY


      # ============================================================
      # 8. DISPLAY REPORT
      # ============================================================

      - name: Display final report
        run: |

          set -e

          cat rklb_option_structure/report.md


      # ============================================================
      # 9. UPLOAD ARTIFACT
      # ============================================================

      - name: Upload option structure artifact
        uses: actions/upload-artifact@v4
        with:
          name: ${{ env.SYMBOL }}-option-structure
          path: rklb_option_structure/
          retention-days: 7


      # ============================================================
      # 10. TELEGRAM
      #
      # This does NOT replace STEP 12.
      #
      # It sends the complete report generated by this workflow.
      # ============================================================

      - name: Send option structure to Telegram
        if: ${{ success() }}
        env:
          TELEGRAM_BOT_TOKEN: ${{ secrets.TELEGRAM_BOT_TOKEN }}
          TELEGRAM_CHAT_ID: ${{ secrets.TELEGRAM_CHAT_ID }}
        run: |

          set -e

          if [ -z "$TELEGRAM_BOT_TOKEN" ]; then
            echo "TELEGRAM_BOT_TOKEN is not configured."
            echo "Skipping Telegram."
            exit 0
          fi

          if [ -z "$TELEGRAM_CHAT_ID" ]; then
            echo "TELEGRAM_CHAT_ID is not configured."
            echo "Skipping Telegram."
            exit 0
          fi

          python - <<'PY'

          import os
          import urllib.parse
          import urllib.request


          token = os.environ[
              "TELEGRAM_BOT_TOKEN"
          ]

          chat_id = os.environ[
              "TELEGRAM_CHAT_ID"
          ]

          file = (
              "rklb_option_structure/"
              "report.md"
          )

          with open(
              file,
              "r",
              encoding="utf-8",
          ) as f:

              text = f.read()

          # Telegram sendMessage limit is roughly
          # 4096 characters.
          #
          # Split into safe chunks.

          chunks = []

          max_length = 3900

          while len(text) > max_length:

              split_at = text.rfind(
                  "\n",
                  0,
                  max_length,
              )

              if split_at <= 0:
                  split_at = max_length

              chunks.append(
                  text[:split_at]
              )

              text = text[
                  split_at:
              ]

          if text:
              chunks.append(text)

          url = (
              f"https://api.telegram.org/"
              f"bot{token}/sendMessage"
          )

          for chunk in chunks:

              payload = urllib.parse.urlencode(
                  {
                      "chat_id": chat_id,
                      "text": chunk,
                  }
              ).encode(
                  "utf-8"
              )

              request = urllib.request.Request(
                  url,
                  data=payload,
                  method="POST",
              )

              with urllib.request.urlopen(
                  request,
                  timeout=30,
              ) as response:

                  result = response.read().decode(
                      "utf-8"
                  )

                  print(
                      "Telegram response:",
                      result[:500],
                  )

          print(
              f"Telegram sent: {len(chunks)} chunks"
          )

          PY
