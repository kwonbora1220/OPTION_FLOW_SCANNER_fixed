name: RKLB - Option Structure $80-$100

on:
  workflow_dispatch:

permissions:
  contents: write

jobs:
  option-structure:
    runs-on: ubuntu-latest

    steps:

      - name: Checkout repository
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install pandas numpy yfinance

      - name: Run RKLB Option Structure Scanner
        env:
          SYMBOL: RKLB
          MIN_STRIKE: "80"
          MAX_STRIKE: "100"
          MAX_DTE: "180"
        run: |
          set -e
          python rklb_option_structure.py

      - name: Upload option structure results
        uses: actions/upload-artifact@v4
        with:
          name: rklb-option-structure
          path: rklb_option_structure/
          if-no-files-found: warn
