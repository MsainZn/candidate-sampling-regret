#!/usr/bin/env bash
# Amazon:Books. Books.csv is the 5-core rating-only file of Amazon Reviews 2023, used as published.
# Writes runs/amazon_books/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset amazon_books \
    --csv data/Books.csv --user user_id --item parent_asin "$@"
