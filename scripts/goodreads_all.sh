#!/usr/bin/env bash
# Goodreads. data/goodreads_all.csv is made by preprocess/goodreads.py.
# Writes runs/goodreads_all/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset goodreads_all \
    --csv data/goodreads_all.csv --user user_id --item item_id "$@"
