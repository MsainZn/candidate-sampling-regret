#!/usr/bin/env bash
# Google:CT. data/google_connecticut.csv is made by preprocess/google_local.py.
# Writes runs/google_connecticut/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset google_connecticut \
    --csv data/google_connecticut.csv --user user_id --item item_id "$@"
