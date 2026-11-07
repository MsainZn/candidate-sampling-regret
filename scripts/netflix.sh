#!/usr/bin/env bash
# Netflix Prize. data/netflix.csv is made by preprocess/netflix.py.
# Writes runs/netflix/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset netflix \
    --csv data/netflix.csv --user user_id --item item_id "$@"
