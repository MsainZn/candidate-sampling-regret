#!/usr/bin/env bash
# MovieLens-20M. ratings.csv from the ml-20m.zip release, used as published.
# Writes runs/ml20m/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset ml20m \
    --csv data/ml-20m/ratings.csv "$@"
