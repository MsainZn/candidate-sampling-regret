#!/usr/bin/env bash
# HotelRec. data/hotelrec.csv is made by preprocess/hotelrec.py.
# Writes runs/hotelrec/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset hotelrec \
    --csv data/hotelrec.csv --user user_id --item item_id "$@"
