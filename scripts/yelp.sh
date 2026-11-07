#!/usr/bin/env bash
# Yelp. The review file of the Yelp Open Dataset, read as JSON lines.
# Writes runs/yelp/. Extra flags are passed through, e.g. --quick for a short test.
set -euo pipefail
cd "$(dirname "$0")/.."
"${PYTHON:-python}" run_experiments.py --dataset yelp \
    --csv data/yelp_academic_dataset_review.json --format jsonl \
    --user user_id --item business_id --rating stars --time date --time-format datetime "$@"
