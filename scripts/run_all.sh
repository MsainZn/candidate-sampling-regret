#!/usr/bin/env bash
# All seven datasets, the quickest first. Each one writes runs/<dataset>/.
set -euo pipefail
cd "$(dirname "$0")"
for d in google_connecticut yelp ml20m netflix amazon_books goodreads_all hotelrec; do
    ./"$d".sh "$@"
done
