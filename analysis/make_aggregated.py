"""Fold the per-dataset CSVs into one file per stage.

    python analysis/make_aggregated.py                  # results/ -> results/aggregated/
    python analysis/make_aggregated.py --results runs   # your own runs -> runs/aggregated/

Each output row is the dataset tag followed by the row of results/<dataset>/<stage>.csv.
"""
import argparse, os

STAGES = ["e0_diagnostics", "e1_arms", "e2_crosscert", "e3_itemonly", "e3b_capacity",
          "e4_invariance", "e7_zbar"]
DATASETS = ["amazon_books", "goodreads_all", "google_connecticut", "hotelrec", "ml20m",
            "netflix", "yelp"]

ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
A = ap.parse_args()
out = os.path.join(A.results, "aggregated")
os.makedirs(out, exist_ok=True)
for st in STAGES:
    lines = []
    for d in DATASETS:
        p = os.path.join(A.results, d, st + ".csv")
        if not os.path.exists(p):
            continue
        rows = open(p, encoding="utf-8").read().splitlines()
        if not lines:
            lines.append("dataset," + rows[0])
        lines += [f"{d},{r}" for r in rows[1:] if r]
    if lines:
        open(os.path.join(out, f"all_{st}.csv"), "w", encoding="utf-8").write("\n".join(lines) + "\n")
        print(f"all_{st}.csv  {len(lines) - 1} rows")
