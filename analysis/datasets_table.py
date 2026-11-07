"""Table 1: the seven datasets before any split.

Records are the interactions in the file that run_experiments.py reads, before the rating cut,
as counted when each file was made (for Google:CT, the lines of the published file, of which
27,166 have no user or business id and are dropped by preprocess/google_local.py). Users, items,
pairs, D and the popularity share come from results/aggregated/all_e0_diagnostics.csv.

    python analysis/datasets_table.py   # writes results/derived/datasets.csv
"""
import csv, os

RECORDS = {
    "amazon_books": 9_488_297,         # Books.csv, Amazon Reviews 2023 5-core rating-only
    "hotelrec": 49_047_625,            # data/hotelrec.csv (placeholder authors dropped)
    "yelp": 6_990_280,                 # yelp_academic_dataset_review.json
    "google_connecticut": 5_181_800,   # lines of review-Connecticut.json.gz
    "goodreads_all": 79_224_894,       # data/goodreads_all.csv (merged genres, rated only)
    "netflix": 20_997_882,             # data/netflix.csv (100,000-user sample)
    "ml20m": 20_000_263,               # ml-20m/ratings.csv
}
E0 = {}
with open("results/aggregated/all_e0_diagnostics.csv", newline="") as fh:
    for r in csv.DictReader(fh):
        E0.setdefault(r["dataset"], {})[r["quantity"]] = float(r["value"])
rows = []
for d, rec in sorted(RECORDS.items(), key=lambda kv: E0[kv[0]]["frac_Lambda_V"]):
    e = E0[d]
    rows.append(dict(dataset=d, records=rec, users=int(e["users"]), items=int(e["items"]),
                     pairs=int(e["interactions"]), kept_pct=round(100 * e["interactions"] / rec, 1),
                     pairs_per_user=round(e["pairs_per_user"], 1), D=round(e["D"], 2),
                     popularity_share=round(e["frac_Lambda_V"], 3)))
    print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
os.makedirs("results/derived", exist_ok=True)
with open("results/derived/datasets.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
