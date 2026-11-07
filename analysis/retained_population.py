"""Table 2: the population kept by the random split and the train-only five-core.

The users, items, training pairs and held-out positives are read from the E1 line of each
console log. The popularity term Lambda_R under uniform candidates is read from
results/aggregated/all_e2_crosscert.csv. Then

    D      = log(users * items / (training + held-out))
    share  = (Lambda_R - (items - 1) / (2 * held-out)) / D

where (items - 1) / (2 * held-out) is the sampling excess of the plug-in Lambda_R.

    python analysis/retained_population.py   # writes results/derived/retained_population.csv
"""
import argparse, csv, glob, math, os, re

DATASETS = ["amazon_books", "hotelrec", "yelp", "google_connecticut", "goodreads_all",
            "netflix", "ml20m"]
ap = argparse.ArgumentParser()
ap.add_argument("--results", default="results")
A = ap.parse_args()

LR = {}
with open(os.path.join(A.results, "aggregated", "all_e2_crosscert.csv"), newline="") as fh:
    for r in csv.DictReader(fh):
        if r["split"] == "random" and r["train_law"] == "[baseline]" and r["cert_ref"] == "uniform":
            LR[r["dataset"]] = float(r["Lambda_R"])

pat = re.compile(r"=== E1 \[random\]\s+train ([\d,]+) test ([\d,]+) users\(N\)=([\d,]+) "
                 r"items\(M\)=([\d,]+)")
rows = []
for d in DATASETS:
    log = sorted(glob.glob(os.path.join(A.results, d, "console_*.log")))[-1]
    tr, te, nu, m = (int(x.replace(",", "")) for x in pat.search(open(log).read()).groups())
    D = math.log(nu * m / (tr + te))
    share = (LR[d] - (m - 1) / (2 * te)) / D
    rows.append(dict(dataset=d, users=nu, items=m, training=tr, held_out=te,
                     D=round(D, 4), Lambda_R=round(LR[d], 4), share=round(share, 4)))
    print(f"{d:20s} users {nu:>10,} items {m:>8,} training {tr:>11,} held-out {te:>10,}"
          f"  D {D:6.2f}  share {share:.3f}")
os.makedirs(os.path.join(A.results, "derived"), exist_ok=True)
with open(os.path.join(A.results, "derived", "retained_population.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader(); w.writerows(rows)
