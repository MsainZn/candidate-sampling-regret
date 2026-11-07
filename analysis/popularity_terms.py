"""Table 9: popularity terms under uniform candidates and the adjusted score of the
uniform-trained model, in percent of log K.

  surrogate   the `honest` column, the score that the item-only scoring function
              log(P_tr / R) reaches with K = 1024 candidates (Monte Carlo, 200,000 positives)
  Lambda_R    the popularity term of the held-out positives, from item counts
  main        four-seed mean of L_hat - Lambda_R for the uniform-trained model
  larger      the same for the larger model (dimension 128, twelve epochs, free per-item scalar)

    python analysis/popularity_terms.py   # writes results/derived/popularity_terms.csv
"""
import csv, math, os, statistics as st

P = 100 / math.log(1024)
DS = ["amazon_books", "hotelrec", "yelp", "google_connecticut", "goodreads_all", "netflix", "ml20m"]
rd = lambda f: list(csv.DictReader(open(os.path.join("results/aggregated", f), newline="")))
e2, e3b = rd("all_e2_crosscert.csv"), rd("all_e3b_capacity.csv")
rows = []
for d in DS:
    b = [r for r in e2 if r["dataset"] == d and r["train_law"] == "[baseline]" and r["cert_ref"] == "uniform"][0]
    LR = float(b["Lambda_R"])
    main = st.mean(float(r["L_hat"]) for r in e2 if r["dataset"] == d and r["train_law"] == "uniform"
                   and r["cert_ref"] == "uniform") - LR
    big = float([r for r in e3b if r["dataset"] == d and r["train_law"] == "uniform"
                 and r["cert_ref"] == "uniform"][0]["L_hat"]) - LR
    rows.append(dict(dataset=d, surrogate=round(P * float(b["honest"]), 1), Lambda_R=round(P * LR, 1),
                     adjusted_main=round(P * main, 1), adjusted_larger=round(P * big, 1)))
    print("  ".join(f"{k}={v}" for k, v in rows[-1].items()))
os.makedirs("results/derived", exist_ok=True)
with open("results/derived/popularity_terms.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
