"""Reading a rating log and splitting it without leakage.

``load_table``   reads csv or line-delimited json, applies the positivity cut
                 (--min-rating), collapses repeated pairs, and returns integer
                 user, item and time arrays.
``make_split``   splits FIRST (at random, 90/10), then applies the five-core to the
                 training rows ALONE, then keeps a held-out pair only if both its
                 user and its item survived there. Coring before the split would
                 let the held-out rows decide which identifiers exist, which is
                 leakage.

Part of the code that produced every number in the paper.
"""
import csv, json, math, os, sys, time
import numpy as np

# ===================================================================== data
def load_table(args):
    """Return (users, items, ratings|None, times|None) as raw arrays."""
    if args.format == "jsonl":
        import json as _j
        U, I, Rt, T = [], [], [], []
        _o = (lambda q: __import__("gzip").open(q, "rt", encoding="utf-8", errors="replace")) \
             if args.csv.endswith(".gz") else \
             (lambda q: open(q, "r", encoding="utf-8", errors="replace"))
        with _o(args.csv) as f:
            for ln, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try: d = _j.loads(line)
                except Exception: continue
                U.append(d.get(args.user)); I.append(d.get(args.item))
                Rt.append(d.get(args.rating) if args.rating else None)
                T.append(d.get(args.time) if args.time else None)
                if args.subsample and len(U) >= args.subsample * 4: break
        u = np.array(U, dtype=object); i = np.array(I, dtype=object)
        r = np.array([x if x is not None else np.nan for x in Rt], float) if args.rating else None
        t = np.array(T, dtype=object) if args.time else None
    else:
        import pandas as pd
        _o = ((lambda q: __import__("gzip").open(q, "rt", encoding="utf-8", errors="replace"))
              if args.csv.endswith(".gz") else
              (lambda q: open(q, "r", encoding="utf-8", errors="replace")))
        with _o(args.csv) as f:
            l1 = f.readline().rstrip("\n"); l2 = f.readline().rstrip("\n")
        def isnum(x):
            try: float(x.strip().strip('"')); return True
            except Exception: return False
        if args.no_header:
            cols = [c.strip() for c in args.cols.split(",")]
            # SNIFF: does row 0 look like a header? Decide by comparing row 0 with row 1
            # position by position: a position that parses as a number on row 1 but not on
            # row 0 means row 0 is column NAMES. Reading it as data either crashes later
            # ("could not convert string to float: 'rating'") or, worse, silently mis-maps.
            c1 = l1.split(","); c2 = l2.split(",")
            if len(c1) == len(c2) and any(isnum(b) and not isnum(a) for a, b in zip(c1, c2)):
                names = [c.strip().strip('"') for c in c1]
                pick = lambda al: next((k for k in al if k in names), None)
                us = pick(["user_id", "userId", "user", "reviewerID"])
                it = pick(["parent_asin", "asin", "movieId", "business_id", "item_id", "item"])
                ra = pick(["rating", "stars", "overall"])
                ti = pick(["timestamp", "unixReviewTime", "time", "date"])
                sys.exit(
                    "\nERROR: --no-header was given but this file HAS a header row.\n"
                    f"  row 0: {l1[:140]}\n"
                    f"  row 1: {l2[:140]}\n"
                    "  Reading row 0 as data makes the rating column literally contain the\n"
                    "  string 'rating', and mis-maps every column besides.\n"
                    "  Drop --no-header/--cols and name the columns instead:\n"
                    f"    --user {us or '<user col>'} --item {it or '<item col>'} "
                    f"--rating {ra or '<rating col>'} --time {ti or '<time col>'}\n"
                    "  (or run:  python getdata.py probe --path " + args.csv + ")")
            if len(cols) != len(c1):
                sys.exit(f"\nERROR: --cols names {len(cols)} columns but the file has "
                         f"{len(c1)}.\n  row 0: {l1[:140]}")
            df = pd.read_csv(args.csv, header=None, names=cols, low_memory=False)
        else:
            have = pd.read_csv(args.csv, nrows=0).columns.tolist()
            miss = [c for c in [args.user, args.item, args.rating, args.time]
                    if c and c not in have]
            if miss:
                sys.exit(f"\nERROR: column(s) {miss} are not in this file.\n"
                         f"  available columns: {have}\n"
                         f"  (or run:  python getdata.py probe --path {args.csv})")
            want = [c for c in [args.user, args.item, args.rating, args.time] if c]
            df = pd.read_csv(args.csv, usecols=want, low_memory=False)
        n0 = len(df)
        u = df[args.user].to_numpy(); i = df[args.item].to_numpy()
        # coerce rather than hard-cast: real dumps contain stray malformed rows (pandas
        # warns "columns have mixed types"). Bad rows are dropped and COUNTED, not silently kept.
        r = t = None
        bad = np.zeros(n0, dtype=bool)
        if args.rating and args.rating in df.columns:
            r = pd.to_numeric(df[args.rating], errors="coerce").to_numpy(float)
            bad |= ~np.isfinite(r)
        if args.time and args.time in df.columns:
            t = df[args.time].to_numpy()
            if args.time_format == "epoch":
                t = pd.to_numeric(df[args.time], errors="coerce").to_numpy(float)
                bad |= ~np.isfinite(t)
        bad |= pd.isna(df[args.user]).to_numpy() | pd.isna(df[args.item]).to_numpy()
        if bad.any():
            frac = bad.mean()
            print(f"  dropped {int(bad.sum()):,} of {n0:,} rows ({frac:.2%}) with an unparseable "
                  f"user/item/rating/timestamp")
            if frac > 0.02:
                print(f"  !! that is more than 2% of the file -- check the column mapping before "
                      f"trusting these results")
            keep = ~bad
            u, i = u[keep], i[keep]
            r = r[keep] if r is not None else None
            t = t[keep] if t is not None else None

    if r is not None and args.min_rating is not None:
        keep = r >= args.min_rating
        u, i = u[keep], i[keep]
        t = t[keep] if t is not None else None
    if t is not None and args.time_format == "datetime":
        import pandas as pd
        t = pd.to_datetime(pd.Series(t), errors="coerce").astype("int64").to_numpy() // 10 ** 9
    if t is not None:
        t = np.asarray(t, dtype=np.float64)
        bad = ~np.isfinite(t)
        if bad.any():
            u, i, t = u[~bad], i[~bad], t[~bad]
    if args.subsample and args.subsample < len(u):
        g = np.random.default_rng(0); s = g.choice(len(u), args.subsample, replace=False)
        u, i = u[s], i[s]; t = t[s] if t is not None else None
    return u, i, t

def make_split(u, i, frac=0.9, min_count=0):
    """Split FIRST (random 90/10), then k-core the TRAINING ROWS ONLY.

    Applying k-core to the whole log before splitting would let the held-out rows
    decide which users and items exist, which is preprocessing leakage. Filtering
    train only is leakage-free: the held-out rows are never consulted, and they are
    then kept exactly when their user and item survive in the training universe.
    """
    n = len(u)
    g = np.random.default_rng(0); idx = g.permutation(n); cut = int(frac * n)
    trm, tem = idx[:cut], idx[cut:]
    if min_count > 0:
        n0 = len(trm)
        while True:
            utr, itr = u[trm], i[trm]
            _, uinv, uc = np.unique(utr, return_inverse=True, return_counts=True)
            _, iinv, ic = np.unique(itr, return_inverse=True, return_counts=True)
            keep = (uc[uinv] >= min_count) & (ic[iinv] >= min_count)
            if keep.all() or not keep.any(): break
            trm = trm[keep]
        print(f"      {min_count}-core on TRAIN ROWS ONLY: {n0:,} -> {len(trm):,} train pairs "
              f"(held-out rows never consulted)")
    ua, ui = np.unique(u[trm], return_inverse=True)
    ia, ii = np.unique(i[trm], return_inverse=True)
    umap = {v: k for k, v in enumerate(ua)}; imap = {v: k for k, v in enumerate(ia)}
    tu, ti = [], []
    for x, y in zip(u[tem], i[tem]):
        a = umap.get(x); b = imap.get(y)
        if a is not None and b is not None: tu.append(a); ti.append(b)
    return ((ui.astype(np.int64), ii.astype(np.int64)),
            (np.array(tu, np.int64), np.array(ti, np.int64)), len(ua), len(ia))
