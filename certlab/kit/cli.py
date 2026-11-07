"""Command line: one invocation runs one dataset.

``main`` parses the flags, reads the log, creates the output directory, runs the
requested stages in order and writes ``meta.json`` beside the CSVs. The defaults are the
settings of the paper; ``scripts/`` holds the invocation for each of the seven datasets.

Part of the code that produced every number in the paper.
"""
import argparse, json, math, os, sys, time, datetime
import numpy as np

from .data import load_table
from .report import Sink, summarise, Tee
from .selftest import selftest
from .stages import stage_e0, stage_e1_e2_e3, stage_e3b, stage_e4_e7

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--dataset", help="short tag used in every output path, e.g. ml20m")
    ap.add_argument("--csv")
    ap.add_argument("--format", default="csv", choices=["csv", "jsonl"])
    ap.add_argument("--no-header", action="store_true")
    ap.add_argument("--cols", default="item,user,rating,timestamp",
                    help="column order for --no-header csv")
    ap.add_argument("--user", default="userId"); ap.add_argument("--item", default="movieId")
    ap.add_argument("--rating", default="rating"); ap.add_argument("--time", default="timestamp")
    ap.add_argument("--time-format", default="epoch", choices=["epoch", "datetime"])
    ap.add_argument("--min-rating", type=float, default=4.0,
                    help='threshold on the rating column; pass --rating "" to disable')
    ap.add_argument("--subsample", type=int, default=0)
    ap.add_argument("--min-count", type=int, default=5,
                    help="iterative k-core applied to the TRAINING ROWS ONLY, after the split, "
                         "so the held-out rows never influence preprocessing. The paper uses 5 "
                         "on every dataset; 0 = off.")
    ap.add_argument("--dup-policy", default="first", choices=["first", "last"],
                    help="which occurrence of a repeated (user,item) pair to keep. "
                         "first = when the pair entered the log (default, the natural clock "
                         "for a set-membership target); last = most recent touch.")
    ap.add_argument("--keep-duplicates", action="store_true",
                    help="do NOT collapse repeat (user,item) rows. Off by default: the estimand "
                         "is set membership, so p counts distinct pairs.")
    ap.add_argument("--stages", default="e0,e1,e2,e3,e3b,e4,e7",
                    help="stages to run; e3b is the larger-model check (it retrains at "
                         "cap-dim and cap-epochs)")
    ap.add_argument("--cap-dim", type=int, default=128); ap.add_argument("--cap-epochs", type=int, default=12)
    ap.add_argument("--e3b-cert-n", type=int, help="override cert-n for e3b")
    ap.add_argument("--cap-item-bias", type=float, default=5.0)
    ap.add_argument("--seeds-e1", type=int, default=4); ap.add_argument("--seeds-e4", type=int, default=2)
    ap.add_argument("--dim", type=int, default=64); ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--K", type=int, default=1024); ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--epochs", type=int, default=6); ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--cert-K", type=int, default=1024); ap.add_argument("--cert-n", type=int, default=110000)
    ap.add_argument("--e1-cert-n", type=int, help="override cert-n for E1")
    ap.add_argument("--e2-cert-n", type=int, help="override cert-n for E2")
    ap.add_argument("--calib-n", type=int, default=30000); ap.add_argument("--cert-chunk", type=int, default=1024)
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25")
    ap.add_argument("--preregister-alpha", type=float, default=1.0,
                    help="the ONE reference whose 95% LCB is quotable without a multiplicity "
                         "correction. Fixed before the run. 1.0 = training popularity.")
    ap.add_argument("--out", default="runs",
                    help="output folder (the published results are in results/)")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run: 500k rows, 2 epochs, 1 seed, 3 alphas")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.selftest: sys.exit(selftest())
    if not (args.dataset and args.csv): sys.exit("need --dataset and --csv (or --selftest)")
    if args.quick:
        args.subsample = args.subsample or 500000; args.epochs = 2
        args.seeds_e1 = 1; args.seeds_e4 = 1; args.alphas = "0,0.5,1.0"
        args.cert_n = 20000; args.calib_n = 8000
    args.e1_cert_n = args.e1_cert_n or args.cert_n
    args.e2_cert_n = args.e2_cert_n or args.cert_n
    if args.rating == "": args.rating = None; args.min_rating = None
    if args.time == "": args.time = None
    if args.no_header:
        cols = [c.strip() for c in args.cols.split(",")]
        def pick(cur, aliases):
            if cur and cur in cols: return cur
            for a in aliases:
                if a in cols: return a
            return None
        args.user = pick(args.user, ["user", "userId", "user_id", "reviewerID"])
        args.item = pick(args.item, ["item", "movieId", "item_id", "asin", "business_id"])
        args.rating = pick(args.rating, ["rating", "stars", "overall"]) if args.rating else None
        args.time = pick(args.time, ["timestamp", "time", "date", "unixReviewTime"]) if args.time else None
        miss = [n for n, v in [("user", args.user), ("item", args.item)] if v is None]
        if miss:
            sys.exit(f"--no-header: --cols {cols} has no column for {miss}. "
                     f"Rename them in --cols, or pass --user/--item to name them explicitly.")
        if args.rating is None: args.min_rating = None
        print(f"  --no-header column map: user={args.user} item={args.item} "
              f"rating={args.rating} time={args.time}")
    stages = set(s.strip() for s in args.stages.split(","))

    root = os.path.join(args.out, args.dataset); ckdir = os.path.join(root, "ckpt")
    os.makedirs(ckdir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    sys.stdout = Tee(os.path.join(root, f"console_{stamp}.log"))

    needs_torch = bool({"e1", "e2", "e3", "e3b", "e4", "e7"} & stages)
    torch = None; dev = "cpu"
    if needs_torch:
        import torch
        dev = ("mps" if torch.backends.mps.is_available()
               else "cuda" if torch.cuda.is_available() else "cpu")
        print(f"runall.py  dataset={args.dataset}  device={dev}  torch {torch.__version__}")
    else:
        print(f"runall.py  dataset={args.dataset}  (e0 only -- torch not needed)")
    print(f"  stages={sorted(stages)}  splits=random  seeds e1={args.seeds_e1} e4={args.seeds_e4}")
    print(f"  dim={args.dim} tau={args.tau} K={args.K} epochs={args.epochs} "
          f"cert_K={args.cert_K} cert_n={args.cert_n} calib_n={args.calib_n}")
    t0 = time.time()

    print("\nloading ...", flush=True)
    u, i, t = load_table(args)
    n_raw = len(u)
    if not args.keep_duplicates:
        # Y = 1{(u,v) in log} is set membership, so p must count DISTINCT pairs.
        # Keeping duplicates makes p = |log|/(N*M) exceed the distinct-pair rate and
        # drives I_+(U;V) = D - Lambda_U - Lambda_V negative.
        # WHICH occurrence to keep: FIRST (default) keeps a pair as it entered the log,
        # LAST its most recent repeat. Neither leaks: the discarded occurrence is
        # deleted, so the model never sees the pair it is tested on. The timestamps are
        # used only here, to order the log before repeated pairs are collapsed.
        order = np.argsort(t, kind="stable") if t is not None else np.arange(n_raw)
        uu, ii = u[order], i[order]
        key = np.array([hash((a, b)) for a, b in zip(uu, ii)]) if uu.dtype == object else \
              (np.asarray(uu, np.int64).astype(np.float64) * (np.asarray(ii, np.int64).max() + 1)
               + np.asarray(ii, np.int64))
        if args.dup_policy == "first":
            _, first = np.unique(key, return_index=True)     # key is time-sorted ascending
            sel = np.sort(first)
        else:
            _, last = np.unique(key[::-1], return_index=True)
            sel = np.sort(len(key) - 1 - last)
        u, i = uu[sel], ii[sel]
        t = t[order][sel] if t is not None else None
    n_after_dedup = len(u)
    print("  ESTIMAND: a held-out pair is kept only if BOTH its user and its item appear in\n"
          "    the training universe, so the target is logged membership AMONG USERS AND ITEMS\n"
          "    OBSERVED IN TRAINING (warm-start), not over the full catalogue. p and D are\n"
          "    computed inside that universe.")
    print(f"  {n_raw:,} rows after filtering -> {len(u):,} distinct (user,item) pairs"
          f"  ({n_raw-len(u):,} duplicates collapsed, policy={args.dup_policy})")
    args.full_users = int(len(np.unique(u))); args.full_items = int(len(np.unique(i)))
    args.n_pairs = int(len(u))
    print(f"  full log universe: users {args.full_users:,}  items {args.full_items:,}  "
          f"pairs {args.n_pairs:,}  ({args.n_pairs/args.full_users:.1f} pairs/user, "
          f"{args.n_pairs/args.full_items:.1f} pairs/item)")
    sink = Sink(root, args.dataset)

    e0rows = e0 = None; e1 = e2 = e3 = e4 = e7 = []
    if "e0" in stages:
        e0rows, e0 = stage_e0(u, i, t, args, sink)
        sink.write("e0_diagnostics.csv", e0rows)
    if "e1" in stages:
        e1, e2, e3 = stage_e1_e2_e3(u, i, t, args, sink, ckdir, dev, torch,
                                    "e2" in stages, "e3" in stages)
        sink.write("e1_arms.csv", e1)
        if e2: sink.write("e2_crosscert.csv", e2)
        if e3:
            print("  NOTE e3 is a diagnostic: zbar comes from the scored users, so it gives "
                  "no lower bound.")
            sink.write("e3_itemonly.csv", e3)
    if "e3b" in stages:
        e3b = stage_e3b(u, i, t, args, sink, ckdir, dev, torch)
        if e3b: sink.write("e3b_capacity.csv", e3b)
    if "e4" in stages:
        e4, e7 = stage_e4_e7(u, i, t, args, sink, ckdir, dev, torch, "e7" in stages)
        sink.write("e4_invariance.csv", e4)
        if e7: sink.write("e7_zbar.csv", e7)
    sink.flush_long()

    meta = dict(dataset=args.dataset, csv=args.csv, stamp=stamp,
                # explicit, in pipeline order, so the provenance cannot read as contradictory
                rows_after_rating_filter=int(n_raw),
                pairs_after_dedup=int(n_after_dedup),
                # k-core is per-split and train-only, so there is no global post-kcore count;
                # the per-split train sizes are printed and stored in the stage CSVs
                kcore_scope="train_rows_only_after_split" if args.min_count else "off",
                dup_policy=args.dup_policy, min_count=int(args.min_count),
                full_users=int(args.full_users), full_items=int(args.full_items),
                pairs_per_user=round(args.n_pairs / args.full_users, 3),
                pairs_per_item=round(args.n_pairs / args.full_items, 3),
                estimand="logged membership among users and items observed in training "
                         "(warm-start); held-out pairs with an unseen user or item are dropped",
                interactions=int(len(u)), has_time=bool(t is not None),
                config={k: v for k, v in vars(args).items() if k != "split_list"},
                e0=e0, minutes=round((time.time() - t0) / 60, 1))
    with open(os.path.join(root, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    summarise(e1, e2, e4, e7, sink, args)
    print(f"\ndone in {(time.time()-t0)/60:.1f} min. everything under {root}/")
    print("paste the PASTE-ME SUMMARY block; the CSVs are for plotting.")
