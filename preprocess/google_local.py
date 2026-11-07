#!/usr/bin/env python3
"""
preprocess/google_local.py -- converter for the Google Local review data (McAuley Lab, 2021).

SEPARATE FILE ON PURPOSE, like prepdata_extra.py. run_experiments.py is not read, not imported
and not modified by this script. Nothing about the five finished datasets changes.

WHY A CONVERTER AT ALL
  run_experiments.py can read gzipped JSONL directly (--format jsonl), so in principle the raw
  state file could be handed to it. Two things make a converter the safer path:

  1. ids. user_id is a 21-digit decimal and gmap_id a hex string such as
     "0x89c2...:0xc1f0...". Whether the JSON quotes user_id decides whether numpy
     gets an object array of str or a ragged mix of int and str, and a mixed object
     array breaks np.unique on the first str/int comparison. Dense integer recoding
     removes the question, exactly as prepdata.py does for Steam.
  2. size. A state file is several GB of review TEXT, of which this study uses four
     scalar fields. Converting once turns a multi-GB stream into a small CSV, so the
     seven stages of run_experiments.py do not re-parse the text seven times.

WHAT IS *NOT* DONE HERE, ON PURPOSE
  No rating threshold. The rating is written through and the pipeline applies --min-rating
  4.0 itself, exactly as it does for ML-20M, Netflix and Amazon. Nothing is
  pre-filtered, so the positivity rule stays in one place.

  No k-core. The pipeline applies a five-core to the TRAINING ROWS ONLY, after the split.
  This is why the 10-core files published alongside (review-<State>_10.json.gz) must
  NOT be used: their core is computed over all reviews, held-out ones
  included, which is precisely the preprocessing leakage the protocol is built to
  avoid, and it would also double-filter against the pipeline's own core. Use the full
  per-state file.

  No de-duplication. The pipeline collapses repeat (user, item) pairs under --dup-policy.

TIMESTAMPS
  Google Local reports `time` in MILLISECONDS. This converter writes SECONDS, which
  is the unit ML-20M and Netflix already feed the pipeline, and the unit precheck.py assumes
  when it prints a date range (it raises "year 41971 is out of range" on milliseconds).
  Emitting seconds therefore needs no change to any existing tool.

  The conversion is safe by construction: --dup-policy first sorts by the timestamps,
  and that order is invariant under any strictly increasing rescaling. Sub-second ordering is lost, which cannot matter at
  this granularity. The script verifies the incoming unit rather than assuming it, and
  refuses to guess if the span is implausible.

USAGE
  python preprocess/google_local.py list
  python preprocess/google_local.py survey --src data/review-Connecticut.json.gz
  python preprocess/google_local.py prep   --src data/review-Connecticut.json.gz \
                                   --out data/google_connecticut.csv
  python preprocess/google_local.py prep   --src data/review-Colorado.json.gz \
                                   --out data/google_colorado.csv --sample-frac 0.35

SOURCE
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/
  files  review-<State>.json.gz      (full; use this one)
         review-<State>_10.json.gz   (10-core; do NOT use, see above)
  cite   Li, Shang and McAuley, "UCTopic", ACL 2022; and
         Yan, He, Li, Zhang and McAuley, "Personalized Showcases", SIGIR 2023.
"""
import argparse, collections, datetime, gzip, hashlib, json, os, sys, time, zlib

MAX_EXACT = 2 ** 53
BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal"

# published review counts, for choosing a slice before downloading anything
STATES = [
    ("Vermont", 852_203), ("Alaska", 1_051_246), ("North Dakota", 1_109_558),
    ("Wyoming", 1_141_421), ("South Dakota", 1_452_599), ("Rhode Island", 1_777_094),
    ("Delaware", 1_885_948), ("District of Columbia", 1_894_317),
    ("Montana", 1_933_939), ("Maine", 2_214_773), ("New Hampshire", 2_648_081),
    ("Nebraska", 3_286_810), ("Hawaii", 3_111_531), ("Mississippi", 3_861_771),
    ("Idaho", 3_892_636), ("New Mexico", 4_705_389), ("Iowa", 4_838_887),
    ("Connecticut", 5_181_800), ("Arkansas", 5_106_056), ("Kansas", 5_546_880),
    ("Nevada", 8_833_403), ("Alabama", 8_967_499), ("Utah", 9_081_167),
    ("Oregon", 11_012_170), ("Indiana", 12_865_167), ("Colorado", 15_681_222),
    ("Arizona", 18_375_050), ("Illinois", 23_096_838), ("Georgia", 24_060_125),
    ("Florida", 61_803_524), ("California", 70_529_977),
]

# the five finished datasets, for judging whether a slice lands in a comparable range
REFERENCE = [("Amazon-Games", 654_867), ("Yelp", 4_558_128), ("Amazon-Books", 8_038_735),
             ("ML-20M", 9_995_410), ("Netflix (sample)", 11_875_534)]


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") \
        else open(p, "r", encoding="utf-8", errors="replace")


class Coder:
    """First-appearance dense integer recoding. Deterministic for a given input order."""
    def __init__(self): self.m = {}
    def __call__(self, k):
        c = self.m.get(k)
        if c is None:
            c = len(self.m); self.m[k] = c
        return c


def keep_user(uid, frac):
    """Deterministic, streaming, reproducible: CRC32 of the id, no first pass, no RNG
    state, identical on every machine and every re-run. Same formula prepdata_extra.py
    uses for Netflix. A sampled user keeps their ENTIRE history, so per-user density,
    and hence p and D, are preserved in expectation; sampling rows would not preserve them.

    Parameterised as a FRACTION rather than a user count, because Google Local publishes
    review counts per state but not user counts, so a target count could only be turned
    into a rate by guessing the denominator."""
    if frac >= 1.0:
        return True
    return (zlib.crc32(str(uid).encode()) & 0xffffffff) / 2 ** 32 < frac


# ------------------------------------------------------------------ list
def cmd_list(a):
    print(f"\n  Google Local review files      {BASE}/review-<State>.json.gz\n")
    print(f"  {'state':24s}{'reviews':>14s}   {'note':<34s}")
    print("  " + "-" * 74)
    for name, cnt in STATES:
        note = ""
        if 3_000_000 <= cnt <= 16_000_000:
            note = "in range of the five studied"
        elif cnt < 3_000_000:
            note = "small; half-width may not resolve"
        else:
            note = "large; expect a long run"
        print(f"  {name:24s}{cnt:14,}   {note:<34s}")
    print("\n  for scale, distinct positive pairs in the five finished datasets:")
    for n, c in REFERENCE:
        print(f"    {n:20s}{c:12,}")
    print("\n  Amazon-Books, at 8.0M positives, took about 20.6 h at your settings, so a")
    print("  state of roughly 5M reviews is about a 10 h run once the rating filter and")
    print("  the five-core have been applied.\n")
    print("  The sandbox cannot reach mcauleylab.ucsd.edu; the proxy allowlist refuses it.")
    print("  Download in your OWN terminal, for example:\n")
    print(f"    curl -L -o data/review-Connecticut.json.gz \\")
    print(f"      {BASE}/review-Connecticut.json.gz\n")
    print("  Use the plain file, NOT review-<State>_10.json.gz. The 10-core version")
    print("  is cored over held-out reviews too, which is the leakage this protocol")
    print("  exists to avoid, and it would double-filter against the pipeline's own core.\n")


# ------------------------------------------------------------------ scan
def probe_unit(src, n=200_000):
    """Read a prefix and decide whether `time` is in seconds or milliseconds, from the
    span it implies. Returns True for milliseconds. Refuses to guess on a bad span."""
    lo = hi = None; seen = 0
    with opener(src) as f:
        for line in f:
            if seen >= n: break
            try: t = json.loads(line).get("time")
            except Exception: continue
            if t is None: continue
            t = int(t); seen += 1
            lo = t if lo is None or t < lo else lo
            hi = t if hi is None or t > hi else hi
    if lo is None:
        sys.exit("\nERROR: no parseable `time` field in the first "
                 f"{n:,} lines of {src}. Inspect the file before running.")
    yr_ms, yr_s = (hi - lo) / 1000 / 86400 / 365.25, (hi - lo) / 86400 / 365.25
    plaus = lambda v: 1e9 < v < 4e9
    if plaus(lo / 1000) and plaus(hi / 1000):
        print(f"    unit probe over {seen:,} records: MILLISECONDS "
              f"(span {yr_ms:.1f} years); the converter will divide by 1000")
        return True
    if plaus(lo) and plaus(hi):
        print(f"    unit probe over {seen:,} records: SECONDS already "
              f"(span {yr_s:.1f} years); the converter will pass them through")
        return False
    sys.exit(f"\nERROR: `time` spans {lo} .. {hi}, which is not a plausible date range "
             "as either seconds or milliseconds. Inspect the file before running.")


def scan(src, out, frac, limit):
    """One streaming pass. Writes the CSV when out is not None, otherwise only reports."""
    ms = probe_unit(src)
    uc, ic = Coder(), Coder()
    fh = open(out, "w", newline="") if out else None
    if fh:
        fh.write("user_id,item_id,rating,timestamp\n")

    n = written = 0
    drop = collections.Counter()
    rate_hist = collections.Counter()
    tmin = tmax = None
    uid_types, iid_types = collections.Counter(), collections.Counter()
    pairs = set() if out is None else None     # survey only: measure the dedup effect
    ge4 = 0
    t0 = time.time()

    with opener(src) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            if limit and n > limit:
                n -= 1
                break
            try:
                d = json.loads(line)
            except Exception:
                drop["unparseable json"] += 1
                continue
            u, i = d.get("user_id"), d.get("gmap_id")
            r, t = d.get("rating"), d.get("time")
            if u is None or i is None:
                drop["missing user_id or gmap_id"] += 1
                continue
            if r is None:
                drop["missing rating"] += 1
                continue
            if t is None:
                drop["missing time"] += 1
                continue
            try:
                r = float(r); t = int(t)
            except (TypeError, ValueError):
                drop["unparseable rating or time"] += 1
                continue
            if not (0.0 < r <= 5.0):
                drop["rating outside (0,5]"] += 1
                continue
            uid_types[type(u).__name__] += 1
            iid_types[type(i).__name__] += 1
            if not keep_user(u, frac):
                drop["not in user sample"] += 1
                continue
            rate_hist[r] += 1
            if r >= 4.0:
                ge4 += 1
            tmin = t if tmin is None or t < tmin else tmin
            tmax = t if tmax is None or t > tmax else tmax
            t_out = t // 1000 if ms else t          # emit seconds, see the header
            ui, ii = uc(u), ic(i)
            if pairs is not None:
                pairs.add((ui, ii))
            if fh:
                fh.write(f"{ui},{ii},{r:g},{t_out}\n")
            written += 1
            if n % 2_000_000 == 0:
                print(f"    {n:,} lines  ({time.time()-t0:.0f}s)", flush=True)
    if fh:
        fh.close()
    return dict(n=n, written=written, drop=drop, rate_hist=rate_hist, ge4=ge4, ms=ms,
                tmin=tmin, tmax=tmax, users=len(uc.m), items=len(ic.m),
                uid_types=uid_types, iid_types=iid_types,
                distinct_pairs=(len(pairs) if pairs is not None else None),
                secs=time.time() - t0)


def ymd(t, ms):
    return datetime.datetime.utcfromtimestamp(t / (1000.0 if ms else 1.0)).strftime("%Y-%m-%d")


def report(src, s, frac):
    print(f"\n  read {s['n']:,} lines from {src} in {s['secs']:.0f}s")
    if s["drop"]:
        print("  dropped:")
        for k, v in s["drop"].most_common():
            print(f"    {v:12,}  {k}")
    print(f"  kept {s['written']:,} rows   users {s['users']:,}   items {s['items']:,}")
    if s["users"]:
        print(f"    {s['written']/s['users']:.1f} rows/user   "
              f"{s['written']/max(s['items'],1):.1f} rows/item")

    print("\n  id types as parsed from the JSON:")
    print(f"    user_id  {dict(s['uid_types'])}")
    print(f"    gmap_id  {dict(s['iid_types'])}")
    if len(s["uid_types"]) > 1 or len(s["iid_types"]) > 1:
        print("    !! mixed types in one field. This is exactly why the ids are recoded;")
        print("       handing the raw file to the pipeline's jsonl path would build a mixed")
        print("       object array and np.unique would raise on the first comparison.")

    print("\n  rating histogram:")
    tot = sum(s["rate_hist"].values()) or 1
    for r in sorted(s["rate_hist"]):
        print(f"    {r:>4}  {s['rate_hist'][r]:12,}  {100*s['rate_hist'][r]/tot:5.1f}%")
    print(f"    rating >= 4 : {s['ge4']:,}  ({100*s['ge4']/tot:.1f}% of kept rows)")

    if s["tmin"] is not None:
        div = 1000.0 if s["ms"] else 1.0
        yrs = (s["tmax"] - s["tmin"]) / div / 86400 / 365.25
        print(f"\n  time span {ymd(s['tmin'], s['ms'])} .. {ymd(s['tmax'], s['ms'])}"
              f"  ({yrs:.1f} years)")
        print(f"    written to the CSV in SECONDS: {int(s['tmin']/div)} .. {int(s['tmax']/div)}")
        if s["tmin"] == s["tmax"]:
            print("    !! all timestamps identical: the time field was not read.")
    umax, imax = s["users"] - 1, s["items"] - 1
    key_max = umax * (imax + 1) + imax if s["users"] and s["items"] else 0
    print(f"\n  duplicate-key precision: max key {key_max:,}  limit 2^53 = {MAX_EXACT:,}"
          f"  -> {'SAFE' if key_max < MAX_EXACT else 'UNSAFE'}")
    if key_max >= MAX_EXACT:
        sys.exit("\nERROR: key exceeds 2^53 even after recoding. Do not run.")

    if s["distinct_pairs"] is not None:
        dup = s["written"] - s["distinct_pairs"]
        print(f"\n  distinct (user,item) pairs {s['distinct_pairs']:,}  "
              f"({dup:,} repeats, {100*dup/max(s['written'],1):.2f}%, collapsed by the pipeline)")
        est = int(s["distinct_pairs"] * s["ge4"] / max(s["written"], 1))
        print(f"  rough positives after the rating filter and the collapse: {est:,}")
        lo = min(c for _, c in REFERENCE); hi = max(c for _, c in REFERENCE)
        verdict = "inside" if lo <= est <= hi else ("below" if est < lo else "above")
        print(f"  that is {verdict} the range of the five finished datasets "
              f"({lo:,} to {hi:,}).")


# ------------------------------------------------------------------ survey / prep
def cmd_survey(a):
    print(f"\n  SURVEY (no file written)")
    s = scan(a.src, None, a.sample_frac, a.limit)
    report(a.src, s, a.sample_frac)
    print("\n  nothing was written. Re-run with `prep --out <path>` when the numbers look right.\n")


def cmd_prep(a):
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"\nERROR: {a.out} exists. Pass --force to overwrite, or choose another name.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    print(f"\n  PREP  {a.src}  ->  {a.out}")
    s = scan(a.src, a.out, a.sample_frac, a.limit)
    report(a.src, s, a.sample_frac)

    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    man = dict(
        source=a.src, source_url=f"{BASE}/{os.path.basename(a.src)}",
        output=a.out, rows_written=s["written"], lines_read=s["n"],
        users=s["users"], items=s["items"], has_timestamp=True,
        time_unit_in_source="milliseconds" if s["ms"] else "seconds",
        time_unit_written="seconds", rating_threshold_applied=False, kcore_applied=False,
        dedup_applied=False, ids="dense integer recoding, first appearance order",
        sample_frac=a.sample_frac if a.sample_frac < 1.0 else None,
        dropped={k: v for k, v in s["drop"].items()},
        sha256=h.hexdigest(), bytes=os.path.getsize(a.out),
        made=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n  wrote {a.out}  ({os.path.getsize(a.out)/1e6:.1f} MB)")
    print(f"  wrote {a.out}.manifest.json   sha256 {man['sha256'][:16]}...")

    tag = "google_" + os.path.basename(a.out).replace("google_", "").replace(".csv", "")
    print("\n  NEXT, in order:\n")
    print("  1. validate with the pipeline's own code, training nothing:\n")
    print(f"     python precheck.py --dataset {tag} --csv {a.out} \\")
    print(f"       --user user_id --item item_id --rating rating --time timestamp \\")
    print(f"       --min-rating 4.0 --min-count 5 --out results_final_{tag}\n")
    print("  2. only if that says GO, run it:\n")
    print(f"     python run_experiments.py --dataset {tag} --csv {a.out} \\")
    print(f"       --user user_id --item item_id --rating rating --time timestamp \\")
    print(f"       --min-rating 4.0 --out results_final_{tag} \\")
    print(f"       --stages e0,e1,e2,e3,e3b,e4,e7 \\")
    print(f"       --min-count 5 --dup-policy first --preregister-alpha 1.0\n")
    print("  those flags are the ones the five finished datasets used. Changing any of")
    print("  them would make this dataset incomparable with them.\n")


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("list", help="states, sizes, and the download command")
p.set_defaults(fn=cmd_list)
for name, fn in (("survey", cmd_survey), ("prep", cmd_prep)):
    q = sub.add_parser(name)
    q.add_argument("--src", required=True, help="review-<State>.json.gz as downloaded")
    q.add_argument("--sample-frac", type=float, default=1.0,
                   help="keep this fraction of USERS by CRC of the id, whole histories "
                        "intact (default 1.0, keep everyone)")
    q.add_argument("--limit", type=int, default=0, help="stop after N lines (a smoke test)")
    if name == "prep":
        q.add_argument("--out", required=True)
        q.add_argument("--force", action="store_true")
    q.set_defaults(fn=fn)
a = ap.parse_args()
a.fn(a)
