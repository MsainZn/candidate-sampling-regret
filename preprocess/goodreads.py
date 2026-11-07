#!/usr/bin/env python3
"""
preprocess/goodreads.py -- all eight Goodreads genre files -> one CSV.

Merges them, removes the cross-genre duplicates exactly, drops rating 0, and writes
the four columns run_experiments.py reads. No sampling, no genre selection, no options that
change what the corpus is.

run_experiments.py is not read, not imported and not modified.

WHAT IT DOES, IN ORDER
  1. validates the date parser against strptime before trusting it on 200M rows
  2. pass one: parses each genre file once, drops rating 0 and unusable rows,
     buffers the rest to a temporary file, and densely recodes user and book ids
  3. builds an exact int64 collapse key, uid*n_books+iid, and finds the first
     occurrence of each (user, book) pair
  4. pass two: rewrites the buffer keeping only those first occurrences
  5. reports the counts and writes a manifest beside the CSV

WHY TWO PASSES
  209,530,114 rows cannot be deduplicated in one pass without holding them all in
  memory. The key array alone is 8 bytes a row; a Python set of the same keys would
  be about 60. Re-reading a plain temporary file is far cheaper than decompressing
  and re-parsing 10 GB of gzip a second time.

THE DEDUP IS EXACT
  Ids are densely recoded first, so the key is an integer, not a hash. No collisions,
  no record lost. After it runs no (user, book) pair repeats, so the pipeline's own
  --dup-policy first is a no-op on the output.

RATING 0
  A Goodreads interaction is a shelf-add that may carry no rating; 0 means unrated.
  Those rows are dropped. They would have fallen out under --min-rating 4.0 anyway,
  so no result changes; the corpus table stops counting shelf events as interactions.

BUDGET
  RAM   about 1.5 GB for the key array plus the id maps
  DISK  a temporary buffer roughly the size of the final CSV, plus the CSV
  TIME  a few hours for the parse, dominated by gzip and json

USAGE
  python preprocess/goodreads.py list
  python preprocess/goodreads.py prep --dir data/goodreads --out data/goodreads_all.csv

SOURCE
  https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/
  cite  Wan and McAuley, "Item Recommendation on Monotonic Behavior Chains", RecSys 2018;
        Wan, Misra, Nakashole and McAuley, "Fine-Grained Spoiler Detection from
        Large-Scale Review Corpora", ACL 2019.
"""
import argparse, calendar, collections, datetime, gzip, hashlib, json, math
import os, sys, tempfile, time

import numpy as np

MAX_EXACT = 2 ** 53
BASE = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre"

GENRES = [
    ("poetry", 2_734_350, "145M"),
    ("comics_graphic", 7_347_630, "369M"),
    ("children", 10_059_349, "488M"),
    ("mystery_thriller_crime", 24_799_896, "1.2G"),
    ("history_biography", 31_479_229, "1.6G"),
    ("young_adult", 34_919_254, "1.7G"),
    ("romance", 42_792_856, "2.1G"),
    ("fantasy_paranormal", 55_397_550, "2.6G"),
]
GENRE_NAMES = [g for g, _, _ in GENRES]
TOTAL_ROWS = sum(c for _, c, _ in GENRES)

ML_PAIRS, ML_MINUTES = 9.0e6, 581.0
CERT_K, TAU, DELTA = 1024, 0.1, 0.05
CHUNK = 8_000_000

MON = {m: k + 1 for k, m in enumerate(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
     "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])}


def hw(n):
    return (math.log(CERT_K) + 2.0 / TAU) * math.sqrt(math.log(2 / DELTA) / (2 * n))


def ymd(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")


def human(n):
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def parse_added(s):
    """'Thu Aug 17 15:15:28 -0700 2017' -> epoch seconds. Fixed-offset slicing; about
    thirty times faster than strptime, which matters over 200M rows. Validated against
    strptime by check_parser() before the bulk pass runs."""
    try:
        mon = MON[s[4:7]]
        day = int(s[8:10]); hh = int(s[11:13]); mm = int(s[14:16]); ss = int(s[17:19])
        sign = -1 if s[20] == "-" else 1
        off = sign * (int(s[21:23]) * 3600 + int(s[23:25]) * 60)
        yr = int(s[26:30])
        return calendar.timegm((yr, mon, day, hh, mm, ss, 0, 0, 0)) - off
    except (KeyError, ValueError, IndexError):
        return None


def check_parser(path, n=20_000):
    import datetime as _dt
    ok = bad = 0
    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
        for k, line in enumerate(f):
            if k >= n:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            s = d.get("date_added") or ""
            if not s:
                continue
            fast = parse_added(s)
            try:
                slow = int(_dt.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp())
            except ValueError:
                continue
            if fast == slow:
                ok += 1
            else:
                bad += 1
    if bad:
        sys.exit(f"\nERROR: the fast date parser disagrees with strptime on {bad} of "
                 f"{ok+bad} records. Stopping; the timestamps would be wrong.")
    print(f"  date parser validated against strptime on {ok:,} records: exact")


def src_path(d, genre):
    for name in (f"goodreads_interactions_{genre}.json.gz",
                 f"goodreads_interactions_{genre}.json"):
        p = os.path.join(d, name)
        if os.path.exists(p):
            return p
    return None


def opener(p):
    return gzip.open(p, "rt", encoding="utf-8", errors="replace") if p.endswith(".gz") \
        else open(p, "r", encoding="utf-8", errors="replace")


class Coder:
    __slots__ = ("m",)

    def __init__(self):
        self.m = {}

    def __call__(self, k):
        m = self.m
        c = m.get(k)
        if c is None:
            c = len(m)
            m[k] = c
        return c


class KeyBuf:
    """A growing int64 array in chunks: 8 bytes a key instead of the ~60 a Python
    list or set would cost."""
    def __init__(self):
        self.chunks = []
        self.cur = np.empty(CHUNK, dtype=np.int64)
        self.n = 0

    def append(self, v):
        if self.n == CHUNK:
            self.chunks.append(self.cur)
            self.cur = np.empty(CHUNK, dtype=np.int64)
            self.n = 0
        self.cur[self.n] = v
        self.n += 1

    def finish(self):
        self.chunks.append(self.cur[:self.n])
        out = np.concatenate(self.chunks) if len(self.chunks) > 1 else self.chunks[0]
        self.chunks = self.cur = None
        return out


# ------------------------------------------------------------------ list
def cmd_list(a):
    print(f"\n  Goodreads per-genre interactions   {BASE}/\n")
    print(f"  {'genre':26s}{'interactions':>14s}{'gz':>8s}")
    print("  " + "-" * 48)
    for g, c, s in GENRES:
        print(f"  {g:26s}{c:14,}{s:>8s}")
    print("  " + "-" * 48)
    print(f"  {'COMBINED':26s}{TOTAL_ROWS:14,}{'~10.2G':>8s}")
    print("\n  That sum counts cross-genre repeats; the distinct total is lower and")
    print("  'prep' reports it exactly.\n")
    print("  Download:\n")
    print("    mkdir -p data/goodreads")
    for g, _, _ in GENRES:
        print(f"    curl -L -C - -o data/goodreads/goodreads_interactions_{g}.json.gz \\")
        print(f"      {BASE}/goodreads_interactions_{g}.json.gz")
    print("\n  Do NOT use goodreads_interactions.csv from the parent directory: it has")
    print("  no date field, which orders repeated pairs before they are collapsed.\n")


# ------------------------------------------------------------------ prep
def cmd_prep(a):
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"\nERROR: {a.out} exists. Pass --force to overwrite.")
    if not os.path.isdir(a.dir):
        sys.exit(f"\nERROR: {a.dir} is not a directory. Run 'list' for the downloads.")
    paths, missing = {}, []
    for g in GENRE_NAMES:
        p = src_path(a.dir, g)
        (paths.__setitem__(g, p) if p else missing.append(g))
    if missing:
        sys.exit(f"\nERROR: missing {len(missing)} genre file(s) in {a.dir}:\n"
                 + "".join(f"    goodreads_interactions_{g}.json.gz\n" for g in missing)
                 + "  All eight are required. Run 'list' for the download commands.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    tmpdir = a.tmpdir or os.path.dirname(os.path.abspath(a.out))
    fd, tmp = tempfile.mkstemp(prefix="goodreads_all_", suffix=".tmp", dir=tmpdir)
    os.close(fd)

    print(f"\n  PREP  8 genre files  ->  {a.out}")
    print(f"  buffer: {tmp}")
    check_parser(paths[GENRE_NAMES[0]])

    uc, ic = Coder(), Coder()
    drop = collections.Counter()
    read = kept = unrated = 0
    per_genre = {}
    tmin = tmax = None
    t0 = time.time()

    # ---------------- pass 1
    print("\n  PASS 1  parse, drop rating 0, buffer\n")
    print(f"    {'genre':26s}{'lines':>14s}{'kept':>14s}{'elapsed':>10s}")
    print("    " + "-" * 64)
    tf = open(tmp, "w", buffering=1 << 20, newline="")
    for g in GENRE_NAMES:
        gread = gkept = 0
        with opener(paths[g]) as f:
            for line in f:
                gread += 1
                if a.limit and gread > a.limit:
                    gread -= 1
                    break
                try:
                    d = json.loads(line)
                except Exception:
                    drop["unparseable json"] += 1
                    continue
                u = d.get("user_id"); i = d.get("book_id")
                if not u or not i:
                    drop["missing user_id or book_id"] += 1
                    continue
                r = d.get("rating")
                try:
                    r = float(r)
                except (TypeError, ValueError):
                    drop["missing or unparseable rating"] += 1
                    continue
                if not (0.0 <= r <= 5.0):
                    drop["rating outside [0,5]"] += 1
                    continue
                if r == 0.0:
                    unrated += 1
                    drop["rating 0, unrated shelf-add"] += 1
                    continue
                s = d.get("date_added") or ""
                t = parse_added(s) if s else None
                if t is None:
                    drop["missing or unparseable date_added"] += 1
                    continue
                if not (0 < t < 4_000_000_000):
                    drop["date outside a plausible range"] += 1
                    continue
                tf.write(f"{uc(u)},{ic(i)},{r:g},{t}\n")
                tmin = t if tmin is None or t < tmin else tmin
                tmax = t if tmax is None or t > tmax else tmax
                gkept += 1
        read += gread
        kept += gkept
        per_genre[g] = (gread, gkept)
        print(f"    {g:26s}{gread:14,}{gkept:14,}{time.time()-t0:9.0f}s", flush=True)
    tf.close()

    nusers, nitems = len(uc.m), len(ic.m)
    del uc, ic
    print(f"\n    read {read:,} lines, buffered {kept:,} rows")
    print(f"    users {nusers:,}   books {nitems:,}   buffer {human(os.path.getsize(tmp))}")

    key_max = (nusers - 1) * nitems + (nitems - 1) if nusers and nitems else 0
    if key_max >= 2 ** 63 - 1:
        os.remove(tmp)
        sys.exit("\nERROR: the exact collapse key overflows int64.")
    print(f"    exact key max {key_max:,}   int64 limit {2**63-1:,}   SAFE")
    if key_max >= MAX_EXACT:
        print(f"    (above 2^53, which is fine here: this key is int64 and never a float.")
        print(f"     the pipeline recodes to its own dense ids before its float64 key, and")
        print(f"     precheck.py checks that separately.)")

    # ---------------- dedup
    print("\n  DEDUP  exact, on the int64 key\n")
    print(f"    building the key array ({kept*8/1e9:.2f} GB) ...", flush=True)
    keys = KeyBuf()
    with open(tmp, buffering=1 << 20) as f:
        for line in f:
            u, i, _ = line.split(",", 2)
            keys.append(int(u) * nitems + int(i))
    arr = keys.finish()
    print(f"    finding first occurrences ...", flush=True)
    _, first_idx = np.unique(arr, return_index=True)
    distinct = int(first_idx.size)
    dupes = int(arr.size - distinct)
    mask = np.zeros(arr.size, dtype=bool)
    mask[first_idx] = True
    del arr, first_idx

    print(f"\n    cross-genre duplicate pairs removed  {dupes:,}"
          f"  ({100*dupes/max(kept,1):.2f}% of buffered rows)")
    print(f"    distinct (user, book) pairs written  {distinct:,}")

    # ---------------- pass 2
    print("\n  PASS 2  rewrite, keeping the first occurrence of each pair\n")
    hist = collections.Counter()
    per_item = collections.Counter()
    per_user = collections.Counter()
    written = conflicts = 0
    sample = {}
    with open(tmp, buffering=1 << 20) as f, \
         open(a.out, "w", buffering=1 << 20, newline="") as out:
        out.write("user_id,item_id,rating,timestamp\n")
        for k, line in enumerate(f):
            if not mask[k]:
                if sample:
                    u, i, r, t = line.rstrip("\n").split(",")
                    prev = sample.get((u, i))
                    if prev is not None and prev != (r, t):
                        conflicts += 1
                continue
            out.write(line)
            u, i, r, t = line.rstrip("\n").split(",")
            if len(sample) < a.conflict_sample:
                sample[(u, i)] = (r, t)
            rv = float(r)
            hist[rv] += 1
            if rv >= 4.0:
                per_item[i] += 1
                per_user[u] += 1
            written += 1
    os.remove(tmp)
    del mask

    if conflicts:
        print(f"    !! {conflicts:,} removed repeats disagreed with the row kept, on")
        print("       rating or timestamp. The kept copy is the first in read order,")
        print("       genres in the order 'list' shows. Report this if it is material.")
    else:
        print(f"    every removed repeat matched the kept row exactly "
              f"(checked against the first {min(a.conflict_sample, distinct):,} pairs)")
    print(f"    no (user, book) pair repeats in the output, so the pipeline's")
    print(f"    --dup-policy first is a no-op on this file")

    # ---------------- report
    tot = sum(hist.values()) or 1
    print("\n  RESULT\n")
    print(f"    rows written {written:,}   users {nusers:,}   books {nitems:,}")
    print(f"    {written/max(nusers,1):.2f} rows/user   {written/max(nitems,1):.1f} rows/book")
    print("\n    rating histogram:")
    for r in sorted(hist):
        bar = "#" * int(40 * hist[r] / max(hist.values()))
        print(f"      {r:>4.0f}  {hist[r]:14,}  {100*hist[r]/tot:5.1f}%  {bar}")
    ge4 = sum(v for r, v in hist.items() if r >= 4.0)
    print(f"      at >= 4 : {ge4:,} ({100*ge4/tot:.1f}%)   this is what --min-rating 4.0 keeps")
    print(f"      rating 0 dropped in pass 1: {unrated:,}")

    if drop:
        print("\n    dropped in pass 1:")
        for k, v in drop.most_common():
            print(f"      {v:14,}  {k}")

    if tmin is not None:
        print(f"\n    date_added span {ymd(tmin)} .. {ymd(tmax)}  "
              f"({(tmax-tmin)/86400/365.25:.1f} years)")
        if tmin == tmax:
            sys.exit("\nERROR: all timestamps identical; the date field was not read.")

    ni, nu = len(per_item), len(per_user)
    print(f"\n    positives at >= 4 : {ge4:,}")
    if ni:
        print(f"      {ge4/ni:.1f} per book over {ni:,} books   (study range 16.3 to 672)")
    if nu:
        print(f"      {ge4/nu:.2f} per user over {nu:,} users   (study range 3.11 to 119)")
    warm = 0.75
    n_test = int(ge4 * 0.10 * warm)
    steps = int(ge4 * 0.90 * warm / 4096)
    mins = ML_MINUTES * (ge4 * 0.9 * warm) / ML_PAIRS
    print(f"\n    forecast: ~{n_test:,} warm test pairs, n_cert {min(110_000, n_test):,}, "
          f"h ~{hw(min(110_000, max(n_test,1))):.4f}")
    print(f"              ~{steps:,} steps/epoch at batch 4096   (ML-20M got 2,197)")
    rt = f"{mins/60:.1f} hours" if mins < 2880 else f"{mins/1440:.1f} days"
    print(f"              ~{rt} for the stage list   (ML-20M took 9.7 hours)")
    print("              precheck.py measures all of this for real, next.")

    # ---------------- manifest
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    man = dict(
        dataset="goodreads_all", source_dir=a.dir, source_url=BASE,
        citation="Wan and McAuley, Item Recommendation on Monotonic Behavior Chains, "
                 "RecSys 2018; Wan, Misra, Nakashole and McAuley, Fine-Grained Spoiler "
                 "Detection from Large-Scale Review Corpora, ACL 2019",
        genres=GENRE_NAMES, genres_combined=8,
        published_rows_all_genres=TOTAL_ROWS,
        lines_read=read, rows_after_filters=kept,
        per_genre={g: dict(lines=v[0], kept=v[1]) for g, v in per_genre.items()},
        cross_genre_duplicates_removed=dupes, duplicate_conflicts=conflicts,
        dedup_key="exact uid*n_books+iid in int64; no hashing, no collisions",
        dedup_scope="across all eight genre files",
        unrated_rows_dropped=unrated,
        unrated_note="rating 0 means unrated. Dropping them changes no result, because "
                     "the pipeline applies --min-rating 4.0 before E0 and they would fall out "
                     "there. It changes the raw corpus table only.",
        sampling="none; the full corpus",
        output=a.out, rows_written=written,
        users=nusers, items=nitems, distinct_pairs=distinct,
        positives_ge4=ge4,
        has_timestamp=True, time_field="date_added", time_unit_written="seconds",
        time_span=[ymd(tmin), ymd(tmax)] if tmin is not None else None,
        rating_threshold_applied=False, kcore_applied=False,
        dedup_applied="cross-genre, exact; the pipeline's --dup-policy first is a no-op after it",
        ids="dense integer recoding, first appearance across the combined stream",
        dropped=dict(drop), sha256=h.hexdigest(),
        bytes=os.path.getsize(a.out),
        elapsed_seconds=round(time.time() - t0, 1),
        made=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n    wrote {a.out}  ({human(os.path.getsize(a.out))})")
    print(f"    wrote {a.out}.manifest.json   sha256 {man['sha256'][:16]}...")
    print(f"    total elapsed {(time.time()-t0)/60:.0f} min\n")


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)

p = sub.add_parser("list", help="the eight genres, sizes, and the download commands")
p.set_defaults(fn=cmd_list)

p = sub.add_parser("prep", help="merge all eight, dedup exactly, drop rating 0")
p.add_argument("--dir", required=True,
               help="directory holding all eight goodreads_interactions_<genre>.json.gz")
p.add_argument("--out", required=True)
p.add_argument("--tmpdir", default=None,
               help="where to put the pass-1 buffer; needs room for roughly the "
                    "size of the final CSV. Default: beside the output.")
p.add_argument("--conflict-sample", type=int, default=5_000_000,
               help="distinct pairs remembered when checking that removed repeats "
                    "really are identical")
p.add_argument("--limit", type=int, default=0,
               help="stop after N lines per genre file; a smoke test, not for real use")
p.add_argument("--force", action="store_true")
p.set_defaults(fn=cmd_prep)

a = ap.parse_args()
a.fn(a)
