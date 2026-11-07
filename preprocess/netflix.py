#!/usr/bin/env python3
"""Netflix Prize -> the four-column CSV the pipeline reads.

The Prize data ships as combined_data_*.txt, where a line ending in a colon names a movie
and the lines under it are its ratings, so it is not a table and needs converting. Ratings
are written through unchanged; the positivity cut is the pipeline's, not this script's.

    python preprocess/netflix.py --src-dir data/netflix --out data/netflix.csv --survey
    python preprocess/netflix.py --src-dir data/netflix --out data/netflix.csv

``--survey`` reports what the source contains and writes nothing.
"""
import argparse, csv, datetime, glob, hashlib, json, os, sys, time, zlib


def iso_epoch(s):
    """'2009-05-04T23:08:57Z' or '2005-09-06' -> epoch seconds."""
    s = s.strip()
    if not s:
        return None
    try:
        return datetime.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except Exception:
        pass
    for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.datetime.strptime(s, fmt).replace(
                tzinfo=datetime.timezone.utc).timestamp()
        except Exception:
            continue
    return None


class Coder:
    def __init__(self): self.m = {}
    def __call__(self, k):
        c = self.m.get(k)
        if c is None:
            c = len(self.m); self.m[k] = c
        return c



def finish(out, n_rows, n_users, n_items, has_time, src, extra):
    umax, imax = n_users - 1, n_items - 1
    key_max = umax * (imax + 1) + imax
    print("\n  DUPLICATE-KEY PRECISION CHECK (run_experiments.py keys repeat pairs as")
    print("    user*(item_max+1) + item in float64; exact only below 2^53)")
    print(f"    users {n_users:,}  items {n_items:,}  max key {key_max:,.0f}  -> "
          f"{'SAFE' if key_max < MAX_EXACT else 'UNSAFE'}")
    if key_max >= MAX_EXACT:
        sys.exit("\nERROR: key exceeds 2^53. Do not run.")
    h = hashlib.sha256()
    with open(out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    man = dict(source=src, output=out, rows_written=n_rows, users=n_users, items=n_items,
               has_timestamp=has_time, sha256=h.hexdigest(), bytes=os.path.getsize(out),
               made=datetime.datetime.now().isoformat(timespec="seconds"), **extra)
    with open(out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n  wrote {out}  ({n_rows:,} rows, {os.path.getsize(out)/1e9:.2f} GB)")
    print(f"  wrote {out}.manifest.json   sha256 {man['sha256'][:16]}...")
    print(f"  users {n_users:,}   items {n_items:,}   "
          f"{n_rows/max(n_users,1):.1f} rows/user   {n_rows/max(n_items,1):.1f} rows/item")


# ------------------------------------------------------------------ Last.fm-1K

def netflix_files(d):
    fs = sorted(glob.glob(os.path.join(d, "combined_data_*.txt")))
    if not fs:
        sys.exit(f"\nERROR: no combined_data_*.txt under {d}\n"
                 f"  after unzipping you should have combined_data_1.txt .. _4.txt")
    return fs



def keep_user(uid, frac):
    """Deterministic, streaming, reproducible: CRC32 of the id, no first pass, no RNG
    state, identical on every machine and every re-run."""
    return (zlib.crc32(uid.encode()) & 0xffffffff) / 2 ** 32 < frac



def cmd_netflix(a):
    fs = netflix_files(a.src_dir)
    print(f"  found {len(fs)} files: " + ", ".join(os.path.basename(x) for x in fs))
    TOTAL_USERS = 480189                      # published count, used only to set the rate
    if a.survey:
        n = 0; users = set(); movies = 0; hist = {}
        t0 = time.time()
        for p in fs:
            with open(p, "r") as f:
                for line in f:
                    if line.rstrip().endswith(":"):
                        movies += 1; continue
                    parts = line.rstrip("\n").split(",")
                    if len(parts) != 3: continue
                    n += 1; users.add(parts[0])
                    hist[parts[1]] = hist.get(parts[1], 0) + 1
        print(f"  {n:,} ratings, {len(users):,} users, {movies:,} movies "
              f"({time.time()-t0:.0f}s)")
        print(f"\n  RATING HISTOGRAM")
        for k in sorted(hist):
            print(f"    {k} stars {hist[k]:>14,}  ({hist[k]/n:.1%})")
        pos = sum(v for k, v in hist.items() if float(k) >= 4.0)
        print(f"    >= 4 stars {pos:>12,}  ({pos/n:.1%})  <- what the pipeline keeps")
        print(f"\n  USER-SAMPLE PLAN  (a sampled user keeps their WHOLE history)")
        print(f"    {'users kept':>12}{'est. positives':>17}{'est. hours':>12}")
        for k in (40000, 60000, 100000, 150000, len(users)):
            est = pos * k / max(len(users), 1)
            print(f"    {k:>12,}{est:>17,.0f}{581*est*0.9/9.0e6/60:>12.1f}")
        print("\n  Pick a row near ML-20M's ~10M positives, then re-run with --sample-users N.")
        return
    frac = min(1.0, a.sample_users / TOTAL_USERS)
    print(f"  sampling {a.sample_users:,} of ~{TOTAL_USERS:,} users "
          f"(rate {frac:.4f}, deterministic CRC32)")
    uc, ic = Coder(), Coder()
    n = 0; movie = None
    t0 = time.time()
    with open(a.out, "w", newline="") as g:
        w = csv.writer(g); w.writerow(["user_id", "item_id", "rating", "timestamp"])
        for p in fs:
            with open(p, "r") as f:
                for line in f:
                    s = line.rstrip("\n")
                    if s.endswith(":"):
                        movie = s[:-1]; continue
                    parts = s.split(",")
                    if len(parts) != 3 or movie is None: continue
                    uid, r, dt = parts
                    if frac < 1.0 and not keep_user(uid, frac): continue
                    e = iso_epoch(dt)
                    if e is None: continue
                    w.writerow([uc(uid), ic(movie), float(r), int(e)])
                    n += 1
            print(f"    {os.path.basename(p)} done, {n:,} rows so far "
                  f"({time.time()-t0:.0f}s)", flush=True)
    finish(a.out, n, len(uc.m), len(ic.m), True, a.src_dir,
           dict(sample_users_target=a.sample_users, sample_rate=frac,
                sampling="deterministic CRC32 of the raw user id; sampled users keep their "
                         "ENTIRE history, so per-user density is preserved",
                rule="ALL ratings written through; the pipeline applies --min-rating 4.0 itself, "
                     "exactly as for ML-20M"))
    print(f"\n  run it with --min-rating 4.0, the same rule as ML-20M/Yelp/Amazon.")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", required=True,
                    help="directory holding the combined_data_*.txt files")
    ap.add_argument("--out", default="data/netflix.csv")
    ap.add_argument("--sample-users", type=int, default=100000)
    ap.add_argument("--survey", action="store_true",
                    help="report what the source contains and write nothing")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    cmd_netflix(a)


if __name__ == "__main__":
    main()
