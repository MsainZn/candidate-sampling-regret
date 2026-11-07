#!/usr/bin/env python3
"""
preprocess/hotelrec.py -- HotelRec (TripAdvisor, ~50M reviews) -> the CSV the pipeline reads.

run_experiments.py is not read, not imported and not modified.

THE RECORD, verified against your copy of data/HotelRec.txt

  {"hotel_url": "Hotel_Review-g194775-d1121769-Reviews-Hotel_Baltic-...html",
   "author": "violettaf340", "date": "2019-01-01T00:00:00", "rating": 5.0,
   "title": "...", "text": "...", "property_dict": {...}}

  Exactly one key order over 20,000 randomly sampled records, and the four fields this
  study needs all come before "text", which is most of each 1,088-byte line. So the
  parser extracts them by substring and never touches the review body, with a
  json.loads fallback and a validation pass that compares the two.

  user    author      item  the -dNNNNN- hotel id inside hotel_url
  rating  rating      time  date, ISO 8601, converted to epoch seconds

THREE THINGS THIS FILE GETS RIGHT THAT A NAIVE CONVERTER WOULD NOT

  1. "/undefined" IS NOT A USER. It is the placeholder TripAdvisor leaves for deleted
     or anonymous reviewers, and it accounts for 1.96% of a 300,000-row random sample
     spread over the whole file, i.e. roughly 900,000 rows. Kept, it becomes a single
     user with a million reviews: it would dominate the user marginal, wreck every
     per-user statistic in E0, and do it silently. Rows whose author is "/undefined"
     or "null" are dropped. --keep-placeholder-users overrides that, and you would
     have to defend it.

  2. THE CLOCK IS MONTH-GRANULAR. 99.6% of dates are the first of the month at
     00:00:00, because TripAdvisor publishes "January 2019" and not a day. The date
     only orders repeated (user, hotel) pairs before they are collapsed, for which
     month ties are harmless.

  3. THE ITEM IS THE HOTEL ID, NOT THE URL. Every one of 400,000 consecutive records
     carried a -dNNNNN- component, which is TripAdvisor's stable hotel key. Using the
     full URL as the item id would work but is fragile to the slug text changing, and
     it makes the collapse key enormous. The extraction rate is reported; if any
     record lacks the id the full URL is used for that row and the count is printed.

RATING
  Clean integers 1 to 5, no rescaling, so --min-rating 4.0 means here exactly what it
  means on MovieLens, Netflix, Yelp and Amazon. 78.2% of the sample clears it, which is
  what you expect from hotel reviews.

  property_dict holds sub-ratings (sleep quality, value, rooms, service, cleanliness,
  location) and is NOT used. An average of sub-ratings would be a different construct
  from every other dataset in the study, and it is absent from 28.7% of records.

WHAT IS NOT DONE HERE, ON PURPOSE
  No rating threshold, no k-core, no de-duplication of (user, hotel) repeats. The pipeline
  applies --min-rating 4.0, a five-core to the TRAINING ROWS ONLY after the split, and
  --dup-policy first.

BUDGET
  The file is 50.4 GB and about 46 million records. One pass. RAM is dominated by the
  id maps, so it scales with the number of distinct authors; expect a few GB. Time is
  dominated by reading 50 GB, not by parsing, because the review text is skipped.

USAGE
  python preprocess/hotelrec.py inspect --src data/HotelRec.txt
  python preprocess/hotelrec.py prep    --src data/HotelRec.txt --out data/hotelrec.csv

SOURCE
  https://github.com/diegoantognini/HotelRec   (academic research use only)
  cite  Antognini and Faltings, "HotelRec: a Novel Very Large-Scale Hotel
        Recommendation Dataset", LREC 2020.
"""
import argparse, collections, datetime, hashlib, json, math, os, random, re, sys, time

import numpy as np

MAX_EXACT = 2 ** 53
REPO = "https://github.com/diegoantognini/HotelRec"
PUB_REVIEWS = 50_000_000            # "50 million reviews", the only count the README states

PLACEHOLDER_USERS = {"/undefined", "null", "", "None", None}

REFERENCE = [("Yelp", 4_558_128, 3.11, 30.9), ("Amazon-Games", 654_867, 6.97, 25.6),
             ("Amazon-Books", 8_038_735, 10.39, 16.3), ("ML-20M", 9_995_410, 72.28, 482.4),
             ("Netflix", 11_875_534, 118.99, 672.3)]

ML_PAIRS, ML_MINUTES = 9.0e6, 581.0
CERT_K, TAU, DELTA = 1024, 0.1, 0.05
CHUNK = 8_000_000

DID = re.compile(rb'-d(\d+)-')
A_KEY = b'"author": "'
D_KEY = b'"date": "'
R_KEY = b'"rating": '
U_KEY = b'"hotel_url": "'


def hw(n):
    return (math.log(CERT_K) + 2.0 / TAU) * math.sqrt(math.log(2 / DELTA) / (2 * n))


def ymd(t):
    return datetime.datetime.fromtimestamp(t, datetime.timezone.utc).strftime("%Y-%m-%d")


def human(n):
    return f"{n/1e9:.1f} GB" if n >= 1e9 else f"{n/1e6:.0f} MB"


def fast_fields(line):
    """Pull hotel_url, author, date, rating out of a raw JSON line without parsing it.

    Returns (item, author, date, rating) or None if the line does not have the shape
    this expects. Never touches "text", which is most of the line."""
    try:
        a = line.index(U_KEY) + len(U_KEY)
        b = line.index(b'"', a)
        url = line[a:b]
        a = line.index(A_KEY, b) + len(A_KEY)
        b = line.index(b'"', a)
        author = line[a:b]
        a = line.index(D_KEY, b) + len(D_KEY)
        b = line.index(b'"', a)
        date = line[a:b]
        a = line.index(R_KEY, b) + len(R_KEY)
        b = a
        while b < len(line) and line[b] in b"0123456789.-eE":
            b += 1
        rating = line[a:b]
        return url, author, date, rating
    except ValueError:
        return None


def validate_parser(path, n=20_000, seed=11):
    """Compare the substring parser with json.loads on random records before trusting
    it on 46 million. Same discipline as the Goodreads date parser check."""
    rng = random.Random(seed)
    sz = os.path.getsize(path)
    ok = fell_back = mismatch = 0
    with open(path, "rb") as f:
        for _ in range(n):
            f.seek(rng.randrange(0, max(sz - 4096, 1)))
            f.readline()
            line = f.readline()
            if not line.strip():
                continue
            try:
                d = json.loads(line)
            except Exception:
                continue
            got = fast_fields(line)
            if got is None:
                fell_back += 1
                continue
            url, author, date, rating = got
            same = (url.decode("utf-8", "replace") == (d.get("hotel_url") or "")
                    and author.decode("utf-8", "replace") == str(d.get("author"))
                    and date.decode("utf-8", "replace") == (d.get("date") or "")
                    and abs(float(rating) - float(d.get("rating"))) < 1e-9)
            ok += same
            mismatch += (not same)
    if mismatch:
        sys.exit(f"\nERROR: the fast parser disagrees with json.loads on {mismatch} of "
                 f"{ok+mismatch} sampled records. Stopping rather than writing a file\n"
                 "  whose fields may be mis-assigned.")
    print(f"  fast parser validated against json.loads on {ok:,} random records: exact"
          + (f"   ({fell_back} fell back)" if fell_back else ""))


def parse_iso(s):
    """b'2019-01-01T00:00:00' -> epoch seconds UTC. Fixed-offset slicing."""
    try:
        y = int(s[0:4]); mo = int(s[5:7]); d = int(s[8:10])
        hh = int(s[11:13]); mm = int(s[14:16]); ss = int(s[17:19])
        return int(datetime.datetime(y, mo, d, hh, mm, ss,
                                     tzinfo=datetime.timezone.utc).timestamp())
    except (ValueError, IndexError):
        return None


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


class TimeBuf:
    """int64 timestamps in chunks, for the split diagnostic without a second pass."""
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


# ------------------------------------------------------------------ inspect
def cmd_inspect(a):
    if not os.path.exists(a.src):
        sys.exit(f"\nERROR: {a.src} does not exist.")
    sz = os.path.getsize(a.src)
    print(f"\n  INSPECT  {a.src}  ({human(sz)})   nothing is written\n")
    validate_parser(a.src, a.sample_validate)

    rng = random.Random(7)
    auth = collections.Counter(); item = collections.Counter()
    rat = collections.Counter(); day = collections.Counter(); yr = collections.Counter()
    tod = collections.Counter()
    n = blen = noid = 0
    with open(a.src, "rb") as f:
        for _ in range(a.sample):
            f.seek(rng.randrange(0, max(sz - 4096, 1)))
            f.readline()
            line = f.readline()
            if not line.strip():
                continue
            got = fast_fields(line)
            if got is None:
                continue
            url, author, date, rating = got
            blen += len(line); n += 1
            auth[author.decode("utf-8", "replace")] += 1
            m = DID.search(url)
            if m:
                item[m.group(1)] += 1
            else:
                noid += 1
            try:
                rat[float(rating)] += 1
            except ValueError:
                pass
            if len(date) >= 19:
                yr[date[:4].decode()] += 1
                day[date[8:10].decode()] += 1
                tod[date[11:19].decode()] += 1

    print(f"\n  RANDOM-OFFSET SAMPLE over the whole file ({n:,} records)")
    print(f"    mean bytes/record {blen/max(n,1):.0f}"
          f"   -> estimated total records {sz/(blen/max(n,1)):,.0f}"
          f"   (published: ~{PUB_REVIEWS:,})")

    print("\n  AUTHOR FIELD")
    print(f"    distinct authors in the sample {len(auth):,}")
    ph = sum(auth.get(k, 0) for k in PLACEHOLDER_USERS if isinstance(k, str))
    print(f"    top values: {auth.most_common(4)}")
    print(f"    placeholder rows ('/undefined', 'null'): {ph:,}"
          f"  ({100*ph/max(n,1):.2f}% of sampled rows)")
    if ph:
        print("    These are NOT users. Kept, they collapse into one account with about")
        print(f"    {int(ph/max(n,1)*sz/(blen/max(n,1))):,} reviews and dominate every")
        print("    per-user statistic in E0. 'prep' drops them unless you override.")
    print("    (A random-offset sample nearly always draws distinct users, so the")
    print("     distinct count above is NOT an estimate of the user total. 'prep'")
    print("     counts users exactly.)")

    print("\n  ITEM FIELD")
    print(f"    hotel ids extracted from hotel_url: {sum(item.values()):,}"
          f"   without a -dNNN- id: {noid:,}")
    print(f"    distinct hotels in the sample {len(item):,}")

    tot = sum(rat.values()) or 1
    print("\n  RATING")
    for k in sorted(rat):
        bar = "#" * int(40 * rat[k] / max(rat.values()))
        print(f"    {k:>4.0f}  {rat[k]:10,}  {100*rat[k]/tot:5.1f}%  {bar}")
    ge4 = sum(v for k, v in rat.items() if k >= 4.0)
    print(f"    at >= 4 : {ge4:,} ({100*ge4/tot:.1f}%)   what --min-rating 4.0 keeps")
    if max(rat) > 5.0 or min(rat) < 1.0:
        print("    !! values outside 1-5. Check before converting.")

    dtot = sum(day.values()) or 1
    print("\n  CLOCK")
    print(f"    years {min(yr)} .. {max(yr)}")
    print(f"    day-of-month '01': {100*day.get('01',0)/dtot:.1f}%"
          f"   time '00:00:00': {100*tod.get('00:00:00',0)/max(sum(tod.values()),1):.1f}%")
    if day.get("01", 0) / dtot > 0.9:
        print("    The clock is MONTH-GRANULAR: TripAdvisor publishes the month, not the")
        print("    day. Ordering is correct but heavily tied.")

    print("\n  NEXT\n")
    print(f"    python preprocess/hotelrec.py prep --src {a.src} --out data/hotelrec.csv\n")


# ------------------------------------------------------------------ prep
def cmd_prep(a):
    if not os.path.exists(a.src):
        sys.exit(f"\nERROR: {a.src} does not exist.")
    if os.path.exists(a.out) and not a.force:
        sys.exit(f"\nERROR: {a.out} exists. Pass --force to overwrite.")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sz = os.path.getsize(a.src)
    print(f"\n  PREP  {a.src} ({human(sz)})  ->  {a.out}")
    validate_parser(a.src, a.sample_validate)

    uc, ic = Coder(), Coder()
    drop = collections.Counter()
    hist = collections.Counter()
    times = TimeBuf()
    per_item = collections.Counter()
    per_user = collections.Counter()
    read = written = placeholders = noid = 0
    tmin = tmax = None
    t0 = time.time()

    print(f"\n  ONE PASS over {human(sz)}\n")
    with open(a.src, "rb") as f, open(a.out, "w", buffering=1 << 20, newline="") as out:
        out.write("user_id,item_id,rating,timestamp\n")
        for line in f:
            read += 1
            if a.limit and read > a.limit:
                read -= 1
                break
            got = fast_fields(line)
            if got is None:
                try:
                    d = json.loads(line)
                    got = ((d.get("hotel_url") or "").encode(),
                           str(d.get("author")).encode(),
                           (d.get("date") or "").encode(),
                           str(d.get("rating")).encode())
                except Exception:
                    drop["unparseable record"] += 1
                    continue
            url, author, date, rating = got

            au = author.decode("utf-8", "replace")
            if au in PLACEHOLDER_USERS:
                placeholders += 1
                if not a.keep_placeholder_users:
                    drop["placeholder author ('/undefined' or 'null')"] += 1
                    continue
            m = DID.search(url)
            if m:
                it = m.group(1).decode()
            else:
                noid += 1
                it = url.decode("utf-8", "replace")
                if not it:
                    drop["blank hotel_url"] += 1
                    continue
            try:
                r = float(rating)
            except ValueError:
                drop["unparseable rating"] += 1
                continue
            if not (1.0 <= r <= 5.0):
                drop["rating outside [1,5]"] += 1
                continue
            t = parse_iso(date)
            if t is None:
                drop["unparseable date"] += 1
                continue
            if not (0 < t < 4_000_000_000):
                drop["date outside a plausible range"] += 1
                continue

            uid, iid = uc(au), ic(it)
            out.write(f"{uid},{iid},{r:g},{t}\n")
            times.append(t)
            hist[r] += 1
            if r >= 4.0:
                per_item[iid] += 1
                per_user[uid] += 1
            tmin = t if tmin is None or t < tmin else tmin
            tmax = t if tmax is None or t > tmax else tmax
            written += 1
            if read % 2_000_000 == 0:
                el = time.time() - t0
                print(f"    {read:,} records  {written:,} written  "
                      f"{el:.0f}s  ({100*f.tell()/sz:.1f}% of the file)", flush=True)

    nusers, nitems = len(uc.m), len(ic.m)
    del uc, ic
    tarr = times.finish()

    # ---------------- report
    print(f"\n  read {read:,} records in {(time.time()-t0)/60:.0f} min")
    if drop:
        print("  dropped:")
        for k, v in drop.most_common():
            print(f"    {v:14,}  {k}")
    print(f"  wrote {written:,} rows   users {nusers:,}   hotels {nitems:,}")
    print(f"    {written/max(nusers,1):.2f} rows/user   {written/max(nitems,1):.1f} rows/hotel")
    if abs(read - PUB_REVIEWS) / PUB_REVIEWS < 0.10:
        print(f"    record count is within 10% of the published ~{PUB_REVIEWS:,}")
    else:
        print(f"    !! published count is ~{PUB_REVIEWS:,}; this file has {read:,}."
              " Check the download.")
    print(f"  placeholder-author rows seen: {placeholders:,}"
          f"  ({'KEPT' if a.keep_placeholder_users else 'dropped'})")
    if noid:
        print(f"  records with no -dNNN- hotel id, keyed by full URL: {noid:,}")

    tot = sum(hist.values()) or 1
    print("\n  rating histogram:")
    for r in sorted(hist):
        bar = "#" * int(40 * hist[r] / max(hist.values()))
        print(f"    {r:>4.0f}  {hist[r]:14,}  {100*hist[r]/tot:5.1f}%  {bar}")
    ge4 = sum(v for r, v in hist.items() if r >= 4.0)
    print(f"    at >= 4 : {ge4:,} ({100*ge4/tot:.1f}%)   what --min-rating 4.0 keeps")

    # ---------------- the month-granularity diagnostic
    print(f"\n  CLOCK  {ymd(tmin)} .. {ymd(tmax)}  "
          f"({(tmax-tmin)/86400/365.25:.1f} years)")
    ndistinct = int(np.unique(tarr).size)
    print(f"    distinct timestamp values: {ndistinct:,} over {written:,} rows"
          f"   ({written/max(ndistinct,1):,.0f} rows per value)")
    del tarr

    key_max = (nusers - 1) * nitems + (nitems - 1) if nusers and nitems else 0
    print(f"\n  duplicate-key precision: max key {key_max:,}   2^53 = {MAX_EXACT:,}"
          f"   -> {'SAFE' if key_max < MAX_EXACT else 'UNSAFE'}")
    if key_max >= MAX_EXACT:
        sys.exit("\nERROR: the pipeline's float64 collapse key would alias distinct pairs.")

    ni, nu = len(per_item), len(per_user)
    print(f"\n  positives at >= 4 : {ge4:,}")
    if ni:
        print(f"    {ge4/ni:.1f} per hotel over {ni:,} hotels   (study range 16.3 to 672)")
    if nu:
        print(f"    {ge4/nu:.2f} per user over {nu:,} users   (study range 3.11 to 119)")
    thin_u = sum(1 for c in per_user.values() if c < 5)
    thin_i = sum(1 for c in per_item.values() if c < 5)
    print("\n  FIVE-CORE EXPOSURE (one pass; the pipeline iterates, precheck measures it)")
    print(f"    users with fewer than 5 positives:  {thin_u:,} of {nu:,} "
          f"({100*thin_u/max(nu,1):.1f}%)")
    print(f"    hotels with fewer than 5 positives: {thin_i:,} of {ni:,} "
          f"({100*thin_i/max(ni,1):.1f}%)")

    warm = 0.75
    n_test = int(ge4 * 0.10 * warm)
    steps = int(ge4 * 0.90 * warm / 4096)
    mins = ML_MINUTES * (ge4 * 0.9 * warm) / ML_PAIRS
    print(f"\n  FORECAST (warm-start share assumed {warm:.0%}; precheck measures it)")
    print(f"    test pairs ~{n_test:,}   n_cert {min(110_000, max(n_test,1)):,}"
          f"   h ~{hw(min(110_000, max(n_test,1))):.4f}   (powered four: 0.1103)")
    print(f"    steps/epoch ~{steps:,} at batch 4096   (ML-20M 2,197; Games 144)")
    rt = f"{mins/60:.1f} hours" if mins < 2880 else f"{mins/1440:.1f} days"
    print(f"    runtime ~{rt} for the stage list   (ML-20M took 9.7 hours)")

    # ---------------- manifest
    h = hashlib.sha256()
    with open(a.out, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    man = dict(
        dataset="hotelrec", source=a.src, repository=REPO,
        citation="Antognini and Faltings, HotelRec: a Novel Very Large-Scale Hotel "
                 "Recommendation Dataset, LREC 2020",
        licence="academic research use only",
        published_reviews=PUB_REVIEWS, records_read=read, rows_written=written,
        user_field="author", item_field="hotel id (-dNNN- extracted from hotel_url)",
        records_without_hotel_id=noid,
        placeholder_author_rows=placeholders,
        placeholder_policy=("kept" if a.keep_placeholder_users else
                            "dropped: '/undefined' and 'null' are TripAdvisor "
                            "placeholders for deleted or anonymous reviewers, not users"),
        rating_field="rating", rating_scale="1-5 integers, used unchanged",
        subratings_used=False,
        subratings_note="property_dict holds sleep quality, value, rooms, service, "
                        "cleanliness and location. Not used: an average of sub-ratings "
                        "would be a different construct from every other dataset, and "
                        "the field is absent from about 29% of records.",
        time_field="date", time_unit_written="seconds",
        time_span=[ymd(tmin), ymd(tmax)] if tmin is not None else None,
        distinct_timestamps=ndistinct,
        clock_granularity="month: TripAdvisor publishes the month, so ~99.6% of dates "
                          "are day 01 at 00:00:00",
        users=nusers, items=nitems, positives_ge4=ge4,
        sampling="none; the full corpus",
        rating_threshold_applied=False, kcore_applied=False, dedup_applied=False,
        ids="dense integer recoding, first appearance order",
        dropped=dict(drop), sha256=h.hexdigest(),
        bytes=os.path.getsize(a.out),
        elapsed_seconds=round(time.time() - t0, 1),
        made=datetime.datetime.now().isoformat(timespec="seconds"))
    with open(a.out + ".manifest.json", "w") as f:
        json.dump(man, f, indent=2)
    print(f"\n  wrote {a.out}  ({human(os.path.getsize(a.out))})")
    print(f"  wrote {a.out}.manifest.json   sha256 {man['sha256'][:16]}...")
    print(f"  total elapsed {(time.time()-t0)/60:.0f} min\n")


ap = argparse.ArgumentParser(description=__doc__,
                             formatter_class=argparse.RawDescriptionHelpFormatter)
sub = ap.add_subparsers(dest="cmd", required=True)

p = sub.add_parser("inspect", help="sample the file: parser check, authors, clock, ratings")
p.add_argument("--src", default="data/HotelRec.txt")
p.add_argument("--sample", type=int, default=300_000)
p.add_argument("--sample-validate", type=int, default=20_000)
p.set_defaults(fn=cmd_inspect)

p = sub.add_parser("prep", help="convert the full file to the CSV the pipeline reads")
p.add_argument("--src", default="data/HotelRec.txt")
p.add_argument("--out", required=True)
p.add_argument("--keep-placeholder-users", action="store_true",
               help="keep rows whose author is '/undefined' or 'null'. Off by default: "
                    "they are placeholders, not users, and keeping them creates one "
                    "account with about a million reviews.")
p.add_argument("--sample-validate", type=int, default=20_000)
p.add_argument("--limit", type=int, default=0, help="stop after N records (a smoke test)")
p.add_argument("--force", action="store_true")
p.set_defaults(fn=cmd_prep)

a = ap.parse_args()
a.fn(a)
