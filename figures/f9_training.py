"""Training behaviour and the main results table, from the archived random-split runs.
No model is trained or rescored: per-epoch losses and times are read from the console logs written
during the published runs (results/<dataset>/console_*.log), and the held-out metrics
from results/aggregated.

  figs/f9_training.pdf   training loss per epoch of the four training configurations, one panel per
                         dataset (four-seed means); same rcParams and layout as f2_alpha_sweep
  --tex DIR              results_table.tex (main results table body), capacity_table rows, and
                         training_numbers.json with every number quoted in the text

Usage (from the recommender folder):
    python3 figures/f9_training.py [--runs results] [--out results/figures] [--tex DIR] [--png]
"""
import argparse, csv, glob, json, math, os, re, statistics as st
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
BLACK, GREY, WHITE, GRID = "#000000", "#9a9a9a", "#ffffff", "#cccccc"
plt.rcParams.update({                       # identical to f1_f2_crossover_and_sweep.py
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage{amsmath}", r"\usepackage{amssymb}"]),
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK,
    "axes.linewidth": 0.7,
    "xtick.color": BLACK, "ytick.color": BLACK,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.top": True, "ytick.right": True,
    "grid.color": GRID, "grid.linewidth": 0.4,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "legend.frameon": False, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default="results")
ap.add_argument("--out", default="results/figures")
ap.add_argument("--tex", default="results/derived")
ap.add_argument("--png", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

TW = 6.27
P = 100.0 / math.log(1024)
DS = ["amazon_books", "hotelrec", "yelp", "google_connecticut", "goodreads_all", "netflix", "ml20m"]
NICE = {"ml20m": "ML-20M", "netflix": "Netflix", "yelp": "Yelp", "amazon_books": "Amazon:Books",
        "goodreads_all": "Goodreads", "google_connecticut": "Google:CT", "hotelrec": "HotelRec"}
TEXN = {"ml20m": "ML-20M", "netflix": "Netflix", "yelp": "Yelp", "amazon_books": "Amazon:Books",
        "goodreads_all": "Goodreads", "google_connecticut": "Google:CT", "hotelrec": "HotelRec"}
ARMS = ["uniform", "inbatch_logq", "popularity", "inbatch"]
ARM_TEX = {"uniform": "U", "popularity": "P", "inbatch": "B", "inbatch_logq": "Q"}
HW_RANK = 1.2452     # simultaneous paired half-width for NDCG@20 / HR@10 terms in [0,1], points (f8_decision.py)

# ---------------------------------------------------------------- per-epoch logs (random split only)
EP = {}          # (dataset, stage, arm, seed) -> [(epoch, loss, seconds)]
DEVICE = {}
for d in DS:
    f = sorted(glob.glob(os.path.join(A.runs, d, "console_*.log")))[-1]
    cur = None
    for line in open(f):
        m = re.search(r"device=(\S+)", line)
        if m and d not in DEVICE:
            DEVICE[d] = m.group(1)
        m = re.search(r"\[(\S+?)\] training", line)
        if m:
            cur = m.group(1)
            continue
        m = re.search(r"epoch (\d+)/(\d+) loss=([\d.]+) \((\d+)s\)", line)
        if m and cur and "_random_" in cur:
            g = re.match(r".+?_random_(e1|e3b|e4)_(.+?)_s(\d+)_", cur)
            EP.setdefault((d, g.group(1), g.group(2), int(g.group(3))), []).append(
                (int(m.group(1)), float(m.group(3)), int(m.group(4))))
for k, v in EP.items():
    assert [e for e, _, _ in v] == list(range(1, len(v) + 1)), k
E1 = {(k[0], k[2], k[3]): v for k, v in EP.items() if k[1] == "e1"}
assert len(E1) == len(DS) * len(ARMS) * 4 and all(len(v) == 6 for v in E1.values())

def curve(d, a):
    return [st.mean(E1[(d, a, s)][e][1] for s in range(4)) for e in range(6)]

def final(d, a):
    return st.mean(E1[(d, a, s)][-1][1] for s in range(4))

def secs(d, a):
    return st.mean(x[2] for s in range(4) for x in E1[(d, a, s)])

# ---------------------------------------------------------------- held-out metrics (random split)
def rows(name):
    with open(os.path.join(A.runs, "aggregated", name), newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "random"]

MET = {}
for r in rows("all_e2_crosscert.csv"):
    if r["seed"] and r["L_hat"] and r["train_law"] in ARMS:
        for m, scale in (("L_hat", P), ("ndcg20", 100), ("hr10", 100)):
            MET.setdefault((r["dataset"], r["train_law"], r["cert_ref"], m), []).append(scale * float(r[m]))
assert all(len(v) == 4 for v in MET.values())
SD = {k: st.pstdev(v) for k, v in MET.items()}
MET = {k: st.mean(v) for k, v in MET.items()}
TWO_H = 2 * P * float(rows("all_e2_crosscert.csv")[0]["half_width"])

CAP = {}
for r in rows("all_e3b_capacity.csv"):
    for m, scale in (("L_hat", P), ("ndcg20", 100)):
        CAP[(r["dataset"], r["train_law"], r["cert_ref"], m)] = scale * float(r[m])
TWO_H_CAP = 2 * P * float(rows("all_e3b_capacity.csv")[0]["half_width"])

# ---------------------------------------------------------------- figure: training loss per epoch
STYLE = {"uniform": (BLUE, "o", "-", BLUE), "inbatch_logq": (BLUE, "o", (0, (1.8, 1.4)), WHITE),
         "popularity": (ORANGE, "s", "-", ORANGE), "inbatch": (ORANGE, "s", (0, (1.8, 1.4)), WHITE)}
LABEL = {"uniform": "uniform training", "inbatch_logq": "corrected in-batch training",
         "popularity": "popularity training", "inbatch": "in-batch training"}
NCOL, FH = 4, 0.95 + 1.95 * 2
fig = plt.figure(figsize=(TW, FH))
gs = GridSpec(2, 2 * NCOL, figure=fig)
axes = [fig.add_subplot(gs[0, 2 * j:2 * j + 2]) for j in range(4)]
axes += [fig.add_subplot(gs[1, 2 * j + 1:2 * j + 3]) for j in range(3)]
ep = list(range(1, 7))
for ax, d in zip(axes, DS):
    for a in ARMS:
        col, mk, ls, mfc = STYLE[a]
        ax.plot(ep, curve(d, a), color=col, marker=mk, ms=3.0, lw=1.1, ls=ls, mfc=mfc, mec=col, mew=0.7, zorder=3)
    ax.set_title(NICE[d], pad=3.5)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.6, 6.4)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(4))
    ax.tick_params(axis="y", pad=2)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
axes[0].set_ylabel("training loss (nats)", labelpad=2)
axes[4].set_ylabel("training loss (nats)", labelpad=2)
fig.supxlabel("epoch", y=0.42 / FH, fontsize=9)
leg = fig.legend(handles=[Line2D([], [], color=STYLE[a][0], marker=STYLE[a][1], ms=3.0, mfc=STYLE[a][3],
                                 lw=1.1, ls=STYLE[a][2], label=LABEL[a]) for a in ARMS],
                 loc="upper center", bbox_to_anchor=(0.5, 0.26 / FH), ncol=4, handletextpad=0.6, columnspacing=2.0,
                 frameon=True, fancybox=False, facecolor=WHITE, edgecolor=BLACK, framealpha=1.0)
leg.get_frame().set_linewidth(0.6)
fig.subplots_adjust(wspace=0.95, hspace=0.36, bottom=0.78 / FH)
fig.savefig(os.path.join(A.out, "f9_training.pdf"))
if A.png:
    fig.savefig(os.path.join(A.out, "f9_training.png"), dpi=170)
plt.close(fig)
print("wrote f9_training to", A.out)

# ---------------------------------------------------------------- numbers and table
def fmt(x):
    return f"{x:.1f}" if abs(x) >= 10 else f"{x:.2f}"

# Goodreads ran on a different machine from the other six datasets. Its seconds per epoch in Table 4 are
# those predicted for the machine that ran the other six: a per-pair cost linear in the embedding-table
# rows (N + M of the random-split training set) is fitted to their uniform epochs, and each configuration
# adds its mean per-pair overhead over uniform on those six datasets.
SIZE = {}
for d in DS:
    t = open(sorted(glob.glob(os.path.join(A.runs, d, "console_*.log")))[-1]).read()
    m = re.search(r"=== E1 \[random\]\s+train ([\d,]+) test [\d,]+ users\(N\)=([\d,]+) items\(M\)=([\d,]+)", t)
    SIZE[d] = tuple(int(x.replace(",", "")) for x in m.groups())
REF = [d for d in DS if DEVICE[d] == "mps"]
def _fit():
    xs = [(SIZE[d][1] + SIZE[d][2]) / 1e6 for d in REF]
    ys = [secs(d, "uniform") / SIZE[d][0] for d in REF]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return my - b * mx, b
C0, C1 = _fit()
OVER = {a: st.mean((secs(d, a) - secs(d, "uniform")) / SIZE[d][0] for d in REF) for a in ARMS}
def table_secs(d, a):
    if d in REF:
        return secs(d, a)
    n, N, M = SIZE[d]
    return (C0 + C1 * (N + M) / 1e6) * n + OVER[a] * n

def results_table():
    """Table 4 body: training loss after the last epoch, mean seconds per epoch (table_secs), and the contrastive score
    (% of log K) under the three candidate laws; bold marks every configuration within 2h of the best
    under that law. NDCG@20 and HR@10 are in the payoff table written by f8_decision.py."""
    out = []
    LAWS = ("uniform", "sqrt_pop", "popularity")
    for d in DS:
        best = {}
        for law in LAWS:
            top = max(MET[(d, a, law, "L_hat")] for a in ARMS)
            best[law] = {a for a in ARMS if top - MET[(d, a, law, "L_hat")] <= TWO_H}
        for i, a in enumerate(["uniform", "popularity", "inbatch", "inbatch_logq"]):
            cells = [TEXN[d] if i == 0 else "", ARM_TEX[a], f"{final(d, a):.2f}", f"{table_secs(d, a):.0f}"]
            for law in LAWS:
                x = fmt(MET[(d, a, law, "L_hat")])
                cells.append(r"\textbf{" + x + "}" if a in best[law] else x)
            out.append(" & ".join(cells) + r" \\")
        out.append(r"\midrule")
    return "\n".join(out[:-1])

Q = {"device": DEVICE, "two_h": TWO_H, "two_h_capacity": TWO_H_CAP}
Q["final_loss"] = {f"{d}|{a}": final(d, a) for d in DS for a in ARMS}
Q["sec_per_epoch"] = {f"{d}|{a}": secs(d, a) for d in DS for a in ARMS}
Q["last_epoch_drop"] = {d: st.mean(curve(d, a)[4] - curve(d, a)[5] for a in ARMS) for d in DS}
Q["first_to_last_drop"] = {d: st.mean(curve(d, a)[0] - curve(d, a)[5] for a in ARMS) for d in DS}
Q["logq_highest_final_loss"] = sum(1 for d in DS if max(ARMS, key=lambda a: final(d, a)) == "inbatch_logq")
Q["uniform_lowest_final_loss"] = sum(1 for d in DS if min(ARMS, key=lambda a: final(d, a)) == "uniform")
Q["pop_minus_inbatch_final_loss_max"] = max(abs(final(d, "popularity") - final(d, "inbatch")) for d in DS)
Q["table_sec_per_epoch"] = {f"{d}|{a}": table_secs(d, a) for d in DS for a in ARMS}
Q["inbatch_time_ratio"] = {d: table_secs(d, "inbatch") / table_secs(d, "uniform") for d in DS}
Q["logq_time_ratio"] = {d: table_secs(d, "inbatch_logq") / table_secs(d, "uniform") for d in DS}
Q["popularity_time_ratio"] = {d: table_secs(d, "popularity") / table_secs(d, "uniform") for d in DS}
Q["metrics"] = {f"{d}|{a}|{law}|{m}": MET[(d, a, law, m)] for d in DS for a in ARMS
                for law in ("uniform", "sqrt_pop", "popularity") for m in ("L_hat", "ndcg20", "hr10")}
Q["max_seed_sd_ndcg"] = max(v for k, v in SD.items() if k[3] == "ndcg20")
Q["logq_tied_best_ndcg_uniform"] = sum(1 for d in DS if max(MET[(d, a, "uniform", "ndcg20")] for a in ARMS)
                                       - MET[(d, "inbatch_logq", "uniform", "ndcg20")] <= HW_RANK)
cap = {}
for d in DS:
    g = lambda a, law, m: CAP[(d, a, law, m)]
    cap[d] = {"ndcg_gap_uniform": g("uniform", "uniform", "ndcg20") - g("popularity", "uniform", "ndcg20"),
              "ndcg_gap_popularity": g("uniform", "popularity", "ndcg20") - g("popularity", "popularity", "ndcg20"),
              "score_gap_uniform": g("uniform", "uniform", "L_hat") - g("popularity", "uniform", "L_hat"),
              "score_gap_popularity": g("uniform", "popularity", "L_hat") - g("popularity", "popularity", "L_hat"),
              "ndcg_gap_uniform_6ep": MET[(d, "uniform", "uniform", "ndcg20")] - MET[(d, "popularity", "uniform", "ndcg20")],
              "ndcg_gap_popularity_6ep": MET[(d, "uniform", "popularity", "ndcg20")] - MET[(d, "popularity", "popularity", "ndcg20")],
              "logq_minus_uniform_ndcg_uniform": g("inbatch_logq", "uniform", "ndcg20") - g("uniform", "uniform", "ndcg20")}
Q["capacity"] = cap
for k, v in Q.items():
    if not isinstance(v, dict) or k in ("device",):
        print(k, v)
print("last-epoch drop", {d: round(v, 3) for d, v in Q["last_epoch_drop"].items()})
print("time ratios inbatch", {d: round(v, 2) for d, v in Q["inbatch_time_ratio"].items()})
print("time ratios pop", {d: round(v, 2) for d, v in Q["popularity_time_ratio"].items()})
for d in DS:
    print(d, {k: round(v, 1) for k, v in cap[d].items()})
if A.tex:
    os.makedirs(A.tex, exist_ok=True)
    open(os.path.join(A.tex, "results_table.tex"), "w").write(results_table() + "\n")
    json.dump(Q, open(os.path.join(A.tex, "training_numbers.json"), "w"), indent=1)
    print("wrote results_table.tex and training_numbers.json to", A.tex)
