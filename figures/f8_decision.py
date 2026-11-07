"""Rebuild figs/f8_regret.pdf, and optionally the decision tables, from the archived
random-split cross-certification runs (all_e2_crosscert.csv). No model is trained or rescored.

The selection decision: four fitted training configurations (uniform, popularity, in-batch,
in-batch + log Q) are the alternatives, the three certification references R_alpha with
alpha in {0, 0.5, 1} are the candidate laws, and sampled NDCG@20 (or HR@10) is the payoff.
For configuration a under law s, the regret is r(a, s) = max_b v(b, s) - v(a, s), with v the
four-seed mean in percent. Three rules for an uncertain law are compared: minimax regret, maximin
over raw payoffs, and expected payoff (equal weights, and the break-even weight on the popularity
law between the best uniform-like and the best popularity-like configuration). Differences are judged against the simultaneous paired half-width
for per-positive terms in [0, 1]: 2 * sqrt(log(2m / delta) / (2n)), with n = 110,000 scored
positives, delta = 0.05 and m = 7 datasets x 3 laws x 6 configuration pairs = 126.

Usage (from the recommender folder):
    python3 figures/f8_decision.py [--bundle results/aggregated] [--out results/figures]
                                   [--tex DIR] [--png]
"""
import argparse, csv, json, math, os, statistics as st
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt

BLUE, ORANGE = "#2a78d6", "#eb6834"
LBLUE, LORANGE = "#8fbbe9", "#f4b08e"
BLACK, GREY, WHITE, GRID = "#000000", "#9a9a9a", "#ffffff", "#cccccc"
plt.rcParams.update({
    "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([r"\usepackage{amsmath}", r"\usepackage{amssymb}"]),
    "font.family": "serif", "font.size": 8,
    "axes.titlesize": 9, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": BLACK, "axes.labelcolor": BLACK, "axes.edgecolor": BLACK,
    "axes.linewidth": 0.7, "xtick.color": BLACK, "ytick.color": BLACK,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "xtick.major.size": 2.6, "ytick.major.size": 2.6,
    "xtick.top": False, "ytick.right": True,
    "grid.color": GRID, "grid.linewidth": 0.4,
    "figure.facecolor": WHITE, "axes.facecolor": WHITE, "savefig.facecolor": WHITE,
    "legend.frameon": False, "lines.linewidth": 1.1,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})

ap = argparse.ArgumentParser()
ap.add_argument("--bundle", default="results/aggregated")
ap.add_argument("--out", default="results/figures")
ap.add_argument("--tex", default="results/derived", help="also write the LaTeX table bodies and a JSON of the quoted numbers here")
ap.add_argument("--png", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

TW = 6.27
DS = ["amazon_books", "hotelrec", "yelp", "google_connecticut", "goodreads_all", "netflix", "ml20m"]
NICE = {"amazon_books": "Amazon:Books", "hotelrec": "HotelRec", "yelp": "Yelp",
        "google_connecticut": "Google:CT", "goodreads_all": "Goodreads", "netflix": "Netflix", "ml20m": "ML-20M"}
ARMS = ["uniform", "popularity", "inbatch", "inbatch_logq"]
ARM_TEX = {"uniform": "U", "popularity": "P", "inbatch": "B", "inbatch_logq": "Q"}
ARM_LONG = {"uniform": "uniform", "popularity": "popularity", "inbatch": "in-batch", "inbatch_logq": r"in-batch $+\log Q$"}
LAWS = ["uniform", "sqrt_pop", "popularity"]
LAW_TEX = {"uniform": r"Uniform ($\alpha=0$)", "sqrt_pop": r"$\sqrt{\text{popularity}}$ ($\alpha=0.5$)", "popularity": r"Popularity ($\alpha=1$)"}
N, DELTA = 110000, 0.05
M = len(DS) * len(LAWS) * (len(ARMS) * (len(ARMS) - 1) // 2)
HW = 100 * 2 * math.sqrt(math.log(2 * M / DELTA) / (2 * N))    # percentage points

with open(os.path.join(A.bundle, "all_e2_crosscert.csv"), newline="") as fh:
    rows = [r for r in csv.DictReader(fh)
            if r["split"] == "random" and r["train_law"] in ARMS and r["ndcg20"]]
V = {}
for metric in ["ndcg20", "hr10"]:
    acc = {}
    for r in rows:
        acc.setdefault((r["dataset"], r["train_law"], r["cert_ref"]), []).append(float(r[metric]))
    assert all(len(v) == 4 for v in acc.values()) and len(acc) == len(DS) * len(ARMS) * len(LAWS)
    V[metric] = {k: 100 * st.mean(v) for k, v in acc.items()}


def analyse(metric):
    v = V[metric]
    best = {(d, s): max(v[(d, a, s)] for a in ARMS) for d in DS for s in LAWS}
    reg = {(d, a, s): best[(d, s)] - v[(d, a, s)] for d in DS for a in ARMS for s in LAWS}
    tied = {(d, s): [a for a in ARMS if reg[(d, a, s)] <= HW] for d in DS for s in LAWS}
    worst = {(d, a): max(reg[(d, a, s)] for s in LAWS) for d in DS for a in ARMS}
    mmr = {d: min(worst[(d, a)] for a in ARMS) for d in DS}
    # every regret is a paired difference under one law, so all regrets lie within HW of their population
    # values simultaneously; two worst-case regrets (possibly taken under different laws) are therefore
    # separated only when they differ by more than 2 * HW
    mm_set = {d: [a for a in ARMS if worst[(d, a)] - mmr[d] <= 2 * HW] for d in DS}
    # sensitivity of the minimax-regret set to the declared laws: drop one law at a time
    def mm_subset(laws):
        w = {(d, a): max(reg[(d, a, s)] for s in laws) for d in DS for a in ARMS}
        return {d: [a for a in ARMS if w[(d, a)] - min(w[(d, b)] for b in ARMS) <= 2 * HW] for d in DS}
    mm_drop = {s: mm_subset([t for t in LAWS if t != s]) for s in LAWS}
    # selecting the best configuration under law s, then judging it under law t
    pick = {(d, s): max(ARMS, key=lambda a: v[(d, a, s)]) for d in DS for s in LAWS}
    loss = {(s, t): [reg[(d, pick[(d, s)], t)] for d in DS] for s in LAWS for t in LAWS}
    rev = sum(1 for d in DS if v[(d, "uniform", "uniform")] > v[(d, "popularity", "uniform")]
              and v[(d, "popularity", "popularity")] > v[(d, "uniform", "popularity")])
    # the other two classical rules: maximin over raw payoffs and expected payoff with equal weights
    low = {(d, a): min(v[(d, a, s)] for s in LAWS) for d in DS for a in ARMS}
    # maximin compares levels that may sit under different laws. Each level lies within HW / 2 of its
    # population value simultaneously (a [0,1] term, fewer statements than the paired ones), so two
    # lowest levels are separated only when they differ by more than HW, the tolerance used below
    maximin = {d: max(ARMS, key=lambda a: low[(d, a)]) for d in DS}
    ev = {(d, a): st.mean(v[(d, a, s)] for s in LAWS) for d in DS for a in ARMS}
    ev_pick = {d: max(ARMS, key=lambda a: ev[(d, a)]) for d in DS}
    UL, PL = ("uniform", "inbatch_logq"), ("popularity", "inbatch")
    ev_margin = {d: max(ev[(d, a)] for a in UL) - max(ev[(d, a)] for a in PL) for d in DS}
    # break-even weight w on the popularity law (1 - w on the uniform law) at which the best
    # popularity-like configuration first matches the best uniform-like one in expected payoff
    def gap(d, w):
        e = lambda a: (1 - w) * v[(d, a, "uniform")] + w * v[(d, a, "popularity")]
        return max(e(a) for a in UL) - max(e(a) for a in PL)
    breakeven = {d: next(i / 100000 for i in range(100001) if gap(d, i / 100000) <= 0) for d in DS}
    drop = min(v[(d, a, "uniform")] - v[(d, a, "popularity")] for d in DS for a in ARMS)
    return dict(v=v, reg=reg, tied=tied, worst=worst, mmr=mmr, mm_set=mm_set, pick=pick, loss=loss, rev=rev,
                maximin=maximin, ev_pick=ev_pick, ev_margin=ev_margin, breakeven=breakeven, drop=drop,
                mm_drop=mm_drop)


R = {m: analyse(m) for m in ["ndcg20", "hr10"]}


def decision_table(metric):
    v, tied = R[metric]["v"], R[metric]["tied"]
    out = []
    for d in DS:
        cells = []
        for s in LAWS:
            for a in ARMS:
                x = f"{v[(d, a, s)]:.1f}" if v[(d, a, s)] >= 10 else f"{v[(d, a, s)]:.2f}"   # two decimals below 10, as in the original reversal table
                cells.append(r"\textbf{" + x + "}" if a in tied[(d, s)] else x)
        out.append(NICE[d] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def loss_table(metric):
    loss = R[metric]["loss"]
    out = []
    for s in LAWS:
        cells = []
        for t in LAWS:
            xs = loss[(s, t)]
            cells.append("--" if s == t else f"{st.mean(xs):.1f} ({max(xs):.1f})")
        out.append(LAW_TEX[s] + " & " + " & ".join(cells) + r" \\")
    return "\n".join(out)


def summary_table(metric):
    """One row per dataset: worst-case regret of each configuration (bold: the minimax-regret set,
    within 2 * HW of the smallest worst-case regret),
    the configurations chosen by maximin and by equal-weight expected payoff (ties within the
    half-width listed together), and the break-even weight on the popularity law."""
    x, v = R[metric], V[metric]
    out = []
    for d in DS:
        low = {a: min(v[(d, a, s)] for s in LAWS) for a in ARMS}
        ev = {a: st.mean(v[(d, a, s)] for s in LAWS) for a in ARMS}
        mm = [a for a in ARMS if max(low.values()) - low[a] <= HW]
        ew = [a for a in ARMS if max(ev.values()) - ev[a] <= HW]
        cells = [NICE[d]]
        for a in ["uniform", "popularity", "inbatch", "inbatch_logq"]:
            w = x["worst"][(d, a)]
            s = f"{w:.1f}" if w >= 10 else f"{w:.2f}"
            cells.append(r"\textbf{" + s + "}" if a in x["mm_set"][d] else s)
        order = ["uniform", "popularity", "inbatch", "inbatch_logq"]
        cells.append(", ".join(ARM_TEX[a] for a in order if a in mm))
        cells.append(", ".join(ARM_TEX[a] for a in order if a in ew))
        cells.append(f"{x['breakeven'][d]:.2f}")
        out.append(" & ".join(cells) + r" \\")
    return "\n".join(out)


# ------------------------------------------------------------------ figure: worst-case regret
r = R["ndcg20"]
fig, ax = plt.subplots(figsize=(TW, 2.45))
w = 0.19
cols = {"uniform": BLUE, "inbatch_logq": LBLUE, "popularity": ORANGE, "inbatch": LORANGE}
order = ["uniform", "inbatch_logq", "popularity", "inbatch"]
for j, a in enumerate(order):
    xs = [i + (j - 1.5) * w for i in range(len(DS))]
    ax.bar(xs, [r["worst"][(d, a)] for d in DS], width=w * 0.92, color=cols[a],
           label=ARM_LONG[a], zorder=2)
ax.axhline(HW, color=GREY, lw=0.9, ls=(0, (3, 2)), zorder=3, label=r"paired half-width")
ax.set_xticks(range(len(DS)))
ax.set_xticklabels([NICE[d].replace(":", r"\:") for d in DS])
ax.set_xlim(-0.6, len(DS) - 0.4)
ax.set_ylabel(r"worst-case NDCG@20 regret (points)")
ax.yaxis.grid(True, zorder=0)
ax.set_axisbelow(True)
ax.tick_params(axis="x", length=0)
ax.legend(ncol=5, loc="lower left", bbox_to_anchor=(0.0, 1.0), handlelength=1.4, columnspacing=1.0, fontsize=7)
fig.savefig(os.path.join(A.out, "f8_regret.pdf"))
if A.png:
    fig.savefig(os.path.join(A.out, "f8_regret.png"), dpi=170)
plt.close(fig)
print("wrote f8_regret to", A.out)

if A.tex:
    os.makedirs(A.tex, exist_ok=True)
    for m in ["ndcg20", "hr10"]:
        open(os.path.join(A.tex, f"decision_{m}.tex"), "w").write(decision_table(m) + "\n")
        open(os.path.join(A.tex, f"loss_{m}.tex"), "w").write(loss_table(m) + "\n")
        open(os.path.join(A.tex, f"decision_summary_{m}.tex"), "w").write(summary_table(m) + "\n")
    def merged(fn):
        a, b = fn("ndcg20").split("\n"), fn("hr10").split("\n")
        out = []
        for ra, rb in zip(a, b):
            name, rest_a = ra.split(" & ", 1)
            _, rest_b = rb.split(" & ", 1)
            out += [name + " & NDCG@20 & " + rest_a, " & HR@10 & " + rest_b, r"\midrule"]
        return "\n".join(out[:-1])
    open(os.path.join(A.tex, "payoff_merged.tex"), "w").write(merged(decision_table) + "\n")
    open(os.path.join(A.tex, "summary_merged.tex"), "w").write(merged(summary_table) + "\n")
    q = {"half_width": HW, "m": M}
    for m in ["ndcg20", "hr10"]:
        x = R[m]
        q[m] = {"reversals": x["rev"],
                "tied_best": {f"{d}|{s}": x["tied"][(d, s)] for d in DS for s in LAWS},
                "minimax_set": x["mm_set"], "minimax_regret": x["mmr"],
                "worst": {f"{d}|{a}": x["worst"][(d, a)] for d in DS for a in ARMS},
                "loss_mean_max": {f"{s}->{t}": [st.mean(x["loss"][(s, t)]), max(x["loss"][(s, t)]), min(x["loss"][(s, t)])]
                                  for s in LAWS for t in LAWS if s != t},
                "pick": {f"{d}|{s}": x["pick"][(d, s)] for d in DS for s in LAWS},
                "maximin_pick": x["maximin"], "equal_weight_pick": x["ev_pick"], "equal_weight_margin": x["ev_margin"],
                "breakeven_weight_popularity": x["breakeven"], "min_drop_uniform_to_popularity": x["drop"],
                "minimax_set_without": {s: x["mm_drop"][s] for s in LAWS}}
    json.dump(q, open(os.path.join(A.tex, "decision_numbers.json"), "w"), indent=1)
    print("wrote tables and numbers to", A.tex)
