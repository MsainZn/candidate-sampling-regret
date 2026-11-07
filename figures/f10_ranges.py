"""Rebuild figs/f10_ranges.pdf: how far the contrastive score moves across sampler choices, before and
after subtracting the popularity surrogate, from the archived random-split runs. Same palette, rcParams
and marker encoding as f1_f2_crossover_and_sweep.py (filled circle: raw score; open circle: adjusted
score), one row per dataset as in f1_crossover.

  (a) matched sweep: one practitioner moves training sampler and candidate law together, alpha in
      {0, 0.25, ..., 1.25}; range of the two-seed mean over the six settings
  (b) uniform-trained model, frozen, scored under alpha in {0, 0.5, 1}; range of the four-seed mean
  (c) popularity-trained model, the same
The surrogate-adjusted score is L_hat minus the training-popularity surrogate of the same candidate law.
Arrows point from the raw to the surrogate-adjusted range. The grey band marks ranges below the paired
half-width 2h (3.18 points).

Usage (from the recommender folder):
    python3 figures/f10_ranges.py [--bundle results/aggregated] [--out results/figures] [--png]
"""
import argparse, csv, math, os, statistics as st
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLUE, ORANGE = "#2a78d6", "#eb6834"
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
ap.add_argument("--bundle", default="results/aggregated")
ap.add_argument("--out", default="results/figures")
ap.add_argument("--png", action="store_true")
A = ap.parse_args()
os.makedirs(A.out, exist_ok=True)

TW = 6.27
P = 100.0 / math.log(1024)
DS = ["amazon_books", "hotelrec", "yelp", "google_connecticut", "goodreads_all", "netflix", "ml20m"]
NICE = {"ml20m": "ML-20M", "netflix": "Netflix", "yelp": "Yelp", "amazon_books": "Amazon:Books",
        "goodreads_all": "Goodreads", "google_connecticut": "Google:CT", "hotelrec": "HotelRec"}

def rows(f):
    with open(os.path.join(A.bundle, f), newline="") as fh:
        return [r for r in csv.DictReader(fh) if r["split"] == "random"]

# (a) matched sweep
S4 = {}
for r in rows("all_e4_invariance.csv"):
    S4.setdefault(r["dataset"], {}).setdefault(float(r["alpha"]), []).append(r)
def sweep(d):
    al = sorted(S4[d])
    raw = [P * st.mean(float(r["L_hat"]) for r in S4[d][a]) for a in al]
    adj = [P * st.mean(float(r["adjusted_honest"]) for r in S4[d][a]) for a in al]
    return max(raw) - min(raw), max(adj) - min(adj)

# (b), (c) frozen models under three candidate laws
e2 = rows("all_e2_crosscert.csv")
HON = {(r["dataset"], r["cert_ref"]): P * float(r["honest"]) for r in e2 if r["train_law"] == "[baseline]"}
SC = {}
for r in e2:
    if r["seed"] and r["L_hat"] and r["train_law"] in ("uniform", "popularity"):
        SC.setdefault((r["dataset"], r["train_law"], r["cert_ref"]), []).append(P * float(r["L_hat"]))
LAWS = ["uniform", "sqrt_pop", "popularity"]
def frozen(d, arm):
    raw = [st.mean(SC[(d, arm, s)]) for s in LAWS]
    adj = [x - HON[(d, s)] for x, s in zip(raw, LAWS)]
    return max(raw) - min(raw), max(adj) - min(adj)
TWO_H = 2 * P * float(e2[0]["half_width"])

PANELS = [("(a) Matched sweep", sweep), ("(b) Uniform-trained model", lambda d: frozen(d, "uniform")),
          ("(c) Popularity-trained model", lambda d: frozen(d, "popularity"))]
fig, axes = plt.subplots(1, 3, figsize=(TW, 2.55), sharey=True)
ys = list(range(len(DS)))[::-1]
for ax, (title, fn) in zip(axes, PANELS):
    ax.axvspan(0, TWO_H, color=GREY, alpha=0.22, lw=0, zorder=0)
    for y, d in zip(ys, DS):
        raw, adj = fn(d)
        ax.annotate("", xy=(adj, y), xytext=(raw, y), zorder=2,
                    arrowprops=dict(arrowstyle="-|>", color=GREY, lw=0.9, mutation_scale=8, shrinkA=3.2, shrinkB=3.2))
        ax.plot(raw, y, "o", ms=4.2, color=BLUE, mfc=BLUE, mec=BLUE, mew=0.7, zorder=3)
        ax.plot(adj, y, "o", ms=4.2, color=BLUE, mfc=WHITE, mec=BLUE, mew=0.9, zorder=3)
        print("%-28s %-20s raw %5.1f  adjusted %5.1f" % (title, d, raw, adj))
    ax.set_title(title)
    ax.set_xlim(0, 48)
    ax.set_yticks(ys)
    ax.set_yticklabels([NICE[d] for d in DS])
    ax.set_ylim(-0.6, len(DS) - 0.4)
    ax.grid(axis="x", zorder=0)
    ax.set_axisbelow(True)
fig.supxlabel(r"range of the score across settings (\% of $\log K$)", y=0.05, fontsize=8.5)
leg = fig.legend(handles=[
    Line2D([], [], color=BLUE, marker="o", ls="", ms=4.2, mfc=BLUE, label=r"raw $\widehat{\mathcal{L}}$"),
    Line2D([], [], color=BLUE, marker="o", ls="", ms=4.2, mfc=WHITE, mew=0.9,
           label=r"surrogate-adjusted $\widehat{\mathcal{L}}-\widetilde\Lambda_R$"),
    Patch(facecolor=GREY, alpha=0.22, label=r"below $2h$")],
    loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.07), handletextpad=0.5, columnspacing=2.0)
fig.subplots_adjust(wspace=0.10, bottom=0.2)
fig.savefig(os.path.join(A.out, "f10_ranges.pdf"))
if A.png:
    fig.savefig(os.path.join(A.out, "f10_ranges.png"), dpi=170)
plt.close(fig)
print("2h", TWO_H, "; wrote f10_ranges to", A.out)
