"""Rebuild figs/f11_reversal.pdf: the reversal of the preferred model as spider charts.

Each spoke is a dataset. The blue polygon joins the four-seed mean of the uniform-trained model and the
orange polygon that of the popularity-trained model. Columns: uniform candidates and popularity
candidates; rows: NDCG@20 (%) and contrastive score (% of log K). Under uniform candidates the blue
polygon encloses the orange one, and under popularity candidates the orange encloses the blue. Same
rcParams, colours and markers as f1_f2_crossover_and_sweep.py.

Usage (from the recommender folder):
    python3 figures/f11_reversal.py [--bundle results/aggregated] [--out results/figures] [--png]
"""
import argparse, csv, math, os, statistics as st
import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

BLACK, GREY, WHITE, GRID = "#000000", "#9a9a9a", "#ffffff", "#cccccc"
DATASET_COLORS = {"amazon_books": "#4c78a8", "hotelrec": "#f58518", "yelp": "#54a24b",   # f3_structure.py
                  "google_connecticut": "#e45756", "goodreads_all": "#72b7b2", "netflix": "#b279a2",
                  "ml20m": "#ff9da6"}
plt.rcParams.update({
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
with open(os.path.join(A.bundle, "all_e2_crosscert.csv"), newline="") as fh:
    rows = [r for r in csv.DictReader(fh) if r["split"] == "random"]
V = {}
for r in rows:
    if r["seed"] and r["L_hat"] and r["train_law"] in ("uniform", "popularity"):
        for m, s in (("L_hat", P), ("ndcg20", 100)):
            V.setdefault((r["dataset"], r["train_law"], r["cert_ref"], m), []).append(s * float(r[m]))
V = {k: st.mean(v) for k, v in V.items()}
import numpy as np
BLUE, ORANGE = "#2a78d6", "#eb6834"
SHORT = {"ml20m": "ML-20M", "netflix": "Netflix", "yelp": "Yelp", "amazon_books": "Amazon:Books",
         "goodreads_all": "Goodreads", "google_connecticut": "Google:CT", "hotelrec": "HotelRec"}
ang = np.linspace(0, 2 * np.pi, len(DS), endpoint=False)
closed = lambda v: list(v) + [v[0]]
ROWS = [("ndcg20", r"NDCG@20 (\%)", {"uniform": (66, [20, 40, 60]), "popularity": (34, [10, 20, 30])}),
        ("L_hat", r"Contrastive score (\% of $\log K$, $\log K=6.93$ nats)",
         {"uniform": (66, [20, 40, 60]), "popularity": (36, [10, 20, 30])})]
COLS = [("uniform", "uniform candidates"), ("popularity", "popularity candidates")]

def web(ax, r, **kw):
    """polygonal grid line at radius r through the spokes"""
    ax.plot(closed(ang), [r] * (len(ang) + 1), **kw)

fig, axes = plt.subplots(2, 2, figsize=(TW, 6.9), subplot_kw=dict(projection="polar"))
for i, (m, rowlab, scale) in enumerate(ROWS):
    for j, (law, coltitle) in enumerate(COLS):
        ax = axes[i][j]
        ax.set_theta_offset(np.pi / 2); ax.set_theta_direction(-1)
        rmax, ticks = scale[law]
        ax.grid(False); ax.spines["polar"].set_visible(False)
        for t in ticks:
            web(ax, t, color=GRID, lw=0.5, zorder=1)
        web(ax, rmax, color=GREY, lw=0.7, zorder=1)
        for th in ang:
            ax.plot([th, th], [0, rmax], color=GRID, lw=0.5, zorder=1)
        for arm, col, mk in (("popularity", ORANGE, "s"), ("uniform", BLUE, "o")):
            v = [V[(d, arm, law, m)] for d in DS]
            ax.plot(closed(ang), closed(v), color=col, lw=1.3, marker=mk, ms=3.6, zorder=3)
            ax.fill(closed(ang), closed(v), color=col, alpha=0.12, zorder=2)
        inner, outer = ("popularity", "uniform") if law == "uniform" else ("uniform", "popularity")
        assert all(V[(d, outer, law, m)] > V[(d, inner, law, m)] for d in DS), (m, law)
        ax.set_xticks(ang); ax.set_xticklabels([SHORT[d] for d in DS], fontsize=8)
        ax.tick_params(axis="x", pad=5)
        ax.set_xlim(0, 2 * np.pi); ax.set_ylim(0, rmax)
        ax.set_yticks(ticks); ax.set_yticklabels([str(t) for t in ticks], fontsize=7, color="#555555")
        ax.set_rlabel_position(360 / 14)
        ax.text(0.5, -0.11, "(%s) %s" % ("abcd"[2 * i + j], coltitle.capitalize()), transform=ax.transAxes,
                ha="center", va="top", fontsize=9)
    y = axes[i][0].get_position().y0
fig.subplots_adjust(left=0.08, right=0.98, top=0.97, bottom=0.12, wspace=0.42, hspace=0.40)
for i, (m, rowlab, scale) in enumerate(ROWS):
    pos = axes[i][0].get_position()
    fig.text(0.015, (pos.y0 + pos.y1) / 2, rowlab, rotation=90, ha="center", va="center", fontsize=9)
mid = (axes[0][0].get_position().y0 + axes[1][0].get_position().y1) / 2 - 0.02
fig.add_artist(Line2D([0.03, 0.98], [mid, mid], transform=fig.transFigure, color=GREY, lw=0.6))
fig.legend(handles=[Line2D([], [], color=BLUE, marker="o", ms=3.6, lw=1.3, label="uniform-trained model"),
                    Line2D([], [], color=ORANGE, marker="s", ms=3.6, lw=1.3, label="popularity-trained model")],
           loc="lower center", ncol=2, bbox_to_anchor=(0.53, 0.0), fontsize=8)
fig.savefig(os.path.join(A.out, "f11_reversal.pdf"))
if A.png:
    fig.savefig(os.path.join(A.out, "f11_reversal.png"), dpi=170)
plt.close(fig)
print("wrote f11_reversal to", A.out)
