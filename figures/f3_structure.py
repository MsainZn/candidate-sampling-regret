
"""
Regenerate figs/f3_structure.pdf as a two-panel figure.

Panel (a): the two selection coordinates, with each dataset assigned
a distinct colour and identified by an in-panel legend.

Panel (b): for each dataset the identifier-level decomposition

    D = Lambda_U + Lambda_V + I_+(U;V)

drawn in the same three colours and the same stacking order as the
schematic bar of Figure 1, so the two figures read as the schematic
and its measurement.

Reads results/aggregated/all_e0_diagnostics.csv only.
Touches no other figure and no other script.
"""

import argparse
import collections
import csv
import math
import os
import subprocess

import matplotlib

matplotlib.use("pgf")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# ---------------------------------------------------------------------------
# Palettes
# ---------------------------------------------------------------------------
# Figure 1 palette: one colour per decomposition term.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
GREEN = "#1baf7a"

BLACK = "#000000"
GREY = "#9a9a9a"
WHITE = "#ffffff"
GRID = "#cccccc"

# One distinct colour per dataset for panel (a).
DATASET_COLORS = {
    "amazon_books": "#4c78a8",
    "hotelrec": "#f58518",
    "yelp": "#54a24b",
    "google_connecticut": "#e45756",
    "goodreads_all": "#72b7b2",
    "netflix": "#b279a2",
    "ml20m": "#ff9da6",
}


# ---------------------------------------------------------------------------
# Matplotlib configuration
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "pgf.rcfonts": False,
    "pgf.preamble": "\n".join([
        r"\usepackage{amsmath}",
        r"\usepackage{amssymb}",
    ]),
    "font.family": "serif",
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "text.color": BLACK,
    "axes.labelcolor": BLACK,
    "axes.edgecolor": BLACK,
    "axes.linewidth": 0.7,
    "xtick.color": BLACK,
    "ytick.color": BLACK,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 2.6,
    "ytick.major.size": 2.6,
    "xtick.top": True,
    "ytick.right": True,
    "grid.color": GRID,
    "grid.linewidth": 0.4,
    "figure.facecolor": WHITE,
    "axes.facecolor": WHITE,
    "savefig.facecolor": WHITE,
    "legend.frameon": False,
    "lines.linewidth": 1.1,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


TW = 6.27


# ---------------------------------------------------------------------------
# Display names
# ---------------------------------------------------------------------------
NICE = {
    "ml20m": "ML-20M",
    "netflix": "Netflix",
    "yelp": "Yelp",
    "amazon_books": "Amazon:Books",
    "goodreads_all": "Goodreads",
    "google_connecticut": "Google:CT",
    "hotelrec": "HotelRec",
}


def nice(t):
    return NICE.get(t, t.replace("_", "-").title())


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------
ap = argparse.ArgumentParser()

ap.add_argument(
    "--bundle",
    default="results/aggregated",
)

ap.add_argument(
    "--out",
    default="results/figures",
)

ap.add_argument(
    "--png",
    action="store_true",
)

A = ap.parse_args()

os.makedirs(A.out, exist_ok=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def F(x):
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Read diagnostics
# ---------------------------------------------------------------------------
E0 = collections.defaultdict(dict)

with open(
    os.path.join(A.bundle, "all_e0_diagnostics.csv"),
    newline="",
) as fh:
    for r in csv.DictReader(fh):
        E0[r["dataset"]][r["quantity"]] = F(r["value"])


# ---------------------------------------------------------------------------
# Dataset order
# ---------------------------------------------------------------------------
ORDER = [
    "amazon_books",
    "hotelrec",
    "yelp",
    "google_connecticut",
    "goodreads_all",
    "netflix",
    "ml20m",
]

ORDER = [
    d
    for d in ORDER
    if d in E0 and E0[d].get("D")
]

assert len(ORDER) == 7, (
    "expected seven corpora, found %d" % len(ORDER)
)


# ---------------------------------------------------------------------------
# Validate decomposition
# ---------------------------------------------------------------------------
for d in ORDER:
    s = (
        E0[d]["Lambda_U"]
        + E0[d]["Lambda_V"]
        + E0[d]["I_plus_UV"]
    )
    assert abs(s - E0[d]["D"]) < 1e-6, (
        "%s: decomposition does not close" % d
    )


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (axa, axb) = plt.subplots(
    1,
    2,
    figsize=(TW, 3.25),
    gridspec_kw={
        "width_ratios": [1.0, 1.05],
    },
)


# ===========================================================================
# PANEL (a): selection coordinates
# ===========================================================================
for d in ORDER:
    D_ = E0[d]["D"]
    LV = E0[d]["Lambda_V"]
    ppu = E0[d]["pairs_per_user"]

    axa.scatter(
        D_,
        LV / D_,
        s=18 + 26 * math.log10(ppu + 1),
        facecolor=DATASET_COLORS[d],
        edgecolor=BLACK,
        linewidth=0.5,
        zorder=3,
    )


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
axa.set_xlabel(
    r"$D=\log(1/p)$ (nats)",
    labelpad=2,
)

axa.set_ylabel(
    r"popularity share $\Lambda_V/D$",
    labelpad=2,
)


_xs = [E0[d]["D"] for d in ORDER]
_ys = [E0[d]["Lambda_V"] / E0[d]["D"] for d in ORDER]

_xr = max(_xs) - min(_xs)
_yr = max(_ys) - min(_ys)

axa.set_xlim(
    min(_xs) - 0.14 * _xr,
    max(_xs) + 0.22 * _xr,
)

axa.set_ylim(
    max(0.0, min(_ys) - 0.24 * _yr),
    max(_ys) + 0.16 * _yr,
)

axa.grid(True)
axa.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Dataset legend: bottom-left, inside panel (a)
# ---------------------------------------------------------------------------
dataset_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        linestyle="",
        markersize=5.5,
        markerfacecolor=DATASET_COLORS[d],
        markeredgecolor=BLACK,
        markeredgewidth=0.5,
        label=nice(d),
    )
    for d in ORDER
]

axa.legend(
    handles=dataset_handles,
    loc="lower left",
    bbox_to_anchor=(0.02, 0.02),
    ncol=1,
    frameon=True,
    facecolor=WHITE,
    edgecolor=GRID,
    framealpha=0.92,
    borderpad=0.45,
    labelspacing=0.25,
    handletextpad=0.45,
    handlelength=1.1,
)


# ---------------------------------------------------------------------------
# Marker-size explanation
# ---------------------------------------------------------------------------
axa.text(
    0.5,
    0.5,
    r"marker area $\propto\log(1+\text{pairs per user})$",
    transform=axa.transAxes,
    ha="center",
    va="bottom",
    fontsize=7,
    color=BLACK,
    bbox=dict(
        facecolor=WHITE,
        edgecolor=BLACK,
        linewidth=0.5,
        boxstyle="square,pad=0.25",
    ),
)


# ===========================================================================
# PANEL (b): identifier-level decomposition
# ===========================================================================
SEG = [
    (
        "Lambda_U",
        BLUE,
        r"$\Lambda_U$ user activity",
    ),
    (
        "Lambda_V",
        ORANGE,
        r"$\Lambda_V$ item popularity",
    ),
    (
        "I_plus_UV",
        GREEN,
        r"$I_+(U;V)$ user--item" "\n" r"dependence",
    ),
]


# Reverse order so first dataset appears at top.
y = list(range(len(ORDER)))[::-1]


for yi, d in zip(y, ORDER):
    left = 0.0

    for q, fc, _ in SEG:
        axb.barh(
            yi,
            E0[d][q],
            left=left,
            height=0.62,
            facecolor=fc,
            edgecolor=WHITE,
            linewidth=0.5,
            zorder=3,
        )

        left += E0[d][q]


# ---------------------------------------------------------------------------
# Axes
# ---------------------------------------------------------------------------
axb.set_yticks(y)

axb.set_yticklabels(
    [nice(d) for d in ORDER]
)

axb.set_xlabel(
    "nats",
    labelpad=2,
)

# Explicitly extend right panel to 14 nats.
axb.set_xlim(0, 14)

axb.set_ylim(
    -0.65,
    len(ORDER) - 0.35,
)

axb.xaxis.grid(True)
axb.set_axisbelow(True)

axb.tick_params(
    axis="y",
    length=0,
)


# ---------------------------------------------------------------------------
# Decomposition legend: bottom-right, inside panel (b)
# ---------------------------------------------------------------------------
axb.legend(
    handles=[
        Patch(
            facecolor=fc,
            edgecolor=WHITE,
            linewidth=0.5,
            label=l,
        )
        for _, fc, l in SEG
    ],
    loc="lower right",
    bbox_to_anchor=(1.0, 0.02),
    ncol=1,
    frameon=True,
    facecolor=WHITE,
    edgecolor=GRID,
    framealpha=0.92,
    borderpad=0.35,
    handletextpad=0.5,
    handlelength=1.3,
    labelspacing=0.25,
)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
fig.subplots_adjust(
    left=0.09,
    right=0.98,
    top=0.98,
    bottom=0.20,
    wspace=0.46,
)


# ===========================================================================
# PANEL CAPTIONS BELOW THE PLOTS
#
# Center captions using the actual subplot positions after the layout
# has been finalized, rather than hard-coded figure coordinates.
# ===========================================================================
pos_a = axa.get_position()
pos_b = axb.get_position()

fig.text(
    pos_a.x0 + pos_a.width / 2,
    0.045,
    r"(a) selection coordinates",
    ha="center",
    va="bottom",
    fontsize=9,
    color=BLACK,
)

fig.text(
    pos_b.x0 + pos_b.width / 2,
    0.045,
    r"(b) how each dataset divides its $D$",
    ha="center",
    va="bottom",
    fontsize=9,
    color=BLACK,
)


# ---------------------------------------------------------------------------
# Save PDF
# ---------------------------------------------------------------------------
fig.savefig(
    "%s/f3_structure.pdf" % A.out
)

plt.close(fig)


# ---------------------------------------------------------------------------
# Optional PNG
# ---------------------------------------------------------------------------
if A.png:

    subprocess.run(
        [
            "pdftoppm",
            "-png",
            "-r",
            "170",
            "-singlefile",
            "%s/f3_structure.pdf" % A.out,
            "%s/f3_structure" % A.out,
        ],
        check=False,
    )


print(
    "  wrote %s/f3_structure.pdf"
    % A.out
)
