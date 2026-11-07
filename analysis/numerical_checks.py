#!/usr/bin/env python3
"""Appendix D: numerical checks of the formal results on exactly enumerable models.

Every population is a discrete law on a small alphabet, so every score is summed over all
candidate sets rather than estimated. Random laws are Dirichlet draws from a fixed seed, so a
run prints the same numbers every time, and each check has its own generator, so --quick
changes only the number of random draws.

    python analysis/numerical_checks.py           # all checks, about two minutes
    python analysis/numerical_checks.py --quick   # a tenth of the random draws

Item (d) is in analysis/numerical_check_d.py.
"""
import argparse
import itertools
import math
import os
import sys

import numpy as np

TOL = 1e-10
ap = argparse.ArgumentParser()
ap.add_argument("--quick", action="store_true")
ap.add_argument("--seed", type=int, default=0)
A = ap.parse_args()
SCALE = 10 if A.quick else 1
rng = None


def seed(k):
    global rng
    rng = np.random.default_rng([A.seed, k])


# ------------------------------------------------------------------ basic quantities
def kl(q, r):
    q, r = np.asarray(q, float).ravel(), np.asarray(r, float).ravel()
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / r[m])))


def hb(p):
    return 0.0 if p <= 0 or p >= 1 else -p * math.log(p) - (1 - p) * math.log(1 - p)


_COMP = {}


def compositions(n, k):
    """All count vectors of n candidates over k symbols, with log multinomial coefficients."""
    key = (n, k)
    if key not in _COMP:
        rows = []
        for cut in itertools.combinations(range(n + k - 1), k - 1):
            prev, c = -1, []
            for x in cut + (n + k - 1,):
                c.append(x - prev - 1)
                prev = x
            rows.append(c)
        c = np.array(rows, dtype=float)
        logcoef = math.lgamma(n + 1) - np.sum([[math.lgamma(v + 1) for v in r] for r in rows], axis=1)
        _COMP[key] = (c, np.asarray(logcoef))
    return _COMP[key]


def score(tpos, tneg, Q, R, K):
    """Exact L_K: the positive drawn from Q and scored by tpos, K-1 candidates drawn from R and
    scored by tneg. With tpos = tneg = T this is Definition A.3."""
    Q, R = np.asarray(Q, float).ravel(), np.asarray(R, float).ravel()
    s = R > 0
    c, logcoef = compositions(K - 1, int(s.sum()))
    pmf = np.exp(logcoef + c @ np.log(R[s]))
    tp, tn = np.asarray(tpos, float).ravel(), np.asarray(tneg, float).ravel()
    m = max(tp.max(), tn[s].max())
    den = np.exp(tp - m)[:, None] + (c @ np.exp(tn[s] - m))[None, :]
    val = math.log(K) + (tp - m)[:, None] - np.log(den)
    q = Q > 0
    return float(Q[q] @ (val[q] @ pmf))


def score_tuples(T, Q, R, K):
    """L_K and I(J; X_1..K) by enumeration of every K-tuple of draws and every positive slot."""
    Q, R, T = map(lambda a: np.asarray(a, float), (Q, R, T))
    a = len(Q)
    L = 0.0
    I = 0.0
    for x in itertools.product(range(a), repeat=K):
        x = np.array(x)
        pj = np.array([Q[x[j]] * np.prod(np.delete(R[x], j)) / K for j in range(K)])
        px = pj.sum()
        if px == 0:
            continue
        e = np.exp(T[x] - T[x].max())
        L += float(np.sum(pj * np.log(K * e / e.sum())))
        nz = pj > 0
        I += float(np.sum(pj[nz] * np.log(pj[nz] / (px / K))))
    return L, I


def population(mu, mv, independent=True, eta=None, a=1.0):
    """P*(u, v, y) as an array (mu, mv, 2). Independent-candidate form when independent=True."""
    if independent:
        puv = np.outer(rng.dirichlet(np.ones(mu) * a), rng.dirichlet(np.ones(mv) * a))
    else:
        puv = rng.dirichlet(np.ones(mu * mv) * a).reshape(mu, mv)
    if eta is None:
        eta = rng.uniform(0.01, 0.99, size=(mu, mv)) ** rng.uniform(0.5, 3.0)
    return np.stack([puv * (1 - eta), puv * eta], axis=-1)


def encode(P, fz, fc):
    """Push P*(u, v, y) through deterministic encoders z = fz[u], c = fc[v]."""
    out = np.zeros((max(fz) + 1, max(fc) + 1, 2))
    for u in range(P.shape[0]):
        for v in range(P.shape[1]):
            out[fz[u], fc[v]] += P[u, v]
    return out


def parts(P):
    """p, P+, P-, P* and eta on the representation pairs."""
    pstar = P.sum(-1)
    p = P[..., 1].sum()
    pos = P[..., 1] / p
    neg = P[..., 0] / (1 - p)
    eta = np.where(pstar > 0, P[..., 1] / np.where(pstar > 0, pstar, 1), 0)
    return p, pos, neg, pstar, eta


def mutual_info_y(P):
    p, pos, neg, pstar, eta = parts(P)
    return hb(p) - float(np.sum(pstar * np.vectorize(hb)(eta)))


def mi(joint):
    joint = joint / joint.sum()
    return kl(joint, np.outer(joint.sum(1), joint.sum(0)))


def report(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


results = []

# ------------------------------------------------------------------ (a) Lemma A.5
print("(a) Lemma A.5, by enumeration over all K-tuples")
seed(1)
n_conf, worst_gap, worst_opt = 0, -np.inf, 0.0
for size in (2, 3, 4, 5):
    for K in (2, 3, 4, 5):
        for _ in range(20):
            Q, R = rng.dirichlet(np.ones(size)), rng.dirichlet(np.ones(size))
            lr = np.log(Q / R)
            cap = min(kl(Q, R), math.log(K))
            for T in (lr, 0.5 * lr, 2.0 * lr, -lr, rng.normal(0, 2, size)):
                L, I = score_tuples(T, Q, R, K)
                worst_gap = max(worst_gap, L - I, I - cap)
                n_conf += 1
            Lopt, Iopt = score_tuples(lr, Q, R, K)
            worst_opt = max(worst_opt, abs(Lopt - Iopt))
results.append(report("L_K <= I(J;X) <= min(KL, log K), and the log-ratio attains I(J;X)",
                      worst_gap <= TOL and worst_opt <= TOL,
                      f"{n_conf} configurations, largest violation {max(worst_gap, 0):.1e}, "
                      f"log-ratio gap {worst_opt:.1e}"))

# ------------------------------------------------------------------ (b), (c) Theorem A.6, Proposition A.7
print("(b) Theorem A.6 and (c) Proposition A.7")
seed(2)


def theorem_draw(independent):
    mu, mv = rng.integers(2, 4, size=2)
    P = population(mu, mv, independent)
    fz = rng.integers(0, rng.integers(1, mu + 1), size=mu)
    fc = rng.integers(0, rng.integers(1, mv + 1), size=mv)
    Pz = encode(P, fz, fc)
    p, pos, neg, pstar, eta = parts(Pz)
    I = mutual_info_y(Pz)
    keep = pstar.ravel() > 0                      # pairs the encoders actually produce
    pos, neg, pstar, eta = (a.ravel()[keep] for a in (pos, neg, pstar, eta))
    D, D0 = kl(pos, pstar), kl(neg, pstar)
    K = int(rng.integers(2, 5))
    g = rng.normal(0, 2, pos.size)
    L = score(g, g, pos, pstar, K)
    lr = np.log(pos / pstar)
    Lopt = score(lr, lr, pos, pstar, K)
    IS = mutual_info_y(P)                       # S = (U, V) itself
    ident = abs(I - (p * D + (1 - p) * D0))
    bound = max(p * max(L, Lopt) - p * D, p * D - I, (IS - I) - (IS - p * max(L, 0)))
    var = float(np.sum(pstar * (eta - p) ** 2))
    tight1 = (1 - p) * D0 - var / (1 - p)
    kappa = eta.max() / p
    tight2 = ((I - p * D) / (p * D) - 2 * kappa * p / (1 - p)) if D > 1e-12 else -1.0
    return ident, bound, tight1, tight2


for label, n, indep in (("independent-candidate", 200000 // SCALE, True), ("dependent", 60000 // SCALE, False)):
    worst = np.full(4, -np.inf)
    for _ in range(n):
        worst = np.maximum(worst, theorem_draw(indep))
    results.append(report(f"I = pD + (1-p)D0 and I >= pD >= pL_K, {label} populations",
                          worst[0] <= TOL and worst[1] <= TOL,
                          f"{n} draws, identity error {worst[0]:.1e}, largest violation {max(worst[1], 0):.1e}"))
    results.append(report(f"slack bounds of Proposition A.7, {label} populations",
                          worst[2] <= TOL and worst[3] <= TOL,
                          f"{n} draws, largest violations {max(worst[2], 0):.1e} and {max(worst[3], 0):.1e}"))
sl = []
for p in (1e-1, 1e-2, 1e-3, 1e-4, 1e-5):
    I = hb(p)                                   # eta in {0, 1}: Y is fixed by (Z, C)
    sl.append((I - p * math.log(1 / p)) / I)
print("      eta in {0,1}: share of I left as slack at p = 1e-1 ... 1e-5: "
      + ", ".join(f"{s:.3f}" for s in sl))

# ------------------------------------------------------------------ (d)
print("(d) Proposition A.10: see analysis/numerical_check_d.py")

# ------------------------------------------------------------------ (e) Remark A.13
print("(e) Remark A.13")
worst = 0.0
for e1 in np.linspace(0.05, 0.95, 7):
    for e0 in np.linspace(0.05, 0.95, 7):
        P = np.stack([0.25 * np.array([[1 - e1, 1 - e0], [1 - e0, 1 - e1]]),
                      0.25 * np.array([[e1, e0], [e0, e1]])], axis=-1)
        worst = max(worst, mi(P.sum(-1)))
P = np.stack([0.25 * np.array([[0.1, 0.9], [0.9, 0.1]]), 0.25 * np.array([[0.9, 0.1], [0.1, 0.9]])], axis=-1)
p = P[..., 1].sum()
results.append(report("I*(Z;C) = 0 for every (eta1, eta0)", worst <= TOL, f"largest {worst:.1e}"))
print(f"      (eta1, eta0) = (0.9, 0.1): I*(Y;Z,C) = {mutual_info_y(P) / hb(p):.2f} H_b(p)")

# ------------------------------------------------------------------ (f) Corollary A.15, Propositions A.16, A.9
print("(f) Corollary A.15, Proposition A.16 and Proposition A.9")
seed(6)


def newton_max(f, x0, iters=60, h=1e-5):
    """Maximise a smooth concave function by damped Newton steps with finite differences;
    the first coordinate is held at 0, which removes the constant the scores ignore."""
    x = np.array(x0, float)
    x[0] = 0.0
    d = len(x) - 1
    for _ in range(iters):
        g, H = np.zeros(d), np.zeros((d, d))
        f0 = f(x)
        for i in range(d):
            e = np.zeros(len(x)); e[i + 1] = h
            g[i] = (f(x + e) - f(x - e)) / (2 * h)
            for j in range(i, d):
                e2 = np.zeros(len(x)); e2[j + 1] = h
                H[i, j] = H[j, i] = (f(x + e + e2) - f(x + e - e2) - f(x - e + e2) + f(x - e - e2)) / (4 * h * h)
        w, V = np.linalg.eigh(H)
        step = V @ ((V.T @ g) / np.maximum(-w, 1e-6))
        step *= min(1.0, 5.0 / max(np.abs(step).max(), 1e-300))
        t = 1.0
        while t > 1e-10:
            fn = f(np.concatenate([[0.0], x[1:] + t * step]))
            if np.isfinite(fn) and fn >= f0 - 1e-15:
                break
            t /= 2
        x[1:] += t * step
        if np.abs(t * step).max() < 1e-12:
            break
    return x


slacks = []
for _ in range(200):
    mc = int(rng.integers(2, 5))
    Qc, Rc = rng.dirichlet(np.ones(mc)), rng.dirichlet(np.ones(mc))
    K = int(rng.integers(2, 6))
    t = newton_max(lambda t: score(t, t, Qc, Rc, K), rng.normal(0, 1, mc))
    sup = score(t, t, Qc, Rc, K)
    slacks.append(kl(Qc, Rc) - sup)
results.append(report("KL(P+(C)||P*(C)) - sup_t L_K(t) >= 0", min(slacks) >= -TOL,
                      f"200 optimisations, minimum slack {min(slacks):.1e}"))
worst = -np.inf
seed(7)
for _ in range(40000 // SCALE):
    mu, mv = rng.integers(2, 4, size=2)
    P = population(mu, mv, True)
    p, pos, neg, pstar, eta = parts(P)
    pz, pc = pos.sum(1), pos.sum(0)
    lz, lc = kl(pz, pstar.sum(1)), kl(pc, pstar.sum(0))
    Ipos = mi(pos)
    K = int(rng.integers(2, 5))
    g = rng.normal(0, 2, (mu, mv))
    joint = score(g.ravel(), g.ravel(), pos, pstar, K)
    R = rng.dirichlet(np.ones(mv))
    anc = sum(pz[z] * score(g[z], g[z], pos[z] / pz[z], R, K) for z in range(mu))
    worst = max(worst, max(joint - lc, 0) - (Ipos + lz), max(anc - kl(pc, R), 0) - Ipos)
results.append(report("adjusted bounds, joint and anchored", worst <= TOL,
                      f"{40000 // SCALE} draws, largest violation {max(worst, 0):.1e}"))
dec_prod, dec_dep = 0.0, 0.0
seed(8)
for _ in range(1000 // SCALE):
    for indep in (True, False):
        P = population(3, 3, indep)
        p, pos, neg, pstar, eta = parts(P)
        gap = abs(kl(pos, pstar) - (mi(pos) + kl(pos.sum(1), pstar.sum(1)) + kl(pos.sum(0), pstar.sum(0))))
        if indep:
            dec_prod = max(dec_prod, gap)
        else:
            dec_dep = max(dec_dep, gap)
results.append(report("decomposition holds under a product candidate law", dec_prod <= TOL,
                      f"largest error {dec_prod:.1e}"))
results.append(report("decomposition fails under a dependent candidate law", dec_dep > 1e-3,
                      f"largest error {dec_dep:.2f} nats"))
w = np.arange(1, 10001, dtype=float)
zipf = [kl((w ** -z) / (w ** -z).sum(), np.full(10000, 1e-4)) for z in (0.5, 1.0, 1.5)]
print("      Zipf catalogue of 10^4 items against uniform candidates: "
      + ", ".join(f"{v:.2f}" for v in zipf) + " nats at exponent 0.5, 1.0, 1.5;"
      f" log 4096 leaves {math.log(4096) - zipf[2]:.2f} nats at 1.5")

# ------------------------------------------------------------------ (g) Appendix A.10, proposal correction
print("(g) proposal correction")
m, K = 4, 4
Pst, Pp, R = np.array([0.4, 0.3, 0.2, 0.1]), np.array([0.1, 0.2, 0.3, 0.4]), np.full(4, 0.25)
lw = np.log(R / Pst)
target = np.log(Pp / Pst) - np.log(Pp[0] / Pst[0])
g_all = newton_max(lambda g: score(g - lw, g - lw, Pp, R, K), np.zeros(m))
g_neg = newton_max(lambda g: score(g, g - lw, Pp, R, K), np.zeros(m))
e_all, e_neg = np.abs(g_all - target).max(), np.abs(g_neg - target).max()
results.append(report("all-slot correction recovers log dP+/dP*", e_all < 1e-5,
                      f"largest error {e_all:.1e}; negative slots only {e_neg:.2f} nats"))

# ------------------------------------------------------------------ (h) Proposition A.20
print("(h) Proposition A.20")
for z in (0.0, 0.5, 1.0, 1.5):
    w = np.arange(1, 2001, dtype=float) ** -z
    print(f"      Zipf exponent {z}: macro-versus-micro {kl(np.full(2000, 1 / 2000), w / w.sum()):.3f} nats")
e0 = os.path.join("results", "aggregated", "all_e0_diagnostics.csv")
if os.path.exists(e0):
    import csv
    v = [float(r["value"]) for r in csv.DictReader(open(e0)) if r["quantity"] == "KL_macro_micro"]
    print(f"      the seven datasets: {min(v):.3f} to {max(v):.3f} nats")

# ------------------------------------------------------------------ (i) Proposition B.1
print("(i) Proposition B.1")
seed(9)
G, K, m = 2.5, 8, 6
lo, hi = np.inf, -np.inf
for _ in range(200):
    g = rng.uniform(-G, G, m)
    Q, R = rng.dirichlet(np.ones(m)), rng.dirichlet(np.ones(m))
    x = np.concatenate([rng.choice(m, size=(20000, 1), p=Q), rng.choice(m, size=(20000, K - 1), p=R)], axis=1)
    e = g[x]
    terms = math.log(K) + e[:, 0] - (e.max(1) + np.log(np.exp(e - e.max(1, keepdims=True)).sum(1)))
    lo, hi = min(lo, terms.min()), max(hi, terms.max())
results.append(report("per-positive terms lie in [-2G, log K]", lo >= -2 * G and hi <= math.log(K),
                      f"observed range [{lo:.2f}, {hi:.2f}] inside [{-2 * G:.2f}, {math.log(K):.2f}]"))

# ------------------------------------------------------------------ further enumerations
print("Further enumerations")
worst_ok, tightest = True, None
for p in (1e-1, 1e-2, 1e-3):
    for zeta in (0.0, 1.0):
        for K in (2, 4, 8):
            w = np.arange(1, 4, dtype=float) ** -zeta
            pu = pv = w / w.sum()
            aff = 1.0 + 2.0 * np.eye(3)
            eta = p * aff / float(pu @ aff @ pv)
            P = np.stack([np.outer(pu, pv) * (1 - eta), np.outer(pu, pv) * eta], axis=-1)
            pp, pos, neg, pstar, et = parts(P)
            I, D = mutual_info_y(P), kl(pos, pstar)
            lr = np.log(pos / pstar)
            L = score(lr, lr, pos, pstar, K)
            kappa = et.max() / pp
            rel, bound = (I - pp * D) / (pp * D), 2 * kappa * pp / (1 - pp)
            worst_ok &= pp * L <= I + TOL and rel <= bound + TOL
            if tightest is None or rel / bound > tightest[0] / tightest[1]:
                tightest = (rel, bound, pp, kappa)
results.append(report("pL_K <= I and relative slack within Proposition A.7, 18 configurations", worst_ok,
                      f"tightest {tightest[0]:.3f} against {tightest[1]:.3f} at p = {tightest[2]:g}, "
                      f"kappa = {tightest[3]:.2f}"))

eta = np.array([[0.1, 0.1], [0.1, 0.9]])               # uniform users and items, one preferred pair
P = np.stack([0.25 * (1 - eta), 0.25 * eta], axis=-1)
p, Q, neg, Rstar, eta = parts(P)
Rpop = np.outer(Q.sum(1), Q.sum(0))
T = np.log(Q / Rpop).ravel()                         # optimal for R_pop
off = kl(Q, Rstar) - kl(Q, Rpop)
d = [score(T, T, Q, Rstar, K) - score(T, T, Q, Rpop, K) for K in (2, 4, 8)]
results.append(report("finite-K score differences are not divergence differences", all(x * off < 0 for x in d),
                      f"KL(Q||R*) = {kl(Q, Rstar):.6f}, KL(Q||R_pop) = {kl(Q, Rpop):.6f}, offset {off:.6f}, "
                      f"score differences " + ", ".join(f"{x:.4f}" for x in d) + " at K = 2, 4, 8"))

pw, pu_w, pv_w = np.array([0.5, 0.5]), np.array([[0.8, 0.2], [0.2, 0.8]]), np.array([[0.8, 0.2], [0.2, 0.8]])
puv = np.einsum("w,wu,wv->uv", pw, pu_w, pv_w)
P = np.stack([puv * 0.9, puv * 0.1], axis=-1)        # Y independent of (Z, C)
p, pos, neg, pstar, eta = parts(P)
lr = np.log(pos / pstar)
sup = max(score(lr, lr, pos, pstar, K) for K in (2, 4, 8))
results.append(report("shared nuisance with Y independent of (Z, C)", abs(p * sup) <= TOL and abs(mutual_info_y(P)) <= TOL,
                      f"I+(Z;C) = {mi(pos):.4f}, p sup L_K = {p * sup:.1e}, I*(Y;Z,C) = {mutual_info_y(P):.1e}"))

w = np.ones(3) / 3
eta = 0.1 * (1.0 + 2.0 * np.eye(3)) / float(w @ (1.0 + 2.0 * np.eye(3)) @ w)
P = np.stack([np.outer(w, w) * (1 - eta), np.outer(w, w) * eta], axis=-1)
p, pos, neg, pstar, _ = parts(P)
D = kl(pos, pstar)
lr = np.log(pos / pstar)
ratios = [score(lr, lr, pos, pstar, K) / D for K in (2, 4, 8, 16)]
results.append(report("with the candidate law correct, L_K/D rises with K", all(np.diff(ratios) > 0),
                      ", ".join(f"{r:.3f}" for r in ratios) + f" at K = 2, 4, 8, 16 for p = {p:g}"))

print("\nALL PASS" if all(results) else "\nSOME FAIL")
sys.exit(0 if all(results) else 1)
