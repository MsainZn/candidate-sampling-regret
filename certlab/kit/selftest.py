"""Self-checks that run without any dataset.

``selftest`` exercises the core identities on small synthetic inputs. Run it first
after installing, with ``python run_experiments.py --selftest``; it needs no data and
finishes in seconds.

Part of the code that produced every number in the paper.
"""
import math
import numpy as np

from .core import contrastive_terms, certify, KL, Hb, ref_alpha, cdf_of, sample_from
from .model import item_only_L

# ===================================================================== selftest
def selftest():
    ok = True
    rng = np.random.default_rng(0)
    import itertools
    n_al, K = 5, 4
    Q = rng.dirichlet(np.ones(n_al)); R = rng.dirichlet(np.ones(n_al)); T = rng.normal(0, 1, n_al)
    def exact(Q, R, T, K):
        eT = np.exp(T); m = K - 1
        lf = np.array([math.lgamma(k + 1) for k in range(m + 1)])
        def comps(tot, parts):
            if parts == 1: yield (tot,); return
            for f in range(tot + 1):
                for rest in comps(tot - f, parts - 1): yield (f,) + rest
        s = 0.0
        for c in comps(m, n_al):
            c = np.array(c); sup = c > 0
            w = math.exp(lf[m] - lf[c].sum() + float((c[sup] * np.log(R[sup])).sum()))
            s += w * float((Q * (T - np.log(eT + float((c * eT).sum())))).sum())
        return math.log(K) + s
    ex = exact(Q, R, T, K); Nn = 2_000_000
    mc = contrastive_terms(T[rng.choice(n_al, Nn, p=Q)], T[rng.choice(n_al, (Nn, K - 1), p=R)])
    se = mc.std(ddof=1) / math.sqrt(Nn); good = abs(ex - mc.mean()) < 4 * se
    ok &= good
    print(f"T1 contrastive_terms vs exact enumeration: exact={ex:.6f} mc={mc.mean():.6f} "
          f"(+/-{1.96*se:.6f})  {'PASS' if good else 'FAIL'}")
    M = 3000
    deg = rng.pareto(1.1, M) + 1; Ptr = deg / deg.sum()
    Pp = deg * np.exp(rng.normal(0, .9, M)); Pp /= Pp.sum(); drift = KL(Pp, Ptr)
    class A: cert_K = 1024
    a = A()
    print(f"T2 identity  oracle-honest == KL(P_+||P_tr) = {drift:.4f}")
    for al in [0, 0.5, 1.0, 1.25]:
        R_ = ref_alpha(Ptr, al)
        o, _ = item_only_L(np.log((Pp + 1e-300) / R_), Pp, R_, a, n=30000)
        h, _ = item_only_L(np.log((Ptr + 1e-300) / R_), Pp, R_, a, n=30000)
        d = o - h - drift; g2 = abs(d) < 0.06; ok &= g2
        print(f"   alpha={al:<5g} orc={o:.4f} hon={h:.4f} orc-hon={o-h:.4f} dev={d:+.4f} "
              f"{'PASS' if g2 else 'FAIL'}")
    R1 = ref_alpha(Ptr, 1.0)
    h1, _ = item_only_L(np.log((Ptr + 1e-300) / R1), Pp, R1, a, n=20000)
    g3 = abs(h1) < 1e-9; ok &= g3
    print(f"T3 honest baseline at alpha=1 is exactly 0: {h1:.2e}  {'PASS' if g3 else 'FAIL'}")
    g4 = np.abs(ref_alpha(Ptr, 0.0) - 1.0 / M).max() < 1e-15
    g5 = np.abs(ref_alpha(Ptr, 1.0) - Ptr).max() < 1e-15; ok &= g4 and g5
    print(f"T4 alpha=0 is uniform and alpha=1 is P_tr: {'PASS' if g4 and g5 else 'FAIL'}")
    d, nu, ni, tau = 16, 4000, 500, 0.1
    Z = rng.normal(size=(nu, d)); Z /= np.linalg.norm(Z, axis=1, keepdims=True)
    C = rng.normal(size=(ni, d)); C /= np.linalg.norm(C, axis=1, keepdims=True)
    cer = rng.choice(nu, 2500, replace=False); zc = Z[cer].mean(0); zt = Z.mean(0)
    direct = ((Z[cer] - zt) @ C.T / tau).mean(0); closed = (C @ (zc - zt)) / tau
    g6 = np.abs(direct - closed).max() < 1e-12; ok &= g6
    print(f"T5 item-only audit closed form: max|direct-closed|={np.abs(direct-closed).max():.2e}  "
          f"{'PASS' if g6 else 'FAIL'}")
    g7 = abs(((Z[cer] - zc) @ C.T / tau).mean(0)).max() < 1e-12; ok &= g7
    print(f"T6 residual with matched zbar has zero item-only component: "
          f"{abs(((Z[cer]-zc)@C.T/tau).mean(0)).max():.2e}  {'PASS' if g7 else 'FAIL'}")
    p = 3.4884e-3; hb = Hb(p)
    g8 = abs(hb - 0.023221) < 1e-5; ok &= g8
    print(f"T7 H_b(p) at p=3.4884e-3: {hb:.6f} (expect 0.023221)  {'PASS' if g8 else 'FAIL'}")
    print("\nALL PASS" if ok else "\nSOME FAIL")
    return 0 if ok else 1
