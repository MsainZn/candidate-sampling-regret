"""Collecting rows and writing the per-stage CSVs.

``Sink``       accumulates tidy rows and writes one CSV per stage plus the long-format
               ``metrics_long.csv`` that every plotting script reads.
``summarise``  the end-of-run console report: arms, cross-certification, the sweep and
               the residual rows, with the half-widths that apply to each.
``Tee``        mirrors stdout into ``console_<stamp>.log`` so a run is self-documenting.

Part of the code that produced every number in the paper.
"""
import csv, json, math, os, sys
import numpy as np

from .core import KL

# ===================================================================== output
class Sink:
    """Accumulates tidy rows and writes one CSV per stage plus a combined long file."""
    def __init__(self, root, dataset):
        self.root = root; self.ds = dataset; self.rows = []
        os.makedirs(root, exist_ok=True)
    def add(self, experiment, metric, value, **kw):
        r = dict(dataset=self.ds, experiment=experiment, split=kw.pop("split", ""),
                 train_law=kw.pop("train_law", ""), cert_ref=kw.pop("cert_ref", ""),
                 alpha=kw.pop("alpha", ""), seed=kw.pop("seed", ""),
                 n_seeds=kw.pop("n_seeds", ""), metric=metric, value=value)
        r.update(kw); self.rows.append(r)
    def write(self, name, rows):
        if not rows: return
        keys = list(dict.fromkeys(k for r in rows for k in r))
        p = os.path.join(self.root, name)
        with open(p, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, lineterminator="\n"); w.writeheader()
            for r in rows: w.writerow(r)
        print(f"  wrote {p}  ({len(rows)} rows)")
    def flush_long(self):
        self.write("metrics_long.csv", self.rows)


# ===================================================================== summary
def summarise(e1, e2, e4, e7, sink, args):
    def agg(rows, keyf, valf):
        d = {}
        for r in rows:
            try: v = float(valf(r))
            except (TypeError, ValueError): continue
            d.setdefault(keyf(r), []).append(v)
        return {k: (float(np.mean(v)), float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, len(v))
                for k, v in d.items()}
    print("\n" + "=" * 100)
    print(f"PASTE-ME SUMMARY   dataset={args.dataset}")
    if getattr(args, "quick", False):
        print("*** --quick RUN: reduced n, epochs, seeds and alphas. SMOKE TEST ONLY. ***")
        print("*** These numbers are NOT REPORTABLE. Re-run without --quick for the paper. ***")
    print("=" * 100)
    if e1:
        print("\n[E1] four training laws, certified under UNIFORM negatives")
        A = agg(e1, lambda r: (r['split'], r['train_law']), lambda r: r['L_hat'])
        Nd = agg(e1, lambda r: (r['split'], r['train_law']), lambda r: r['ndcg20'])
        Ad = agg(e1, lambda r: (r['split'], r['train_law']), lambda r: r['adjusted'])
        print(f"  {'split':<10}{'law':<15}{'L_hat':>18}{'adjusted':>11}{'NDCG@20':>10}{'seeds':>7}")
        for k in sorted(A):
            m, s, n = A[k]
            print(f"  {k[0]:<10}{k[1]:<15}{m:>10.4f} +/-{s:<6.4f}{Ad[k][0]:>11.4f}{Nd[k][0]:>10.4f}{n:>7}")
        for spl in sorted({r['split'] for r in e1}):
            # PAIRED analysis: arms at the same seed now share their initialisation, so the
            # right statistic is the per-seed difference, not the difference of means.
            by = {}
            for r in e1:
                if r['split'] == spl: by.setdefault(r['train_law'], {})[r['seed']] = float(r['L_hat'])
            def paired(a, bb):
                if a not in by or bb not in by: return None
                ks = sorted(set(by[a]) & set(by[bb]))
                if not ks: return None
                d = np.array([by[a][k] - by[bb][k] for k in ks])
                return d, ks
            pr = paired('uniform', 'popularity')
            if pr:
                d, ks = pr
                sd = float(d.std(ddof=1)) if len(d) > 1 else float('nan')
                se = sd / math.sqrt(len(d)) if len(d) > 1 else float('nan')
                print(f"  [{spl}] PAIRED Delta(uniform - popularity) over {len(d)} seeds: "
                      f"{d.mean():+.4f} +/- {sd:.4f} (SD)"
                      + (f", SE {se:.4f}" if se == se else "")
                      + f"   per seed: {np.array2string(d, precision=4)}")
            pr = paired('inbatch_logq', 'inbatch')
            pu = paired('uniform', 'inbatch')
            if pr and pu:
                num, ks1 = pr; den, ks2 = pu
                ks = sorted(set(ks1) & set(ks2))
                num = np.array([by['inbatch_logq'][k] - by['inbatch'][k] for k in ks])
                den = np.array([by['uniform'][k] - by['inbatch'][k] for k in ks])
                ok = np.abs(den) > 1e-6
                rec = num[ok] / den[ok]
                rat = np.array([by['uniform'][k] / by['inbatch'][k] for k in ks
                                if abs(by['inbatch'][k]) > 1e-6])
                if len(rat):
                    print(f"  [{spl}] PAIRED in-batch deficit ratio over {len(rat)} seeds: "
                          f"{rat.mean():.2f}x +/- {(rat.std(ddof=1) if len(rat)>1 else 0):.2f}")
                if len(rec):
                    print(f"  [{spl}] PAIRED logQ recovery over {len(rec)} seeds: "
                          f"{rec.mean():.1%} +/- {(rec.std(ddof=1) if len(rec)>1 else 0):.1%}")
    if e2:
        print("\n[E2] cross-certification, L_hat (rows = training law, cols = reference)")
        refs = ["uniform", "sqrt_pop", "popularity"]
        M_ = agg([r for r in e2 if r['train_law'] != "[baseline]"],
                 lambda r: (r['split'], r['train_law'], r['cert_ref']), lambda r: r['L_hat'])
        for spl in sorted({k[0] for k in M_}):
            print(f"  --- {spl}")
            print(f"    {'law':<15}" + "".join(f"{c:>14}" for c in refs))
            for law in ["uniform", "popularity", "inbatch", "inbatch_logq"]:
                cells = [M_.get((spl, law, c)) for c in refs]
                if not any(cells): continue
                print(f"    {law:<15}" + "".join(f"{(c[0] if c else float('nan')):>14.4f}" for c in cells))
            B = {r['cert_ref']: r for r in e2 if r['train_law'] == "[baseline]" and r['split'] == spl}
            for lbl, key in [("[Lambda_R]", 'Lambda_R'), ("[oracle]", 'oracle'), ("[honest]", 'honest')]:
                print(f"    {lbl:<15}" + "".join(
                    f"{float(B[c][key]):>14.4f}" if c in B and B[c][key] != "" else f"{'':>14}" for c in refs))
            byc = {}
            for r in e2:
                if r['split'] == spl and r['train_law'] != "[baseline]" and r['L_hat'] != "":
                    byc.setdefault((r['train_law'], r['cert_ref']), {})[r['seed']] = float(r['L_hat'])
            a, bb = ('popularity', 'popularity'), ('uniform', 'popularity')
            if a in byc and bb in byc:
                ks = sorted(set(byc[a]) & set(byc[bb]))
                d = np.array([byc[a][k] - byc[bb][k] for k in ks])
                if len(d):
                    print(f"    PAIRED diagonal dominance under R=popularity over {len(d)} "
                          f"seeds: {d.mean():+.4f} +/- "
                          f"{(d.std(ddof=1) if len(d)>1 else 0):.4f} (SD)"
                          f"   per seed: {np.array2string(d, precision=4)}")
    if e4:
        print("\n[E4] matched train+certify sweep over R_alpha ~ P_tr^alpha")
        for spl in sorted({r['split'] for r in e4}):
            rs = [r for r in e4 if r['split'] == spl]
            A = agg(rs, lambda r: r['alpha'], lambda r: r['L_hat'])
            H = agg(rs, lambda r: r['alpha'], lambda r: r['adjusted_honest'])
            O = agg(rs, lambda r: r['alpha'], lambda r: r['adjusted_oracle_lcb'])
            dv = agg(rs, lambda r: r['alpha'], lambda r: r['identity_dev'])
            als = sorted(A)
            print(f"  --- {spl}   drift={rs[0]['drift']:.4f}  hw={rs[0]['half_width']:.4f}")
            print(f"    {'alpha':>7}{'L_hat':>10}{'adj_honest':>13}{'adj_oracle_LCB':>16}"
                  f"{'identity_dev':>14}{'seeds':>7}")
            for a in als:
                print(f"    {a:>7g}{A[a][0]:>10.4f}{H[a][0]:>13.4f}{O[a][0]:>16.4f}"
                      f"{dv[a][0]:>+14.4f}{A[a][2]:>7}")
            hv = np.array([H[a][0] for a in als]); nz = sum(1 for a in als if O[a][0] > 0)
            print(f"    adj_honest spread={hv.max()-hv.min():.4f} (hw={rs[0]['half_width']:.4f})"
                  f"   corr(alpha)={np.corrcoef(als,hv)[0,1]:+.3f}"
                  f"   oracle-adjusted nonzero: {nz}/{len(als)}")
    if e7:
        print("\n[E7] zbar variants -- AUDIT then calibrated residual (read calib_FIXED only)")
        for spl in sorted({r['split'] for r in e7}):
            rs = [r for r in e7 if r['split'] == spl]
            print(f"  --- {spl}   target range(log P_+) = {rs[0]['target_range']:.2f} nats")
            print(f"    {'zbar':<14}{'item_only_range':>17}{'item_only_R2':>14}"
                  f"{'spread(L)':>11}{'min LCB':>10}{'max LCB':>10}")
            for v in ["cert_LEAKY", "train", "calib_FIXED"]:
                q = [r for r in rs if r['zbar'] == v]
                if not q: continue
                L = agg(q, lambda r: r['alpha'], lambda r: r['L_hat'])
                lc = agg(q, lambda r: r['alpha'], lambda r: r['L_lcb'])
                Lv = np.array([L[a][0] for a in sorted(L)]); lv = np.array([lc[a][0] for a in sorted(lc)])
                io = float(np.mean([r['item_only_range'] for r in q]))
                r2 = float(np.mean([r['item_only_r2'] for r in q]))
                print(f"    {v:<14}{io:>17.3f}{r2:>14.3f}{Lv.max()-Lv.min():>11.4f}"
                      f"{lv.min():>10.4f}{lv.max():>10.4f}")
            q = [r for r in rs if r['zbar'] == "calib_FIXED"]
            if q:
                io = float(np.mean([r['item_only_range'] for r in q]))
                tgt = float(q[0]['target_range']); frac = io / tgt if tgt > 0 else 1.0
                lc = agg(q, lambda r: r['alpha'], lambda r: r['L_lcb'])
                lb = agg(q, lambda r: r['alpha'], lambda r: r['L_lcb_bonf'])
                lv = np.array([lc[a][0] for a in sorted(lc)])
                bv = np.array([lb[a][0] for a in sorted(lb)])
                nref = int(q[0]['n_refs']); hwb = float(q[0]['half_width_bonf'])
                D = math.log(1 / [r for r in e4 if r['split'] == spl][0]['p']) if e4 else float('nan')
                sound = frac < 0.05
                print(f"    READ-OFF calib_FIXED: item-only {io:.3f} nats = {frac:.2%} of the "
                      f"{tgt:.2f}-nat target range -> {'SOUND' if sound else 'NOT item-free'}")
                if sound:
                    pa = float(args.preregister_alpha)
                    if pa in lc:
                        print(f"      (1) PRE-REGISTERED alpha={pa:g} (fixed by "
                              f"--preregister-alpha before the run), 95% LCB = "
                              f"{lc[pa][0]:.4f} nats"
                              + (f" = {lc[pa][0]/D:.2%} of D" if D == D else "")
                              + "   <- this one needs no multiplicity correction")
                    else:
                        print(f"      (1) --preregister-alpha {pa:g} is not in --alphas; no "
                              f"pre-registered figure available")
                    ba = max(lc, key=lambda k: lc[k][0])
                    print(f"          [POST-HOC best alpha={ba:g} gives {lc[ba][0]:.4f} -- "
                          f"selected after seeing the results, NOT quotable as a 95% bound]")
                    # (2) per-seed: min over alpha within one trained model, Bonferroni over refs
                    per_seed = {}
                    for r in q:
                        per_seed.setdefault(r['seed'], []).append(float(r['L_lcb_bonf']))
                    ps = {k: min(v) for k, v in per_seed.items()}
                    print(f"      (2) PER SEED, simultaneous over {nref} refs "
                          f"(delta=0.05/{nref}, hw {hwb:.4f}): "
                          + "  ".join(f"s{k}={v:.4f}" for k, v in sorted(ps.items())))
                    # (3) joint over the whole alpha x seed grid
                    jv = np.array([float(r['L_lcb_joint']) for r in q])
                    ncell = int(q[0]['n_cells']); hwj = float(q[0]['half_width_joint'])
                    print(f"      (3) JOINT over all {ncell} (alpha,seed) cells "
                          f"(delta=0.05/{ncell}, hw {hwj:.4f}): min LCB = {jv.min():.4f} nats"
                          + (f" = {jv.min()/D:.2%} of D" if D == D else ""))
                    print(f"          (3) covers the {ncell} FIXED (alpha,seed) cells, conditional "
                          f"on the calibration procedure; it does not license any further "
                          f"data-dependent selection.")
                    print(f"      QUOTE (1) if the alpha was pre-registered, else (3). The uncorrected "
                          f"min of {nref} per-seed-averaged 95% LCBs is {lv.min():.4f}; averaging "
                          f"LCBs across seeds bounds only the seed-AVERAGE of L, and its joint "
                          f"coverage is 1-{ncell}*0.05 = {max(0.0,1-ncell*0.05):.0%}, so it is not "
                          f"a simultaneous bound at all.")
                else:
                    print(f"      DO NOT QUOTE this row. Raise --calib-n (zbar is estimated from it) "
                          f"or the test split is too small for this audit.")
    print("\n" + "=" * 100)


# ===================================================================== main
class Tee:
    def __init__(self, path):
        self.f = open(path, "w"); self.o = sys.stdout
    def write(self, s): self.o.write(s); self.f.write(s); self.f.flush()
    def flush(self): self.o.flush(); self.f.flush()
