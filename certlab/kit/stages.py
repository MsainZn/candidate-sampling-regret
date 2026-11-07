"""The experiment stages, one function per stage.

``stage_e0``          estimand diagnostics with no training at all: p, D, Lambda_U,
                      Lambda_V, I_+(U;V), candidate-set conditions.
``stage_e1_e2_e3``    fits the four training arms, certifies them under a common
                      uniform reference (e1), re-scores the SAME frozen models under
                      every certification reference (e2), and runs the item-only
                      diagnostic (e3).
``stage_e3b``         the enlarged-critic stress test: refits at --cap-dim over
                      --cap-epochs with a free per-item scalar.
``stage_e4_e7``       the matched sweep over R_alpha, training and certification
                      moved together, plus the residual diagnostic.
``prep``/``split_ok``/``containment_check``  the guards each stage runs first.

Part of the code that produced every number in the paper.
"""
import json, math, os, time
import numpy as np

from .core import KL, Hb, certify, ref_alpha, contrastive_terms, cdf_of, sample_from
from .data import make_split
from .model import get_model, score_pairs, item_only_L, best_beta, seed_everything, train, BETAS

# ===================================================================== stages
def stage_e0(u, i, t, args, sink):
    """Identified-estimand diagnostics on the whole log. No split, no model."""
    ua, ui = np.unique(u, return_inverse=True); ia, ii = np.unique(i, return_inverse=True)
    N, M, L = len(ua), len(ia), len(ui)
    p = L / (N * M)
    PU = np.bincount(ui, minlength=N).astype(float); PU /= PU.sum()
    PV = np.bincount(ii, minlength=M).astype(float); PV /= PV.sum()
    LamV = KL(PV, np.full(M, 1.0 / M)); LamU = KL(PU, np.full(N, 1.0 / N))
    D = math.log(1.0 / p); Ipl = D - LamU - LamV
    # macro (per-user) vs micro (per-interaction)
    deg = np.bincount(ui, minlength=N).astype(float)
    macro = np.full(N, 1.0 / N); micro = deg / deg.sum()
    out = dict(users=N, items=M, interactions=L, p=p, H_b_p=Hb(p), D=D,
               Lambda_V=LamV, Lambda_U=LamU, I_plus_UV=Ipl,
               frac_Lambda_V=LamV / D, frac_Lambda_U=LamU / D, frac_I_plus=Ipl / D,
               K_for_all_Hb=math.exp(Hb(p) / p), K_for_nonvacuous=math.exp(LamV),
               KL_macro_micro=KL(macro, micro), decomposition_residual=abs(LamU + LamV + Ipl - D),
               pairs_per_user=L / N, pairs_per_item=L / M)
    rows = [dict(quantity=k, value=v) for k, v in out.items()]
    for k, v in out.items(): sink.add("e0", k, v, split="full_log")
    print("\n  E0 identified-estimand diagnostics")
    for k, v in out.items():
        print(f"    {k:<24} {v:,.6g}" if abs(v) < 1e6 else f"    {k:<24} {v:,.0f}")
    print(f"    CAVEAT on frac_I_plus: for P_+ uniform over the L distinct pairs,")
    print(f"      I_+(U;V) = H(U) + H(V) - log L, and H(U) = {math.log(N)-LamU:.4f} nats here.")
    print(f"      With {L/N:.1f} pairs per user, knowing the user nearly determines the item, so")
    print(f"      a large frac_I_plus reflects SPARSITY (the N/L ratio), not richer preference")
    print(f"      structure. Do NOT compare frac_I_plus across datasets without reporting")
    print(f"      pairs_per_user beside it.")
    return rows, out

MIN_TEST = 1000

def containment_check(args, sp, N, M, n_tr, n_te, n_pairs):
    """Print-only invariant audit. The train universe is a SUBSET of the full log, so
    its user and item counts can never exceed the E0 counts. Printing both, labelled,
    with retention, makes a transposed reading of the header impossible."""
    fu = getattr(args, "full_users", None); fi = getattr(args, "full_items", None)
    if fu is None: return
    n_held = n_pairs - n_train_rows(args, n_pairs)
    print(f"      CONTAINMENT [{sp}]  (train universe must be a subset of the full log)")
    print(f"        USERS : full log {fu:>12,}   train universe {N:>12,}   "
          f"retention {N/fu:6.1%}   subset? {'YES' if N <= fu else 'NO  <-- IMPOSSIBLE, BUG'}")
    print(f"        ITEMS : full log {fi:>12,}   train universe {M:>12,}   "
          f"retention {M/fi:6.1%}   subset? {'YES' if M <= fi else 'NO  <-- IMPOSSIBLE, BUG'}")
    if N > fu or M > fi:
        print("        !! CONTAINMENT VIOLATED. Stop and debug the split; do not use these results.")
    if n_held > 0:
        print(f"        TEST  : held-out pairs {n_held:>12,}   "
              f"kept after the warm-start filter {n_te:>10,}   retention {n_te/n_held:6.1%}")

def n_train_rows(args, n_pairs):
    return int(0.9 * n_pairs)

def split_ok(te, sp, args):
    """A held-out row survives only if BOTH its user and item appear in train.
    On very sparse data that can empty the test set -- say so instead of crashing."""
    n = len(te[0])
    if n >= MIN_TEST: return True
    print(f"\n  !! SKIPPING the {sp} split: only {n:,} usable test pairs (need {MIN_TEST:,}).")
    print(f"     Held-out rows are dropped when their user or item is unseen in train, which")
    print(f"     on sparse data removes almost everything. Fixes, in order of preference:")
    print(f"       --min-count 5     iterative k-core, the standard preprocessing")
    print(f"       drop --subsample  if you are subsampling, it makes this much worse")
    return False

def prep(u, i, args):
    tr, te, N, M = make_split(u, i, min_count=args.min_count)
    P_tr = np.bincount(tr[1], minlength=M).astype(float); P_tr /= P_tr.sum()
    P_pos = np.bincount(te[1], minlength=M).astype(float); P_pos /= P_pos.sum()
    p_star = len(te[0]) / (N * M)
    return tr, te, N, M, P_tr, P_pos, p_star

def stage_e1_e2_e3(u, i, t, args, sink, ckdir, dev, torch, do_e2, do_e3):
    ARMS = ["uniform", "popularity", "inbatch", "inbatch_logq"]
    e1, e2, e3 = [], [], []
    for sp in ["random"]:
        tr, te, N, M, P_tr, P_pos, p_star = prep(u, i, args)
        if not split_ok(te, sp, args): continue
        drift = KL(P_pos, P_tr)
        refs = {"uniform": ref_alpha(P_tr, 0.0), "sqrt_pop": ref_alpha(P_tr, 0.5),
                "popularity": ref_alpha(P_tr, 1.0)}
        n_cert = min(args.e1_cert_n, len(te[0]))
        n_cert2 = min(args.e2_cert_n, len(te[0]))
        hw = (math.log(args.cert_K) + 2 / args.tau) * math.sqrt(math.log(2 / 0.05) / (2 * n_cert))
        hw2 = (math.log(args.cert_K) + 2 / args.tau) * math.sqrt(math.log(2 / 0.05) / (2 * n_cert2))
        containment_check(args, sp, N, M, len(tr[0]), len(te[0]), getattr(args, "n_pairs", 0))
        print(f"\n  === E1 [{sp}]  train {len(tr[0]):,} test {len(te[0]):,} "
              f"users(N)={N:,} items(M)={M:,}  p={p_star:.4e} D={math.log(1/p_star):.4f} "
              f"drift={drift:.4f} hw={hw:.4f}")
        for nm, R in refs.items():
            LR = KL(P_pos, R + 1e-300)
            orc, _ = item_only_L(np.log((P_pos + 1e-300) / R), P_pos, R, args)
            hon, _ = item_only_L(np.log((P_tr + 1e-300) / R), P_pos, R, args)
            for k, v in [("Lambda_R", LR), ("oracle_baseline", orc), ("honest_baseline", hon)]:
                sink.add("e2", k, v, split=sp, cert_ref=nm)
            e2.append(dict(split=sp, train_law="[baseline]", cert_ref=nm, seed="",
                           Lambda_R=LR, oracle=orc, honest=hon, L_hat="", L_lcb="",
                           adjusted="", ndcg20="", hr10="", half_width=hw2))
        g = np.random.default_rng(12345); perm = g.permutation(len(te[0]))
        sel = perm[:n_cert]; sel2 = perm[:n_cert2]
        if n_cert != n_cert2:
            print(f"      E1 n={n_cert:,} (hw {hw:.4f})   E2 n={n_cert2:,} (hw {hw2:.4f})")
        for arm in ARMS:
            for sd in range(args.seeds_e1):
                args.seed = sd
                tag = f"{args.dataset}_{sp}_e1_{arm}_s{sd}_d{args.dim}_e{args.epochs}"
                U, I, _ = get_model(tag, arm, tr, N, M, args, P_tr, dev, torch, ckdir)
                # ---- E1: certify under uniform
                Ru = refs["uniform"]; LRu = KL(P_pos, Ru + 1e-300)
                P, Nn, rk = score_pairs(U, I, te[0], te[1], sel, Ru, args, dev, torch)
                c = certify(contrastive_terms(P, Nn), args.cert_K, 1 / args.tau, p_star, LRu)
                nd = float(np.where(rk <= 20, 1.0 / np.log2(rk + 1), 0.0).mean())
                hr = float((rk <= 10).mean())
                e1.append(dict(split=sp, train_law=arm, seed=sd, cert_ref="uniform",
                               L_hat=c['L_hat'], L_lcb=c['L_lcb'], half_width=c['half_width'],
                               adjusted=c['adjusted'], certified_info=c['certified_info'],
                               Lambda_R=LRu, ndcg20=nd, hr10=hr, p=p_star,
                               D=math.log(1 / p_star), drift=drift))
                for k in ("L_hat", "L_lcb", "adjusted", "certified_info"):
                    sink.add("e1", k, c[k], split=sp, train_law=arm, seed=sd, cert_ref="uniform")
                sink.add("e1", "ndcg20", nd, split=sp, train_law=arm, seed=sd, cert_ref="uniform")
                sink.add("e1", "hr10", hr, split=sp, train_law=arm, seed=sd, cert_ref="uniform")
                print(f"    [{sp}/{arm}/s{sd}] L_hat={c['L_hat']:.4f} LCB={c['L_lcb']:.4f} "
                      f"adj={c['adjusted']:.4f} NDCG@20={nd:.4f}", flush=True)
                # ---- E2: cross-certify under every reference
                if do_e2:
                    for nm, R in refs.items():
                        if nm == "uniform" and n_cert == n_cert2:
                            cc, ndc, hrc = c, nd, hr
                        else:
                            LR = KL(P_pos, R + 1e-300)
                            P2, N2, r2 = score_pairs(U, I, te[0], te[1], sel2, R, args, dev, torch)
                            cc = certify(contrastive_terms(P2, N2), args.cert_K, 1 / args.tau, p_star, LR)
                            ndc = float(np.where(r2 <= 20, 1.0 / np.log2(r2 + 1), 0.0).mean())
                            hrc = float((r2 <= 10).mean())
                        e2.append(dict(split=sp, train_law=arm, cert_ref=nm, seed=sd,
                                       Lambda_R=KL(P_pos, R + 1e-300), oracle="", honest="",
                                       L_hat=cc['L_hat'], L_lcb=cc['L_lcb'],
                                       adjusted=cc['adjusted'], ndcg20=ndc, hr10=hrc,
                                       half_width=cc['half_width']))
                        for k in ("L_hat", "L_lcb", "adjusted"):
                            sink.add("e2", k, cc[k], split=sp, train_law=arm, seed=sd, cert_ref=nm)
                        sink.add("e2", "ndcg20", ndc, split=sp, train_law=arm, seed=sd, cert_ref=nm)
                # ---- E3: item-only component of this critic
                if do_e3 and sd == 0:
                    with torch.no_grad():
                        zb = torch.nn.functional.normalize(
                            U(torch.as_tensor(te[0][sel], device=dev)), dim=1).mean(0)
                        sbar = ((torch.nn.functional.normalize(I.weight, dim=1) @ zb)
                                / args.tau).float().cpu().numpy().astype(np.float64)
                    for lbl, Pp in [("log_P_pos", P_pos), ("log_P_train", P_tr)]:
                        mk = Pp > 0; y = np.log(Pp[mk]); x = sbar[mk]
                        r = float(np.corrcoef(x, y)[0, 1])
                        e3.append(dict(split=sp, train_law=arm, target=lbl, corr=r, r2=r * r,
                                       model_range=float(x.max() - x.min()),
                                       target_range=float(y.max() - y.min())))
                        sink.add("e3", f"r2_vs_{lbl}", r * r, split=sp, train_law=arm, seed=0)
                del U, I
    return e1, e2, e3

def stage_e3b(u, i, t, args, sink, ckdir, dev, torch):
    """The larger-model check: more dims, more epochs, and a free per-item scalar, so the
    critic CAN represent the item-only solution it is being scored against. The Hoeffding
    range grows to G = 1/tau + B, which is reported."""
    ARMS = ["uniform", "popularity", "inbatch_logq"]
    rows = []
    B = float(args.cap_item_bias); G = 1.0 / args.tau + max(0.0, B)
    for sp in ["random"]:
        tr, te, N, M, P_tr, P_pos, p_star = prep(u, i, args)
        if not split_ok(te, sp, args): continue
        refs = {"uniform": ref_alpha(P_tr, 0.0), "sqrt_pop": ref_alpha(P_tr, 0.5),
                "popularity": ref_alpha(P_tr, 1.0)}
        n_cert = min(args.e3b_cert_n or args.cert_n, len(te[0]))
        hw = (math.log(args.cert_K) + 2 * G) * math.sqrt(math.log(2 / 0.05) / (2 * n_cert))
        g = np.random.default_rng(12345); sel = g.permutation(len(te[0]))[:n_cert]
        base = {}
        for nm, R in refs.items():
            LR = KL(P_pos, R + 1e-300)
            orc, ose = item_only_L(np.log((P_pos + 1e-300) / R), P_pos, R, args)
            hon, hse = item_only_L(np.log((P_tr + 1e-300) / R), P_pos, R, args)
            base[nm] = dict(Lambda=LR, oracle=orc, honest=hon, orc_mc=ose, hon_mc=hse)
        containment_check(args, sp, N, M, len(tr[0]), len(te[0]), getattr(args, "n_pairs", 0))
        print(f"\n  === E3b CAPACITY CONTROL [{sp}]  dim={args.cap_dim} epochs={args.cap_epochs} "
              f"item_bias={B:g} -> G={G:.1f}  hw={hw:.4f}")
        for arm in ARMS:
            args.seed = 0
            tag = (f"{args.dataset}_{sp}_e3b_{arm}_s0_d{args.cap_dim}_e{args.cap_epochs}"
                   f"_b{B:g}")
            U, I, braw = get_model(tag, arm, tr, N, M, args, P_tr, dev, torch, ckdir,
                                   dim=args.cap_dim, epochs=args.cap_epochs, item_bias=B)
            bb = None
            if braw is not None:
                with torch.no_grad(): bb = B * torch.tanh(braw)
            for nm, R in refs.items():
                P, Nn, rk = score_pairs(U, I, te[0], te[1], sel, R, args, dev, torch, bias=bb)
                c = certify(contrastive_terms(P, Nn), args.cert_K, G, p_star, base[nm]['Lambda'])
                nd = float(np.where(rk <= 20, 1.0 / np.log2(rk + 1), 0.0).mean())
                # point estimate
                pt_h = c['L_hat'] - base[nm]['honest']; pt_o = c['L_hat'] - base[nm]['oracle']
                # CONSERVATIVE: use the model's LCB and subtract the baseline's own MC error,
                # which is NOT inside the Hoeffding half-width. Verdicts use these.
                cs_h = c['L_lcb'] - base[nm]['honest'] - base[nm]['hon_mc']
                cs_o = c['L_lcb'] - base[nm]['oracle'] - base[nm]['orc_mc']
                rows.append(dict(split=sp, train_law=arm, cert_ref=nm, dim=args.cap_dim,
                                 epochs=args.cap_epochs, item_bias=B, G=G,
                                 Lambda_R=base[nm]['Lambda'], oracle=base[nm]['oracle'],
                                 honest=base[nm]['honest'], oracle_mc95=base[nm]['orc_mc'],
                                 honest_mc95=base[nm]['hon_mc'],
                                 L_hat=c['L_hat'], L_lcb=c['L_lcb'],
                                 vs_honest_point=pt_h, vs_oracle_point=pt_o,
                                 vs_honest_lcb=cs_h, vs_oracle_lcb=cs_o,
                                 adjusted=c['adjusted'], ndcg20=nd, half_width=c['half_width']))
                for k, v in [("L_hat", c['L_hat']), ("L_lcb", c['L_lcb']),
                             ("vs_honest_point", pt_h), ("vs_oracle_point", pt_o),
                             ("vs_honest_lcb", cs_h), ("vs_oracle_lcb", cs_o),
                             ("adjusted", c['adjusted'])]:
                    sink.add("e3b", k, v, split=sp, train_law=arm, cert_ref=nm, seed=0)
                print(f"    [{sp}/{arm}/R={nm}] L={c['L_hat']:.4f} LCB={c['L_lcb']:.4f} | "
                      f"point -hon={pt_h:+.4f} -orc={pt_o:+.4f} | "
                      f"conservative -hon={cs_h:+.4f} -orc={cs_o:+.4f}", flush=True)
            del U, I
    if rows:
        hw = rows[0]['half_width']
        print(f"\n  E3b VERDICT   PER SPLIT (half-width {hw:.4f}; verdicts use the CONSERVATIVE")
        print(f"    column L_lcb - baseline - baseline_MC95, so the model's Hoeffding interval AND")
        print(f"    the baseline's Monte-Carlo error are both already inside the number)")
        for spl in sorted({r['split'] for r in rows}):
            rs = [r for r in rows if r['split'] == spl]
            p1 = max(r['vs_honest_point'] for r in rs); p2 = max(r['vs_oracle_point'] for r in rs)
            q1 = max(r['vs_honest_lcb'] for r in rs);   q2 = max(r['vs_oracle_lcb'] for r in rs)
            print(f"    --- {spl}")
            print(f"      vs HONEST baseline: point {p1:+.4f}   conservative {q1:+.4f}  -> "
                  + ("model carries real personalised signal" if q1 > 0 else
                     "does NOT clear a train-popularity lookup table once uncertainty is "
                     "charged: fix capacity before concluding anything"))
            print(f"      vs ORACLE floor   : point {p2:+.4f}   conservative {q2:+.4f}  -> "
                  + ("clears the floor; the adjusted certificate is NON-EMPTY on this split" if q2 > 0
                     else "nothing clears the floor even with extra capacity and a free item "
                          "scalar; the capacity confound is dead and the zero is reportable"))
    return rows

def stage_e4_e7(u, i, t, args, sink, ckdir, dev, torch, do_e7):
    e4, e7 = [], []
    alphas = [float(x) for x in args.alphas.split(",")]
    for sp in ["random"]:
        tr, te, N, M, P_tr, P_pos, p_star = prep(u, i, args)
        if not split_ok(te, sp, args): continue
        te_u, te_i = te; drift = KL(P_pos, P_tr)
        g = np.random.default_rng(12345); perm = g.permutation(len(te_u))
        cal = perm[:args.calib_n]
        cer = perm[args.calib_n:args.calib_n + args.cert_n]
        if len(cer) < 1000:
            cal = perm[:max(1000, len(perm) // 5)]; cer = perm[len(cal):]
        clip = G = 1.0 / args.tau
        hw = (math.log(args.cert_K) + 2 * G) * math.sqrt(math.log(2 / 0.05) / (2 * len(cer)))
        train_users = np.unique(tr[0])
        mk = P_pos > 0; logPpos = np.log(P_pos[mk])
        containment_check(args, sp, N, M, len(tr[0]), len(te_u), getattr(args, "n_pairs", 0))
        print(f"\n  === E4 [{sp}]  calib {len(cal):,} cert {len(cer):,} (disjoint) hw={hw:.4f} "
              f"drift={drift:.4f}  range(log P_+)={logPpos.max()-logPpos.min():.2f}")
        if len(cal) < 20000:
            print(f"      !! calib n={len(cal):,} is small; zbar is estimated from it, so the E7 "
                  f"item-free audit may refuse the calib_FIXED row. Raise --calib-n if the test "
                  f"split allows.")
        for a in alphas:
            R = ref_alpha(P_tr, a); LR = KL(P_pos, R + 1e-300)
            orc, orse = item_only_L(np.log((P_pos + 1e-300) / R), P_pos, R, args)
            hon, honse = item_only_L(np.log((P_tr + 1e-300) / R), P_pos, R, args)
            if a == alphas[0] and max(orse, honse) > 0.02:
                print(f"      note: baseline MC error +/-{max(orse,honse):.4f} adds to the "
                      f"Hoeffding half-width in adjusted_honest")
            for sd in range(args.seeds_e4):
                args.seed = sd
                tag = f"{args.dataset}_{sp}_e4_a{a:g}_s{sd}_d{args.dim}_e{args.epochs}"
                U, I, _ = get_model(tag, R, tr, N, M, args, P_tr, dev, torch, ckdir)
                P, Nn, rk = score_pairs(U, I, te_u, te_i, cer, R, args, dev, torch)
                c = certify(contrastive_terms(P, Nn), args.cert_K, G, p_star, LR)
                nd = float(np.where(rk <= 20, 1.0 / np.log2(rk + 1), 0.0).mean())
                e4.append(dict(split=sp, alpha=a, seed=sd, Lambda_R=LR, oracle=orc,
                               oracle_mc95=orse, honest=hon, L_hat=c['L_hat'], L_lcb=c['L_lcb'],
                               adjusted_honest=c['L_hat'] - hon, honest_mc95=honse,
                               adjusted_oracle_lcb=c['adjusted'],
                               identity_orc_minus_hon=orc - hon, drift=drift,
                               identity_dev=(orc - hon) - drift, ndcg20=nd,
                               half_width=c['half_width'], p=p_star, D=math.log(1 / p_star)))
                for k, v in [("L_hat", c['L_hat']), ("L_lcb", c['L_lcb']),
                             ("adjusted_honest", c['L_hat'] - hon),
                             ("adjusted_oracle_lcb", c['adjusted']), ("Lambda_R", LR),
                             ("oracle_baseline", orc), ("honest_baseline", hon),
                             ("identity_dev", (orc - hon) - drift), ("ndcg20", nd)]:
                    sink.add("e4", k, v, split=sp, alpha=a, seed=sd, cert_ref=f"alpha={a:g}")
                print(f"    [{sp}/a{a:g}/s{sd}] L={c['L_hat']:.4f} LCB={c['L_lcb']:.4f} "
                      f"adj_hon={c['L_hat']-hon:.4f} adj_orc={c['adjusted']:.4f}", flush=True)
                # ---------- E7: three zbar variants + calibrated residual
                if do_e7:
                    nz = lambda ix: torch.nn.functional.normalize(
                        U(torch.as_tensor(ix, device=dev)), dim=1).mean(0)
                    with torch.no_grad():
                        zb = {"cert_LEAKY": nz(te_u[cer]), "train": nz(train_users),
                              "calib_FIXED": nz(te_u[cal])}
                        zc = zb["cert_LEAKY"]
                        Cn = torch.nn.functional.normalize(I.weight, dim=1)
                    for vname, z in zb.items():
                        with torch.no_grad():
                            mv = ((Cn @ (zc - z)) / args.tau).float().cpu().numpy().astype(np.float64)
                        io = float(mv[mk].max() - mv[mk].min())
                        io_r2 = float(np.corrcoef(mv[mk], logPpos)[0, 1] ** 2) if io > 1e-12 else 0.0
                        cp_, cn_, _ = score_pairs(U, I, te_u, te_i, cal, R, args, dev, torch, z)
                        b = best_beta(cp_, cn_, clip); del cp_, cn_
                        sp_, sn_, _ = score_pairs(U, I, te_u, te_i, cer, R, args, dev, torch, z)
                        cc = certify(contrastive_terms(np.clip(b * sp_, -clip, clip),
                                                       np.clip(b * sn_, -clip, clip)),
                                     args.cert_K, G, p_star, 0.0)
                        c1 = certify(contrastive_terms(np.clip(sp_, -clip, clip),
                                                       np.clip(sn_, -clip, clip)),
                                     args.cert_K, G, p_star, 0.0)
                        del sp_, sn_
                        n_ref = len([float(x) for x in args.alphas.split(",")])
                        n_cell = n_ref * max(1, args.seeds_e4)   # alpha x seed grid
                        hw_b = (math.log(args.cert_K) + 2 * G) * math.sqrt(
                            math.log(2 / (0.05 / n_ref)) / (2 * len(cer)))
                        hw_j = (math.log(args.cert_K) + 2 * G) * math.sqrt(
                            math.log(2 / (0.05 / n_cell)) / (2 * len(cer)))
                        e7.append(dict(split=sp, alpha=a, seed=sd, zbar=vname, beta_star=b,
                                       L_lcb_bonf=cc['L_hat'] - hw_b, half_width_bonf=hw_b,
                                       L_lcb_joint=cc['L_hat'] - hw_j, half_width_joint=hw_j,
                                       n_refs=n_ref, n_cells=n_cell,
                                       item_only_range=io, item_only_r2=io_r2,
                                       target_range=float(logPpos.max() - logPpos.min()),
                                       L_hat=cc['L_hat'], L_lcb=cc['L_lcb'],
                                       L_hat_uncalib=c1['L_hat'], Lambda_R=LR,
                                       half_width=cc['half_width']))
                        for k, v in [("resid_L_hat", cc['L_hat']), ("resid_L_lcb", cc['L_lcb']),
                                     ("resid_L_lcb_bonf", cc['L_hat'] - hw_b),
                                     ("resid_L_lcb_joint", cc['L_hat'] - hw_j),
                                     ("resid_L_hat_uncalib", c1['L_hat']), ("beta_star", b),
                                     ("item_only_range", io), ("item_only_r2", io_r2)]:
                            sink.add("e7", k, v, split=sp, alpha=a, seed=sd, cert_ref=vname)
                del U, I
    return e4, e7
