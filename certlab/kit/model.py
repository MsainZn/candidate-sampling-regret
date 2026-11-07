"""The two-tower critic: fitting it, scoring with it, and the item-only baseline.

``seed_everything``  fixes python, numpy and torch seeds for one fit.
``train``            fits normalised user and item embedding tables by ascending the
                     K-slot contrastive objective. One candidate set is shared by a
                     batch and the user anchor enters all K slots, which is the
                     anchored protocol of the paper rather than the
                     joint one.
``get_model``        loads a checkpoint when one exists, so a re-run never retrains.
``score_pairs``      the certification pass: frozen model, fresh candidates per
                     positive, no gradients.
``item_only_L``      the score an item-only critic could reach, the benchmark the
                     adjustment is measured against.
``best_beta``        the temperature used by the residual diagnostic, chosen on the
                     calibration split only.

Part of the code that produced every number in the paper.
"""
import math, os, time
import numpy as np

from .core import cdf_of, sample_from, ref_alpha, contrastive_terms

# ===================================================================== model
def seed_everything(seed, torch):
    """Re-seed immediately before each model is built. Seeding once in main() is NOT
    enough: training arms sequentially advances the global RNG, so arm 2 would get a
    different initialisation from arm 1 and the arms would stop being a paired
    comparison. Re-seeding per model makes every arm at a given seed start from the
    SAME weights, so the only difference between arms is the sampling law."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if hasattr(torch, "cuda") and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def train(law_or_R, tr, N, M, args, P_tr, dim, epochs, item_bias, dev, torch):
    """law_or_R: one of the named laws, or a probability vector to sample negatives from."""
    tr_u, tr_i = tr
    seed_everything(args.seed, torch)
    U = torch.nn.Embedding(N, dim).to(dev); I = torch.nn.Embedding(M, dim).to(dev)
    torch.nn.init.normal_(U.weight, std=0.1); torch.nn.init.normal_(I.weight, std=0.1)
    params = list(U.parameters()) + list(I.parameters())
    braw = None
    if item_bias > 0:
        braw = torch.nn.Parameter(torch.zeros(M, device=dev)); params.append(braw)
    opt = torch.optim.Adam(params, lr=args.lr)
    n = len(tr_u); rng = np.random.default_rng(args.seed)
    logq = np.log(P_tr * M + 1e-12)
    named = isinstance(law_or_R, str)
    cdf = None if named else cdf_of(law_or_R)
    cdf_pop = cdf_of(P_tr)
    for ep in range(epochs):
        perm = rng.permutation(n); tot = 0.0; nb = 0; t0 = time.time()
        for s in range(0, n, args.batch):
            b = perm[s:s + args.batch]
            if len(b) < 8: continue
            if named and law_or_R in ("inbatch", "inbatch_logq") and len(b) < args.K:
                continue        # a partial final batch would give this arm a smaller K
            bu = torch.as_tensor(tr_u[b], device=dev); bi = torch.as_tensor(tr_i[b], device=dev)
            inbatch = named and law_or_R in ("inbatch", "inbatch_logq")
            if inbatch:
                # TRUE in-batch negatives: the shared candidate set is drawn from the
                # OTHER ROWS OF THIS BATCH, WITHOUT replacement, and each row's own
                # item is masked out of its own negative slots (self-collision).
                # Sampling batch rows WITH replacement instead would make this arm an
                # i.i.d. draw from P_train(V) -- i.e. a duplicate of the popularity arm.
                k = min(args.K - 1, len(b))
                negs = tr_i[rng.choice(b, size=k, replace=False)]
            elif not named:                    negs = sample_from(cdf, args.K - 1, rng)
            elif law_or_R == "uniform":        negs = rng.integers(0, M, size=args.K - 1)
            else:                              negs = sample_from(cdf_pop, args.K - 1, rng)
            negs = np.asarray(negs, dtype=np.int64); nt = torch.as_tensor(negs, device=dev)
            zu = torch.nn.functional.normalize(U(bu), dim=1)
            cp = torch.nn.functional.normalize(I(bi), dim=1)
            cn = torch.nn.functional.normalize(I(nt), dim=1)
            sp = (zu * cp).sum(1) / args.tau
            sn = (zu @ cn.T) / args.tau
            if braw is not None:                     # free per-item scalar, clamped to [-B,B]
                bb = item_bias * torch.tanh(braw)
                sp = sp + bb[bi]; sn = sn + bb[nt][None, :]
            if named and law_or_R == "inbatch_logq":
                sp = sp - torch.as_tensor(logq[tr_i[b]], device=dev, dtype=sp.dtype)
                sn = sn - torch.as_tensor(logq[negs], device=dev, dtype=sn.dtype)[None, :]
            if inbatch:
                # mask a row's own item wherever it landed in the shared negative set
                coll = (negs[None, :] == tr_i[b][:, None])
                if coll.any():
                    sn = sn + torch.as_tensor(np.where(coll, -1e9, 0.0), device=dev, dtype=sn.dtype)
            logits = torch.cat([sp[:, None], sn], dim=1)
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(len(b), dtype=torch.long, device=dev))
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss.detach()); nb += 1
        print(f"      epoch {ep+1}/{epochs} loss={tot/max(nb,1):.4f} ({time.time()-t0:.0f}s)", flush=True)
    return U, I, braw

def get_model(tag, law_or_R, tr, N, M, args, P_tr, dev, torch, ckdir,
              dim=None, epochs=None, item_bias=0.0):
    dim = dim or args.dim; epochs = epochs or args.epochs
    path = os.path.join(ckdir, tag + ".pt")
    if os.path.exists(path):
        st = torch.load(path, map_location=dev)
        U = torch.nn.Embedding(N, dim).to(dev); I = torch.nn.Embedding(M, dim).to(dev)
        U.load_state_dict(st['U']); I.load_state_dict(st['I'])
        b = st.get('b')
        braw = torch.nn.Parameter(b.to(dev)) if b is not None else None
        print(f"    [{tag}] loaded", flush=True); return U, I, braw
    print(f"    [{tag}] training", flush=True)
    U, I, braw = train(law_or_R, tr, N, M, args, P_tr, dim, epochs, item_bias, dev, torch)
    torch.save({'U': U.state_dict(), 'I': I.state_dict(),
                'b': (braw.detach().cpu() if braw is not None else None)}, path)
    return U, I, braw

def score_pairs(U, I, te_u, te_i, idx, R, args, dev, torch, zbar=None, bias=None):
    rng = np.random.default_rng(args.seed + 4242)
    Kc = args.cert_K; cdf = cdf_of(R)
    P = np.empty(len(idx)); Nn = np.empty((len(idx), Kc - 1), dtype=np.float32)
    rk = np.empty(len(idx), dtype=np.int64)
    with torch.no_grad():
        for s in range(0, len(idx), args.cert_chunk):
            b = idx[s:s + args.cert_chunk]; nb = len(b)
            negs = sample_from(cdf, (nb, Kc - 1), rng).astype(np.int64)
            zu = torch.nn.functional.normalize(U(torch.as_tensor(te_u[b], device=dev)), dim=1)
            if zbar is not None: zu = zu - zbar[None, :]
            ti = torch.as_tensor(te_i[b], device=dev); tn = torch.as_tensor(negs.ravel(), device=dev)
            cp = torch.nn.functional.normalize(I(ti), dim=1)
            cn = torch.nn.functional.normalize(I(tn), dim=1).reshape(nb, Kc - 1, -1)
            spt = (zu * cp).sum(1) / args.tau
            snt = torch.einsum('bd,bkd->bk', zu, cn) / args.tau
            if bias is not None:
                spt = spt + bias[ti]; snt = snt + bias[tn].reshape(nb, Kc - 1)
            sp = spt.float().cpu().numpy(); sn = snt.float().cpu().numpy()
            P[s:s + nb] = sp; Nn[s:s + nb] = sn
            rk[s:s + nb] = 1 + (sn > sp[:, None]).sum(1)
    return P, Nn, rk

def item_only_L(t, P_pos, R, args, n=200000, seed=7):
    """Returns (value, 95% MC half-width). The MC error is separate from the Hoeffding
    half-width and is NOT inside it, so adjusted_honest = L_hat - honest carries both."""
    rng = np.random.default_rng(seed)
    cp = cdf_of(P_pos); cr = cdf_of(R); acc = []
    for s in range(0, n, 10000):
        c = min(10000, n - s)
        acc.append(contrastive_terms(t[sample_from(cp, c, rng)],
                                     t[sample_from(cr, (c, args.cert_K - 1), rng)]))
    tm = np.concatenate(acc)
    return float(tm.mean()), float(1.96 * tm.std(ddof=1) / math.sqrt(n))

BETAS = np.exp(np.linspace(math.log(0.02), math.log(6.0), 45))

def best_beta(pos, neg, clip):
    vals = [contrastive_terms(np.clip(b * pos, -clip, clip), np.clip(b * neg, -clip, clip)).mean()
            for b in BETAS]
    j = int(np.argmax(vals)); return float(BETAS[j])
