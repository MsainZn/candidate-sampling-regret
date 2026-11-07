"""Core quantities: the contrastive score, its certificate, and the reference family.

``contrastive_terms``  the per-positive term log K + T(x+) - logsumexp over the K candidates.
``certify``            turns those terms into a mean, a Hoeffding half-width and the
                       item-adjusted score; ``Lambda`` is the term subtracted.
``KL``, ``Hb``         Kullback-Leibler divergence and the binary entropy.
``ref_alpha``          the matched family R_alpha proportional to P_tr ** alpha,
                       running from catalogue-uniform at alpha=0 to training
                       popularity at alpha=1.
``cdf_of``/``sample_from``  inverse-CDF sampling of negatives from a discrete law.

Part of the code that produced every number in the paper.
"""
import math
import numpy as np

# ===================================================================== core math
def contrastive_terms(pos, neg):
    """log K + T(pos) - logsumexp over the K candidates. K = 1 + n_negatives."""
    K = neg.shape[1] + 1
    allv = np.concatenate([np.asarray(pos, np.float64)[:, None], np.asarray(neg, np.float64)], axis=1)
    m = allv.max(axis=1, keepdims=True)
    lse = m[:, 0] + np.log(np.exp(allv - m).sum(axis=1))
    return math.log(K) + allv[:, 0] - lse

def certify(terms, K, G, p, Lambda=0.0, delta=0.05):
    n = len(terms); Lhat = float(terms.mean())
    half = (math.log(K) + 2.0 * G) * math.sqrt(math.log(2.0 / delta) / (2.0 * n))
    lcb = Lhat - half
    return dict(n=n, L_hat=Lhat, half_width=half, L_lcb=lcb,
                certified_info=p * max(lcb, 0.0), adjusted=max(lcb - Lambda, 0.0))

def KL(q, r):
    q = np.asarray(q, float); r = np.asarray(r, float); m = q > 0
    return float((q[m] * np.log(q[m] / r[m])).sum())

def Hb(p):
    return float(-p * math.log(p) - (1 - p) * math.log(1 - p))

def ref_alpha(P_tr, a):
    R = P_tr ** a
    return R / R.sum()

def sample_from(cdf, shape, rng):
    return np.searchsorted(cdf, rng.random(shape), side="right").clip(0, len(cdf) - 1)

def cdf_of(P):
    c = np.cumsum(P); return c / c[-1]
