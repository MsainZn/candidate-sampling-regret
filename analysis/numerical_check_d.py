"""Appendix D(d): exact finite-K contrastive score on a binary instance.

L_K(T; Q, R) = log K + sum_x Q(x) [ T(x) - E_n log( e^{T(x)} + n e^{T(a)} + (K-1-n) e^{T(b)} ) ],
where the K-1 candidates are drawn independently from R and enter only through the count
n ~ Binomial(K-1, R(a)). The binomial expectation is summed exactly in double precision, so no
quantity is estimated; for K <= 8 the result is also checked against enumeration of all K-tuples.

Usage: python3 analysis/numerical_check_d.py
"""
import itertools
import math

import numpy as np


def score(T, Q, R, K):
    n = np.arange(K, dtype=float)
    lg = np.array([math.lgamma(i + 1) for i in range(K)])          # lg[i] = log(i!)
    logp = lg[K - 1] - lg[:K] - lg[::-1][:K] + n * math.log(R[0]) + (K - 1 - n) * math.log(R[1])
    p = np.exp(logp)
    out = math.log(K)
    for x in (0, 1):
        den = np.log(math.exp(T[x]) + n * math.exp(T[0]) + (K - 1 - n) * math.exp(T[1]))
        out += Q[x] * (T[x] - float(np.dot(p, den)))
    return out


def score_enum(T, Q, R, K):
    """Brute force over the positive slot and all K-1 candidates (small K only)."""
    out = 0.0
    for x in (0, 1):
        for c in itertools.product((0, 1), repeat=K - 1):
            w = Q[x] * math.prod(R[y] for y in c)
            out += w * (math.log(K) + T[x] - math.log(math.exp(T[x]) + sum(math.exp(T[y]) for y in c)))
    return out


def kl(Q, R):
    return sum(q * math.log(q / r) for q, r in zip(Q, R))


def dv(T, Q, R):
    return sum(q * t for q, t in zip(Q, T)) - math.log(sum(r * math.exp(t) for r, t in zip(R, T)))


if __name__ == "__main__":
    Q, R = (0.9, 0.1), (0.25, 0.75)
    t_opt = tuple(math.log(q / r) for q, r in zip(Q, R))   # log density ratio, optimal as K grows
    t_fix = (0.0, 2.0)                                      # a fixed, misspecified scoring function
    for K in (2, 4, 8):
        for T in (t_opt, t_fix):
            assert abs(score(T, Q, R, K) - score_enum(T, Q, R, K)) < 1e-12
    print(f"KL(Q||R) = {kl(Q, R):.6f}   DV(t_fix) = {dv(t_fix, Q, R):.6f}")
    prev = -1e9
    for K in (2, 4, 8, 32, 128, 512, 2048, 8192, 32768):
        s_opt, s_fix = score(t_opt, Q, R, K), score(t_fix, Q, R, K)
        assert s_opt <= min(math.log(K), kl(Q, R)) and s_opt > prev
        prev = s_opt
        print(f"K={K:6d}  optimal {s_opt:.6f}   fixed {s_fix:.6f}   KL - optimal {kl(Q, R) - s_opt:.2e}")
