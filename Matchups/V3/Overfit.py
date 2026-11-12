r"""
Overfitting estimates for model selection on one holdout (reading: Bailey, Borwein, Lopez de Prado, Zhu 2014-17;
White 2000 reality check; Romano-Wolf).

PboCscv: Combinatorially symmetric cross-validation. The selection-split games are cut into S contiguous date
blocks; for every way of choosing S/2 blocks as in-sample, pick the trial with the best in-sample mean loss and
record its relative rank out of sample. PBO = share of splits where the in-sample winner is below the
out-of-sample median (logit of relative rank <= 0).

SelectionAdjusted: a max-statistic (White / Romano-Wolf style) day-clustered bootstrap of the gain over a baseline
across ALL trials, giving a simultaneous lower bound for the best trial's gain that accounts for having picked it.
"""

from itertools import combinations

import numpy as np


def PboCscv(L, blocks=16):
    """L: (trials x games) loss matrix, games in date order. Returns PBO, logits, and the median OOS rank."""
    L = np.asarray(L, float)
    T, N = L.shape
    edges = np.linspace(0, N, blocks + 1).astype(int)
    B = np.array([L[:, a:b].sum(1) for a, b in zip(edges[:-1], edges[1:])]).T     # trials x blocks (sums)
    nb = np.array([b - a for a, b in zip(edges[:-1], edges[1:])], float)
    logits = []
    for ins in combinations(range(blocks), blocks // 2):
        mask = np.zeros(blocks, bool)
        mask[list(ins)] = True
        is_mean = B[:, mask].sum(1) / nb[mask].sum()
        oos_mean = B[:, ~mask].sum(1) / nb[~mask].sum()
        best = int(np.argmin(is_mean))
        # relative rank of the IS winner out of sample (1 = best); omega in (0, 1)
        rank = 1 + np.sum(oos_mean < oos_mean[best]) + 0.5 * (np.sum(oos_mean == oos_mean[best]) - 1)
        omega = 1.0 - (rank - 0.5) / T
        logits.append(np.log(omega / (1 - omega)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0)), "n_splits": len(logits), "n_trials": T,
            "median_logit": float(np.median(logits))}


def SelectionAdjusted(L, base, clusters, reps=2000, seed=11):
    """Gain_k = mean(base - L_k). Returns the best trial index, its gain, naive CI and the max-statistic
    simultaneous 95% lower bound (gain_best - q95 of max_k (gain*_k - gain_k))."""
    L, base = np.asarray(L, float), np.asarray(base, float)
    G = base[None, :] - L                                   # trials x games, positive = better than base
    keys, inv = np.unique(np.asarray(clusters), return_inverse=True)
    k = len(keys)
    S = np.array([np.bincount(inv, weights=g, minlength=k) for g in G])   # trials x days
    C = np.bincount(inv, minlength=k).astype(float)
    gain = S.sum(1) / C.sum()
    rng = np.random.default_rng(seed)
    mx, best_boot = [], []
    best = int(np.argmax(gain))
    for _ in range(reps):
        d = rng.integers(0, k, size=k)
        gb = S[:, d].sum(1) / C[d].sum()
        mx.append(np.max(gb - gain))
        best_boot.append(gb[best])
    q = float(np.quantile(mx, 0.95))
    return {"best": best, "gain": float(gain[best]), "naive_lo": float(np.quantile(best_boot, 0.025)),
            "naive_hi": float(np.quantile(best_boot, 0.975)), "adjusted_lo": float(gain[best] - q),
            "gains": gain.tolist()}
