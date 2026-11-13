"""Statistics: clustered bootstraps, drawdown, crossings, PBO (CSCV). numpy only."""

import itertools
import math

import numpy as np


def RatioBootstrap(num_by_cluster, den_by_cluster, reps=2000, seed=7):
    """CI for sum(num)/sum(den), resampling clusters (days or games) with replacement."""
    num = np.asarray(num_by_cluster, float)
    den = np.asarray(den_by_cluster, float)
    if len(num) == 0 or den.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(num), size=(reps, len(num)))
    n, d = num[idx].sum(1), den[idx].sum(1)
    ok = d > 0
    vals = n[ok] / d[ok]
    return float(num.sum() / den.sum()), float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def CellBootstrap(sum_gc, cnt_gc, reps=1000, seed=11):
    """sum_gc, cnt_gc: games x cells. Returns point, lo, hi per cell of sum/cnt with game resampling."""
    sum_gc = np.asarray(sum_gc, float)
    cnt_gc = np.asarray(cnt_gc, float)
    rng = np.random.default_rng(seed)
    g = sum_gc.shape[0]
    lo = np.full(sum_gc.shape[1], np.nan)
    hi = np.full(sum_gc.shape[1], np.nan)
    point = np.where(cnt_gc.sum(0) > 0, sum_gc.sum(0) / np.maximum(cnt_gc.sum(0), 1e-12), np.nan)
    draws = []
    for _ in range(reps):
        w = np.bincount(rng.integers(0, g, g), minlength=g).astype(float)
        s, c = w @ sum_gc, w @ cnt_gc
        draws.append(np.where(c > 0, s / np.maximum(c, 1e-12), np.nan))
    draws = np.array(draws)
    for j in range(sum_gc.shape[1]):
        col = draws[:, j][~np.isnan(draws[:, j])]
        if len(col) > 20:
            lo[j], hi[j] = np.quantile(col, 0.025), np.quantile(col, 0.975)
    return point, lo, hi


def MaxDrawdown(pnls):
    peak = cur = dd = 0.0
    for x in pnls:
        cur += x
        peak = max(peak, cur)
        dd = max(dd, peak - cur)
    return dd


def Traversals(series, lo, hi):
    """Count full lo<->hi traversals (reach one bound, later reach the other) and bound crossings."""
    state = None
    trav = crosses = 0
    prev = None
    for x in series:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            continue
        if prev is not None:
            if (prev - lo) * (x - lo) < 0 or (prev > lo and x == lo):
                crosses += 1
            if (prev - hi) * (x - hi) < 0 or (prev < hi and x == hi):
                crosses += 1
        if x <= lo:
            if state == "hi":
                trav += 1
            state = "lo"
        elif x >= hi:
            if state == "lo":
                trav += 1
            state = "hi"
        prev = x
    return trav, crosses


def Pbo(perf, n_splits=16, max_combos=2000, seed=3):
    """Probability of backtest overfitting (Bailey et al. CSCV).

    perf: T x N matrix (time blocks x variants), e.g. per-day P&L. Returns (pbo, n_combos).
    """
    perf = np.asarray(perf, float)
    t, n = perf.shape
    if n < 2 or t < n_splits:
        return float("nan"), 0
    blocks = np.array_split(np.arange(t), n_splits)
    combos = list(itertools.combinations(range(n_splits), n_splits // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_combos:
        combos = [combos[i] for i in rng.choice(len(combos), max_combos, replace=False)]
    below = 0
    for comb in combos:
        ins = np.concatenate([blocks[i] for i in comb])
        oos = np.concatenate([blocks[i] for i in range(n_splits) if i not in comb])
        best = int(np.argmax(perf[ins].mean(0)))
        oos_mean = perf[oos].mean(0)
        rank = (oos_mean < oos_mean[best]).sum() / (n - 1) if n > 1 else 0.5
        if rank <= 0.5:
            below += 1
    return below / len(combos), len(combos)