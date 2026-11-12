r"""
Shared evaluation harness (reading: Brier 1950; Gneiting-Raftery 2007; Gneiting et al. 2007; Dimitriadis et al. 2021).

Every forecast row carries a margin distribution: Normal(margin_mean, margin_var), or a Normal mixture in
row["mix"] = (weights, means, variances). From it:
  - log loss and Brier of p_home = P(margin > 0)
  - CRPS of the margin distribution (closed form for Normal and Normal mixtures, Grimit et al. 2006)
  - CORP reliability (isotonic / PAV recalibration) with the Brier decomposition  S = MCB - DSC + UNC
  - ECE (10 bins), PIT histogram, 80% interval coverage, spread-skill ratio
  - posterior predictive checks: margin SD, share of 20+ point games, home win rate (predicted vs actual)
PairedBootstrap compares two forecasters on the same games with a day-clustered bootstrap.
"""

import math

import numpy as np

SQRT2 = math.sqrt(2.0)
INV_SQRT_PI = 1.0 / math.sqrt(math.pi)


def Erf(x):
    """Vectorized erf (Abramowitz-Stegun 7.1.26, |error| < 1.5e-7)."""
    x = np.asarray(x, float)
    s = np.sign(x)
    a = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * a)
    y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * np.exp(-a * a)
    return s * y


def Phi(x):
    return 0.5 * (1.0 + Erf(np.asarray(x, float) / SQRT2))


def phi(x):
    x = np.asarray(x, float)
    return np.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def InvPhi(p):
    """Acklam's rational approximation of the standard normal quantile (|error| < 1.2e-9)."""
    p = min(max(float(p), 1e-12), 1 - 1e-12)
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00]
    if p < 0.02425:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
    if p > 1 - 0.02425:
        return -InvPhi(1 - p)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def NormalCrps(y, mu, var):
    y, mu, s = np.asarray(y, float), np.asarray(mu, float), np.sqrt(np.asarray(var, float))
    z = (y - mu) / s
    return s * (z * (2 * Phi(z) - 1) + 2 * phi(z) - INV_SQRT_PI)


def _A(m, v):
    s = np.sqrt(v)
    return 2 * s * phi(m / s) + m * (2 * Phi(m / s) - 1)


def MixCrps(y, w, mu, var):
    w, mu, var = np.asarray(w, float), np.asarray(mu, float), np.asarray(var, float)
    t1 = float(np.sum(w * _A(y - mu, var)))
    dm = mu[:, None] - mu[None, :]
    dv = var[:, None] + var[None, :]
    t2 = float(np.sum(np.outer(w, w) * _A(dm, dv)))
    return t1 - 0.5 * t2


def RowCrps(r):
    if r.get("mix"):
        w, mu, var = r["mix"]
        return MixCrps(r["margin"], w, mu, var)
    return float(NormalCrps(r["margin"], r["margin_mean"], r["margin_var"]))


def RowPit(r):
    if r.get("mix"):
        w, mu, var = (np.asarray(x, float) for x in r["mix"])
        return float(np.sum(w * Phi((r["margin"] - mu) / np.sqrt(var))))
    return float(Phi((r["margin"] - r["margin_mean"]) / math.sqrt(r["margin_var"])))


def RowTail(r, k=20.0):
    """P(|margin| >= k) under the forecast."""
    if r.get("mix"):
        w, mu, var = (np.asarray(x, float) for x in r["mix"])
        s = np.sqrt(var)
        return float(np.sum(w * (1 - Phi((k - mu) / s) + Phi((-k - mu) / s))))
    s = math.sqrt(r["margin_var"])
    return float(1 - Phi((k - r["margin_mean"]) / s) + Phi((-k - r["margin_mean"]) / s))


def LossVectors(rows):
    p = np.clip(np.array([r["p_home"] for r in rows], float), 1e-6, 1 - 1e-6)
    y = np.array([r["home_win"] for r in rows], float)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    br = (p - y) ** 2
    crps = np.array([RowCrps(r) for r in rows])
    return {"log_loss": ll, "brier": br, "crps": crps}


def Pav(p, y):
    """Isotonic (pool-adjacent-violators) regression of y on p: CORP recalibrated probabilities."""
    order = np.argsort(p, kind="mergesort")
    ys = y[order].astype(float)
    vals, wts, sizes = [], [], []
    for v in ys:
        vals.append(v)
        wts.append(1.0)
        sizes.append(1)
        while len(vals) > 1 and vals[-2] >= vals[-1]:
            v2, w2, s2 = vals.pop(), wts.pop(), sizes.pop()
            v1, w1, s1 = vals.pop(), wts.pop(), sizes.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
            sizes.append(s1 + s2)
    fit = np.repeat(vals, sizes)
    out = np.empty_like(fit)
    out[order] = fit
    return out


def Corp(p, y):
    """CORP decomposition of the Brier score: BS = MCB - DSC + UNC, plus a reliability table."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    rc = Pav(p, y)
    bs = float(np.mean((p - y) ** 2))
    bs_rc = float(np.mean((rc - y) ** 2))
    ybar = float(y.mean())
    unc = float(np.mean((ybar - y) ** 2))
    table = []
    for lo in np.arange(0, 1, 0.1):
        m = (p >= lo) & (p < lo + 0.1 + (1e-9 if lo >= 0.9 else 0))
        if m.any():
            table.append((round(float(lo), 1), int(m.sum()), round(float(p[m].mean()), 3), round(float(rc[m].mean()), 3), round(float(y[m].mean()), 3)))
    return {"mcb": bs - bs_rc, "dsc": unc - bs_rc, "unc": unc, "table": table}


def Ece(p, y, bins=10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.clip((p * bins).astype(int), 0, bins - 1)
    return float(sum((idx == b).mean() * abs(p[idx == b].mean() - y[idx == b].mean()) for b in range(bins) if (idx == b).any()))


def Score(rows):
    if not rows:
        return {}
    L = LossVectors(rows)
    p = np.array([r["p_home"] for r in rows], float)
    y = np.array([r["home_win"] for r in rows], float)
    margin = np.array([r["margin"] for r in rows], float)
    mu = np.array([r["margin_mean"] for r in rows], float)
    var = np.array([r["margin_var"] for r in rows], float)
    pit = np.array([RowPit(r) for r in rows])
    hist = np.histogram(pit, bins=10, range=(0, 1))[0] / len(pit)
    corp = Corp(p, y)
    cov80 = float(np.mean(np.abs(margin - mu) <= 1.2816 * np.sqrt(var))) if not rows[0].get("mix") else float(np.mean((pit > 0.1) & (pit < 0.9)))
    return {"n": len(rows), "log_loss": float(L["log_loss"].mean()), "brier": float(L["brier"].mean()),
            "crps": float(L["crps"].mean()), "ece": Ece(p, y), "mcb": corp["mcb"], "dsc": corp["dsc"], "unc": corp["unc"],
            "reliability": corp["table"], "pit": [round(float(h), 3) for h in hist],
            "pit_max_dev": float(np.max(np.abs(hist - 0.1))), "coverage80": cov80,
            "spread_skill": float(np.sqrt(np.mean((margin - mu) ** 2)) / np.sqrt(np.mean(var))),
            "rmse_margin": float(np.sqrt(np.mean((margin - mu) ** 2))),
            "ppc_margin_sd_pred": float(np.sqrt(np.mean(var) + np.var(mu))), "ppc_margin_sd_actual": float(np.std(margin)),
            "ppc_20plus_pred": float(np.mean([RowTail(r) for r in rows])), "ppc_20plus_actual": float(np.mean(np.abs(margin) >= 20)),
            "ppc_home_win_pred": float(p.mean()), "ppc_home_win_actual": float(y.mean())}


def PairedBootstrap(loss_model, loss_base, clusters, reps=2000, seed=7):
    """Mean (model - base) with a day-clustered bootstrap 95% CI and P(model better)."""
    d = np.asarray(loss_model, float) - np.asarray(loss_base, float)
    clusters = np.asarray(clusters)
    keys, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=d)
    cnts = np.bincount(inv).astype(float)
    rng = np.random.default_rng(seed)
    k = len(keys)
    draws = rng.integers(0, k, size=(reps, k))
    boot = sums[draws].sum(1) / cnts[draws].sum(1)
    return {"diff": float(d.mean()), "lo": float(np.quantile(boot, 0.025)), "hi": float(np.quantile(boot, 0.975)),
            "p_better": float(np.mean(boot < 0))}


def EloRows(game_rows, elo_p):
    """Phase-1 Elo as a margin distribution: Normal(13 * InvPhi(p), 13^2), which reproduces p exactly (implied)."""
    out = []
    for g in game_rows:
        p = elo_p.get(g["game_id"])
        if p is None:
            continue
        out.append(dict(g, margin_mean=13.0 * InvPhi(p), margin_var=169.0, p_home=float(p)))
    return out
