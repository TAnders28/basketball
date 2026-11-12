"""Scoring helpers: log loss, Brier, calibration, logistic calibration/stacking, date-cluster bootstrap."""

import math
from collections import defaultdict

import numpy as np

EPS = 1e-6


def Logit(p):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    return np.log(p / (1 - p))


def Sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


def LogLoss(p, y):
    p = np.clip(np.asarray(p, dtype=float), EPS, 1 - EPS)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


def Brier(p, y):
    return (np.asarray(p, dtype=float) - np.asarray(y, dtype=float)) ** 2


def FitLogistic(X, y, l2: float = 1e-3, iterations: int = 50):
    """Newton-Raphson logistic regression with an intercept (column of ones added here)."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    y = np.asarray(y, dtype=float)
    w = np.zeros(X.shape[1])
    reg = np.full(X.shape[1], l2)
    reg[0] = 0.0
    for _ in range(iterations):
        p = Sigmoid(X @ w)
        grad = X.T @ (p - y) + reg * w
        hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(reg + 1e-9)
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return w


def PredictLogistic(w, X):
    X = np.column_stack([np.ones(len(X)), np.asarray(X, dtype=float)])
    return Sigmoid(X @ w)


def Calibration(p, y, bins: int = 10) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    out, ece = [], 0.0
    for i in range(bins):
        mask = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= edges[i + 1])
        if mask.sum() == 0:
            continue
        mp, my = float(p[mask].mean()), float(y[mask].mean())
        out.append({"bin_lo": round(edges[i], 2), "bin_hi": round(edges[i + 1], 2), "n": int(mask.sum()),
                    "mean_pred": round(mp, 4), "mean_actual": round(my, 4)})
        ece += mask.sum() / len(p) * abs(mp - my)
    return {"bins": out, "ece": round(float(ece), 4)}


def Metrics(p, y) -> dict:
    p, y = np.asarray(p, dtype=float), np.asarray(y, dtype=float)
    return {"n": int(len(p)), "log_loss": float(LogLoss(p, y).mean()), "brier": float(Brier(p, y).mean()),
            "accuracy": float(((p > 0.5) == (y > 0.5)).mean()), "ece": Calibration(p, y)["ece"]}


def BootstrapDiff(p_model, p_base, y, clusters, reps: int = 2000, seed: int = 7) -> dict:
    """Paired bootstrap over game-date clusters of mean(loss_model - loss_base). Negative = model better."""
    rng = np.random.default_rng(seed)
    d_ll = LogLoss(p_model, y) - LogLoss(p_base, y)
    d_br = Brier(p_model, y) - Brier(p_base, y)
    groups = defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    idx = [np.array(v) for v in groups.values()]
    sums_ll = np.array([d_ll[i].sum() for i in idx])
    sums_br = np.array([d_br[i].sum() for i in idx])
    counts = np.array([len(i) for i in idx])
    draws = rng.integers(0, len(idx), size=(reps, len(idx)))
    boot_ll = sums_ll[draws].sum(1) / counts[draws].sum(1)
    boot_br = sums_br[draws].sum(1) / counts[draws].sum(1)
    return {"d_log_loss": float(d_ll.mean()), "d_ll_ci": [float(np.percentile(boot_ll, 2.5)), float(np.percentile(boot_ll, 97.5))],
            "d_brier": float(d_br.mean()), "d_br_ci": [float(np.percentile(boot_br, 2.5)), float(np.percentile(boot_br, 97.5))],
            "p_model_better_ll": float((boot_ll < 0).mean())}
