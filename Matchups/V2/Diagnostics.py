r"""
Diagnostics used by tests, smoke runs and (later) backtests:
  - Calibration / ECE and reliability bins for win probabilities
  - Posterior predictive checks: central-interval coverage and PIT histogram for Normal predictives
  - Prior sensitivity: how much predictions move when prior scales are multiplied / divided
"""

import math

import numpy as np

from Matchups.V2.Bayes import NormalCdf


def Ece(p, y, bins: int = 10):
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    ece, table = 0.0, []
    for b in range(bins):
        m = idx == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
            table.append((round(edges[b], 2), int(m.sum()), float(p[m].mean()), float(y[m].mean())))
    return float(ece), table


def LogLoss(p, y):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def Coverage(mean, var, actual, level: float = 0.8):
    """Share of actual outcomes inside the central `level` interval of a Normal predictive."""
    z = {0.5: 0.6745, 0.8: 1.2816, 0.9: 1.6449, 0.95: 1.96}[level]
    mean, sd, actual = np.asarray(mean, float), np.sqrt(np.asarray(var, float)), np.asarray(actual, float)
    return float(np.mean(np.abs(actual - mean) <= z * sd))


def Pit(mean, var, actual, bins: int = 10):
    """Probability integral transform histogram; flat (each ~1/bins) means a calibrated predictive."""
    u = NormalCdf((np.asarray(actual, float) - np.asarray(mean, float)) / np.sqrt(np.asarray(var, float)))
    hist, _ = np.histogram(u, bins=bins, range=(0, 1))
    return (hist / max(len(u), 1)).tolist()


def PriorSensitivity(run_fn, scales=(0.5, 1.0, 2.0)):
    """run_fn(scale) -> array of predictions. Returns max/mean absolute change vs scale 1.0 for each scale."""
    base = np.asarray(run_fn(1.0), float)
    out = {}
    for s in scales:
        if s == 1.0:
            continue
        diff = np.abs(np.asarray(run_fn(s), float) - base)
        out[s] = {"max_abs": float(diff.max()), "mean_abs": float(diff.mean())}
    return out
