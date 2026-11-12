r"""
Per-player conjugate skill posteriors with per-stat decay (Efron-Morris / EPM "estimated skills" style).

Counting stats per 36 minutes are Gamma-Poisson: rate ~ Gamma(prior mean m_role, strength k_role in 36-minute
units), updated with (count, minutes / 36). Shooting percentages are Beta-binomial: pct ~ Beta(mean_role,
strength_role). Each stat has its own per-game decay (fast for steals/turnovers, slow for FT%), applied to that
player's sufficient statistics each time he plays, so the posterior tracks current skill. Role hyperpriors
(guard / wing / big) are re-estimated by empirical Bayes (method of moments) at Refit() from players' current
data - everything is built from games already played, so it is walk-forward safe.
"""

from collections import defaultdict

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Bayes import EmpiricalBayesBeta

# stat: (per-game decay, default per-36 prior mean, default strength in 36-minute games)
COUNTS = {"pts": (0.985, 16.0, 3.0), "ast": (0.985, 3.5, 3.0), "oreb": (0.99, 1.6, 3.0), "dreb": (0.99, 5.0, 3.0),
          "stl": (0.975, 1.1, 5.0), "blk": (0.99, 0.7, 4.0), "tov": (0.98, 2.0, 4.0), "fga": (0.985, 13.5, 3.0),
          "fg3a": (0.985, 5.5, 3.0), "fta": (0.98, 3.5, 3.0), "pf": (0.98, 3.2, 4.0)}
# pct: (made col, attempt col, per-game decay, default mean, default strength in attempts)
PCTS = {"fg3": ("fg3m", "fg3a", 0.995, 0.355, 150.0), "ft": ("ftm", "fta", 0.997, 0.78, 100.0),
        "fg2": (None, None, 0.99, 0.53, 120.0)}
FEATURES = list(COUNTS) + list(PCTS) + ["mpg"]


class SkillBook:
    def __init__(self):
        self.c = defaultdict(lambda: defaultdict(lambda: np.zeros(2)))     # player -> stat -> [y, exposure]
        self.mpg = defaultdict(lambda: np.zeros(2))                          # decayed [minutes, games]
        self.hyper = {}                                                      # (stat, role) -> (mean, strength)

    def Update(self, box_rows):
        for r in box_rows:
            p, mins = r["player_id"], float(r["minutes"] or 0)
            if mins <= 0:
                continue
            for s, (dec, _m, _k) in COUNTS.items():
                a = self.c[p][s]
                a *= dec
                a += (float(r.get(s) or 0), mins / 36.0)
            for s, (mc, ac, dec, _m, _k) in PCTS.items():
                if s == "fg2":
                    made, att = (r.get("fgm") or 0) - (r.get("fg3m") or 0), (r.get("fga") or 0) - (r.get("fg3a") or 0)
                else:
                    made, att = r.get(mc) or 0, r.get(ac) or 0
                a = self.c[p][s]
                a *= dec
                a += (float(made), float(att))
            m = self.mpg[p]
            m *= 0.95
            m += (mins, 1.0)

    def Refit(self):
        for role in (0, 1, 2):
            ps = [p for p in self.c if D.RoleBucket(p) == role]
            if len(ps) < 20:
                continue
            for s in COUNTS:
                y = np.array([self.c[p][s][0] for p in ps])
                e = np.array([self.c[p][s][1] for p in ps])
                keep = e > 1.0
                if keep.sum() < 20:
                    continue
                rate = y[keep] / e[keep]
                m = float(y[keep].sum() / e[keep].sum())
                var_true = max(float(np.average((rate - m) ** 2, weights=e[keep])) - m / float(np.mean(e[keep])), 1e-6)
                self.hyper[(s, role)] = (m, float(min(max(m / var_true, 0.5), 200.0)))
            for s in PCTS:
                k = [self.c[p][s][0] for p in ps]
                n = [self.c[p][s][1] for p in ps]
                self.hyper[(s, role)] = EmpiricalBayesBeta(k, n, min_n=20)

    def Rate(self, p, s):
        role = D.RoleBucket(p)
        y, e = self.c[p][s] if p in self.c else (0.0, 0.0)
        if s in COUNTS:
            m, k = self.hyper.get((s, role), (COUNTS[s][1], COUNTS[s][2]))
        else:
            m, k = self.hyper.get((s, role), (PCTS[s][3], PCTS[s][4]))
        a, b = m * k + y, k + e
        return a / b, (a / b ** 2) if s in COUNTS else (a / b) * (1 - a / b) / (b + 1)

    def Vector(self, p):
        v = [self.Rate(p, s)[0] for s in COUNTS] + [self.Rate(p, s)[0] for s in PCTS]
        mm, g = self.mpg[p] if p in self.mpg else (0.0, 0.0)
        return np.array(v + [mm / g if g > 0 else 12.0])

    def Exposure(self, p):
        return float(self.c[p]["pts"][1]) if p in self.c else 0.0
