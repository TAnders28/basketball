r"""
Stacker: leak-free Bayesian logistic stacking of several approaches.

Members predict every game; features are each member's standardized posterior predictive margin
z = mean / sd (missing -> 0). Weights ~ N(prior mean, 0.3^2) where the prior puts weight 1.6 (the logit slope
of a probit at z) on the first member (KalmanTeam by default) and 0 on the rest - so with little data the
stack IS the Kalman model. Refit by Laplace approximation every 7 days on games from strictly earlier days
(Update stores a row only after the game is decided). The predictive integrates the weight posterior with the
probit approximation p = sigmoid(mu / sqrt(1 + pi * s^2 / 8)).
"""

import math

import numpy as np

from Matchups.V2.Base import REGISTRY, Approach, Register
from Matchups.V2.Bayes import LaplaceLogistic

DEFAULT_MEMBERS = ("kalman_team", "stint_rapm", "on_off", "schedule_fatigue", "four_factors", "shot_quality")


@Register
class Stacker(Approach):
    name = "stacker"
    idea = "Rolling Bayesian logistic stack (Laplace) over member posterior-predictive z-scores; prior = Kalman only."
    data_used = ("(members)",)
    covers = ("ensemble",)
    priors = {"weights": ("N([1.6, 0, ...], 0.3^2), intercept N(0, 0.5^2)", "start from the dynamic team model"),
              "min rows": ("150 decided games before the first refit", "avoid fitting on noise")}

    def __init__(self, members=DEFAULT_MEMBERS, refit_every=7, min_rows=150, **params):
        super().__init__(**params)
        self.members = [REGISTRY[m]() if isinstance(m, str) else m for m in members]
        k = len(self.members)
        self.prior_mean = np.r_[0.0, 1.6, np.zeros(k - 1)]
        self.prior_var = np.r_[0.25, np.full(k, 0.09)]
        self.w, self.cov = self.prior_mean.copy(), np.diag(self.prior_var)
        self.X, self.y, self.pending = [], [], {}
        self.refit_every, self.min_rows, self.days = refit_every, min_rows, 999

    def BeforeDay(self, day, gap):
        for m in self.members:
            m.BeforeDay(day, gap)
        self.days += gap
        if self.days >= self.refit_every and len(self.y) >= self.min_rows:
            self.w, self.cov = LaplaceLogistic(np.array(self.X), np.array(self.y), self.prior_mean, self.prior_var)
            self.days = 0

    def Predict(self, view):
        z, outs = [1.0], []
        for m in self.members:
            o = m.Predict(view) or {}
            outs.append(o)
            if "margin_mean" in o and o.get("margin_var", 0) > 0:
                z.append(o["margin_mean"] / math.sqrt(o["margin_var"]))
            else:
                z.append(0.0)
        x = np.array(z)
        mu, s2 = float(x @ self.w), float(x @ self.cov @ x)
        p = 1 / (1 + math.exp(-mu / math.sqrt(1 + math.pi * s2 / 8)))
        self.pending[view.game_id] = x
        first = outs[0] if outs else {}
        res = {"margin_mean": first.get("margin_mean", 0.0), "margin_var": first.get("margin_var", 169.0), "p_home": p,
               "stack_rows": len(self.y)}
        res.update({f"z_{m.name}": float(v) for m, v in zip(self.members, x[1:])})
        res.update({f"w_{m.name}": float(v) for m, v in zip(self.members, self.w[1:])})
        return res

    def Update(self, rec):
        for m in self.members:
            m.Update(rec)
        g = rec.game
        x = self.pending.pop(g["game_id"], None)
        if x is not None:
            self.X.append(x)
            self.y.append(int(g["home_score"] > g["away_score"]))
