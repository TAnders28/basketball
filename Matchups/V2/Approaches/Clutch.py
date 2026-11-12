r"""
Clutch: late-and-close performance as a heavily shrunk Beta-binomial / Gamma-Poisson team effect.

Clutch possessions (last 5 min of Q4/OT, margin <= 5) are scored per team: points per possession for and
against. Priors are strong (equivalent to 400 league-average possessions) because clutch results are mostly
noise. The team's clutch net (posterior mean) matters only in close games, so the feature is weighted by the
Kalman posterior predictive probability that the game ends within 6 points.
"""

import numpy as np

from Matchups.V2.Approaches.Common import LEAGUE_PPP, ResidualOnKalman
from Matchups.V2.Bayes import GammaPoisson, NormalCdf
from Matchups.V2.Base import Register


@Register
class Clutch(ResidualOnKalman):
    name = "clutch"
    idea = "Heavily pooled clutch net rating (Gamma-Poisson), weighted by the posterior chance of a close game."
    data_used = ("pbp", "rotation")
    covers = ("clutch", "rotations_coaching")
    feature_names = ("close_prob_x_clutch_net_diff",)
    prior_var = 1.0
    priors = {"clutch PPP prior": (f"Gamma(mean {LEAGUE_PPP}, strength 400 poss)", "clutch splits regress hard to the mean"),
              "effect coefficient": ("N(0, 1^2)", "per 1.0 PPP of clutch net, scaled by close-game prob")}

    def __init__(self, **params):
        super().__init__(**params)
        s = params.get("strength", 400.0)
        self.off, self.dfn = {}, {}
        self.strength = s

    def Net(self, team):
        o = self.off.get(team)
        d = self.dfn.get(team)
        return (o.Mean() if o else LEAGUE_PPP) - (d.Mean() if d else LEAGUE_PPP)

    def Features(self, view):
        m, v = self.core.Forecast(view.home, view.away)
        sd = np.sqrt(v)
        close = float(NormalCdf((6 - m) / sd) - NormalCdf((-6 - m) / sd))
        return np.array([close * (self.Net(view.home) - self.Net(view.away)) * 10.0])

    def Learn(self, rec):
        for p in rec.possessions:
            if not p["clutch"]:
                continue
            o = self.off.setdefault(p["offense"], GammaPoisson(LEAGUE_PPP, self.strength))
            d = self.dfn.setdefault(p["defense"], GammaPoisson(LEAGUE_PPP, self.strength))
            o.Add(p["pts"], 1.0)
            d.Add(p["pts"], 1.0)
