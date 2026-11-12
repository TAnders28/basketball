r"""
LineupFeatures: lineup construction (spacing, usage overlap, creation, rim protection, size, rebounding,
assist chemistry) as a Bayesian regression on stint net rating.

Each stint side gets features of its five (posterior-mean player rates, see PlayerRates): number of shooters
(3PA/36 >= 4 and 3P% posterior >= .345), usage excess (sum of usage/min over 2.6), creators (AST/36 >= 5),
rim protectors (BLK/36 >= 1.5 or height >= 6'10"), average height vs 78", rebounding sum, and assist-network
density (sum over pairs of decayed passer<->scorer assists). Target: stint net per 100 (home minus away),
features = home five minus away five, possession-weighted. Coefficients ~ N(0, 3^2) per 100 per unit.
Prediction uses expected features under projected minutes (P(on) = minutes / 48; pair terms P(i)P(j)).
"""

from itertools import combinations

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import PaceBook, TeamPossessions
from Matchups.V2.Approaches.PlayerRates import PlayerRates
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BayesLinear

FEATURES = ("home", "shooters", "usage_excess", "creators", "rim_protectors", "height", "rebounding", "assist_density")


@Register
class LineupFeatures(Approach):
    name = "lineup_features"
    idea = "Spacing / usage overlap / creators / rim protection / size / rebounding / assist chemistry -> stint net (Bayesian)."
    data_used = ("rotation", "pbp", "player_games", "player_bio")
    covers = ("teammate_synergy", "shot_quality", "rebounding", "rotations_coaching")
    priors = {"each feature coefficient": ("N(0, 3^2) per 100 per unit", "lineup effects are a few points per 100"),
              "player rates": ("Gamma / Beta posteriors pooled by role", "see PlayerRates")}

    def __init__(self, forget=0.998, refit_every=3, **params):
        super().__init__(**params)
        self.rates = PlayerRates()
        self.reg = BayesLinear(len(FEATURES), 9.0, sigma2=2 * 12500.0)
        self.pace = PaceBook()
        self.forget, self.refit_every, self.days = forget, refit_every, 999

    def FiveFeatures(self, weights: dict) -> np.ndarray:
        """weights: {player: P(on court)} (1.0 for an observed five)."""
        r = self.rates
        shooters = sum(w for p, w in weights.items() if r.Shooter(p))
        usage = sum(w * r.R(p, "usage").Mean() for p, w in weights.items())
        creators = sum(w for p, w in weights.items() if 36 * r.R(p, "ast").Mean() >= 5.0)
        rim = sum(w for p, w in weights.items() if 36 * r.R(p, "blk").Mean() >= 1.5 or D.Heights().get(p, 78) >= 82)
        height = sum(w * (D.Heights().get(p, 78.0) - 78.0) for p, w in weights.items()) / 5.0
        reb = sum(w * r.R(p, "reb").Mean() for p, w in weights.items())
        links = sum(weights[a] * weights[b] * r.Link(a, b) for a, b in combinations(sorted(weights), 2)) / 10.0
        return np.array([shooters, max(usage - 2.6, 0.0), creators, rim, height, reb, links])

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        self.rates.Decay(f)
        self.reg.Decay(f)
        self.days += gap
        if self.days >= self.refit_every:
            self.reg.Fit()
            self.days = 0

    def Predict(self, view):
        mh, ma = view.Minutes("home"), view.Minutes("away")
        wh = {p: min(m / 48, 1.0) for p, (m, _v) in mh.items()}
        wa = {p: min(m / 48, 1.0) for p, (m, _v) in ma.items()}
        x = np.concatenate([[1.0], self.FiveFeatures(wh) - self.FiveFeatures(wa)]) if wh and wa else np.zeros(len(FEATURES))
        m100, v100 = self.reg.Predict(x)
        v100 -= self.reg.Sigma2()
        pace, _ = self.pace.Game(view.home, view.away)
        return self.Output(pace / 100 * m100, (pace / 100) ** 2 * max(v100, 0) + MARGIN_NOISE_VAR,
                           **{f"x_{n}": float(v) for n, v in zip(FEATURES, x)})

    def Update(self, rec):
        for s in rec.stints:
            if len(s["home5"]) != 5 or len(s["away5"]) != 5 or s["home_poss"] <= 0 or s["away_poss"] <= 0:
                continue
            y = 100 * (s["home_pts"] / s["home_poss"] - s["away_pts"] / s["away_poss"])
            x = np.concatenate([[1.0], self.FiveFeatures({p: 1.0 for p in s["home5"]}) - self.FiveFeatures({p: 1.0 for p in s["away5"]})])
            self.reg.Add(x, y, w=0.5 * (s["home_poss"] + s["away_poss"]))
        self.rates.Apply(rec)
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
