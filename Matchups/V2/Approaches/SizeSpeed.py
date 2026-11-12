r"""
SizeSpeed: a Bayesian Poisson GLM for size / speed / strength / age mismatches at the matchup level.

Each matchup row (offender o guarded by d for c partial possessions, y offender points):
  y ~ Poisson(c * r_o * exp(b . x_od)),  x = [height diff / 3 in, weight diff / 20 lb, speed diff (prior-season
  AVG_SPEED_OFF of o minus AVG_SPEED_DEF of d, mph), age diff / 5 yr, (height diff)^2 / 9]
r_o is the offender's Gamma-Poisson posterior rate (the offset), so b only explains how the *matchup* changes
his usual scoring. b ~ N(0, 0.1^2) on the log scale (a mismatch moves scoring by ~10% at most a priori).
Fit by Laplace approximation (MAP + Hessian) weekly on decayed rows (capped). Team points integrate over
size-kernel assignments; the coefficient covariance propagates to the predictive via the delta method.
"""

import math
from collections import deque

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import GammaPoisson, LaplacePoisson

HOME_EDGE = 2.5


def Num(x, default):
    """Float or default for missing / blank / NaN values (bio and tracking tables have gaps)."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return default
    return v if math.isfinite(v) else default
NAMES = ("height_diff", "weight_diff", "speed_diff", "age_diff", "height_diff_sq")


@Register
class SizeSpeed(Approach):
    name = "size_speed"
    idea = "Bayesian Poisson GLM (Laplace) of matchup scoring on height/weight/speed/age mismatches, offender-rate offset."
    data_used = ("box_matchups", "player_bio", "tracking")
    covers = ("matchups", "defense_schemes")
    priors = {"mismatch coefficients": ("N(0, 0.1^2) log-rate", "a mismatch moves scoring ~10% a priori"),
              "offender rate offset": ("Gamma(0.22, 300 poss)", "usual scoring per partial possession; ~4 games to move halfway")}

    def __init__(self, max_rows=60000, refit_every=7, forget=0.998, **params):
        super().__init__(**params)
        self.rows = deque(maxlen=max_rows)
        self.b = np.zeros(len(NAMES))
        self.cov = np.eye(len(NAMES)) * 0.01
        self.off_rate = {}
        self.refit_every, self.days, self.forget = refit_every, 999, forget
        self.pace = PaceBook()
        self.season, self.speed_off, self.speed_def, self.bio = None, {}, {}, {}

    def LoadPriors(self, season):
        rows = D.SeasonTable(season - 1, "tracking/Player_SpeedDistance.json")
        self.speed_off = {int(r["PLAYER_ID"]): r.get("AVG_SPEED_OFF") for r in rows}
        self.speed_def = {int(r["PLAYER_ID"]): r.get("AVG_SPEED_DEF") for r in rows}
        self.bio = D.Bio(season - 1)
        self.season = season

    def X(self, o, d):
        ho, hd = D.Heights().get(o, 78.0), D.Heights().get(d, 78.0)

        def W(p):
            return Num(self.bio.get(p, {}).get("PLAYER_WEIGHT"), 215.0)

        def A(p):
            return Num(self.bio.get(p, {}).get("AGE"), 26.0)
        so, sd = Num(self.speed_off.get(o), 4.3), Num(self.speed_def.get(d), 4.0)
        # clip to +-2.5 units (7.5 in) so the quadratic term cannot extrapolate wildly (e.g. vs 7-4 centers)
        hdiff = max(-2.5, min(2.5, (ho - hd) / 3.0))
        return np.array([hdiff, (W(o) - W(d)) / 20.0, (so - 4.3) - (sd - 4.0), (A(o) - A(d)) / 5.0, hdiff * hdiff / 4.0])

    def Rate(self, p):
        g = self.off_rate.get(p)
        return g.Mean() if g else 0.22

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        for g in self.off_rate.values():
            g.Decay(f)
        self.days += gap
        if self.days >= self.refit_every and len(self.rows) > 5000:
            X = np.array([r[0] for r in self.rows])
            y = np.array([r[1] for r in self.rows])
            e = np.array([r[2] for r in self.rows])
            self.b, self.cov = LaplacePoisson(X, y, e, np.zeros(X.shape[1]), 0.01)
            self.days = 0

    def TeamPoints(self, off_min, def_min, pace):
        mean, grad = 0.0, np.zeros(len(NAMES))
        if not off_min or not def_min:          # a team with no games yet: league-average scoring
            return pace * 1.14, grad
        dm = {d: m for d, (m, _v) in def_min.items()}
        for o, (m, _v) in off_min.items():
            poss = m / 48 * pace
            ho = D.Heights().get(o, 78.0)
            ker = {d: mm * math.exp(-((ho - D.Heights().get(d, 78.0)) ** 2) / 18.0) for d, mm in dm.items()}
            z = sum(ker.values()) or 1.0
            for d, k in ker.items():
                x = self.X(o, d)
                val = poss * (k / z) * self.Rate(o) * math.exp(float(x @ self.b))
                mean += val
                grad += val * x
        return mean, grad

    def Predict(self, view):
        if self.season != view.season:
            self.LoadPriors(view.season)
        pace, _ = self.pace.Game(view.home, view.away)
        hp, hg = self.TeamPoints(view.Minutes("home"), view.Minutes("away"), pace)
        ap, ag = self.TeamPoints(view.Minutes("away"), view.Minutes("home"), pace)
        g = hg - ag
        return self.Output(hp - ap + HOME_EDGE, float(g @ self.cov @ g) + MARGIN_NOISE_VAR, home_pts=hp, away_pts=ap,
                           **{f"b_{n}": float(v) for n, v in zip(NAMES, self.b)})

    def Update(self, rec):
        if self.season != rec.game["season"]:
            self.LoadPriors(rec.game["season"])
        for r in rec.matchups or []:
            if r["poss"] <= 0:
                continue
            self.rows.append((self.X(r["off"], r["def"]), r["pts"], r["poss"] * self.Rate(r["off"])))
        for r in rec.matchups or []:
            self.off_rate.setdefault(r["off"], GammaPoisson(0.22, 300.0)).Add(r["pts"], r["poss"])
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
