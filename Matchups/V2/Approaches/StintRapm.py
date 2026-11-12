r"""
StintRapm: hierarchical Bayesian offense/defense player ratings from 10-player stints (possession level).

Model (per stint side): 100 * pts/poss - 100 * league_ppp = home_edge*(+-1) + sum_off O_p + sum_def D_p + noise,
noise variance 12,500 / possessions (one possession's points have sd ~1.1). Priors are hierarchical:
O_p, D_p ~ N(m_role(p), tau^2), where m_role is the posterior mean of the player's role bucket (guard / wing /
big by height) re-estimated by empirical Bayes at every refit, optionally replaced by an external player prior
(TrackingPrior). tau = 3 per 100 is typical RAPM-scale skill spread. Sufficient statistics are decayed (half-life
~1 season) so ratings move over time. Prediction integrates over projected minutes: E[on court] = minutes / 48,
and the posterior covariance gives rating uncertainty in the margin.
"""

import math

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import LEAGUE_PPP, Index, PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BayesLinear

POSS_VAR_PER100 = 12500.0


@Register
class StintRapm(Approach):
    name = "stint_rapm"
    idea = "Hierarchical Bayesian O/D player ratings on possession-weighted stints, role-pooled priors, decayed."
    data_used = ("rotation", "pbp", "player_bio")
    covers = ("team_strength", "possession", "rotations_coaching")
    priors = {"player O, D": ("N(role mean, 3^2) per 100 poss", "typical RAPM spread; role mean pooled by height bucket"),
              "home edge": ("N(0, 2^2) per 100 per side", "~1.5 per 100 per side observed"),
              "noise": ("12,500 / possessions (per 100 scale)", "variance of points on one possession ~1.25"),
              "forgetting": ("0.998 per day (half-life ~350 days)", "skills drift; older seasons count less")}

    def __init__(self, capacity=1500, tau=3.0, forget=0.998, refit_every=7, prior_provider=None, **params):
        super().__init__(**params)
        self.ix = Index(capacity)
        self.dim = 1 + 2 * capacity
        v0 = np.full(self.dim, tau ** 2)
        v0[0] = 4.0
        self.reg = BayesLinear(self.dim, v0, sigma2=POSS_VAR_PER100)
        self.forget, self.refit_every, self.tau = forget, refit_every, tau
        self.pace = PaceBook()
        self.days_since_fit = 999
        self.prior_provider = prior_provider
        self.season = None

    def O(self, p, create=True):
        i = self.ix.Get(p, create)
        return None if i is None else 1 + 2 * i

    def RolePriors(self):
        n = len(self.ix)
        if n == 0:
            return
        means = self.reg.mean
        m0 = self.reg.m0.copy()
        for side in (0, 1):
            sums, counts = np.zeros(3), np.zeros(3)
            for p, i in self.ix.ids.items():
                b = D.RoleBucket(p)
                sums[b] += means[1 + 2 * i + side]
                counts[b] += 1
            role_mean = sums / np.maximum(counts, 1)
            for p, i in self.ix.ids.items():
                m0[1 + 2 * i + side] = role_mean[D.RoleBucket(p)]
        if self.prior_provider is not None:
            for p, i in self.ix.ids.items():
                pr = self.prior_provider(p)
                if pr is not None:
                    m0[1 + 2 * i], m0[2 + 2 * i] = pr
        self.reg.SetPriorMean(m0)

    def BeforeDay(self, day, gap):
        self.reg.Decay(self.forget ** gap)
        self.days_since_fit += gap
        if self.reg.dirty and self.days_since_fit >= self.refit_every:
            self.RolePriors()
            self.reg.Fit(active=1 + 2 * len(self.ix))
            self.days_since_fit = 0

    def SideVector(self, off_minutes, def_minutes, home_sign):
        idx, vals = [0], [float(home_sign)]
        for p, (m, _v) in off_minutes.items():
            j = self.O(p, create=False)
            if j is not None:
                idx.append(j)
                vals.append(min(m / 48.0, 1.0))
        for p, (m, _v) in def_minutes.items():
            j = self.O(p, create=False)
            if j is not None:
                idx.append(j + 1)
                vals.append(min(m / 48.0, 1.0))
        return idx, vals

    def MarginMoments(self, view, mh=None, ma=None):
        mh, ma = mh or view.Minutes("home"), ma or view.Minutes("away")
        hi, hv = self.SideVector(mh, ma, +1)
        ai, av = self.SideVector(ma, mh, -1)
        diff = {}
        for i, v in zip(hi, hv):
            diff[i] = diff.get(i, 0.0) + v
        for i, v in zip(ai, av):
            diff[i] = diff.get(i, 0.0) - v
        idx = np.array(list(diff))
        vals = np.array([diff[i] for i in idx])
        mean100 = float(vals @ self.reg.mean[idx])
        var100 = float(vals @ self.reg.cov[np.ix_(idx, idx)] @ vals)
        pace_m, pace_v = self.pace.Game(view.home, view.away)
        mean = pace_m / 100.0 * mean100
        var = (pace_m / 100.0) ** 2 * var100 + (mean100 / 100.0) ** 2 * pace_v
        return mean, var

    def Predict(self, view):
        mean, var = self.MarginMoments(view)
        return self.Output(mean, var + MARGIN_NOISE_VAR, rating_var=var)

    def StintRows(self, rec):
        """(index list, value list, y per 100, possessions) for each stint side."""
        out = []
        for s in rec.stints:
            for off5, def5, pts, poss, sign in ((s["home5"], s["away5"], s["home_pts"], s["home_poss"], 1),
                                                (s["away5"], s["home5"], s["away_pts"], s["away_poss"], -1)):
                if poss <= 0 or len(off5) != 5 or len(def5) != 5:
                    continue
                idx, vals = [0], [float(sign)]
                for p in off5:
                    j = self.O(p)
                    if j is not None:
                        idx.append(j)
                        vals.append(1.0)
                for p in def5:
                    j = self.O(p)
                    if j is not None:
                        idx.append(j + 1)
                        vals.append(1.0)
                out.append((idx, vals, 100.0 * pts / poss - 100.0 * LEAGUE_PPP, poss, off5, def5, sign))
        return out

    def Update(self, rec):
        for idx, vals, y, w, *_ in self.StintRows(rec):
            self.reg.AddSparse(idx, vals, y, w)
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
