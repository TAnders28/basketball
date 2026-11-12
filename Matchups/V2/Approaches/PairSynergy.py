r"""
PairSynergy: teammate-pair interaction terms on top of StintRapm, with a strong shrinkage prior.

For every teammate pair that has shared at least `min_shared` possessions on the floor, two terms are added:
offensive synergy (both on offense) and defensive synergy (both on defense). The target is each stint side's
StintRapm residual (actual - posterior mean at that time), so the pair terms only explain what individual
ratings cannot. Prior: N(0, 1^2) per 100 - far tighter than players' 3^2, because most pairs have no true
synergy. tau_pair is re-estimated by empirical Bayes (method of moments on posterior means) and reported.
Prediction: P(both on) ~ (m_i/48)(m_j/48), an independence assumption (coaches stagger stars - a weakness).
"""

from collections import defaultdict
from itertools import combinations

import numpy as np

from Matchups.V2.Approaches.Common import Index, LearnPriorScale
from Matchups.V2.Approaches.StintRapm import POSS_VAR_PER100, StintRapm
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BayesLinear


@Register
class PairSynergy(Approach):
    name = "pair_synergy"
    idea = "Shrunk teammate-pair offensive/defensive synergy on StintRapm residuals (empirical-Bayes tau)."
    data_used = ("rotation", "pbp", "player_bio")
    covers = ("teammate_synergy", "team_strength")
    priors = {"pair synergy": ("N(0, 1^2) per 100", "most pairs have ~no synergy; heavy shrinkage"),
              "pair inclusion": ("shared possessions >= 300", "below that, the term is pure noise")}

    def __init__(self, capacity=2500, min_shared=300, tau_pair=1.0, refit_every=7, **params):
        super().__init__(**params)
        self.base = StintRapm(refit_every=refit_every)
        self.pairs = Index(capacity)
        self.shared = defaultdict(float)
        self.reg = BayesLinear(2 * capacity, tau_pair ** 2, sigma2=POSS_VAR_PER100)
        self.min_shared, self.refit_every, self.days = min_shared, refit_every, 999
        self.tau_hat = None
        self.tau_pair, self.dropped, self.drop_tau = tau_pair, False, params.get("drop_tau", 0.1)
        self.v0_base = np.full(2 * capacity, tau_pair ** 2)

    def BeforeDay(self, day, gap):
        self.base.BeforeDay(day, gap)
        self.reg.Decay(self.base.forget ** gap)
        self.days += gap
        if self.reg.dirty and self.days >= self.refit_every:
            n = 2 * len(self.pairs)
            self.reg.Fit(active=max(n, 1))
            if n > 10:
                # learned pair-effect variance (evidence fixed point); drop the family if it collapses
                s = LearnPriorScale(self.reg, self.v0_base, dims=n)
                self.tau_hat = float(np.sqrt(s) * self.tau_pair)
                self.dropped = self.tau_hat < self.drop_tau
            self.days = 0

    def PairCols(self, five, side, create):
        cols = []
        for a, b in combinations(sorted(five), 2):
            key = (a, b)
            if create:
                self.shared[key] += 1.0
                if self.shared[key] < self.min_shared:
                    continue
            i = self.pairs.Get(key, create)
            if i is not None:
                cols.append(2 * i + side)
        return cols

    def Predict(self, view):
        mh, ma = view.Minutes("home"), view.Minutes("away")
        mean, var = self.base.MarginMoments(view, mh, ma)
        pace_m, _ = self.base.pace.Game(view.home, view.away)
        idx, vals = {}, {}
        for off, dfn, sign in ((mh, ma, 1.0), (ma, mh, -1.0)):
            for a, b in combinations(sorted(off), 2):
                i = self.pairs.Get((a, b), create=False)
                if i is not None:
                    idx[2 * i] = idx.get(2 * i, 0.0) + sign * (off[a][0] / 48) * (off[b][0] / 48)
            for a, b in combinations(sorted(dfn), 2):
                i = self.pairs.Get((a, b), create=False)
                if i is not None:
                    idx[2 * i + 1] = idx.get(2 * i + 1, 0.0) + sign * (dfn[a][0] / 48) * (dfn[b][0] / 48)
        syn_mean = syn_var = 0.0
        if idx and not self.dropped:
            ii = np.array(list(idx))
            vv = np.array([idx[i] for i in ii])
            syn_mean = float(vv @ self.reg.mean[ii]) * pace_m / 100
            syn_var = float(vv @ self.reg.cov[np.ix_(ii, ii)] @ vv) * (pace_m / 100) ** 2
        return self.Output(mean + syn_mean, var + syn_var + MARGIN_NOISE_VAR, synergy=syn_mean,
                           n_pairs=len(idx), tau_pair_hat=self.tau_hat or float("nan"),
                           term_dropped=int(self.dropped))

    def Update(self, rec):
        for idx, vals, y, w, off5, def5, _sign in self.base.StintRows(rec):
            resid = y - float(np.dot(vals, self.base.reg.mean[idx]))
            cols = self.PairCols(off5, 0, True) + [c for c in self.PairCols(def5, 1, False)]
            if cols:
                self.reg.AddSparse(cols, np.ones(len(cols)), resid, w)
        self.base.Update(rec)
