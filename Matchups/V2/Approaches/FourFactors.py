r"""
FourFactors: a possession-chain model from play-by-play with Beta-binomial rates for every link, combined
offense-vs-defense on the log-odds scale (log5), and posterior predictive variance by the delta method.

Per team, offense and defense (allowed), conjugate rates updated walk-forward from possessions:
  turnover rate       Beta(0.135, 400) per possession
  transition rate     Beta(0.15, 400)  share of possessions with a shot within 8 s of a live-ball start
  transition PPP      Gamma(1.20, 150); half-court PPP is built from the chain below
  shot quality        Beta(0.53, 400)  "points per FGA / 2" (an eFG-style make rate)
  offensive rebound   Beta(0.25, 400)  per missed FGA
  FT points / poss    Gamma(0.19, 400)
Matchup rate p_AB = logistic(logit p_off_A + logit p_def_B - logit p_league). Half-court PPP =
(1 - tov) * 2 * efg * (1 + oreb * (1 - efg)) + ft. Game PPP = t * transition PPP + (1 - t) * half-court PPP.
"""

import math
from collections import defaultdict

import numpy as np

from Matchups.V2.Approaches.Common import PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BetaBinomial, GammaPoisson

HOME_EDGE = 2.5
BETA_PRIORS = {"tov": 0.135, "trans": 0.15, "efg": 0.53, "oreb": 0.25}
GAMMA_PRIORS = {"trans_ppp": (1.20, 150.0), "ft": (0.19, 400.0)}


def Logit(p):
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def Sig(z):
    return 1 / (1 + math.exp(-z))


@Register
class FourFactors(Approach):
    name = "four_factors"
    idea = "Possession chain (TOV, transition, eFG, OREB, FT) with Beta/Gamma posteriors, log5 offense-vs-defense, delta-method variance."
    data_used = ("pbp", "rotation")
    covers = ("possession", "turnovers", "rebounding", "transition", "shot_quality")
    priors = {k: (f"Beta({v}, 400)", "league-typical rate, ~1 week of games to move") for k, v in BETA_PRIORS.items()}
    priors.update({k: (f"Gamma({m}, {s})", "") for k, (m, s) in GAMMA_PRIORS.items()})

    def __init__(self, strength=400.0, forget=0.998, **params):
        super().__init__(**params)
        self.b = {side: {k: defaultdict(lambda k=k: BetaBinomial(BETA_PRIORS[k], strength)) for k in BETA_PRIORS} for side in ("off", "def")}
        self.g = {side: {k: defaultdict(lambda k=k: GammaPoisson(*GAMMA_PRIORS[k])) for k in GAMMA_PRIORS} for side in ("off", "def")}
        self.lg = {k: BetaBinomial(v, 2000.0) for k, v in BETA_PRIORS.items()}
        self.pace, self.forget = PaceBook(), forget

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        for side in self.b.values():
            for d in side.values():
                for x in d.values():
                    x.Decay(f)
        for side in self.g.values():
            for d in side.values():
                for x in d.values():
                    x.Decay(f)

    def Rate(self, k, off, dfn):
        po, pd, pl = self.b["off"][k][off], self.b["def"][k][dfn], self.lg[k].Mean()
        z = Logit(po.Mean()) + Logit(pd.Mean()) - Logit(pl)
        return Sig(z), po.Var() + pd.Var()

    def Ppp(self, off, dfn, bump=None):
        vals, vars_ = {}, {}
        for k in BETA_PRIORS:
            vals[k], vars_[k] = self.Rate(k, off, dfn)
        for k in GAMMA_PRIORS:
            go, gd = self.g["off"][k][off], self.g["def"][k][dfn]
            vals[k] = 0.5 * (go.Mean() + gd.Mean())
            vars_[k] = 0.25 * (go.Var() + gd.Var())
        if bump:
            vals[bump[0]] += bump[1]
        half = (1 - vals["tov"]) * 2 * vals["efg"] * (1 + vals["oreb"] * (1 - vals["efg"])) + vals["ft"]
        return vals["trans"] * vals["trans_ppp"] + (1 - vals["trans"]) * half, vals, vars_

    def Moments(self, off, dfn):
        m, vals, vars_ = self.Ppp(off, dfn)
        var = 0.0
        for k, v in vars_.items():
            h = math.sqrt(max(v, 1e-12))
            var += (self.Ppp(off, dfn, (k, h))[0] - m) ** 2
        return m, var, vals

    def Predict(self, view):
        pace, _ = self.pace.Game(view.home, view.away)
        hm, hv, hvals = self.Moments(view.home, view.away)
        am, av, avals = self.Moments(view.away, view.home)
        return self.Output(pace * (hm - am) + HOME_EDGE, pace ** 2 * (hv + av) + MARGIN_NOISE_VAR,
                           home_ppp=hm, away_ppp=am, **{f"home_{k}": v for k, v in hvals.items()},
                           **{f"away_{k}": v for k, v in avals.items()})

    def Update(self, rec):
        for team, dfn in ((rec.home, rec.away), (rec.away, rec.home)):
            ps = [p for p in rec.possessions if p["offense"] == team]
            if not ps:
                continue
            n = len(ps)
            tov = sum(p["tov"] > 0 for p in ps)
            trans = [p for p in ps if p["transition"]]
            fga = sum(p["fga"] for p in ps)
            shot_pts = sum(sh["value"] for p in ps for sh in p["shots"] if sh["made"])
            misses = sum(1 for p in ps for sh in p["shots"] if not sh["made"])
            oreb = sum(p["oreb"] for p in ps)
            ft_pts = sum(p["pts"] for p in ps) - shot_pts
            stats = {"tov": (tov, n), "trans": (len(trans), n), "efg": (shot_pts / 2.0, fga), "oreb": (min(oreb, misses), misses)}
            for k, (kk, nn) in stats.items():
                if nn > 0:
                    self.b["off"][k][team].Add(kk, nn)
                    self.b["def"][k][dfn].Add(kk, nn)
                    self.lg[k].Add(kk, nn)
            if trans:
                tp = sum(p["pts"] for p in trans)
                self.g["off"]["trans_ppp"][team].Add(tp, len(trans))
                self.g["def"]["trans_ppp"][dfn].Add(tp, len(trans))
            self.g["off"]["ft"][team].Add(max(ft_pts, 0), n)
            self.g["def"]["ft"][dfn].Add(max(ft_pts, 0), n)
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
