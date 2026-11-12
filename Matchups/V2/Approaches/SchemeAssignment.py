r"""
SchemeAssignment: a Dirichlet-multinomial defender-assignment model with a team "switchiness" random effect,
plus Gamma-Poisson offender / defender scoring rates, giving matchup-based expected points.

Assignment for offender o against defense T: prior weights a_od = (1 - w_T) * K(height_o, height_d) * m_d
+ w_T * m_d (switch-heavy teams spread assignments by minutes, not by size), with w_T = min(1, 3 * switch_T),
switch_T ~ Beta(0.08, 200) from matchup data. Posterior shares = (history c_od + A * a_od) / (sum c + A), A = 20
possessions of prior concentration. Scoring: offender pts / poss ~ Gamma(role prior, 80 poss); defender
multiplier ~ Gamma(1, 80 poss-equivalents). Team expected points = sum_o poss_o sum_d s_od rate_o mult_d.
Update records assignment error (half-L1 vs actual shares) as a diagnostic.
"""

import math
from collections import defaultdict

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BetaBinomial, GammaPoisson

HOME_EDGE = 2.5
LEAGUE_PPP_POSS = 1.14


@Register
class SchemeAssignment(Approach):
    name = "scheme_assignment"
    idea = "Dirichlet-multinomial assignment with team switchiness random effect + Gamma-Poisson matchup scoring."
    data_used = ("box_matchups", "player_bio")
    covers = ("matchups", "defense_schemes")
    priors = {"assignment concentration": ("A = 20 possessions", "history dominates after a few meetings"),
              "size kernel": ("Gaussian, sd 3 inches", "defenders guard similar-size players"),
              "switch rate": ("Beta(0.08, 200)", "league switch share ~8%"),
              "offender rate": ("Gamma(0.22, 300 poss)", "points per partial possession; ~4 games to move halfway"),
              "defender multiplier": ("Gamma(1, 80)", "most defenders are near average")}

    def __init__(self, conc=20.0, kernel_sd=3.0, forget=0.998, **params):
        super().__init__(**params)
        self.c = defaultdict(float)
        self.switch = defaultdict(lambda: BetaBinomial(0.08, 200.0))
        self.off_rate = {}
        self.def_mult_y, self.def_mult_e = defaultdict(float), defaultdict(float)
        self.conc, self.kernel_sd, self.forget = conc, kernel_sd, forget
        self.pace = PaceBook()
        self.diagnostics = {"assign_l1": [], "assign_l1_prior": []}

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        for k in self.c:
            self.c[k] *= f
        for g in self.off_rate.values():
            g.Decay(f)
        for k in self.def_mult_y:
            self.def_mult_y[k] *= f
            self.def_mult_e[k] *= f

    def Rate(self, p):
        g = self.off_rate.get(p)
        return (g.Mean(), g.Var()) if g else (0.22, 0.22 / 300)

    def Mult(self, d):
        y, e = self.def_mult_y[d], self.def_mult_e[d]
        return (80 + y) / (80 + e), (80 + y) / (80 + e) ** 2

    def Shares(self, o, defense: dict, team, use_history=True):
        ho = D.Heights().get(o, 78.0)
        w = min(1.0, 3 * self.switch[team].Mean())
        prior = {}
        for d, m in defense.items():
            k = math.exp(-((ho - D.Heights().get(d, 78.0)) ** 2) / (2 * self.kernel_sd ** 2))
            prior[d] = ((1 - w) * k + w) * m
        z = sum(prior.values()) or 1.0
        prior = {d: v / z for d, v in prior.items()}
        if not use_history:
            return prior
        hist = {d: self.c.get((o, d), 0.0) for d in defense}
        tot = sum(hist.values())
        return {d: (hist[d] + self.conc * prior[d]) / (tot + self.conc) for d in defense}

    def TeamPoints(self, off_min, def_min, def_team, pace):
        mean = var = 0.0
        if not off_min or not def_min:          # a team with no games yet: league-average scoring, wide variance
            return pace * LEAGUE_PPP_POSS, 100.0
        dm = {d: m for d, (m, _v) in def_min.items()}
        for o, (m, _v) in off_min.items():
            poss = m / 48 * pace
            r, rv = self.Rate(o)
            for d, s in self.Shares(o, dm, def_team).items():
                k, kv = self.Mult(d)
                mean += poss * s * r * k
                var += (poss * s) ** 2 * (rv * k * k + kv * r * r)
        return mean, var

    def Predict(self, view):
        pace, _ = self.pace.Game(view.home, view.away)
        mh, ma = view.Minutes("home"), view.Minutes("away")
        hp, hv = self.TeamPoints(mh, ma, view.away, pace)
        ap, av = self.TeamPoints(ma, mh, view.home, pace)
        return self.Output(hp - ap + HOME_EDGE, hv + av + MARGIN_NOISE_VAR, home_pts=hp, away_pts=ap,
                           home_switch=self.switch[view.home].Mean(), away_switch=self.switch[view.away].Mean())

    def Update(self, rec):
        rows = rec.matchups or []
        actual = defaultdict(lambda: defaultdict(float))
        for r in rows:
            actual[(r["team"], r["off"])][r["def"]] += r["poss"]
        played = defaultdict(dict)
        for r in rec.box:
            played[r["team"]][r["player_id"]] = r["minutes"]
        for (team, o), defs in actual.items():
            tot = sum(defs.values())
            dteam = rec.away if team == rec.home else rec.home
            if tot < 3 or not played[dteam]:
                continue
            for name, hist in (("assign_l1", True), ("assign_l1_prior", False)):
                pred = self.Shares(o, played[dteam], dteam, hist)
                keys = set(pred) | set(defs)
                self.diagnostics[name].append(0.5 * sum(abs(pred.get(k, 0) - defs.get(k, 0) / tot) for k in keys))
        for r in rows:
            dteam = rec.away if r["team"] == rec.home else rec.home
            self.c[(r["off"], r["def"])] += r["poss"]
            self.switch[dteam].Add(min(r["switches"], r["poss"]), r["poss"])
            self.off_rate.setdefault(r["off"], GammaPoisson(0.22, 300.0)).Add(r["pts"], r["poss"])
        # defender multiplier: points allowed vs what the offenders usually score
        for r in rows:
            self.def_mult_y[r["def"]] += r["pts"]
            self.def_mult_e[r["def"]] += r["poss"] * self.Rate(r["off"])[0]
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
