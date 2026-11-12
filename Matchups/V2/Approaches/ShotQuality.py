r"""
ShotQuality: a hierarchical Beta-binomial shot model (league cell -> player cell) and team shot-profile
posteriors, giving expected points from shot quality rather than raw makes.

Cells = shot zone (restricted area, paint non-RA, mid-range, corner 3s, above-the-break 3, backcourt).
League make rate per cell ~ Beta(prior 0.45 / 0.36, strength 500), decayed. Player make rate per cell ~
Beta(mean = league cell rate, strength 60) - partial pooling toward the league. Team offensive shot mix ~
Dirichlet(league mix x 50 + decayed counts); defensive mix allowed likewise; defensive make suppression per
team ~ Gamma(1, 400 expected-make equivalents). FGA per possession and FT points per possession ~ Gamma-Poisson.
Expected points per possession for A vs B = FGA/poss_A * sum_cell mix_AB(cell) * value(cell) * P(make | A's
shooters, B's suppression) + FTpts/poss_A, with mix_AB the average of A's mix and B's allowed mix.
Posterior variances of each Beta / Gamma propagate by the delta method.
"""

from collections import defaultdict

import numpy as np

from Matchups.V2.Approaches.Common import LEAGUE_PACE, PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BetaBinomial, GammaPoisson

CELLS = ("Restricted Area", "In The Paint (Non-RA)", "Mid-Range", "Left Corner 3", "Right Corner 3", "Above the Break 3", "Backcourt")
VALUE = {c: (3 if "3" in c or c == "Backcourt" else 2) for c in CELLS}
LEAGUE_MIX = np.array([0.32, 0.14, 0.12, 0.05, 0.05, 0.31, 0.01])
HOME_EDGE = 2.5


@Register
class ShotQuality(Approach):
    name = "shot_quality"
    idea = "Hierarchical Beta-binomial shot-making by zone (league -> player) x team shot-mix Dirichlets -> expected PPP."
    data_used = ("shots", "pbp", "player_games")
    covers = ("shot_quality", "defense_schemes", "possession")
    priors = {"league cell make %": ("Beta(0.45 two / 0.36 three, 500)", "well known league rates"),
              "player cell make %": ("Beta(league cell, 60)", "~60 attempts before a player's own rate dominates"),
              "shot mix": ("Dirichlet(league mix x 50)", ""), "defensive suppression": ("Gamma(1, 400)", "defense effects on makes are small"),
              "FGA/poss, FT pts/poss": ("Gamma(0.85, 300), Gamma(0.20, 300)", "")}

    def __init__(self, forget=0.998, **params):
        super().__init__(**params)
        self.league = {c: BetaBinomial(0.36 if VALUE[c] == 3 else 0.45, 500.0) for c in CELLS}
        self.player = {}
        self.mix_off = defaultdict(lambda: np.zeros(len(CELLS)))
        self.mix_def = defaultdict(lambda: np.zeros(len(CELLS)))
        self.sup_y, self.sup_e = defaultdict(float), defaultdict(float)
        self.fga = defaultdict(lambda: GammaPoisson(0.85, 300.0))
        self.ftp = defaultdict(lambda: GammaPoisson(0.20, 300.0))
        self.shooters = defaultdict(lambda: defaultdict(float))
        self.pace, self.forget = PaceBook(), forget

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        for b in self.league.values():
            b.Decay(f)
        for b in self.player.values():
            b.Decay(f)
        for d in (self.mix_off, self.mix_def):
            for k in d:
                d[k] *= f
        for k in self.sup_y:
            self.sup_y[k] *= f
            self.sup_e[k] *= f
        for d in (self.fga, self.ftp):
            for g in d.values():
                g.Decay(f)

    def PlayerMake(self, p, c):
        key = (p, c)
        if key not in self.player:
            self.player[key] = BetaBinomial(self.league[c].Mean(), 60.0)
        b = self.player[key]
        return b.Mean(), b.Var()

    def Mix(self, team, side):
        counts = (self.mix_off if side == "off" else self.mix_def)[team]
        a = LEAGUE_MIX * 50 + counts
        return a / a.sum()

    def Suppression(self, team):
        return (400 + self.sup_y[team]) / (400 + self.sup_e[team]), (400 + self.sup_y[team]) / (400 + self.sup_e[team]) ** 2

    def Ppp(self, off, dfn, minutes):
        mix = 0.5 * (self.Mix(off, "off") + self.Mix(dfn, "def"))
        sup, sup_v = self.Suppression(dfn)
        wts = {p: m for p, (m, _v) in minutes.items()} if minutes else {}
        pps, var = 0.0, 0.0
        for ci, c in enumerate(CELLS):
            tot_w = sum(wts.values()) or 1.0
            if wts:
                mk = sum(w * self.PlayerMake(p, c)[0] for p, w in wts.items()) / tot_w
                mk_v = sum((w / tot_w) ** 2 * self.PlayerMake(p, c)[1] for p, w in wts.items())
            else:
                mk, mk_v = self.league[c].Mean(), self.league[c].Var()
            pps += mix[ci] * VALUE[c] * mk * sup
            var += (mix[ci] * VALUE[c]) ** 2 * (mk_v * sup ** 2 + sup_v * mk ** 2)
        fga, ftp = self.fga[off], self.ftp[off]
        mean = fga.Mean() * pps + ftp.Mean()
        return mean, fga.Mean() ** 2 * var + fga.Var() * pps ** 2 + ftp.Var(), pps

    def Predict(self, view):
        pace, pv = self.pace.Game(view.home, view.away)
        hm, hv, hpps = self.Ppp(view.home, view.away, view.Minutes("home"))
        am, av, apps = self.Ppp(view.away, view.home, view.Minutes("away"))
        mean = pace * (hm - am) + HOME_EDGE
        return self.Output(mean, pace ** 2 * (hv + av) + MARGIN_NOISE_VAR, home_xppp=hm, away_xppp=am,
                           home_xpps=hpps, away_xpps=apps)

    def Update(self, rec):
        by_team = defaultdict(list)
        for s in rec.shots:
            if s["zone"] in self.league:
                by_team[s["team"]].append(s)
        for team, shots in by_team.items():
            dfn = rec.away if team == rec.home else rec.home
            exp_makes = 0.0
            for s in shots:
                ci = CELLS.index(s["zone"])
                exp_makes += self.PlayerMake(s["player_id"], s["zone"])[0]
                self.mix_off[team][ci] += 1
                self.mix_def[dfn][ci] += 1
            makes = sum(s["made"] for s in shots)
            self.sup_y[dfn] += makes
            self.sup_e[dfn] += exp_makes
            for s in shots:
                self.league[s["zone"]].Add(s["made"], 1)
                self.player[(s["player_id"], s["zone"])].Add(s["made"], 1)
        for team in (rec.home, rec.away):
            poss = TeamPossessions(rec, team)
            rows = [r for r in rec.box if r["team"] == team]
            if poss > 0 and rows:
                self.fga[team].Add(sum(r["fga"] for r in rows), poss)
                self.ftp[team].Add(sum(r["ftm"] for r in rows), poss)
            self.pace.Update(team, poss if poss > 0 else LEAGUE_PACE)
