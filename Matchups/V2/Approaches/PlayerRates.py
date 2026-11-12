r"""
Player skill rates as conjugate posteriors, pooled by role (guard / wing / big). Shared by several approaches.

Per player: 3P% ~ Beta-binomial (prior: role-bucket league rate, strength 150 attempts), 3PA per minute,
assists, blocks, rebounds, usage events (FGA + 0.44 FTA + TOV) and turnovers per minute ~ Gamma-Poisson with
role-bucket prior means (strength 150-300 minutes). Assist network: decayed assist counts passer -> scorer.
"""

from collections import defaultdict

from Matchups.V2 import DataLayer as D
from Matchups.V2.Bayes import BetaBinomial, GammaPoisson

ROLE_PRIORS = {  # per minute rates by role bucket (guard, wing, big): rough league values, weakly informative
    "fg3a": (0.17, 0.13, 0.05), "ast": (0.17, 0.08, 0.07), "blk": (0.01, 0.017, 0.045), "reb": (0.13, 0.17, 0.30),
    "usage": (0.55, 0.45, 0.40), "tov": (0.06, 0.045, 0.05), "pts": (0.50, 0.45, 0.42)}
ROLE_3P = (0.36, 0.355, 0.31)


class PlayerRates:
    def __init__(self, minutes_strength=200.0, three_strength=150.0):
        self.rate = defaultdict(dict)
        self.three = {}
        self.assists = defaultdict(float)
        self.ms, self.ts = minutes_strength, three_strength

    def R(self, p, stat):
        r = self.rate[p].get(stat)
        if r is None:
            r = self.rate[p][stat] = GammaPoisson(ROLE_PRIORS[stat][D.RoleBucket(p)], self.ms)
        return r

    def Three(self, p):
        if p not in self.three:
            self.three[p] = BetaBinomial(ROLE_3P[D.RoleBucket(p)], self.ts)
        return self.three[p]

    def Decay(self, f):
        for d in self.rate.values():
            for g in d.values():
                g.Decay(f)
        for b in self.three.values():
            b.Decay(f)
        for k in self.assists:
            self.assists[k] *= f

    def Apply(self, rec):
        for r in rec.box:
            p, m = r["player_id"], r["minutes"]
            if m <= 0:
                continue
            for stat, val in (("fg3a", r["fg3a"]), ("ast", r["ast"]), ("blk", r["blk"]), ("reb", r["oreb"] + r["dreb"]),
                              ("usage", r["fga"] + 0.44 * r["fta"] + r["tov"]), ("tov", r["tov"]), ("pts", r["pts"])):
                self.R(p, stat).Add(val, m)
            self.Three(p).Add(r["fg3m"], r["fg3a"])
        for poss in rec.possessions:
            if poss["assister"] and poss["scorer"]:
                self.assists[(poss["assister"], poss["scorer"])] += 1.0

    def Shooter(self, p):
        return 36 * self.R(p, "fg3a").Mean() >= 4.0 and self.Three(p).Mean() >= 0.345

    def Link(self, a, b):
        return self.assists.get((a, b), 0.0) + self.assists.get((b, a), 0.0)
