r"""
SwitchHunt: does an offense hunt the weakest defender, and how exposed is the opponent's weak link?

From matchup rows: each defender's points allowed per partial possession ~ Gamma-Poisson (prior: league rate,
strength 60 poss). Each game, an offense's "hunt share" = share of its matchup possessions spent against the
defense's two weakest on-court defenders (posterior means), versus those defenders' share of defensive time.
Hunting propensity per offense ~ Beta-binomial (prior 0.4, strength 50). Vulnerability per defense = minutes-
weighted gap between its weakest projected defenders and its average defender. Feature = propensity x opponent
vulnerability (home minus away), a Bayesian residual term on Kalman ratings.
"""

from collections import defaultdict

import numpy as np

from Matchups.V2.Approaches.Common import ResidualOnKalman
from Matchups.V2.Bayes import BetaBinomial, GammaPoisson
from Matchups.V2.Base import Register

LEAGUE_DEF_PPP = 0.22


@Register
class SwitchHunt(ResidualOnKalman):
    name = "switch_hunt"
    idea = "Hunting propensity (Beta-binomial) x opponent weak-defender exposure (Gamma-Poisson defender ratings)."
    data_used = ("box_matchups", "player_games")
    covers = ("matchups", "defense_schemes")
    feature_names = ("hunt_x_vuln_diff",)
    prior_var = 25.0
    priors = {"defender pts/poss": ("Gamma(0.22, 60 poss)", "defender-level allowed rates are noisy"),
              "hunt share": ("Beta(mean 0.4, strength 50)", "two weakest of five defenders get ~40% at random"),
              "effect": ("N(0, 5^2) pts per unit", "weak prior, unit = 1 PPP gap x propensity")}

    def __init__(self, **params):
        super().__init__(**params)
        self.def_rate = {}
        self.hunt = defaultdict(lambda: BetaBinomial(0.4, 50.0))

    def Rate(self, p):
        r = self.def_rate.get(p)
        return r.Mean() if r else LEAGUE_DEF_PPP

    def Vulnerability(self, minutes):
        if not minutes:
            return 0.0
        rates = sorted(((self.Rate(p), m) for p, (m, _v) in minutes.items()), reverse=True)
        avg = sum(r * m for r, m in rates) / max(sum(m for _, m in rates), 1e-9)
        worst = rates[:2]
        return sum((r - avg) * m / 48.0 for r, m in worst)

    def Features(self, view):
        mh, ma = view.Minutes("home"), view.Minutes("away")
        h = self.hunt[view.home].Mean() * self.Vulnerability(ma)
        a = self.hunt[view.away].Mean() * self.Vulnerability(mh)
        return np.array([(h - a) * 100.0])

    def Learn(self, rec):
        rows = rec.matchups or []
        by_off_team = defaultdict(list)
        for r in rows:
            by_off_team[r["team"]].append(r)
        for team, rs in by_off_team.items():
            defenders = defaultdict(float)
            for r in rs:
                defenders[r["def"]] += r["poss"]
            if len(defenders) < 5:
                continue
            weakest = sorted(defenders, key=lambda d: -self.Rate(d))[:2]
            total = sum(defenders.values())
            k = sum(defenders[d] for d in weakest)
            self.hunt[team].Add(k / 5.0, total / 5.0)
        for r in rows:
            self.def_rate.setdefault(r["def"], GammaPoisson(LEAGUE_DEF_PPP, 60.0)).Add(r["pts"], r["poss"])
