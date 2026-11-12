r"""
OnOff: injury-aware with/without model. How much does a team's net rating change when each player sits?

For every player j and his team: points scored and allowed per possession with j ON and with j OFF, each a
Gamma-Poisson posterior with prior = the team's overall rate (strength 400 possessions). The shrunk on-off
swing e_j = (net_on - net_off) and j's historical on-court share h_j are tracked. Prediction:
team net = team posterior net + sum_j (projected share_j - h_j) * e_j, so a missing star (projected share 0)
moves the rating by roughly h_j * e_j. Margin = pace * (net_home - net_away) + home edge 2.5. Variances from
the Gamma posteriors propagate to the predictive.
"""

from collections import defaultdict

import numpy as np

from Matchups.V2.Approaches.Common import LEAGUE_PPP, PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import GammaPoisson

HOME_EDGE = 2.5


@Register
class OnOff(Approach):
    name = "on_off"
    idea = "Shrunk on/off net-rating swings per player; projected absences shift team ratings (injury-aware)."
    data_used = ("rotation", "pbp")
    covers = ("teammate_synergy", "rotations_coaching", "team_strength")
    priors = {"team PPP": (f"Gamma(mean {LEAGUE_PPP}, strength 1000 poss)", "prior sd ~0.034 PPP per side -> net sd ~4.7 per 100, the real NBA spread"),
              "on/off PPP": ("Gamma(mean = team rate, strength 400 poss)", "on/off splits are notoriously noisy"),
              "home edge": ("2.5 pts fixed", "kept simple; Kalman learns it elsewhere")}

    def __init__(self, strength=400.0, forget=0.998, **params):
        super().__init__(**params)
        self.team_for = defaultdict(lambda: GammaPoisson(LEAGUE_PPP, 1000.0))
        self.team_against = defaultdict(lambda: GammaPoisson(LEAGUE_PPP, 1000.0))
        self.on = {}      # (team, player) -> [for GP, against GP, poss_on]
        self.off = {}
        self.strength, self.forget = strength, forget
        self.pace = PaceBook()

    def BeforeDay(self, day, gap):
        f = self.forget ** gap
        for d in (self.team_for, self.team_against):
            for g in d.values():
                g.Decay(f)
        for d in (self.on, self.off):
            for v in d.values():
                v[0].Decay(f)
                v[1].Decay(f)
                v[2] *= f

    def Entry(self, store, team, p):
        if (team, p) not in store:
            store[(team, p)] = [GammaPoisson(self.team_for[team].Mean(), self.strength),
                                GammaPoisson(self.team_against[team].Mean(), self.strength), 0.0]
        return store[(team, p)]

    def Swing(self, team, p):
        on, off = self.on.get((team, p)), self.off.get((team, p))
        if not on or not off or on[2] + off[2] <= 0:
            return 0.0, 0.0, 0.0
        e = (on[0].Mean() - on[1].Mean()) - (off[0].Mean() - off[1].Mean())
        var = on[0].Var() + on[1].Var() + off[0].Var() + off[1].Var()
        return e, var, on[2] / (on[2] + off[2])

    def TeamNet(self, team, minutes):
        net = self.team_for[team].Mean() - self.team_against[team].Mean()
        var = self.team_for[team].Var() + self.team_against[team].Var()
        shift = 0.0
        known = {p for (t, p) in self.on if t == team}
        for p in known | set(minutes):
            e, v, h = self.Swing(team, p)
            proj = min(minutes.get(p, (0.0, 0.0))[0] / 48.0, 1.0)
            shift += (proj - h) * e
            var += (proj - h) ** 2 * v
        return net + shift, var, shift

    def Predict(self, view):
        nh, vh, sh = self.TeamNet(view.home, view.Minutes("home"))
        na, va, sa = self.TeamNet(view.away, view.Minutes("away"))
        pace, pv = self.pace.Game(view.home, view.away)
        mean = pace * (nh - na) + HOME_EDGE
        return self.Output(mean, pace ** 2 * (vh + va) + MARGIN_NOISE_VAR, absence_shift_home=pace * sh,
                           absence_shift_away=pace * sa)

    def Update(self, rec):
        for team, opp in ((rec.home, rec.away), (rec.away, rec.home)):
            pf = sum(p["pts"] for p in rec.possessions if p["offense"] == team)
            pa = sum(p["pts"] for p in rec.possessions if p["offense"] == opp)
            nf = sum(1 for p in rec.possessions if p["offense"] == team)
            na = sum(1 for p in rec.possessions if p["offense"] == opp)
            if nf and na:
                self.team_for[team].Add(pf, nf)
                self.team_against[team].Add(pa, na)
            self.pace.Update(team, TeamPossessions(rec, team))
        players = defaultdict(set)
        for s in rec.stints:
            players[rec.home].update(s["home5"])
            players[rec.away].update(s["away5"])
        for s in rec.stints:
            for team, five, pf, pa, nf, na in ((rec.home, s["home5"], s["home_pts"], s["away_pts"], s["home_poss"], s["away_poss"]),
                                               (rec.away, s["away5"], s["away_pts"], s["home_pts"], s["away_poss"], s["home_poss"])):
                if nf <= 0 or na <= 0:
                    continue
                for p in players[team]:
                    e = self.Entry(self.on if p in five else self.off, team, p)
                    e[0].Add(pf, nf)
                    e[1].Add(pa, na)
                    e[2] += 0.5 * (nf + na)
