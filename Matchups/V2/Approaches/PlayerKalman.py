r"""
PlayerKalman: dynamic Bayesian player ratings (Glicko / state-space style, DARKO-like), plus availability scenarios.

Each player has a net rating r_p (points per 100 possessions, on-court contribution) with variance P_p. Every
stint is one Gaussian observation of  100 * (home pts - away pts) / possessions = hca + sum_home r - sum_away r + e,
e ~ N(0, 25,000 / possessions). Updates are the diagonal (Glicko) Kalman approximation: each player's gain is
P_p / (sum of on-court P + P_hca + R). Process noise q per day accrues while a player is idle or playing; variance
is inflated in the offseason (gap > 60 days: +3, means regressed 20% to the role mean), after a team change (+2),
and after a long absence (> 30 days without a game, +1), with a floor of 0.25 (RD floor).
New players start at N(role mean - 0.5, 3^2).

Prediction: team rating = sum_p (expected minutes_p / 48) * r_p, scaled by pace; variance from P (diagonal) plus
13^2 game noise. Modes:
  "minutes"  : RosterBook minutes (players from the last game), the v2 default.
  "scenario" : StatusModel availability probabilities -> enumerated scenarios -> minutes redistribution -> mixture.
With Walk(availability=OracleAvailability) the "minutes" mode becomes the ORACLE known-roster variant.
"""

import math
from collections import defaultdict

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import PaceBook, TeamPossessions
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V3.Availability import Mixture, StatusModel

NET_NOISE_PER100 = 25000.0


@Register
class PlayerKalman(Approach):
    name = "player_kalman"
    idea = "Dynamic (Kalman/Glicko) player net ratings from stints with offseason/trade/absence variance inflation."
    data_used = ("rotation", "pbp", "player_games", "player_bio")
    covers = ("team_strength", "rotations_coaching", "teammate_synergy")
    priors = {"new player": ("N(role mean - 0.5, 3^2) per 100", "unknown players are slightly below average"),
              "process noise": ("0.004 per 100^2 per day", "skill drifts ~1.1 per 100 over a season"),
              "offseason": ("+3 variance, 20% regression to role mean", "aging, roster and role changes"),
              "team change": ("+2 variance", "new role and teammates"),
              "long absence": ("+1 variance after > 30 idle days", "injury return uncertainty"),
              "RD floor": ("P >= 0.25", "never fully certain (Glicko RD floor)"),
              "home edge": ("N(2.5, 1) per 100, learned", "recent NBA home edge")}
    mode = "minutes"

    def __init__(self, prior_var=9.0, q=0.004, offseason_var=3.0, trade_var=2.0, absence_var=1.0, floor=0.25, **params):
        super().__init__(**params)
        self.m, self.P = {}, {}
        self.last, self.team = {}, {}
        self.hca, self.P_hca = 2.5, 1.0
        self.prior_var, self.q, self.offseason_var = prior_var, q, offseason_var
        self.trade_var, self.absence_var, self.floor = trade_var, absence_var, floor
        self.day_index, self.day = 0, None
        self.pace = PaceBook()
        self.status = StatusModel()
        self.role_sum, self.role_n = np.zeros(3), np.zeros(3)

    def RoleMean(self, p):
        b = D.RoleBucket(p)
        return float(self.role_sum[b] / self.role_n[b]) if self.role_n[b] > 20 else 0.0

    def Ensure(self, p):
        if p not in self.m:
            self.m[p], self.P[p], self.last[p] = self.RoleMean(p) - 0.5, self.prior_var, self.day_index

    def Var(self, p):
        if p not in self.m:
            return self.prior_var
        return self.P[p] + self.q * (self.day_index - self.last[p])

    def Mean(self, p):
        return self.m[p] if p in self.m else self.RoleMean(p) - 0.5

    def BeforeDay(self, day, gap):
        self.day_index += gap
        if gap > 60:
            for p in self.m:
                rm = self.RoleMean(p)
                self.m[p] = rm + 0.8 * (self.m[p] - rm)
                self.P[p] += self.offseason_var
            self.P_hca += 0.5
        # role means from current posteriors (for new-player priors)
        self.role_sum[:], self.role_n[:] = 0.0, 0.0
        for p, v in self.m.items():
            b = D.RoleBucket(p)
            self.role_sum[b] += v
            self.role_n[b] += 1

    def TeamMoments(self, minutes):
        mean = var = 0.0
        for p, mm in minutes.items():
            w = min(mm / 48.0, 1.0)
            mean += w * self.Mean(p)
            var += w * w * self.Var(p)
        return mean, var

    def SideMoments(self, view, side):
        team = view.home if side == "home" else view.away
        if self.mode == "scenario":
            sc = self.status.Scenarios(team)
            if sc:
                parts = [(w, *self.TeamMoments(m)) for w, m in sc]
                return Mixture(parts) + (len(sc),)
        return self.TeamMoments({p: m for p, (m, _v) in view.Minutes(side).items()}) + (1,)

    def Predict(self, view):
        hm, hv, hs = self.SideMoments(view, "home")
        am, av, as_ = self.SideMoments(view, "away")
        pace, pace_v = self.pace.Game(view.home, view.away)
        k = pace / 100.0
        net = hm - am + self.hca
        mean = k * net
        var = k * k * (hv + av + self.P_hca) + (net / 100.0) ** 2 * pace_v
        return self.Output(mean, var + MARGIN_NOISE_VAR, rating_var=var, home_scenarios=hs, away_scenarios=as_,
                           hca_per100=self.hca)

    def Observe(self, home5, away5, y, poss):
        ps = list(home5) + list(away5)
        signs = [1.0] * len(home5) + [-1.0] * len(away5)
        for p in ps:
            self.Ensure(p)
            idle = self.day_index - self.last[p]
            self.P[p] += self.q * idle
            self.last[p] = self.day_index
        pred = self.hca + sum(s * self.m[p] for p, s in zip(ps, signs))
        R = NET_NOISE_PER100 / poss
        S = sum(self.P[p] for p in ps) + self.P_hca + R
        v = y - pred
        for p, s in zip(ps, signs):
            k = self.P[p] / S
            self.m[p] += s * k * v
            self.P[p] = max(self.P[p] - self.P[p] * k, self.floor)
        kh = self.P_hca / S
        self.hca += kh * v
        self.P_hca = max(self.P_hca - self.P_hca * kh, 0.05)

    def Update(self, rec):
        # inflation for team changes and long absences, before using this game's stints
        for r in rec.box:
            p, t = r["player_id"], r["team"]
            if p in self.m:
                if self.team.get(p) not in (None, t):
                    self.P[p] += self.trade_var
                idle = self.day_index - self.last[p]
                if 30 < idle <= 60:
                    self.P[p] += self.absence_var
            self.team[p] = t
        for s in rec.stints:
            poss = 0.5 * (s["home_poss"] + s["away_poss"])
            if poss < 2 or len(s["home5"]) != 5 or len(s["away5"]) != 5:
                continue
            y = 100.0 * (s["home_pts"] - s["away_pts"]) / poss
            self.Observe(s["home5"], s["away5"], y, poss)
        for team in (rec.home, rec.away):
            self.pace.Update(team, TeamPossessions(rec, team))
            self.status.Update(team, rec.box)


@Register
class PlayerKalmanScenario(PlayerKalman):
    name = "player_kalman_scen"
    idea = "PlayerKalman with availability scenarios: learned status probabilities, minutes redistribution, mixture rating."
    data_used = PlayerKalman.data_used
    covers = ("team_strength", "rotations_coaching", "availability")
    priors = dict(PlayerKalman.priors, **{"status probability": ("Beta-binomial per missed-games bucket, prior 0.93 / 0.55 / ... / 0.1, strength 30", "no injury feed; learned from appearance history")})
    mode = "scenario"
