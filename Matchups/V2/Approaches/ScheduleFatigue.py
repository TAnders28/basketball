r"""
ScheduleFatigue: rest, density and travel as Bayesian adjustments on top of the Kalman team ratings.

Features (home minus away where it makes sense): rest days (capped 3), back-to-back flags, 3-in-4 and 4-in-6,
travel miles since the last game, time-zone shift, altitude for visitors at DEN/UTA, road-trip length, and
recent minutes load of each team's top-3 players (last 3 games). Coefficients ~ N(0, 2^2) pts: weakly
informative (published B2B effects are ~1-2 pts), learned by conjugate Bayesian regression on Kalman residuals.
"""

from collections import defaultdict, deque

import numpy as np

from Matchups.V2.Approaches.Common import ResidualOnKalman
from Matchups.V2.Base import Register


@Register
class ScheduleFatigue(ResidualOnKalman):
    name = "schedule_fatigue"
    idea = "Rest / back-to-back / 3-in-4 / travel / time zone / altitude / star minutes load as Bayesian residual terms on Kalman."
    data_used = ("games", "player_games", "arenas")
    covers = ("fatigue_schedule", "team_strength")
    feature_names = ("rest_diff", "home_b2b", "away_b2b", "home_3in4", "away_3in4", "away_4in6", "away_travel_k",
                     "home_travel_k", "away_tz_abs", "away_altitude", "away_road_streak", "load_diff")
    prior_var = 4.0
    priors = {"each coefficient": ("N(0, 2^2) pts", "schedule effects are a few points at most"),
              "noise": ("13^2", "single-game margin noise")}

    def __init__(self, **params):
        super().__init__(**params)
        self.recent = defaultdict(lambda: deque(maxlen=3))   # player -> last 3 minutes
        self.team_players = defaultdict(set)

    def Load(self, team):
        loads = sorted((sum(self.recent[p]) for p in self.team_players[team]), reverse=True)[:3]
        return sum(loads) / 100.0

    def Features(self, view):
        h, a = view.Schedule("home"), view.Schedule("away")
        return np.array([min(h["rest"], 3) - min(a["rest"], 3), h["b2b"], a["b2b"], h["three_in_four"], a["three_in_four"],
                         a["four_in_six"], a["travel_miles"] / 1000.0, h["travel_miles"] / 1000.0, abs(a["tz_shift"]),
                         float(a["altitude_ft"] > 4000), min(a["road_streak"], 6) / 3.0,
                         self.Load(view.home) - self.Load(view.away)], dtype=float)

    def Learn(self, rec):
        for r in rec.box:
            self.recent[r["player_id"]].append(r["minutes"])
            self.team_players[r["team"]].add(r["player_id"])
