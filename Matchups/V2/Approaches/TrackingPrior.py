r"""
TrackingPrior: a pregame player-value prior from last season's tracking, hustle, closest-defender and bio data.

Empirical Bayes in two steps, refit at each new season s using ONLY season s-1 tables:
 1. Target: each player's s-1 on-court plus-minus per 48, observation variance ~ 14^2 * 48 / minutes (noisier
    for low minutes). 2. Bayesian linear regression of the target on standardized tracking features, beta ~ N(0, 1.5^2).
The posterior predictive value v_p (pts per 48 on court) is the prior for season s: unknown players (rookies,
no tracking row) get N(-1, 2^2) because new / fringe players are below average on balance. As an approach:
margin = sum(home minutes/48 * v) - sum(away minutes/48 * v) + home edge 2.5. Provider() exposes (O, D) prior
means per 100 for StintRapm (v split evenly between offense and defense).
"""

import math
from collections import defaultdict

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register
from Matchups.V2.Bayes import BayesLinear

PER_MIN = [("tracking/Player_SpeedDistance.json", "DIST_MILES"), ("tracking/Player_Drives.json", "DRIVES"),
           ("tracking/Player_Drives.json", "DRIVE_PTS"), ("tracking/Player_Passing.json", "POTENTIAL_AST"),
           ("tracking/Player_Passing.json", "AST_PTS_CREATED"), ("tracking/Player_Possessions.json", "TIME_OF_POSS"),
           ("tracking/Player_Possessions.json", "TOUCHES"), ("tracking/Player_CatchShoot.json", "CATCH_SHOOT_FGA"),
           ("tracking/Player_PullUpShot.json", "PULL_UP_FGA"), ("tracking/Player_Defense.json", "DEF_RIM_FGA"),
           ("hustle_player.json", "DEFLECTIONS"), ("hustle_player.json", "CONTESTED_SHOTS"),
           ("hustle_player.json", "SCREEN_ASSISTS"), ("hustle_player.json", "BOX_OUTS"), ("hustle_player.json", "CHARGES_DRAWN"),
           ("hustle_player.json", "LOOSE_BALLS_RECOVERED")]
RATES = [("tracking/Player_SpeedDistance.json", "AVG_SPEED_OFF"), ("tracking/Player_SpeedDistance.json", "AVG_SPEED_DEF"),
         ("tracking/Player_CatchShoot.json", "CATCH_SHOOT_EFG_PCT"), ("tracking/Player_PullUpShot.json", "PULL_UP_EFG_PCT"),
         ("tracking/Player_Defense.json", "DEF_RIM_FG_PCT"), ("tracking/Player_Rebounding.json", "OREB_CHANCE_PCT"),
         ("tracking/Player_Rebounding.json", "DREB_CHANCE_PCT"), ("player_bio.json", "PLAYER_HEIGHT_INCHES"),
         ("player_bio.json", "AGE"), ("player_bio.json", "USG_PCT"), ("player_bio.json", "TS_PCT"), ("player_bio.json", "AST_PCT")]
DEFEND = ("defend/Overall.json", "PCT_PLUSMINUS")
UNKNOWN_PRIOR = (-1.0, 4.0)


def Num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def FeatureTable(season: int):
    """{player: feature vector} and names, from season-aggregate tables of `season` (caller enforces prior-only)."""
    feats = defaultdict(dict)
    minutes = {}
    for rel, col in PER_MIN + RATES:
        for r in D.SeasonTable(season, rel):
            p = int(r["PLAYER_ID"])
            if rel.endswith("SpeedDistance.json"):
                minutes[p] = Num(r.get("MIN"))
            feats[p][col if (rel, col) in RATES else "pm_" + col] = Num(r.get(col))
    for r in D.SeasonTable(season, DEFEND[0]):
        feats[int(r["CLOSE_DEF_PERSON_ID"])]["d_pct_plusminus"] = Num(r.get(DEFEND[1]))
    names = [("pm_" + c) for _, c in PER_MIN] + [c for _, c in RATES] + ["d_pct_plusminus"]
    out = {}
    for p, f in feats.items():
        m = minutes.get(p) or float("nan")
        vec = [f.get(n, float("nan")) / m if n.startswith("pm_") and m and m == m else f.get(n, float("nan")) for n in names]
        out[p] = np.array(vec, dtype=float)
    return out, names, minutes


@Register
class TrackingPrior(Approach):
    name = "tracking_prior"
    idea = "Prior-season tracking/hustle/defend/bio -> empirical-Bayes player value prior; usable alone or as StintRapm prior."
    data_used = ("tracking", "hustle", "defend", "player_bio", "player_games")
    covers = ("team_strength", "defense_schemes", "rebounding", "shot_quality")
    priors = {"regression coefficients": ("N(0, 1.5^2) pts/48 per sd of feature", "features are noisy proxies"),
              "unknown players": ("N(-1, 2^2) pts/48", "new/fringe players are below average on balance"),
              "target noise": ("14^2 * 48 / minutes", "on-court +/- per 48 is very noisy at low minutes")}

    def __init__(self, **params):
        super().__init__(**params)
        self.season, self.value, self.value_var, self.fit_info = None, {}, {}, {}

    def Build(self, view):
        season = view.season
        prior_s = season - 1
        feats, names, _mins = FeatureTable(prior_s)
        view.Prior("player_bio.json", prior_s)   # leak guard check (raises if prior_s >= season)
        pm, mins = defaultdict(float), defaultdict(float)
        for rows in D.PlayerBox(prior_s).values():
            for r in rows:
                pm[r["player_id"]] += r["plus_minus"]
                mins[r["player_id"]] += r["minutes"]
        players = [p for p in feats if mins.get(p, 0) >= 200]
        self.value, self.value_var = {}, {}
        if len(players) < 30:
            self.season = season
            return
        X = np.array([feats[p] for p in players])
        mu, sd = np.nanmean(X, 0), np.nanstd(X, 0) + 1e-9
        Z = np.nan_to_num((X - mu) / sd)
        y = np.array([48 * pm[p] / mins[p] for p in players])
        w = np.array([mins[p] / (14 ** 2 * 48) for p in players])
        reg = BayesLinear(Z.shape[1] + 1, np.r_[25.0, np.full(Z.shape[1], 2.25)], sigma2=1.0)
        for zi, yi, wi in zip(Z, y, w):
            reg.Add(np.r_[1.0, zi], yi, wi)
        reg.Fit()
        for p, f in feats.items():
            z = np.nan_to_num((f - mu) / sd)
            m, v = reg.Predict(np.r_[1.0, z])
            self.value[p], self.value_var[p] = m, v - reg.Sigma2()
        top = np.argsort(-np.abs(reg.mean[1:]))[:5]
        self.fit_info = {"season_fit": prior_s, "n_players": len(players),
                         "top_features": [(names[i], round(float(reg.mean[1 + i]), 3)) for i in top]}
        self.season = season

    def Value(self, p):
        return self.value.get(p, UNKNOWN_PRIOR[0]), self.value_var.get(p, UNKNOWN_PRIOR[1])

    def Provider(self):
        def Prior(p):
            if p not in self.value:
                return None
            # on-court +/- per 48 is a five-man number: a lineup of five +v players is +v as a team, so each
            # player's (O - D) share is v / 5, split evenly between offense and defense.
            v = self.value[p]
            return (v / 10.0, -v / 10.0)
        return Prior

    def Predict(self, view):
        if self.season != view.season:
            self.Build(view)
        tot_m, tot_v = 2.5, 0.0
        for side, sign in (("home", 1), ("away", -1)):
            for p, (m, _var) in view.Minutes(side).items():
                v, vv = self.Value(p)
                # team net = minutes-weighted average of on-court +/- per 48 (240 player-minutes per game)
                tot_m += sign * m / 240 * v
                tot_v += (m / 240) ** 2 * vv
        return self.Output(tot_m, tot_v + MARGIN_NOISE_VAR, prior_season=self.fit_info.get("season_fit"))
