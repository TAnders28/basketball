r"""Shared pieces for the approaches: index maps, a pace book, a Kalman team-rating core, residual models."""

import math
from collections import defaultdict

import numpy as np

from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach
from Matchups.V2.Bayes import BayesLinear, Kalman

LEAGUE_PPP = 1.13
LEAGUE_PACE = 99.0


class Index:
    def __init__(self, capacity: int):
        self.ids, self.capacity = {}, capacity

    def Get(self, key, create: bool = True):
        i = self.ids.get(key)
        if i is None and create and len(self.ids) < self.capacity:
            i = self.ids[key] = len(self.ids)
        return i

    def __len__(self):
        return len(self.ids)


def BoxPossessions(box: list, team: str) -> float:
    rows = [r for r in box if r["team"] == team]
    return sum(r["fga"] + 0.44 * r["fta"] - r["oreb"] + r["tov"] for r in rows)


def TeamPossessions(rec, team: str) -> float:
    if rec.possessions:
        return float(sum(1 for p in rec.possessions if p["offense"] == team))
    return BoxPossessions(rec.box, team) if rec.box else LEAGUE_PACE


class PaceBook:
    """Normal-Normal pace per team: prior N(99, 4^2), observation noise 5^2 per game, forgetting via alpha."""

    def __init__(self, prior_mean=LEAGUE_PACE, prior_var=16.0, obs_var=25.0, forget=0.97):
        self.m, self.v = defaultdict(lambda: prior_mean), defaultdict(lambda: prior_var)
        self.obs_var, self.forget, self.prior_var = obs_var, forget, prior_var

    def Update(self, team, poss):
        v = min(self.v[team] / self.forget, self.prior_var)
        k = v / (v + self.obs_var)
        self.m[team] += k * (poss - self.m[team])
        self.v[team] = (1 - k) * v

    def Game(self, home, away):
        mean = 0.5 * (self.m[home] + self.m[away])
        return mean, 0.25 * (self.v[home] + self.v[away]) + 9.0


class KalmanCore:
    """
    Dynamic offense/defense team ratings (points per game above league mean), a Kalman filter that replaces Elo.
      home_pts - mu =  hca/2 + off_home + def_away      (def > 0 means a leaky defense)
      away_pts - mu = -hca/2 + off_away + def_home
    States: 30 off + 30 def + 1 home-court; random walk with q per day; new season: regress 30% to 0, +var.
    """
    TEAMS = ["ATL", "BOS", "BKN", "CHA", "CHI", "CLE", "DAL", "DEN", "DET", "GSW", "HOU", "IND", "LAC", "LAL", "MEM",
             "MIA", "MIL", "MIN", "NOP", "NYK", "OKC", "ORL", "PHI", "PHX", "POR", "SAC", "SAS", "TOR", "UTA", "WAS"]

    def __init__(self, prior_var=16.0, q_per_day=0.02, obs_var=110.0, hca_prior=2.5, season_regress=0.3, season_var=6.0):
        self.ix = {t: i for i, t in enumerate(self.TEAMS)}
        n = 2 * len(self.TEAMS) + 1
        self.k = Kalman(n, prior_var, q_per_day, obs_var)
        self.k.x[-1] = hca_prior
        self.k.P[-1, -1] = 1.0
        self.mu, self.season = 112.0, None
        self.season_regress, self.season_var = season_regress, season_var

    def H(self, side_off, side_def, sign):
        n = len(self.k.x)
        h = np.zeros(n)
        h[self.ix[side_off]] = 1.0
        h[len(self.TEAMS) + self.ix[side_def]] = 1.0
        h[-1] = 0.5 * sign
        return h

    def NewSeasonCheck(self, season):
        if self.season is not None and season != self.season:
            hca = self.k.x[-1]
            self.k.x[:-1] *= (1 - self.season_regress)
            self.k.x[-1] = hca
            self.k.P[:-1, :-1] += np.eye(len(self.k.x) - 1) * self.season_var
        self.season = season

    def MarginVector(self, home, away):
        return self.H(home, away, 1) - self.H(away, home, -1)

    def Forecast(self, home, away):
        if home not in self.ix or away not in self.ix:
            return 0.0, MARGIN_NOISE_VAR
        h = self.MarginVector(home, away)
        mean = float(h @ self.k.x)
        return mean, float(h @ self.k.P @ h) + MARGIN_NOISE_VAR

    def Update(self, game):
        home, away = game["home_team"], game["away_team"]
        if home not in self.ix or away not in self.ix:
            return
        self.k.Update(self.H(home, away, 1), game["home_score"] - self.mu)
        self.k.Update(self.H(away, home, -1), game["away_score"] - self.mu)
        self.mu += 0.01 * (0.5 * (game["home_score"] + game["away_score"]) - self.mu)


def LearnPriorScale(reg, v0_base, iterations=5, lo=1e-3, hi=4.0, dims=None):
    """
    Evidence (type-II ML) fixed point for a shared prior-variance scale s with prior var = s * v0_base:
      gamma = sum_i (1 - Sigma_ii / (s v0_i))   (effective number of well-determined parameters)
      s     = sum_i (m_i - m0_i)^2 / v0_i / gamma
    (MacKay 1992). Refits the posterior at each step. Returns s, clipped to [lo, hi].
    """
    n = len(v0_base) if dims is None else dims
    s = float(np.mean(reg.v0[:n] / v0_base[:n]))
    for _ in range(iterations):
        m = reg.mean[:n] - reg.m0[:n]
        diag = np.diag(reg.cov)[:n]
        gamma = float(np.sum(1.0 - diag / (s * v0_base[:n])))
        s_new = float(np.sum(m * m / v0_base[:n]) / max(gamma, 1e-6)) if gamma > 1e-6 else lo
        s_new = min(max(s_new, lo), hi)
        if abs(s_new - s) < 1e-4 * s:
            break
        s = s_new
        reg.v0[:n] = v0_base[:n] * s
        reg.dirty = True
        reg.Fit(active=n if dims is not None else None)
    return s


class ResidualOnKalman(Approach):
    """
    Base for "adjustment" approaches: margin = Kalman margin + x'beta, with beta ~ N(0, prior_var) learned by
    Bayesian linear regression on the Kalman residual (actual margin - Kalman pregame mean). Subclasses supply
    Features(view) (pregame) and may override Learn(rec) for extra state.
    """
    feature_names = ()
    prior_var = 4.0

    def __init__(self, **params):
        super().__init__(**params)
        self.core = KalmanCore()
        n = len(self.feature_names)
        pv = params.get("prior_var", self.prior_var)
        self.v0_base = (np.full(n, float(pv)) if np.isscalar(pv) else np.asarray(pv, float).copy()) * params.get("prior_scale", 1.0)
        self.reg = BayesLinear(n, self.v0_base.copy(), prior_mean=getattr(self, "prior_mean", None), sigma2=MARGIN_NOISE_VAR)
        # learned prior variance (type-II ML / MacKay evidence updates on a shared scale s): prior var = s * v0.
        # If s collapses below drop_below the whole term family is dropped (adjustment 0).
        self.learn_tau = params.get("learn_tau", True)
        self.min_evidence_games = params.get("min_evidence_games", 200)
        self.drop_below = params.get("drop_below", 0.05)
        self.tau_scale, self.dropped = 1.0, False
        self.pending = {}
        self.forget = params.get("forget", 0.999)

    def Features(self, view) -> np.ndarray:
        raise NotImplementedError

    def BeforeDay(self, day, gap):
        self.core.k.Predict(gap)
        self.reg.Decay(self.forget ** gap)
        self.reg.Fit()
        if self.learn_tau and self.reg.sw >= self.min_evidence_games:
            self.tau_scale = LearnPriorScale(self.reg, self.v0_base)
            self.dropped = self.tau_scale < self.drop_below

    def Predict(self, view):
        self.core.NewSeasonCheck(view.season)
        base_m, base_v = self.core.Forecast(view.home, view.away)
        x = self.Features(view)
        adj, adj_v = self.reg.Predict(x)
        adj_v -= self.reg.Sigma2()     # parameter uncertainty only; noise already in base_v
        if self.dropped:
            adj, adj_v = 0.0, 0.0
        self.pending[view.game_id] = (x, base_m)
        return self.Output(base_m + adj, base_v + max(adj_v, 0.0), base_margin=base_m, adjustment=adj,
                           tau_scale=self.tau_scale, term_dropped=int(self.dropped),
                           **{f"f_{n}": float(v) for n, v in zip(self.feature_names, x)})

    def Learn(self, rec):
        pass

    def Update(self, rec):
        g = rec.game
        if g["game_id"] in self.pending:
            x, base_m = self.pending.pop(g["game_id"])
            self.reg.Add(x, (g["home_score"] - g["away_score"]) - base_m)
        else:
            self.core.NewSeasonCheck(g["season"])
        self.core.Update(g)
        self.Learn(rec)
