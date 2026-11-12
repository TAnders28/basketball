r"""
KalmanTeam: dynamic Bayesian team ratings (state-space), the Bayesian replacement for Elo.

Idea: each team has an offense and a defense rating (points above the league mean) that drift as a random walk
(q per day). Every game gives two Gaussian observations (each team's score). The Kalman filter keeps a full
posterior covariance, so the predictive margin variance = rating uncertainty + single-game noise, and the win
probability comes from that posterior predictive (not a fixed Elo logistic curve).
"""

from Matchups.V2.Approaches.Common import KalmanCore
from Matchups.V2.Base import Approach, Register


@Register
class KalmanTeam(Approach):
    name = "kalman_team"
    idea = "Dynamic offense/defense team ratings via a Kalman filter; posterior predictive margin and win prob."
    data_used = ("games",)
    covers = ("team_strength",)
    priors = {"rating prior sd": ("4 pts", "NBA team strengths span about +-10 pts; 4 pts is weakly informative"),
              "process noise q": ("0.02 pts^2/day", "ratings drift ~1.3 pts sd over a 90-day stretch"),
              "score noise": ("110 pts^2 per team score (sd ~10.5)", "single-team scoring sd in the NBA"),
              "home court prior": ("N(2.5, 1)", "recent NBA home edge ~2-3 pts"),
              "season carryover": ("regress 30% to mean, +6 pts^2 var", "roster turnover between seasons")}

    def __init__(self, **params):
        super().__init__(**params)
        self.core = KalmanCore(q_per_day=params.get("q", 0.02), prior_var=params.get("prior_var", 16.0),
                               obs_var=params.get("obs_var", 110.0))

    def BeforeDay(self, day, gap):
        self.core.k.Predict(gap)

    def Predict(self, view):
        self.core.NewSeasonCheck(view.season)
        m, v = self.core.Forecast(view.home, view.away)
        return self.Output(m, v, rating_var=v - 169.0)

    def Update(self, rec):
        self.core.NewSeasonCheck(rec.game["season"])
        self.core.Update(rec.game)
