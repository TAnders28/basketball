r"""
LineupVsLineup: the integrated stint-level possession model. Coach-projected minutes (CoachProfile) feed the
hierarchical player ratings plus shrunk pair synergies (PairSynergy -> StintRapm), and minutes uncertainty is
propagated: var = rating covariance term + sum_p (rating_p * pace / 100 / 48)^2 * var(minutes_p) + game noise.
It also lists the most likely five-man units per side (product of on-court probabilities, top 3) for
storyline output.
"""

from itertools import combinations

import numpy as np

from Matchups.V2.Approaches.CoachProfile import CoachProfile
from Matchups.V2.Approaches.PairSynergy import PairSynergy
from Matchups.V2.Base import MARGIN_NOISE_VAR, Approach, Register


class _CoachView:
    def __init__(self, view, coach):
        self._v, self._c = view, coach

    def __getattr__(self, k):
        return getattr(self._v, k)

    def Minutes(self, side):
        return self._c.ProjectMinutes(self._v, side)


def LikelyFives(minutes: dict, k=3):
    ps = sorted(minutes, key=lambda p: -minutes[p][0])[:9]
    scored = []
    for five in combinations(ps, 5):
        prob = np.prod([min(minutes[p][0] / 48, 0.99) for p in five])
        scored.append((prob, five))
    return [f for _p, f in sorted(scored, reverse=True)[:k]]


@Register
class LineupVsLineup(Approach):
    name = "lineup_vs_lineup"
    idea = "Coach-projected minutes x hierarchical player ratings + pair synergies, with minutes uncertainty propagated."
    data_used = ("rotation", "pbp", "player_games", "roster", "player_bio")
    covers = ("rotations_coaching", "teammate_synergy", "team_strength", "possession")
    priors = {"inherits": ("StintRapm, PairSynergy, CoachProfile priors", "composition of pooled models")}

    def __init__(self, **params):
        super().__init__(**params)
        self.coach = CoachProfile(**params)
        self.syn = PairSynergy(**params)

    def BeforeDay(self, day, gap):
        self.coach.BeforeDay(day, gap)
        self.syn.BeforeDay(day, gap)

    def Predict(self, view):
        cv = _CoachView(view, self.coach)
        out = self.syn.Predict(cv)
        base = self.syn.base
        pace, _ = base.pace.Game(view.home, view.away)
        extra = 0.0
        for side in ("home", "away"):
            for p, (m, v) in cv.Minutes(side).items():
                j = base.O(p, create=False)
                if j is None:
                    continue
                r = base.reg.mean[j] - base.reg.mean[j + 1]
                extra += (r * pace / 100 / 48) ** 2 * v
        res = self.Output(out["margin_mean"], out["margin_var"] + extra, synergy=out.get("synergy", 0.0), minutes_var_term=extra)
        res["home_fives"] = [list(f) for f in LikelyFives(cv.Minutes("home"))]
        res["away_fives"] = [list(f) for f in LikelyFives(cv.Minutes("away"))]
        return res

    def Update(self, rec):
        self.coach.Update(rec)
        self.syn.Update(rec)
