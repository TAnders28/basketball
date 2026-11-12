r"""
CoachProfile: coaching preferences as hierarchical random effects, used to project tonight's minutes.

Per team-game we observe: rotation depth (players >= 10 min), minutes concentration (HHI of minute shares),
starters' share (top-5 minutes), pace (possessions), close-game tightening (share of clutch on-court slots used
by that game's top-5 players), timeouts and coach's challenges, and starters' minutes on back-to-backs relative
to their usual. Each metric for coach c: x ~ N(theta_c, s^2), theta_c ~ N(mu, tau^2). mu and tau^2 are
estimated across coaches by empirical Bayes (method of moments) every refit; s^2 from pooled within-coach
variance. Posterior theta_c shrinks small-sample coaches toward the league. Coaches come from the roster page
(head coach), so a profile follows a coach across teams; a mid-season change is a known weakness.

Minutes projection: start from the RosterBook projection, re-weight minutes by power gamma so the HHI matches
the coach's posterior HHI, and on a back-to-back scale the top five by the coach's posterior B2B ratio,
renormalized to 240. Outputs coach features and (in diagnostics) projected-vs-actual minutes MAE.
"""

from collections import defaultdict
from datetime import date

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Approaches.Common import TeamPossessions
from Matchups.V2.Base import Approach, Register
from Matchups.V2.Bayes import EmpiricalBayesNormal

METRICS = ("depth", "hhi", "starter_share", "pace", "clutch_top5", "timeouts", "challenges", "b2b_ratio")


class CoachEffects:
    """Running sums per coach per metric with forgetting; posterior via EB Normal-Normal."""

    def __init__(self, forget=0.998):
        self.s = defaultdict(lambda: defaultdict(lambda: np.zeros(3)))   # coach -> metric -> [n, sum, sumsq]
        self.forget = forget
        self.post = {}

    def Add(self, coach, metric, x):
        a = self.s[coach][metric]
        a += (1.0, x, x * x)

    def Decay(self, f):
        for c in self.s.values():
            for a in c.values():
                a *= f

    def Refit(self):
        self.post = {}
        for metric in METRICS:
            coaches = [c for c in self.s if self.s[c][metric][0] >= 1]
            if len(coaches) < 3:
                continue
            n = np.array([self.s[c][metric][0] for c in coaches])
            mean = np.array([self.s[c][metric][1] / self.s[c][metric][0] for c in coaches])
            within = sum(self.s[c][metric][2] - self.s[c][metric][1] ** 2 / self.s[c][metric][0] for c in coaches)
            s2 = max(within / max(n.sum() - len(coaches), 1.0), 1e-6)
            mu, tau2, post_mean, post_var = EmpiricalBayesNormal(mean, s2 / n)
            self.post[metric] = {"mu": mu, "tau2": tau2, "s2": s2,
                                 "coach": {c: (float(m), float(v)) for c, m, v in zip(coaches, post_mean, post_var)}}

    def Get(self, coach, metric, default):
        p = self.post.get(metric)
        if not p:
            return default, float("inf")
        return p["coach"].get(coach, (p["mu"], p["tau2"]))


def Hhi(minutes):
    tot = sum(minutes)
    return sum((m / tot) ** 2 for m in minutes) if tot else 0.0


def MatchHhi(minutes: dict, target: float) -> dict:
    """Re-weight m -> m^gamma (bisection on gamma) so the HHI of minute shares hits `target`; sum stays 240."""
    if not minutes or not np.isfinite(target):
        return minutes
    ps, ms = list(minutes), np.array([max(minutes[p], 0.1) for p in minutes])
    lo, hi = 0.3, 3.0
    for _ in range(30):
        g = 0.5 * (lo + hi)
        w = ms ** g
        w = 240 * w / w.sum()
        w = np.minimum(w, 44.0)
        if Hhi(w) < target:
            lo = g
        else:
            hi = g
    w = ms ** (0.5 * (lo + hi))
    w = np.minimum(240 * w / w.sum(), 44.0)
    w = 240 * w / w.sum()
    return dict(zip(ps, w))


@Register
class CoachProfile(Approach):
    name = "coach_profile"
    idea = "Hierarchical coach random effects (rotation depth, concentration, pace, clutch tightening, B2B rest, timeouts) -> minutes projection."
    data_used = ("player_games", "rotation", "pbp", "roster")
    covers = ("rotations_coaching", "fatigue_schedule")
    priors = {"coach effect": ("N(league mean, tau^2), tau^2 by empirical Bayes", "coaches differ but small samples mislead"),
              "within-coach noise": ("pooled within-coach variance", "game-to-game variation"),
              "unknown coach": ("league mean", "partial pooling default")}

    def __init__(self, forget=0.998, refit_every=7, **params):
        super().__init__(**params)
        self.fx = CoachEffects(forget)
        self.refit_every, self.days = refit_every, 999
        self.usual = defaultdict(float)
        self.last_date = {}
        self.pending = {}
        self.diagnostics = {"mae_base": [], "mae_coach": []}

    def Coach(self, team, season):
        c = D.HeadCoaches(season).get(team)
        return c[0] if c else f"team:{team}"

    def BeforeDay(self, day, gap):
        self.fx.Decay(self.fx.forget ** gap)
        self.days += gap
        if self.days >= self.refit_every:
            self.fx.Refit()
            self.days = 0

    def ProjectMinutes(self, view, side):
        base = view.Minutes(side)
        team = view.home if side == "home" else view.away
        coach = self.Coach(team, view.season)
        target, _ = self.fx.Get(coach, "hhi", float("nan"))
        mins = MatchHhi({p: m for p, (m, _v) in base.items()}, target)
        if view.Schedule(side)["b2b"]:
            ratio, _ = self.fx.Get(coach, "b2b_ratio", 1.0)
            top = sorted(mins, key=lambda p: -mins[p])[:5]
            for p in top:
                mins[p] *= min(max(ratio, 0.8), 1.1)
            tot = sum(mins.values())
            mins = {p: 240 * m / tot for p, m in mins.items()}
        return {p: (m, base[p][1]) for p, m in mins.items()}

    def Predict(self, view):
        out = {}
        for side in ("home", "away"):
            team = view.home if side == "home" else view.away
            coach = self.Coach(team, view.season)
            for metric in METRICS:
                out[f"{side}_{metric}"], _ = self.fx.Get(coach, metric, float("nan"))
            self.pending[(view.game_id, team)] = ({p: m for p, (m, _v) in view.Minutes(side).items()},
                                                  {p: m for p, (m, _v) in self.ProjectMinutes(view, side).items()})
        return out

    def Update(self, rec):
        g = rec.game
        for team in (rec.home, rec.away):
            rows = [r for r in rec.box if r["team"] == team]
            if not rows:
                continue
            coach = self.Coach(team, g["season"])
            mins = sorted((r["minutes"] for r in rows), reverse=True)
            top5 = {r["player_id"] for r in sorted(rows, key=lambda r: -r["minutes"])[:5]}
            self.fx.Add(coach, "depth", sum(m >= 10 for m in mins))
            self.fx.Add(coach, "hhi", Hhi(mins))
            self.fx.Add(coach, "starter_share", sum(mins[:5]) / max(sum(mins), 1))
            self.fx.Add(coach, "pace", TeamPossessions(rec, team))
            clutch = [p for p in rec.possessions if p["clutch"] and team in (p["offense"], p["defense"])]
            if clutch:
                slots = [pl for p in clutch for pl in (p["off5"] if p["offense"] == team else p["def5"])]
                if slots:
                    self.fx.Add(coach, "clutch_top5", sum(pl in top5 for pl in slots) / len(slots))
            ev = rec.events.get(team, {}) if rec.events else {}
            if rec.pbp:
                self.fx.Add(coach, "timeouts", ev.get("timeouts", 0))
                self.fx.Add(coach, "challenges", ev.get("challenges", 0))
            key = (g["game_id"], team)
            if key in self.pending:
                base, coached = self.pending.pop(key)
                actual = {r["player_id"]: r["minutes"] for r in rows}
                for proj, name in ((base, "mae_base"), (coached, "mae_coach")):
                    ps = set(proj) | set(actual)
                    self.diagnostics[name].append(np.mean([abs(proj.get(p, 0) - actual.get(p, 0)) for p in ps]))
            last = self.last_date.get(team)
            is_b2b = last is not None and (date.fromisoformat(g["game_date"]) - date.fromisoformat(last)).days == 1
            starters_usual = [self.usual[p] for p in top5 if self.usual[p] > 0]
            if is_b2b and len(starters_usual) >= 3:
                cur = sum(r["minutes"] for r in rows if r["player_id"] in top5 and self.usual[r["player_id"]] > 0)
                self.fx.Add(coach, "b2b_ratio", cur / sum(starters_usual))
            self.last_date[team] = g["game_date"]
            for r in rows:
                p = r["player_id"]
                self.usual[p] = r["minutes"] if self.usual[p] == 0 else 0.8 * self.usual[p] + 0.2 * r["minutes"]
