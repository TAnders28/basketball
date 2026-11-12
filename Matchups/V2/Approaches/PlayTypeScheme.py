r"""
PlayTypeScheme: play-type mix (Synergy) versus defensive scheme, as Bayesian interaction terms on Kalman.

Prior-season Synergy (team level, offense and defense) gives each team's play-type frequencies and PPP; PPP is
shrunk to the league play-type mean with a Gamma-style prior of 50 possessions. In-season scheme proxies are
conjugate posteriors updated walk-forward:
  switch rate      Beta(0.08, 200): defender switches per matchup possession (BoxScoreMatchupsV3 switchesOn)
  drop proxy       Beta(0.20, 300): share of opponent FGA from mid-range (drop coverage concedes pull-up 2s)
  rim share        Beta(0.30, 300): share of opponent FGA in the restricted area
  turnover forcing Beta(0.14, 300): opponent turnovers per possession
Features (home offense vs away defense minus the reverse): play-type PPP edge, PnR freq x switch rate,
iso freq x switch rate, PnR ball-handler freq x drop proxy, post-up freq x opponent rim share, transition
freq x turnover forcing. Coefficients ~ N(0, 3^2). Synergy is season-aggregate, so only season-1 is used.
"""

from collections import defaultdict

import numpy as np

from Matchups.V2.Approaches.Common import ResidualOnKalman
from Matchups.V2.Bayes import BetaBinomial
from Matchups.V2.Base import Register

PLAY_TYPES = ["Isolation", "Transition", "PRBallHandler", "PRRollMan", "Postup", "Spotup", "Handoff", "Cut", "OffScreen", "OffRebound", "Misc"]
SHRINK_POSS = 50.0


def SynergyProfile(view, season):
    """{team: {"off": {pt: (freq, ppp)}, "def": {pt: (freq, ppp)}}, league ppp per type} from season `season`."""
    prof = defaultdict(lambda: {"off": {}, "def": {}})
    league = {}
    for pt in PLAY_TYPES:
        for side, grouping in (("off", "offensive"), ("def", "defensive")):
            rows = view.Prior(f"synergy/T_{grouping}_{pt}.json", season)
            if not rows:
                continue
            poss = sum(r.get("POSS") or 0 for r in rows)
            pts = sum(r.get("PTS") or 0 for r in rows)
            lg = pts / poss if poss else 1.0
            league[(side, pt)] = lg
            for r in rows:
                n = r.get("POSS") or 0
                ppp = ((r.get("PTS") or 0) + SHRINK_POSS * lg) / (n + SHRINK_POSS)
                prof[r["TEAM_ABBREVIATION"]][side][pt] = (r.get("POSS_PCT") or 0.0, ppp)
    return prof, league


@Register
class PlayTypeScheme(ResidualOnKalman):
    name = "playtype_scheme"
    idea = "Prior-season Synergy play-type mix/PPP x in-season scheme posteriors (switch, drop, rim, TO forcing) as Bayesian terms."
    data_used = ("synergy", "box_matchups", "shots", "pbp")
    covers = ("defense_schemes", "shot_quality", "transition", "turnovers", "matchups")
    feature_names = ("pt_edge", "pnr_x_switch", "iso_x_switch", "pnr_x_drop", "post_x_rim", "trans_x_toforce")
    # pt_edge is in points (prior-season Synergy PPP edge x pace): expect partial carry-over, N(0.3, 0.2^2).
    # Interactions are scaled x100 (typical spread ~5 units) with N(0, 0.15^2) coefficients: a one-sd scheme
    # clash moves the margin ~0.75 pts a priori.
    prior_mean = (0.3, 0.0, 0.0, 0.0, 0.0, 0.0)
    prior_var = (0.04, 0.0225, 0.0225, 0.0225, 0.0225, 0.0225)
    priors = {"pt_edge coefficient": ("N(0.3, 0.2^2)", "prior-season play-type edge carries over partially"),
              "interaction coefficients": ("N(0, 0.15^2) per x100 unit", "one-sd scheme clash ~0.75 pts a priori"),
              "Synergy PPP": ("shrunk to league play-type PPP with 50 poss", "team play-type samples are small"),
              "switch rate": ("Beta(0.08, 200)", ""), "drop proxy": ("Beta(0.20, 300)", "mid-range share allowed"),
              "rim share": ("Beta(0.30, 300)", ""), "TO forcing": ("Beta(0.14, 300)", ""),
              "coefficients": ("N(0, 3^2) pts per unit", "interactions are second-order")}

    def __init__(self, **params):
        super().__init__(**params)
        self.season, self.prof, self.league = None, {}, {}
        self.switch = defaultdict(lambda: BetaBinomial(0.08, 200.0))
        self.drop = defaultdict(lambda: BetaBinomial(0.20, 300.0))
        self.rim = defaultdict(lambda: BetaBinomial(0.30, 300.0))
        self.toforce = defaultdict(lambda: BetaBinomial(0.14, 300.0))

    def Freq(self, team, pt, side="off"):
        return self.prof.get(team, {}).get(side, {}).get(pt, (0.0, self.league.get((side, pt), 1.0)))

    def Edge(self, off, dfn):
        e = 0.0
        for pt in PLAY_TYPES:
            f, ppp = self.Freq(off, pt, "off")
            _fd, ppp_allowed = self.Freq(dfn, pt, "def")
            lo, ld = self.league.get(("off", pt), ppp), self.league.get(("def", pt), ppp_allowed)
            e += f * ((ppp - lo) + (ppp_allowed - ld))
        return e

    def Side(self, off, dfn):
        pnr = self.Freq(off, "PRBallHandler")[0] + self.Freq(off, "PRRollMan")[0]
        return np.array([100 * self.Edge(off, dfn), pnr * self.switch[dfn].Mean() * 10,
                         self.Freq(off, "Isolation")[0] * self.switch[dfn].Mean() * 10,
                         self.Freq(off, "PRBallHandler")[0] * self.drop[dfn].Mean() * 10,
                         self.Freq(off, "Postup")[0] * self.rim[dfn].Mean() * 10,
                         self.Freq(off, "Transition")[0] * self.toforce[dfn].Mean() * 10])

    def Features(self, view):
        if self.season != view.season:
            self.prof, self.league = SynergyProfile(view, view.season - 1)
            self.season = view.season
        x = self.Side(view.home, view.away) - self.Side(view.away, view.home)
        x[1:] *= 100.0
        return x

    def Learn(self, rec):
        for r in rec.matchups or []:
            dfn = rec.away if r["team"] == rec.home else rec.home
            self.switch[dfn].Add(min(r["switches"], r["poss"]), r["poss"])
        by_team = defaultdict(list)
        for s in rec.shots:
            by_team[s["team"]].append(s)
        for team, shots in by_team.items():
            dfn = rec.away if team == rec.home else rec.home
            n = len(shots)
            self.drop[dfn].Add(sum(1 for s in shots if s["zone"] == "Mid-Range"), n)
            self.rim[dfn].Add(sum(1 for s in shots if s["zone"] == "Restricted Area"), n)
        for team in (rec.home, rec.away):
            dfn = rec.away if team == rec.home else rec.home
            ps = [p for p in rec.possessions if p["offense"] == team]
            if ps:
                self.toforce[dfn].Add(sum(p["tov"] > 0 for p in ps), len(ps))
