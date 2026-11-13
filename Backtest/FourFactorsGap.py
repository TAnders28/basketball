"""Leak-free Oliver four-factor team ratings (GAP-style offense and defense).

This is a box-score prototype, distinct from the play-by-play `four_factors`
member already inside stack_global. Ratings are updated only after every game
on a day has been predicted (prior-day ratings).

Factors:
  eFG%  = (FGM + 0.5 * FG3M) / FGA
  TOV%  = TOV / (FGA + 0.44 * FTA + TOV)
  ORB%  = OREB / (OREB + opponent DREB)
  FTR   = FTA / FGA
Offense and defense are tracked separately at home, on the road, and overall.
Home/away venue rates shrink toward the team's overall rate when thin.
"""
from __future__ import annotations

import math
from collections import defaultdict
from datetime import date

LEAGUE = {"efg": 0.540, "tov": 0.130, "orb": 0.270, "ftr": 0.260, "ft": 0.775, "pace": 100.0}
# Prior strength in the unit of the denominator (FGA, possessions, rebound chances, ...).
PRIOR_N = {"efg": 400.0, "tov": 400.0, "orb": 400.0, "ftr": 400.0, "ft": 200.0, "pace": 20.0}
FORGET = 0.997  # per day
HOME_EDGE = 2.5
MARGIN_SIGMA = 12.0
SHRINK_VENUE = 200.0  # venue rate shrinks to overall when venue den is below this


def Clip(p, lo=1e-6, hi=1.0 - 1e-6) -> float:
    return min(max(float(p), lo), hi)


def Phi(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def FactorsFromBoxes(team: dict, opp: dict) -> dict:
    """Four factors plus FT% and possession count from one team's box vs the opponent."""
    fga = float(team.get("FGA") or 0.0)
    fgm = float(team.get("FGM") or 0.0)
    fg3m = float(team.get("FG3M") or 0.0)
    fta = float(team.get("FTA") or 0.0)
    ftm = float(team.get("FTM") or 0.0)
    tov = float(team.get("TOV") or 0.0)
    oreb = float(team.get("OREB") or 0.0)
    dreb_opp = float(opp.get("DREB") or 0.0)
    tov_den = fga + 0.44 * fta + tov
    orb_den = oreb + dreb_opp
    poss = fga + 0.44 * fta - oreb + tov
    minutes = float(team.get("MIN") or 240.0)
    pace = poss * 240.0 / minutes if minutes > 0 else poss
    return {
        "efg": ((fgm + 0.5 * fg3m), fga),
        "tov": (tov, tov_den),
        "orb": (oreb, orb_den),
        "ftr": (fta, fga),
        "ft": (ftm, fta),
        "pace": (pace, 1.0),
        "poss": poss,
    }


def Ppp(efg, tov, orb, ftr, ft) -> float:
    """Expected points per possession from four factors and FT%."""
    efg, tov, orb, ftr, ft = (Clip(efg, 0.05, 0.85), Clip(tov, 0.02, 0.35),
                              Clip(orb, 0.05, 0.50), Clip(ftr, 0.02, 0.80), Clip(ft, 0.40, 0.95))
    miss = Clip(1.0 - efg, 0.15, 0.80)
    shot_end = Clip(1.0 - orb * miss, 0.40, 0.98)
    fga_per_poss = (1.0 - tov) / shot_end
    return fga_per_poss * (2.0 * efg + ftr * ft)


class _Rate:
    def __init__(self, mean: float, n: float):
        self.num = float(mean) * float(n)
        self.den = float(n)

    def Decay(self, f: float) -> None:
        self.num *= f
        self.den *= f

    def Add(self, num: float, den: float) -> None:
        if den <= 0:
            return
        self.num += float(num)
        self.den += float(den)

    def Mean(self) -> float:
        if self.den <= 1e-12:
            return 0.0
        return self.num / self.den


def _NewBook():
    return {k: _Rate(LEAGUE[k], PRIOR_N[k]) for k in LEAGUE}


class FourFactorsGap:
    """Exponentially faded team offense/defense four-factor ratings, venue-split."""

    def __init__(self, forget: float = FORGET, home_edge: float = HOME_EDGE, sigma: float = MARGIN_SIGMA):
        self.forget = forget
        self.home_edge = home_edge
        self.sigma = sigma
        self.off = {venue: defaultdict(_NewBook) for venue in ("all", "home", "away")}
        self.defn = {venue: defaultdict(_NewBook) for venue in ("all", "home", "away")}
        self.league = _NewBook()
        self.last_day = None

    def BeforeDay(self, day: str, gap: int) -> None:
        if gap <= 0:
            self.last_day = day
            return
        f = self.forget ** gap
        for book in (self.off, self.defn):
            for venue in book:
                for rates in book[venue].values():
                    for r in rates.values():
                        r.Decay(f)
        for r in self.league.values():
            r.Decay(f)
        self.last_day = day

    def _Rate(self, book, venue, team, key) -> float:
        overall = book["all"][team][key]
        ven = book[venue][team][key]
        # Shrink venue toward overall, overall already shrinks to the prior.
        w = ven.den / (ven.den + SHRINK_VENUE)
        return w * ven.Mean() + (1.0 - w) * overall.Mean()

    def MatchupRate(self, key: str, off_team: str, def_team: str, off_venue: str, def_venue: str) -> float:
        po = self._Rate(self.off, off_venue, off_team, key)
        pd = self._Rate(self.defn, def_venue, def_team, key)
        pl = self.league[key].Mean()
        # For rates, defense is "allowed" (same direction as offense). Average the two views.
        return 0.5 * (po + pd) if key in ("pace",) else (0.5 * (po + pd) if not pl else (po * pd / pl if pl > 0 else 0.5 * (po + pd)))

    def Predict(self, home: str, away: str) -> dict:
        """Home-win probability and supporting PPP from ratings as of this call (prior day if Update is deferred)."""
        vals_h, vals_a = {}, {}
        for k in ("efg", "tov", "orb", "ftr", "ft"):
            vals_h[k] = self.MatchupRate(k, home, away, "home", "away")
            vals_a[k] = self.MatchupRate(k, away, home, "away", "home")
        pace = 0.5 * (self._Rate(self.off, "home", home, "pace") + self._Rate(self.off, "away", away, "pace"))
        ppp_h = Ppp(vals_h["efg"], vals_h["tov"], vals_h["orb"], vals_h["ftr"], vals_h["ft"])
        ppp_a = Ppp(vals_a["efg"], vals_a["tov"], vals_a["orb"], vals_a["ftr"], vals_a["ft"])
        margin = (ppp_h - ppp_a) * pace + self.home_edge
        p_home = Clip(Phi(margin / self.sigma))
        return {"p_home": p_home, "margin": margin, "ppp_home": ppp_h, "ppp_away": ppp_a, "pace": pace,
                "home_factors": vals_h, "away_factors": vals_a}

    def Update(self, home: str, away: str, home_box: dict, away_box: dict) -> None:
        hf = FactorsFromBoxes(home_box, away_box)
        af = FactorsFromBoxes(away_box, home_box)
        for team, venue, fac, opp_fac in (
            (home, "home", hf, af),
            (away, "away", af, hf),
        ):
            for key in ("efg", "tov", "orb", "ftr", "ft", "pace"):
                num, den = fac[key]
                self.off["all"][team][key].Add(num, den)
                self.off[venue][team][key].Add(num, den)
                # Defense = factors allowed to the opponent.
                onum, oden = opp_fac[key]
                self.defn["all"][team][key].Add(onum, oden)
                self.defn[venue][team][key].Add(onum, oden)
                self.league[key].Add(num, den)
        self.last_day = self.last_day


def WalkRatings(games: list, forget: float = FORGET) -> dict:
    """games: list of dicts with game_id, game_date, home, away, home_box, away_box, sorted by date, id.

    Returns {game_id: predict-dict}. Same-day games all see prior-day ratings.
    """
    model = FourFactorsGap(forget=forget)
    out = {}
    last = None
    # group by day
    days = []
    bucket = defaultdict(list)
    for g in games:
        bucket[g["game_date"]].append(g)
    for day in sorted(bucket):
        gap = (date.fromisoformat(day) - date.fromisoformat(last)).days if last else 0
        model.BeforeDay(day, gap)
        for g in bucket[day]:
            out[g["game_id"]] = model.Predict(g["home"], g["away"])
        for g in bucket[day]:
            model.Update(g["home"], g["away"], g["home_box"], g["away_box"])
        last = day
    return out