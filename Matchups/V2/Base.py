r"""
Shared interface for every v2 approach, plus the one leak-free walk that backtests any of them.

    class MyApproach(Approach):
        name = "my_approach"
        def BeforeDay(self, day, days_since_last): ...   # decay / refit from state built on EARLIER days
        def Predict(self, view: PregameView) -> dict: ... # must not see results; returns margin_mean, margin_var, ...
        def Update(self, rec: GameRecord): ...            # called only after every game of the day was predicted

Walk(approach, games, records) predicts each game day before feeding that day's results, so no approach can
see a result before it predicts. PregameView exposes only pregame facts: teams, date, season, projected minutes
(from games before today), schedule history, and prior-season tables (Prior() refuses the current season).
Every approach's Predict returns a posterior predictive margin (mean, variance) and p_home from it.
"""

import math
from collections import defaultdict
from datetime import date
from itertools import groupby

import numpy as np

from Matchups.V2 import DataLayer as D
from Matchups.V2.Bayes import PredictiveWinProb
from Matchups.V2 import Pbp

REGISTRY = {}
MARGIN_NOISE_VAR = 13.0 ** 2     # single-game margin noise around the true expected margin (NBA ~ 12-14 pts)


def Register(cls):
    REGISTRY[cls.name] = cls
    return cls


class Approach:
    name = "base"
    idea = ""
    data_used = ()
    covers = ()          # parts of the game: shot_quality, possession, rebounding, turnovers, transition,
                         # defense_schemes, rotations_coaching, teammate_synergy, fatigue_schedule, matchups, clutch, team_strength
    priors = {}          # documented priors: name -> (value, rationale)
    min_season = 2023

    def __init__(self, **params):
        self.params = params

    def BeforeDay(self, day: str, days_since_last: float):
        pass

    def Predict(self, view: "PregameView") -> dict:
        raise NotImplementedError

    def Update(self, rec: "GameRecord"):
        pass

    @staticmethod
    def Output(margin_mean: float, margin_var: float, **extra) -> dict:
        return {"margin_mean": float(margin_mean), "margin_var": float(margin_var),
                "p_home": PredictiveWinProb(margin_mean, margin_var), **extra}


class RosterBook:
    """
    Pregame minutes projection with uncertainty (Normal-Normal): each player's minutes-when-active has a prior
    from his role (starter ~30, bench ~16, sd 8) updated with exponentially weighted observations.
    Availability = appeared in the team's most recent game (no injury feed yet; hook: `availability`).
    """
    ALPHA = 0.25

    def __init__(self):
        self.mean, self.var, self.n = {}, {}, defaultdict(float)
        self.last_team, self.team_last = {}, {}
        self.starter_rate = defaultdict(float)
        self.names = {}

    def Apply(self, box_rows: list):
        by_team = defaultdict(list)
        for r in box_rows:
            p = r["player_id"]
            if p not in self.mean:
                self.mean[p], self.var[p] = r["minutes"], 64.0
            else:
                d = r["minutes"] - self.mean[p]
                self.mean[p] += self.ALPHA * d
                self.var[p] = (1 - self.ALPHA) * (self.var[p] + self.ALPHA * d * d)
            self.n[p] += 1
            self.last_team[p] = r["team"]
            self.names[p] = r.get("name")
            by_team[r["team"]].append(p)
        for team, ps in by_team.items():
            self.team_last[team] = ps

    def Project(self, team: str, availability: dict = None) -> dict:
        """{player: (mean minutes, variance)} scaled so the team's expected minutes sum to 240."""
        roster = [p for p in self.team_last.get(team, []) if self.last_team.get(p) == team]
        if availability is not None:
            roster = [p for p in roster if availability.get(p, 1.0) > 0.5] + [p for p, a in availability.items() if a > 0.5 and p not in roster]
        mins = {p: self.mean.get(p, 15.0) for p in roster}
        total = sum(mins.values())
        if total <= 0:
            return {}
        k = 240.0 / total
        return {p: (m * k, self.var.get(p, 64.0) * k * k) for p, m in mins.items()}


class ScheduleBook:
    """Team schedule history (dates and venues of games already played) for rest/travel features."""

    def __init__(self):
        self.history = defaultdict(list)    # team -> [(date, venue team)]

    def Apply(self, game: dict):
        for side in ("home", "away"):
            self.history[game[f"{side}_team"]].append((game["game_date"], game["home_team"]))

    def Context(self, team: str, day: str, venue: str) -> dict:
        hist = self.history.get(team, [])
        d = date.fromisoformat(day)
        last = hist[-1] if hist else None
        rest = (d - date.fromisoformat(last[0])).days - 1 if last else 3
        games_last_4 = sum(1 for g, _ in hist if 0 < (d - date.fromisoformat(g)).days <= 3) + 1
        games_last_6 = sum(1 for g, _ in hist if 0 < (d - date.fromisoformat(g)).days <= 5) + 1
        prev_venue = last[1] if last else venue
        miles = Haversine(D.ARENAS[prev_venue][:2], D.ARENAS[venue][:2]) if prev_venue in D.ARENAS and venue in D.ARENAS else 0.0
        tz_shift = (D.ARENAS[venue][2] - D.ARENAS[prev_venue][2]) if prev_venue in D.ARENAS and venue in D.ARENAS else 0
        road_streak = 0
        for _g, v in reversed(hist):
            if v == team:
                break
            road_streak += 1
        return {"rest": min(rest, 7), "b2b": int(rest == 0), "three_in_four": int(games_last_4 >= 3),
                "four_in_six": int(games_last_6 >= 4), "travel_miles": miles, "tz_shift": tz_shift,
                "altitude_ft": D.ARENAS.get(venue, (0, 0, 0, 0))[3], "road_streak": road_streak if venue != team else 0}


def Haversine(a, b) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 3958.8 * math.asin(math.sqrt(h))


class PregameView:
    """Everything an approach may use to predict: no scores, no box scores, nothing from today."""

    def __init__(self, game: dict, rosters: RosterBook, schedule: ScheduleBook, availability: dict = None):
        self.game_id, self.date, self.season = game["game_id"], game["game_date"], game["season"]
        self.home, self.away = game["home_team"], game["away_team"]
        self._rosters, self._schedule, self._availability = rosters, schedule, availability or {}

    def Minutes(self, side: str) -> dict:
        team = self.home if side == "home" else self.away
        return self._rosters.Project(team, self._availability.get(team))

    def Schedule(self, side: str) -> dict:
        team = self.home if side == "home" else self.away
        return self._schedule.Context(team, self.date, self.home)

    def Prior(self, relpath: str, season: int = None) -> list:
        season = self.season - 1 if season is None else season
        if season >= self.season:
            raise PermissionError("season-aggregate tables are prior-season only (leak guard)")
        return D.SeasonTable(season, relpath)

    def Name(self, player: int) -> str:
        return self._rosters.names.get(player) or str(player)


class GameRecord:
    """A decided game with its data. Derived tables (possessions, stints) are computed lazily."""

    def __init__(self, game: dict, box=None, matchups=None, rotation=None, pbp=None, shots=None,
                 possessions=None, stints=None, events=None):
        self.game = game
        self.box, self.matchups, self.rotation, self.pbp, self.shots = box or [], matchups, rotation, pbp, shots or []
        self._poss, self._stints, self._events = possessions, stints, events

    @property
    def home(self):
        return self.game["home_team"]

    @property
    def away(self):
        return self.game["away_team"]

    @property
    def possessions(self):
        if self._poss is None:
            self._poss = Pbp.BuildPossessions(self.pbp, self.rotation, self.home, self.away) if self.pbp and self.rotation else []
        return self._poss

    @property
    def stints(self):
        if self._stints is None:
            self._stints = Pbp.BuildStints(self.pbp, self.rotation, self.home, self.away, self.possessions) if self.pbp and self.rotation else []
        return self._stints

    @property
    def events(self):
        if self._events is None:
            self._events = Pbp.GameEvents(self.pbp, self.home, self.away) if self.pbp else {}
        return self._events


def RealRecords(game: dict) -> GameRecord:
    s, gid = game["season"], game["game_id"]
    return GameRecord(game, box=D.PlayerBox(s).get(gid, []), matchups=D.BoxMatchups(s, gid),
                      rotation=D.Rotation(s, gid), pbp=D.Pbp(s, gid), shots=D.ShotsByGame(s).get(gid, []))


def LoadGames(seasons) -> list:
    games = []
    for s in seasons:
        for g in D.LoadJson(f"data/season/games_{s}.json", []):
            if g["game_id"].startswith("002") and g.get("home_score") is not None:
                g = dict(g)
                g["season"] = s
                games.append(g)
    return sorted(games, key=lambda g: (g["game_date"], g["game_id"]))


def Walk(approach: Approach, games: list, records=RealRecords, record_from: str = None, max_days: int = None,
         availability=None, progress=None) -> list:
    """Leak-free day loop. Returns one row per predicted game: ids, outcome, and the approach's outputs."""
    rosters, schedule = RosterBook(), ScheduleBook()
    rows, last_day, n_days = [], None, 0
    for day, day_games in groupby(games, key=lambda g: g["game_date"]):
        day_games = list(day_games)
        gap = (date.fromisoformat(day) - date.fromisoformat(last_day)).days if last_day else 0
        approach.BeforeDay(day, gap)
        if record_from is None or day >= record_from:
            for g in day_games:
                view = PregameView(g, rosters, schedule, availability(g) if availability else None)
                out = approach.Predict(view)
                row = {"game_id": g["game_id"], "game_date": day, "season": g["season"], "home": g["home_team"],
                       "away": g["away_team"], "margin": g["home_score"] - g["away_score"],
                       "home_win": int(g["home_score"] > g["away_score"])}
                row.update(out or {})
                rows.append(row)
        recs = [records(g) for g in day_games]
        for rec in recs:
            approach.Update(rec)
        for rec in recs:
            rosters.Apply(rec.box)
            schedule.Apply(rec.game)
        last_day = day
        n_days += 1
        if progress and n_days % 30 == 0:
            progress(f"{approach.name}: {day}")
        if max_days and n_days >= max_days:
            break
    return rows


def ProjectedLineupWeights(minutes: dict) -> dict:
    """Each player's expected share of the 5 floor spots (minutes / 48), clipped to [0, 1]."""
    return {p: min(m / 48.0, 1.0) for p, (m, _v) in minutes.items()}
