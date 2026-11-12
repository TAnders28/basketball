"""
Leak-free walk-forward over regular seasons 2022-23 .. 2026-27 for one model configuration.

    for each game day D (chronological):
        1. decay state to D, refit (state holds ONLY games from days before D)
        2. predict every game on D from pre-tip rosters ("pregame") and from the players who actually
           appeared ("oracle": an injury-report proxy; it uses who played, never how they played)
        3. THEN feed D's results in (box scores, matchup rows, minutes, pace)

The matchup model's only pre-2023 input is the 2022-23 season aggregate, added before the 2023-24 opener.
"""

import math
import time
from itertools import groupby

import numpy as np

from Matchups import Data
from Matchups.Models import MatchupModel, MinutesBook, RapmModel

SEASONS = [2022, 2023, 2024, 2025, 2026]
RECORD_FROM = 2023
MATCHUP_FIRST_SEASON = 2023


def DaysBetween(a: str, b: str) -> int:
    from datetime import date
    return (date.fromisoformat(b) - date.fromisoformat(a)).days


def LoadAll(seasons=SEASONS):
    games, players, poss = [], {}, {}
    for s in seasons:
        for g in Data.RegularGames(s):
            g = dict(g)
            g["season"] = s
            games.append(g)
        players.update(Data.PlayerGames(s))
        poss.update(Data.TeamPossessions(s))
    games.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return games, players, poss


def Rosters(book: MinutesBook, game: dict, player_rows: list) -> dict:
    out = {}
    for side in ("home", "away"):
        team = game[f"{side}_team"]
        out[("pregame", side)] = book.PregameRoster(team, game["game_date"])
        out[("oracle", side)] = book.OracleRoster([r["player_id"] for r in player_rows if r["team"] == team])
    return out


def AssignmentCheck(model: MatchupModel, rows: list, rosters: dict, game: dict) -> dict:
    """Poss-weighted half-L1 distance between predicted and actual defender shares (oracle rosters)."""
    out = {}
    actual = {}
    for r in rows:
        if r["poss"] > 0:
            defs = actual.setdefault((r["team"], r["off"]), {})
            defs[r["def"]] = defs.get(r["def"], 0.0) + r["poss"]
    for mode in ("full", "prior", "hist"):
        num = den = 0.0
        for (team, off), defs in actual.items():
            side_def = "away" if team == game["home_team"] else "home"
            defense = rosters[("oracle", side_def)]
            tot = sum(defs.values())
            if tot < 3 or not defense:
                continue
            pred = model.Shares(off, defense, mode)
            keys = set(pred) | set(defs)
            dist = 0.5 * sum(abs(pred.get(k, 0.0) - defs.get(k, 0.0) / tot) for k in keys)
            num += tot * dist
            den += tot
        out[f"assign_l1_{mode}"] = num / den if den else None
    out["assign_poss"] = sum(sum(v.values()) for v in actual.values())
    return out


def PairDeviance(model: MatchupModel, rows: list) -> dict:
    """Poisson log-likelihood of each pairing's points: full model vs offender-only vs league rate."""
    ll = {"pair": 0.0, "offender": 0.0, "league": 0.0}
    league = math.exp(model.alpha)
    for r in rows:
        c, y = r["poss"], r["pts"]
        if c <= 0:
            continue
        for key, lam in (("pair", model.Lambda(r["off"], r["def"])), ("offender", model.OffenseRate(r["off"])),
                         ("league", league)):
            mu = max(c * lam, 1e-9)
            ll[key] += y * math.log(mu) - mu - math.lgamma(y + 1)
    return {f"pair_ll_{k}": v for k, v in ll.items()}


def PairShape(model: MatchupModel, off: int, dfn: int) -> float:
    """Approximate gamma shape (evidence) behind a pairing rate, used by the simulator."""
    j = model.pair_of.get((off, dfn))
    pair_y = model.y[j] if j is not None else 0.0
    io = model.index.ids.get(off)
    off_y = float(model.off_y[io]) if io is not None and io < len(model.off_y) else 0.0
    return model.k_pair + pair_y + min(off_y, 300.0) / 6.0


def Walk(cfg: dict, record_sim: bool = False, progress=None, data=None) -> dict:
    games, players, poss = data if data is not None else LoadAll()
    book = MinutesBook()
    kind = cfg["model"]
    if kind == "rapm":
        model = RapmModel(cfg["lam"], cfg["half_life"])
    else:
        model = MatchupModel(cfg["k"], cfg["half_life"], k_assign=cfg.get("k_assign", 10.0), target=cfg.get("target", "player"))
    rows_out, sim_inputs = [], {}
    last_date, seeded = None, False
    t0 = time.time()
    for day, day_games in groupby(games, key=lambda g: g["game_date"]):
        day_games = list(day_games)
        season = day_games[0]["season"]
        if last_date is not None:
            model.Decay(DaysBetween(last_date, day))
        if kind == "matchup" and not seeded and season >= MATCHUP_FIRST_SEASON:
            model.AddRows(Data.SeasonMatchupAggregate(MATCHUP_FIRST_SEASON - 1), weight=0.5)
            seeded = True
        last_date = day
        box_today = {g["game_id"]: (Data.BoxMatchups(season, g["game_id"]) if season >= MATCHUP_FIRST_SEASON else None)
                     for g in day_games}
        if season >= RECORD_FROM:
            model.Fit()
            for g in day_games:
                prow = players.get(g["game_id"], [])
                rosters = Rosters(book, g, prow)
                row = {"game_id": g["game_id"], "game_date": g["game_date"], "season": season,
                       "home_team": g["home_team"], "away_team": g["away_team"],
                       "home_win": int(g["home_score"] > g["away_score"]),
                       "margin": g["home_score"] - g["away_score"], "total": g["home_score"] + g["away_score"]}
                for variant in ("pregame", "oracle"):
                    home, away = rosters[(variant, "home")], rosters[(variant, "away")]
                    if kind == "rapm":
                        row[f"pred_margin_{variant}"] = model.PredictMargin(home, away) if home and away else None
                    else:
                        if not (home and away) or not seeded:
                            row[f"pred_margin_{variant}"] = None
                            continue
                        pace = 0.5 * (book.Pace(g["home_team"]) + book.Pace(g["away_team"]))
                        hp, hpairs = model.ExpectedPoints(home, away, pace, detail=True)
                        ap, apairs = model.ExpectedPoints(away, home, pace, detail=True)
                        row[f"pred_home_pts_{variant}"], row[f"pred_away_pts_{variant}"] = hp, ap
                        row[f"pred_margin_{variant}"], row[f"pred_total_{variant}"] = hp - ap, hp + ap
                        if record_sim and season >= 2024:
                            ratio = model.AttributionRatio()
                            sim_inputs[(g["game_id"], variant)] = {
                                "pace": pace, "ratio": ratio,
                                "home": np.array([[p, lam, PairShape(model, o, d)] for o, d, p, lam in hpairs], dtype=np.float64),
                                "away": np.array([[p, lam, PairShape(model, o, d)] for o, d, p, lam in apairs], dtype=np.float64)}
                box = box_today[g["game_id"]]
                if kind == "matchup" and box and seeded and season >= 2024:
                    row.update(AssignmentCheck(model, box, rosters, g))
                    if model.target == "player":
                        row.update(PairDeviance(model, box))
                rows_out.append(row)
        # ---- results of day D enter the state only now ----
        for g in day_games:
            prow = players.get(g["game_id"], [])
            if kind == "rapm":
                if prow:
                    model.Apply(g, prow)
            else:
                box = box_today[g["game_id"]]
                if box:
                    model.AddRows(box)
                    for side in ("home", "away"):
                        team = g[f"{side}_team"]
                        attributed = sum(r["pts"] for r in box if r["team"] == team)
                        if attributed > 0:
                            model.AddTeamPoints(g[f"{side}_score"], attributed)
            book.Apply(g, prow, poss)
        if progress and day.endswith("-01"):
            progress(f"{cfg['name']} at {day} ({time.time() - t0:.0f}s)")
    return {"cfg": cfg, "rows": rows_out, "sim_inputs": sim_inputs, "model": model, "book": book,
            "secs": time.time() - t0}
