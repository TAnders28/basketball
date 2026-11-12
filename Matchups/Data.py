"""Loaders for the cached season, player-game and matchup data (all local, under data/)."""

import json
import os
from collections import defaultdict

SEASON_DIR = os.path.join("data", "season")
MATCHUP_DIR = os.path.join("data", "matchups")


def LoadJson(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def AllGames(season: int) -> list:
    """Every completed game of a season (all types), as in data/season/games_<season>.json."""
    return [g for g in LoadJson(os.path.join(SEASON_DIR, f"games_{season}.json"), [])
            if g.get("home_score") is not None]


def RegularGames(season: int) -> list:
    return sorted((g for g in AllGames(season) if g["game_id"].startswith("002")),
                  key=lambda g: (g["game_date"], g["game_id"]))


def TeamPossessions(season: int) -> dict:
    """{(game_id, team): estimated possessions} from team box scores (FGA + 0.44 FTA - OREB + TOV)."""
    out = {}
    path = os.path.join(SEASON_DIR, f"team_games_{season}.jsonl")
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            try:
                out[(r["GAME_ID"], r["TEAM_ABBREVIATION"])] = r["FGA"] + 0.44 * r["FTA"] - r["OREB"] + r["TOV"]
            except (KeyError, TypeError):
                continue
    return out


def PlayerGames(season: int) -> dict:
    """{game_id: [{player_id, team, minutes, pts, plus_minus, name}]} (regular season)."""
    by_game = defaultdict(list)
    for r in LoadJson(os.path.join(MATCHUP_DIR, f"player_games_{season}.json"), []):
        minutes = r.get("MIN") or 0
        if minutes <= 0:
            continue
        by_game[r["GAME_ID"]].append({"player_id": int(r["PLAYER_ID"]), "team": r["TEAM_ABBREVIATION"],
                                      "minutes": float(minutes), "pts": r.get("PTS") or 0,
                                      "plus_minus": r.get("PLUS_MINUS") or 0, "name": r.get("PLAYER_NAME")})
    return dict(by_game)


def BoxMatchups(season: int, game_id: str):
    """Rows for one game: offense team, offender, defender, partial possessions, offender's points."""
    rows = LoadJson(os.path.join(MATCHUP_DIR, "box", str(season), f"{game_id}.json"))
    if rows is None:
        return None
    return [{"team": r["teamTricode"], "off": int(r["personIdOff"]), "def": int(r["personIdDef"]),
             "poss": float(r.get("partialPossessions") or 0.0), "pts": float(r.get("playerPoints") or 0.0),
             "team_pts": float(r.get("teamPoints") or 0.0), "pos": r.get("positionOff") or "",
             "off_name": r.get("nameIOff"), "def_name": r.get("nameIDef")}
            for r in rows]


def SeasonMatchupAggregate(season: int) -> list:
    """Season-total pairs (used only as a prior for LATER seasons)."""
    return [{"off": int(r["OFF_PLAYER_ID"]), "def": int(r["DEF_PLAYER_ID"]),
             "poss": float(r.get("PARTIAL_POSS") or 0.0), "pts": float(r.get("PLAYER_PTS") or 0.0),
             "team_pts": float(r.get("TEAM_PTS") or 0.0)}
            for r in LoadJson(os.path.join(MATCHUP_DIR, f"season_matchups_{season}.json"), [])]
