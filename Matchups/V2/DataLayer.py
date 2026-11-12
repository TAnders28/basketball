r"""
Data layer for the v2 approaches: one loader per cached table, and a data dictionary describing each.

Leak status of every table (DATA_DICTIONARY[...]["leak"]):
  "dated"         rows belong to one game; usable walk-forward (only after that game is decided)
  "prior-season"  season aggregates with no dates; usable ONLY for later seasons (PregameView.Prior enforces it)
  "static"        facts known before the season (arena coordinates, coaches on the roster page)
"""

import json
import os
from collections import defaultdict
from functools import lru_cache

ROOT1 = os.path.join("data", "matchups")
ROOT2 = os.path.join("data", "matchups2")
SEASON_DIR = os.path.join("data", "season")

TEAM_IDS = {"ATL": 1610612737, "BOS": 1610612738, "BKN": 1610612751, "CHA": 1610612766, "CHI": 1610612741,
            "CLE": 1610612739, "DAL": 1610612742, "DEN": 1610612743, "DET": 1610612765, "GSW": 1610612744,
            "HOU": 1610612745, "IND": 1610612754, "LAC": 1610612746, "LAL": 1610612747, "MEM": 1610612763,
            "MIA": 1610612748, "MIL": 1610612749, "MIN": 1610612750, "NOP": 1610612740, "NYK": 1610612752,
            "OKC": 1610612760, "ORL": 1610612753, "PHI": 1610612755, "PHX": 1610612756, "POR": 1610612757,
            "SAC": 1610612758, "SAS": 1610612759, "TOR": 1610612761, "UTA": 1610612762, "WAS": 1610612764}
TEAM_BY_ID = {v: k for k, v in TEAM_IDS.items()}

# (lat, lon, UTC offset in standard time, altitude ft). Approximate arena locations.
ARENAS = {"ATL": (33.757, -84.396, -5, 1050), "BOS": (42.366, -71.062, -5, 20), "BKN": (40.683, -73.976, -5, 30),
          "CHA": (35.225, -80.839, -5, 750), "CHI": (41.881, -87.674, -6, 600), "CLE": (41.497, -81.688, -5, 650),
          "DAL": (32.790, -96.810, -6, 430), "DEN": (39.749, -105.008, -7, 5280), "DET": (42.341, -83.055, -5, 600),
          "GSW": (37.768, -122.388, -8, 10), "HOU": (29.751, -95.362, -6, 50), "IND": (39.764, -86.156, -5, 715),
          "LAC": (33.945, -118.343, -8, 100), "LAL": (34.043, -118.267, -8, 300), "MEM": (35.138, -90.051, -6, 300),
          "MIA": (25.781, -80.188, -5, 10), "MIL": (43.045, -87.917, -6, 620), "MIN": (44.980, -93.276, -6, 830),
          "NOP": (29.949, -90.082, -6, 5), "NYK": (40.751, -73.993, -5, 30), "OKC": (35.463, -97.515, -6, 1200),
          "ORL": (28.539, -81.384, -5, 100), "PHI": (39.901, -75.172, -5, 40), "PHX": (33.446, -112.071, -7, 1100),
          "POR": (45.532, -122.667, -8, 50), "SAC": (38.580, -121.500, -8, 30), "SAS": (29.427, -98.438, -6, 650),
          "TOR": (43.643, -79.379, -5, 250), "UTA": (40.768, -111.901, -7, 4300), "WAS": (38.898, -77.021, -5, 30)}

DATA_DICTIONARY = {
    "games": {"path": "data/season/games_<season>.json", "source": "LeagueGameFinder via GetSeasonGames.py",
              "grain": "game", "leak": "dated", "key_cols": "game_id, game_date, home_team, away_team, home_score, away_score, home_stats/away_stats"},
    "team_games": {"path": "data/season/team_games_<season>.jsonl", "source": "LeagueGameFinder", "grain": "team-game",
                   "leak": "dated", "key_cols": "FGA, FTA, OREB, TOV (possessions estimate), PTS"},
    "player_games": {"path": "data/matchups/player_games_<season>.json", "source": "LeagueGameLog (P)", "grain": "player-game",
                     "leak": "dated", "key_cols": "MIN, PTS, FGA, FG3A, FG3M, FTA, OREB, DREB, AST, STL, BLK, TOV, PF, PLUS_MINUS"},
    "box_matchups": {"path": "data/matchups/box/<season>/<game_id>.json", "source": "BoxScoreMatchupsV3", "grain": "game x offender x defender",
                     "leak": "dated", "key_cols": "partialPossessions, playerPoints, teamPoints, switchesOn, matchup FGA/3PA, percentage of time"},
    "season_matchups": {"path": "data/matchups/season_matchups_<season>.json", "source": "LeagueSeasonMatchups",
                        "grain": "season x offender x defender", "leak": "prior-season", "key_cols": "PARTIAL_POSS, PLAYER_PTS, TEAM_PTS, MATCHUP_FGA"},
    "rotation": {"path": "data/matchups2/rotation/<season>/<game_id>.json", "source": "GameRotation", "grain": "player stint",
                 "leak": "dated", "key_cols": "PERSON_ID, IN_TIME_REAL, OUT_TIME_REAL (tenths of a second), PT_DIFF, USG_PCT"},
    "pbp": {"path": "data/matchups2/pbp/<season>/<game_id>.json", "source": "PlayByPlayV3", "grain": "event",
            "leak": "dated", "key_cols": "period, clock, teamTricode, personId, actionType, subType, shotResult, shotValue, scoreHome/scoreAway, description"},
    "possessions": {"path": "(derived in memory) Matchups.V2.Pbp.BuildPossessions", "source": "pbp + rotation", "grain": "possession",
                    "leak": "dated", "key_cols": "offense, start/end tenths, start_type, pts, transition, fga, fg3a, fta, oreb, tov, assisted, off5, def5, clutch"},
    "stints": {"path": "(derived in memory) Matchups.V2.Pbp.BuildStints", "source": "pbp + rotation", "grain": "10-player stint",
               "leak": "dated", "key_cols": "home5, away5, seconds, home/away pts and possessions"},
    "shots": {"path": "data/matchups2/season/<season>/shots/<team>.json", "source": "ShotChartDetail (team, all players)",
              "grain": "shot", "leak": "dated", "key_cols": "GAME_ID, GAME_DATE, PLAYER_ID, SHOT_ZONE_BASIC, SHOT_DISTANCE, SHOT_TYPE, ACTION_TYPE, SHOT_MADE_FLAG, LOC_X/Y"},
    "player_bio": {"path": "data/matchups2/season/<season>/player_bio.json", "source": "LeagueDashPlayerBioStats", "grain": "player-season",
                   "leak": "prior-season (height/weight/age are effectively static)", "key_cols": "PLAYER_HEIGHT_INCHES, PLAYER_WEIGHT, AGE, USG_PCT, TS_PCT, AST_PCT, OREB_PCT, DREB_PCT, NET_RATING"},
    "roster": {"path": "data/matchups2/season/<season>/roster/<team>.json", "source": "CommonTeamRoster", "grain": "team-season (players, coaches)",
               "leak": "static (head coach); roster list is the season-end page", "key_cols": "players: PLAYER_ID, POSITION, HEIGHT, WEIGHT, EXP; coaches: COACH_ID, COACH_NAME, COACH_TYPE"},
    "synergy": {"path": "data/matchups2/season/<season>/synergy/<P|T>_<offensive|defensive>_<PlayType>.json", "source": "SynergyPlayTypes",
                "grain": "player or team x play type x side, season", "leak": "prior-season",
                "key_cols": "POSS_PCT, PPP, POSS, PTS, FG_PCT, TOV_POSS_PCT, FT_POSS_PCT, SCORE_POSS_PCT, PERCENTILE"},
    "tracking": {"path": "data/matchups2/season/<season>/tracking/<Player|Team>_<Measure>.json", "source": "LeagueDashPtStats",
                 "grain": "player or team season", "leak": "prior-season",
                 "key_cols": "SpeedDistance (AVG_SPEED, DIST_MILES), Drives, Passing (POTENTIAL_AST, AST_PTS_CREATED), touches, CatchShoot, PullUpShot, Rebounding (contested), Possessions (TIME_OF_POSS), Defense"},
    "defend": {"path": "data/matchups2/season/<season>/defend/<Category>.json", "source": "LeagueDashPtDefend", "grain": "defender season",
               "leak": "prior-season", "key_cols": "D_FGA, D_FG_PCT, NORMAL_FG_PCT, PCT_PLUSMINUS (Overall, 3PT, 2PT, <6ft rim, >15ft)"},
    "hustle": {"path": "data/matchups2/season/<season>/hustle_<player|team>.json", "source": "LeagueHustleStatsPlayer/Team",
               "grain": "player or team season", "leak": "prior-season", "key_cols": "CONTESTED_SHOTS, DEFLECTIONS, CHARGES_DRAWN, SCREEN_ASSISTS, LOOSE_BALLS, BOX_OUTS"},
    "arenas": {"path": "(constant) Matchups.V2.DataLayer.ARENAS", "source": "hand-entered approximations", "grain": "team",
               "leak": "static", "key_cols": "lat, lon, UTC offset, altitude ft"},
}


def LoadJson(path, default=None):
    if not os.path.exists(path):
        return default
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=8)
def PlayerBox(season: int) -> dict:
    """{game_id: [player row dict]} with the full LeagueGameLog stat line (regular season)."""
    out = defaultdict(list)
    for r in LoadJson(os.path.join(ROOT1, f"player_games_{season}.json"), []):
        if not r.get("MIN"):
            continue
        out[r["GAME_ID"]].append({
            "player_id": int(r["PLAYER_ID"]), "team": r["TEAM_ABBREVIATION"], "name": r.get("PLAYER_NAME"),
            "minutes": float(r["MIN"]), "pts": r.get("PTS") or 0, "fgm": r.get("FGM") or 0, "fga": r.get("FGA") or 0,
            "fg3m": r.get("FG3M") or 0, "fg3a": r.get("FG3A") or 0, "ftm": r.get("FTM") or 0, "fta": r.get("FTA") or 0,
            "oreb": r.get("OREB") or 0, "dreb": r.get("DREB") or 0, "ast": r.get("AST") or 0, "stl": r.get("STL") or 0,
            "blk": r.get("BLK") or 0, "tov": r.get("TOV") or 0, "pf": r.get("PF") or 0, "plus_minus": r.get("PLUS_MINUS") or 0})
    return dict(out)


def Rotation(season: int, game_id: str):
    return LoadJson(os.path.join(ROOT2, "rotation", str(season), f"{game_id}.json"))


def Pbp(season: int, game_id: str):
    return LoadJson(os.path.join(ROOT2, "pbp", str(season), f"{game_id}.json"))


def BoxMatchups(season: int, game_id: str):
    rows = LoadJson(os.path.join(ROOT1, "box", str(season), f"{game_id}.json"))
    if rows is None:
        return None
    return [{"team": r["teamTricode"], "off": int(r["personIdOff"]), "def": int(r["personIdDef"]),
             "poss": float(r.get("partialPossessions") or 0.0), "pts": float(r.get("playerPoints") or 0.0),
             "team_pts": float(r.get("teamPoints") or 0.0), "switches": float(r.get("switchesOn") or 0.0),
             "fga": float(r.get("matchupFieldGoalsAttempted") or 0.0), "fg3a": float(r.get("matchupThreePointersAttempted") or 0.0),
             "pos": r.get("positionOff") or ""} for r in rows]


@lru_cache(maxsize=8)
def ShotsByGame(season: int) -> dict:
    """{game_id: [shot]} from the per-team shot charts. Shots are dated, so walk-forward safe."""
    out = defaultdict(list)
    folder = os.path.join(ROOT2, "season", str(season), "shots")
    if not os.path.isdir(folder):
        return {}
    for fn in os.listdir(folder):
        if not fn.endswith(".json"):
            continue
        team = fn[:-5]
        for r in LoadJson(os.path.join(folder, fn), []):
            out[r["GAME_ID"]].append({"team": team, "player_id": int(r["PLAYER_ID"]), "zone": r.get("SHOT_ZONE_BASIC"),
                                      "area": r.get("SHOT_ZONE_AREA"), "dist": r.get("SHOT_DISTANCE") or 0,
                                      "three": (r.get("SHOT_TYPE") or "").startswith("3"), "action": r.get("ACTION_TYPE"),
                                      "made": int(r.get("SHOT_MADE_FLAG") or 0), "period": r.get("PERIOD")})
    return dict(out)


@lru_cache(maxsize=64)
def SeasonTable(season: int, relpath: str, frame: int = 0) -> list:
    """A season-aggregate table, e.g. SeasonTable(2024, 'synergy/T_offensive_PRBallHandler.json')."""
    data = LoadJson(os.path.join(ROOT2, "season", str(season), relpath))
    if data is None:
        return []
    return data[frame] if data and isinstance(data[0], list) else data


@lru_cache(maxsize=16)
def HeadCoaches(season: int) -> dict:
    """{team: (coach_id, coach_name)} from the roster page coaches frame."""
    out = {}
    for team in TEAM_IDS:
        data = LoadJson(os.path.join(ROOT2, "season", str(season), "roster", f"{team}.json"))
        if not data or len(data) < 2:
            continue
        for c in data[1]:
            if c.get("COACH_TYPE") == "Head Coach":
                out[team] = (c["COACH_ID"], c["COACH_NAME"])
                break
    return out


@lru_cache(maxsize=16)
def Bio(season: int) -> dict:
    """{player_id: bio row} (height inches, weight, age, usage, ast%, reb%)."""
    return {int(r["PLAYER_ID"]): r for r in SeasonTable(season, "player_bio.json")}


def Inventory() -> list:
    """Row counts / file counts / bytes for every cached table (for the report's data dictionary)."""
    rows = []
    for name, spec in DATA_DICTIONARY.items():
        base = {"games": SEASON_DIR, "team_games": SEASON_DIR, "player_games": ROOT1, "season_matchups": ROOT1,
                "box_matchups": os.path.join(ROOT1, "box"), "rotation": os.path.join(ROOT2, "rotation"),
                "pbp": os.path.join(ROOT2, "pbp")}.get(name)
        if base is None and name in ("shots", "player_bio", "roster", "synergy", "tracking", "defend", "hustle"):
            base = os.path.join(ROOT2, "season")
        files = size = 0
        if base and os.path.isdir(base):
            key = {"games": "games_", "team_games": "team_games_", "player_games": "player_games_",
                   "season_matchups": "season_matchups_", "player_bio": "player_bio", "hustle": "hustle_"}.get(name)
            sub = {"shots": "shots", "roster": "roster", "synergy": "synergy", "tracking": "tracking", "defend": "defend"}.get(name)
            for dirpath, _dirs, fns in os.walk(base):
                if sub and os.path.basename(dirpath) != sub:
                    continue
                for fn in fns:
                    if fn.endswith(".json") or fn.endswith(".jsonl"):
                        if key and not fn.startswith(key):
                            continue
                        files += 1
                        size += os.path.getsize(os.path.join(dirpath, fn))
        rows.append({"table": name, "files": files, "mb": round(size / 1e6, 1), **spec})
    return rows


@lru_cache(maxsize=1)
def Heights() -> dict:
    """{player_id: height inches} merged over every cached bio season. Height is static, so this is not a leak."""
    out = {}
    for season in range(2018, 2030):
        for pid, r in Bio(season).items():
            h = r.get("PLAYER_HEIGHT_INCHES")
            if h:
                out[pid] = float(h)
    return out


def RoleBucket(player_id: int) -> int:
    """0 guard (< 6'4"), 1 wing (6'4"-6'8"), 2 big (>= 6'9"); unknown -> wing."""
    h = Heights().get(player_id)
    if h is None:
        return 1
    return 0 if h < 76 else 2 if h >= 81 else 1
