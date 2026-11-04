r"""
Pull NBA seasons from stats.nba.com (via nba_api, no API key) and store them as readable JSON / JSONL.

    .venv\Scripts\python.exe GetSeasonGames.py --season 2024              # 2024-25 regular season + playoffs
    .venv\Scripts\python.exe GetSeasonGames.py --season 2023,2024,2025 [--refresh]
    .venv\Scripts\python.exe GetSeasonGames.py --season 2024 --dry-run --limit 10   # print sample, write nothing
    .venv\Scripts\python.exe GetSeasonGames.py --season 2024 --sample 60  # also refresh examples/games_sample.json

A season is named by the year it STARTS: 2024 = the 2024-25 season (like football's season ints).

Per season writes:
    data/season/games_<season>.json         JSON array, one object per game, pretty-printed
                                            (date, home/away team, scores, both teams' box totals)
    data/season/team_games_<season>.jsonl   JSONL, one team-game row per line (raw LeagueGameLog row)
    data/season/pull_summary_<season>.json  counts + pull timestamp

Seasons that already have every output are skipped unless --refresh is given.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

ROOT = os.path.dirname(os.path.abspath(__file__))
SEASON_DIR = os.path.join(ROOT, "data", "season")
EXAMPLES_DIR = os.path.join(ROOT, "examples")
SEASON_TYPES = {"regular": "Regular Season", "playoffs": "Playoffs", "playin": "PlayIn"}
STAT_KEYS = ["FGM", "FGA", "FG_PCT", "FG3M", "FG3A", "FG3_PCT", "FTM", "FTA", "FT_PCT",
             "OREB", "DREB", "REB", "AST", "STL", "BLK", "TOV", "PF", "PTS", "PLUS_MINUS", "MIN"]


def SeasonLabel(season: int) -> str:
    """2024 -> '2024-25' (the format stats.nba.com expects)."""
    return f"{season}-{str(season + 1)[-2:]}"


def CurrentSeason() -> int:
    today = datetime.now()
    return today.year if today.month >= 10 else today.year - 1


def OutputPaths(season: int) -> dict:
    return {
        "games": os.path.join(SEASON_DIR, f"games_{season}.json"),
        "team_games": os.path.join(SEASON_DIR, f"team_games_{season}.jsonl"),
        "summary": os.path.join(SEASON_DIR, f"pull_summary_{season}.json"),
    }


def FetchTeamGameRows(season: int, season_type: str, retries: int = 3) -> list:
    """One row per team per game from LeagueGameLog, as plain dicts."""
    from nba_api.stats.endpoints import leaguegamelog
    for attempt in range(1, retries + 1):
        try:
            payload = leaguegamelog.LeagueGameLog(season=SeasonLabel(season),
                                                  season_type_all_star=season_type,
                                                  timeout=60).get_dict()
            result = payload["resultSets"][0]
            rows = [dict(zip(result["headers"], row)) for row in result["rowSet"]]
            for row in rows:
                row["SEASON_TYPE"] = season_type
            return rows
        except Exception as exc:  # stats.nba.com is flaky; back off and retry
            print(f"  {season_type} attempt {attempt} failed: {exc}")
            time.sleep(3 * attempt)
    raise SystemExit(f"could not pull {SeasonLabel(season)} {season_type} after {retries} tries")


def PairGames(team_rows: list, season: int) -> list:
    """Turn two team-game rows into one game record. Home team's MATCHUP reads 'AAA vs. BBB',
    away team's reads 'BBB @ AAA'."""
    by_game = {}
    for row in team_rows:
        by_game.setdefault(row["GAME_ID"], []).append(row)
    games = []
    for game_id, rows in by_game.items():
        home = next((r for r in rows if " vs. " in r["MATCHUP"]), None)
        away = next((r for r in rows if " @ " in r["MATCHUP"]), None)
        if home is None or away is None:
            continue
        games.append({
            "game_id": game_id,
            "season": season,
            "season_type": home["SEASON_TYPE"],
            "game_date": home["GAME_DATE"],
            "home_team": home["TEAM_ABBREVIATION"],
            "away_team": away["TEAM_ABBREVIATION"],
            "home_team_name": home["TEAM_NAME"],
            "away_team_name": away["TEAM_NAME"],
            "home_score": home["PTS"],
            "away_score": away["PTS"],
            "home_win": None if home["WL"] is None else home["WL"] == "W",
            "home_stats": {k.lower(): home.get(k) for k in STAT_KEYS},
            "away_stats": {k.lower(): away.get(k) for k in STAT_KEYS},
        })
    games.sort(key=lambda g: (g["game_date"], g["game_id"]))
    return games


def WriteJson(records, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)


def WriteJsonl(records: list, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps({k: v for k, v in record.items() if v is not None}, separators=(",", ":")))
            f.write("\n")


def PullSeason(season: int, season_types: list, refresh: bool, dry_run: bool, limit: int, sample: int) -> None:
    paths = OutputPaths(season)
    if not (refresh or dry_run or sample) and all(os.path.exists(p) for p in paths.values()):
        print(f"[{season}] all outputs exist, skipping (use --refresh to re-pull)")
        return

    team_rows = []
    for season_type in season_types:
        print(f"[{season}] {SeasonLabel(season)} {season_type}...")
        team_rows += FetchTeamGameRows(season, season_type)
        time.sleep(1)  # be polite to stats.nba.com
    games = PairGames(team_rows, season)

    if dry_run:
        for game in games[:limit]:
            print(f"  {game['game_date']} {game['away_team']} {game['away_score']} @ "
                  f"{game['home_team']} {game['home_score']} ({game['season_type']})")
        print(f"[{season}] dry run: {len(games)} games / {len(team_rows)} team rows available, nothing written")
        return

    if sample:
        os.makedirs(EXAMPLES_DIR, exist_ok=True)
        sample_path = os.path.join(EXAMPLES_DIR, "games_sample.json")
        WriteJson(games[:sample], sample_path)
        print(f"[{season}] wrote {min(sample, len(games))} games to {sample_path}")

    os.makedirs(SEASON_DIR, exist_ok=True)
    WriteJson(games, paths["games"])
    WriteJsonl(team_rows, paths["team_games"])
    by_type = {}
    for game in games:
        by_type[game["season_type"]] = by_type.get(game["season_type"], 0) + 1
    summary = {
        "season": season,
        "season_label": SeasonLabel(season),
        "pulled_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "games": len(games),
        "games_by_type": by_type,
        "team_game_rows": len(team_rows),
        "first_game_date": games[0]["game_date"] if games else None,
        "last_game_date": games[-1]["game_date"] if games else None,
    }
    WriteJson(summary, paths["summary"])
    print(f"[{season}] {summary['games']} games ({by_type}) "
          f"{summary['first_game_date']} -> {summary['last_game_date']}")


def Main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--season", default=str(CurrentSeason()),
                        help="comma-separated start years, e.g. 2023,2024 (2024 = 2024-25)")
    parser.add_argument("--types", default="regular,playoffs",
                        help=f"comma-separated season types from {list(SEASON_TYPES)}")
    parser.add_argument("--refresh", action="store_true", help="re-pull even if outputs exist")
    parser.add_argument("--dry-run", action="store_true", help="pull and print a few games; write nothing")
    parser.add_argument("--limit", type=int, default=10, help="games to print in --dry-run")
    parser.add_argument("--sample", type=int, default=0, help="also write the first N games to examples/games_sample.json")
    args = parser.parse_args()

    season_types = [SEASON_TYPES[t.strip()] for t in args.types.split(",") if t.strip()]
    for season in (int(s) for s in args.season.split(",") if s.strip()):
        PullSeason(season, season_types, args.refresh, args.dry_run, args.limit, args.sample)


if __name__ == "__main__":
    Main()
