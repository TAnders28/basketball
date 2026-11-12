r"""
Polite, resumable pull of the public nba_api data the matchup models need. Caches under data/matchups/
(gitignored). Sequential, one request at a time, sleeps between calls.

    .venv\Scripts\python.exe -m Matchups.PullData --seasons 2023 2024 2025 2026

Season keys follow the repo (2024 = 2024-25). Pulls, per season:
  - LeagueGameLog (players): one call -> data/matchups/player_games_<season>.json
  - LeagueSeasonMatchups (season aggregate, used ONLY as a prior for the following season)
  - BoxScoreMatchupsV3 for every completed regular-season game in data/season/games_<season>.json
    -> data/matchups/box/<season>/<game_id>.json
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone

DATA_DIR = os.path.join("data", "matchups")
SLEEP_SECS = 0.6
TIMEOUT = 45
MATCHUP_COLS = ["gameId", "teamTricode", "personIdOff", "nameIOff", "positionOff", "personIdDef", "nameIDef",
                "matchupMinutesSort", "partialPossessions", "percentageDefenderTotalTime",
                "percentageOffensiveTotalTime", "switchesOn", "playerPoints", "teamPoints", "matchupAssists",
                "matchupTurnovers", "matchupFieldGoalsMade", "matchupFieldGoalsAttempted",
                "matchupThreePointersMade", "matchupThreePointersAttempted", "matchupFreeThrowsMade",
                "matchupFreeThrowsAttempted", "shootingFouls"]


def SeasonLabel(season: int) -> str:
    return f"{season}-{str(season + 1)[-2:]}"


def Log(msg: str, log_path: str) -> None:
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def CallWithRetry(fn, log_path, tries=4):
    for attempt in range(tries):
        try:
            return fn()
        except Exception as ex:   # network hiccups / throttling: back off
            wait = 5 * (attempt + 1) ** 2
            Log(f"retry {attempt + 1} after {wait}s: {repr(ex)[:160]}", log_path)
            time.sleep(wait)
    return None


def RegularSeasonGameIds(season: int) -> list:
    path = os.path.join("data", "season", f"games_{season}.json")
    games = json.load(open(path, encoding="utf-8"))
    return sorted(g["game_id"] for g in games
                  if g["game_id"].startswith("002") and g.get("home_score") is not None)


def PullPlayerGames(season, log_path, refresh=False):
    from nba_api.stats.endpoints import LeagueGameLog
    out = os.path.join(DATA_DIR, f"player_games_{season}.json")
    if os.path.exists(out) and not refresh:
        return
    obj = CallWithRetry(lambda: LeagueGameLog(season=SeasonLabel(season), player_or_team_abbreviation="P",
                                              season_type_all_star="Regular Season", timeout=TIMEOUT), log_path)
    if obj is None:
        Log(f"player games {season} FAILED", log_path)
        return
    rows = obj.get_data_frames()[0].to_dict("records")
    json.dump(rows, open(out, "w", encoding="utf-8"))
    Log(f"player games {season}: {len(rows)} rows", log_path)
    time.sleep(SLEEP_SECS)


def PullSeasonMatchups(season, log_path):
    from nba_api.stats.endpoints import LeagueSeasonMatchups
    out = os.path.join(DATA_DIR, f"season_matchups_{season}.json")
    if os.path.exists(out):
        return
    obj = CallWithRetry(lambda: LeagueSeasonMatchups(season=SeasonLabel(season), timeout=120), log_path)
    if obj is None:
        Log(f"season matchups {season} FAILED", log_path)
        return
    rows = obj.get_data_frames()[0].to_dict("records")
    json.dump(rows, open(out, "w", encoding="utf-8"))
    Log(f"season matchups {season}: {len(rows)} rows", log_path)
    time.sleep(SLEEP_SECS)


def PullBoxMatchups(season, log_path, limit=None):
    from nba_api.stats.endpoints import BoxScoreMatchupsV3
    folder = os.path.join(DATA_DIR, "box", str(season))
    os.makedirs(folder, exist_ok=True)
    ids = RegularSeasonGameIds(season)
    todo = [g for g in ids if not os.path.exists(os.path.join(folder, f"{g}.json"))]
    if limit:
        todo = todo[:limit]
    Log(f"box matchups {season}: {len(ids)} games, {len(todo)} to pull", log_path)
    t0, done, failed = time.time(), 0, 0
    for gid in todo:
        obj = CallWithRetry(lambda: BoxScoreMatchupsV3(game_id=gid, timeout=TIMEOUT), log_path)
        if obj is None:
            failed += 1
            continue
        frame = obj.get_data_frames()[0]
        cols = [c for c in MATCHUP_COLS if c in frame.columns]
        rows = frame[cols].to_dict("records")
        tmp = os.path.join(folder, f"{gid}.json.tmp")
        json.dump(rows, open(tmp, "w", encoding="utf-8"))
        os.replace(tmp, os.path.join(folder, f"{gid}.json"))
        done += 1
        if done % 50 == 0:
            rate = (time.time() - t0) / done
            Log(f"box {season}: {done}/{len(todo)} ({rate:.2f}s/game, eta {rate * (len(todo) - done) / 60:.0f} min)", log_path)
        time.sleep(SLEEP_SECS)
    Log(f"box matchups {season}: done {done}, failed {failed}, {time.time() - t0:.0f}s", log_path)


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", type=int, nargs="+", required=True)
    parser.add_argument("--prior-seasons", type=int, nargs="*", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--refresh-current", type=int, default=None, help="season whose player log to re-pull")
    args = parser.parse_args()
    os.makedirs(DATA_DIR, exist_ok=True)
    log_path = os.path.join("logs", f"{datetime.now():%Y-%m-%d}_matchups_pull.log")
    import subprocess, sys
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Log(f"START python {sys.version.split()[0]} head {head} args {vars(args)} utc {datetime.now(timezone.utc).isoformat(timespec='seconds')}", log_path)
    t0 = time.time()
    for season in sorted(set(args.seasons) | set(args.prior_seasons)):
        PullPlayerGames(season, log_path, refresh=(season == args.refresh_current))
        PullSeasonMatchups(season, log_path)
    for season in args.seasons:
        PullBoxMatchups(season, log_path, args.limit)
    Log(f"END total {time.time() - t0:.0f}s", log_path)


if __name__ == "__main__":
    Main()
