r"""Grade append-only PredictDay logs without rewriting them.

    .venv\Scripts\python.exe -m Backtest.GradePredictions --season 2026

Writes data/predictions/<season>_grades.json keyed by game_id.
Excludes log rows whose created_at is at/after start_time (post-tip / look-ahead).
"""

import argparse
import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

from Backtest.PredictionLog import GradesPath, ReadPredictionLog

SEASON_DIR = os.path.join("data", "season")
EASTERN = ZoneInfo("America/New_York")


def LoadGames(season: int) -> dict:
    path = os.path.join(SEASON_DIR, f"games_{season}.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        games = json.load(fh)
    return {g["game_id"]: g for g in games}


def ParseIso(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return None


def RowEligible(row: dict) -> tuple:
    """Return (ok, reason). Post-tip creations are excluded."""
    created = ParseIso(row.get("created_at"))
    start = ParseIso(row.get("start_time"))
    if created is None:
        return False, "missing_created_at"
    if start is None:
        # Tip was not parseable to ISO — keep row but flag; still grade if final exists.
        return True, "start_time_unparsed"
    if created.tzinfo is None:
        created = created.replace(tzinfo=EASTERN)
    if start.tzinfo is None:
        start = start.replace(tzinfo=EASTERN)
    if created >= start:
        return False, "created_after_start"
    return True, "ok"


def GradeSeason(season: int) -> dict:
    games = LoadGames(season)
    grades = {}
    excluded = []
    for row in ReadPredictionLog(season):
        ok, reason = RowEligible(row)
        gid = row.get("game_id")
        if not ok:
            excluded.append({"game_id": gid, "reason": reason, "created_at": row.get("created_at")})
            continue
        game = games.get(gid)
        if not game or game.get("home_score") is None:
            grades[gid] = {
                "game_id": gid,
                "status": "ungraded_no_final",
                "pick": row.get("pick"),
                "model_prob": row.get("model_prob"),
                "created_at": row.get("created_at"),
                "start_time": row.get("start_time"),
            }
            continue
        home_won = game["home_score"] > game["away_score"]
        pick = row.get("pick")
        correct = (pick == game["home_team"] and home_won) or (pick == game["away_team"] and not home_won)
        # First eligible row wins for a game_id (append-only may repeat).
        if gid in grades and grades[gid].get("status") == "graded":
            continue
        grades[gid] = {
            "game_id": gid,
            "status": "graded",
            "pick": pick,
            "model_prob": row.get("model_prob"),
            "correct": bool(correct),
            "score": f"{game['away_score']}-{game['home_score']}",
            "created_at": row.get("created_at"),
            "start_time": row.get("start_time"),
            "eligibility": reason,
        }
    summary = {
        "season": season,
        "graded": sum(1 for g in grades.values() if g.get("status") == "graded"),
        "correct": sum(1 for g in grades.values() if g.get("correct") is True),
        "excluded_rows": excluded,
        "grades": grades,
    }
    if summary["graded"]:
        summary["accuracy"] = round(summary["correct"] / summary["graded"], 4)
    else:
        summary["accuracy"] = None
    path = GradesPath(season)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def Main() -> None:
    parser = argparse.ArgumentParser(description="Grade append-only prediction logs")
    parser.add_argument("--season", type=int, required=True)
    args = parser.parse_args()
    summary = GradeSeason(args.season)
    print(f"season {args.season}: graded={summary['graded']} correct={summary['correct']} "
          f"accuracy={summary['accuracy']} excluded={len(summary['excluded_rows'])}")
    print(f"wrote {GradesPath(args.season)}")


if __name__ == "__main__":
    Main()

