"""Append-only season prediction log under data/predictions/."""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
PRED_LOG_DIR = os.path.join("data", "predictions")
MODEL_VERSION = "elo_predictday_v1"


def LogPath(season: int) -> str:
    return os.path.join(PRED_LOG_DIR, f"{season}.jsonl")


def GradesPath(season: int) -> str:
    return os.path.join(PRED_LOG_DIR, f"{season}_grades.json")


def ParseStartTime(game_date: str, tip_et: str):
    """Best-effort America/New_York ISO start; None if tip cannot be parsed."""
    if not tip_et:
        return None
    tip = tip_et.replace("ET", "").strip()
    for fmt in ("%I:%M %p", "%H:%M"):
        try:
            clock = datetime.strptime(tip, fmt).time()
            return datetime(int(game_date[0:4]), int(game_date[5:7]), int(game_date[8:10]),
                            clock.hour, clock.minute, tzinfo=EASTERN).isoformat()
        except ValueError:
            continue
    return None


def AppendPredictionRows(season: int, rows: list, source: str = "PredictDay",
                         created_at: str = None) -> str:
    """Append one JSON object per game. Never rewrites prior lines. Returns log path."""
    os.makedirs(PRED_LOG_DIR, exist_ok=True)
    path = LogPath(season)
    stamp = created_at or datetime.now(EASTERN).isoformat()
    with open(path, "a", encoding="utf-8") as fh:
        for row in rows:
            game_date = row.get("game_date") or row.get("date")
            tip = row.get("tip_et") or row.get("start_time_raw")
            record = {
                "created_at": stamp,
                "game_id": row["game_id"],
                "start_time": row.get("start_time") or ParseStartTime(game_date, tip) or tip,
                "pick": row.get("recommended") or row.get("pick"),
                "model_prob": row.get("recommended_prob") if row.get("recommended_prob") is not None else row.get("model_prob"),
                "market_price": row.get("market_price"),
                "price_time": row.get("price_time"),
                "source": source,
                "model_version": row.get("model_version") or MODEL_VERSION,
                "game_date": game_date,
                "home_team": row.get("home_team"),
                "away_team": row.get("away_team"),
            }
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path


def ReadPredictionLog(season: int) -> list:
    path = LogPath(season)
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows
