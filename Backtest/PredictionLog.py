"""Append-only season prediction log under data/predictions."""

import json
import os
from datetime import datetime
from zoneinfo import ZoneInfo

EASTERN = ZoneInfo("America/New_York")
PRED_LOG_DIR = os.path.join("data", "predictions")
# Primary PredictDay model. Schema v1 rows used model_version=elo_predictday_v1 and the
# core fields only; v2 adds shadow probabilities and an action. Readers must tolerate both.
MODEL_VERSION = "stack_global_predictday_v1"
ELO_MODEL_VERSION = "elo_predictday_v1"
KALMAN_MODEL_VERSION = "kalman_team"
LOG_SCHEMA_VERSION = 2

# Core fields written for every row (schema v1). Unknown extra keys are ignored by old readers.
CORE_FIELDS = (
    "created_at", "game_id", "start_time", "pick", "model_prob",
    "market_price", "price_time", "source", "model_version", "game_date",
    "home_team", "away_team",
)
# Schema v2 extras. Absent on old rows; new rows always include them (None if unknown).
SCHEMA_V2_FIELDS = (
    "log_schema",
    "stack_global_home_prob",
    "kalman_team_home_prob",
    "elo_predictday_v1_home_prob",
    "kalman_team_pick",
    "kalman_team_prob",
    "elo_predictday_v1_pick",
    "elo_predictday_v1_prob",
    "action",
    "action_reason",
    "edge",
    "fee",
    "fee_basis",
)


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


def LogSchema(row: dict) -> int:
    """Schema version of a stored or in-memory row. Old rows without log_schema are v1."""
    value = row.get("log_schema")
    if value is None:
        return 1
    return int(value)


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
                "log_schema": row.get("log_schema") or LOG_SCHEMA_VERSION,
                "stack_global_home_prob": row.get("stack_global_home_prob"),
                "kalman_team_home_prob": row.get("kalman_team_home_prob"),
                "elo_predictday_v1_home_prob": row.get("elo_predictday_v1_home_prob"),
                "kalman_team_pick": row.get("kalman_team_pick"),
                "kalman_team_prob": row.get("kalman_team_prob"),
                "elo_predictday_v1_pick": row.get("elo_predictday_v1_pick"),
                "elo_predictday_v1_prob": row.get("elo_predictday_v1_prob"),
                "action": row.get("action") or "PASS",
                "action_reason": row.get("action_reason") or row.get("reason"),
                "edge": row.get("edge"),
                "fee": row.get("fee"),
                "fee_basis": row.get("fee_basis"),
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