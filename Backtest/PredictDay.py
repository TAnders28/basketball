r"""
Pre-game picks for one day's NBA slate, using the same leak-free FeatureBook + EloModel + strategies
as the backtest (no new model). Only completed games dated BEFORE the target day are fed in.

    .venv\Scripts\python.exe -m Backtest.PredictDay                    # today (Eastern)
    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-04 --season 2026

Needs data/season/games_<season>.json (and games_<season-1>.json to seed Elo) from GetSeasonGames.py.
Schedule comes from stats.nba.com ScoreboardV2 (GameHeader + LineScore for abbreviations / tip time),
with bounded retries and a CDN todaysScoreboard fallback when ScoreboardV2 returns empty/non-JSON.
Writes data/backtest/predictions/picks_<date>.json and .md, and appends data/predictions/<season>.jsonl.
"""

import argparse
import json
import os
import time
import warnings
from datetime import datetime
from zoneinfo import ZoneInfo

from Backtest.Engine import Completed, PreGameView
from Backtest.Features import FeatureBook
from Backtest.Model import EloModel, SeasonFinalRatings
from Backtest.Strategies import DefaultStrategies
from Backtest.PredictionLog import AppendPredictionRows, MODEL_VERSION

SEASON_DIR = os.path.join("data", "season")
PRED_DIR = os.path.join("data", "backtest", "predictions")
EASTERN = ZoneInfo("America/New_York")
SCOREBOARD_RETRIES = 3
SCOREBOARD_RETRY_SLEEP = 1.5


def LoadJson(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def ScoreboardSets(endpoint, game_date: str, retries: int = SCOREBOARD_RETRIES) -> list:
    """Call ScoreboardV2 with bounded retries. Raises a clear error on empty/non-JSON."""
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            raw = endpoint.ScoreboardV2(game_date=game_date, timeout=60).get_dict()
            sets = raw.get("resultSets")
            if not isinstance(sets, list) or not sets:
                raise ValueError("ScoreboardV2 returned empty resultSets")
            return sets
        except Exception as exc:
            last_err = exc
            if attempt < retries:
                time.sleep(SCOREBOARD_RETRY_SLEEP)
    raise RuntimeError(
        f"ScoreboardV2 failed for {game_date} after {retries} attempts "
        f"({type(last_err).__name__}: {last_err})"
    ) from last_err


def FetchSlateFromScoreboard(sets: list, game_date: str) -> list:
    named = {r["name"]: r for r in sets}
    header, lines = named["GameHeader"], named["LineScore"]
    abbr = {}
    for row in lines["rowSet"]:
        rec = dict(zip(lines["headers"], row))
        abbr[rec["TEAM_ID"]] = rec["TEAM_ABBREVIATION"]
    slate = []
    for row in header["rowSet"]:
        rec = dict(zip(header["headers"], row))
        slate.append({
            "game_id": rec["GAME_ID"], "game_date": game_date,
            "home_team": abbr.get(rec["HOME_TEAM_ID"], rec["GAMECODE"].split("/")[1][3:]),
            "away_team": abbr.get(rec["VISITOR_TEAM_ID"], rec["GAMECODE"].split("/")[1][:3]),
            "tip_et": rec["GAME_STATUS_TEXT"].strip(), "season_type": "Regular Season",
            "home_moneyline": None, "away_moneyline": None, "line": "-",
        })
    return slate


def FetchSlateFromCdn(game_date: str) -> list:
    """Fallback using LiveOdds CDN todaysScoreboard (same day only is meaningful)."""
    from LiveOdds.Sources import FetchScoreboard, ReadOnlyHttp
    got = FetchScoreboard(ReadOnlyHttp())
    if got.get("error") or not got.get("body"):
        raise RuntimeError(f"CDN scoreboard fallback failed: {got.get('error') or 'empty body'}")
    board = (got["body"] or {}).get("scoreboard") or {}
    board_date = (board.get("gameDate") or "")[:10]
    if board_date and board_date != game_date:
        raise RuntimeError(
            f"CDN scoreboard date {board_date} != requested {game_date}; "
            "ScoreboardV2 unavailable and CDN cannot fill another day"
        )
    slate = []
    for game in board.get("games") or []:
        home = (game.get("homeTeam") or {}).get("teamTricode")
        away = (game.get("awayTeam") or {}).get("teamTricode")
        tip = game.get("gameStatusText") or game.get("gameEt") or ""
        slate.append({
            "game_id": game.get("gameId"), "game_date": game_date,
            "home_team": home, "away_team": away,
            "tip_et": str(tip).strip(), "season_type": "Regular Season",
            "home_moneyline": None, "away_moneyline": None, "line": "-",
            "slate_source": "cdn_scoreboard",
        })
    return slate


def FetchSlate(game_date: str) -> list:
    """[{game_id, home_team, away_team, tip_et, status, line}] for one date."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nba_api.stats.endpoints import scoreboardv2
    try:
        sets = ScoreboardSets(scoreboardv2, game_date)
        return FetchSlateFromScoreboard(sets, game_date)
    except Exception as primary:
        try:
            slate = FetchSlateFromCdn(game_date)
            if slate:
                return slate
        except Exception as cdn_err:
            raise RuntimeError(
                f"ScoreboardV2 failed ({primary}); CDN fallback failed ({cdn_err})"
            ) from cdn_err
        raise RuntimeError(f"ScoreboardV2 failed ({primary}); CDN returned no games") from primary


def FormatLine(game: dict) -> str:
    """Moneyline pair for the table, or '-' when odds are absent."""
    hm, am = game.get("home_moneyline"), game.get("away_moneyline")
    if hm is None or am is None:
        return "-"
    return f"{am}/{hm}"


def BuildState(season: int, game_date: str):
    prior_games = LoadJson(os.path.join(SEASON_DIR, f"games_{season - 1}.json"))
    model = EloModel(SeasonFinalRatings(prior_games) if prior_games else None)
    features = FeatureBook()
    games = sorted((g for g in LoadJson(os.path.join(SEASON_DIR, f"games_{season}.json"))
                    if Completed(g) and g["game_date"] < game_date), key=lambda g: (g["game_date"], g["game_id"]))
    for game in games:
        features.Apply(game)
        model.Apply(game)
    return features, model, games


def Recommend(view: dict, picks: dict) -> tuple:
    """Elo pick is the headline (best backtest accuracy); confidence from Elo prob, nudged by agreement."""
    side = picks["elo_pick"]
    prob = view[f"model_{side}_prob"]
    agree = [n for n in ("better_record", "rest_advantage") if picks.get(n) == side]
    against = [n for n in ("better_record", "rest_advantage") if picks.get(n) not in (None, side)]
    tier = "high" if prob >= 0.70 else "medium" if prob >= 0.60 else "low"
    if against and tier != "low":
        tier = "medium" if tier == "high" else "low"
    return side, prob, tier, agree, against


def Main() -> None:
    parser = argparse.ArgumentParser(description="Pre-game picks for one NBA day")
    parser.add_argument("--date", default=datetime.now(EASTERN).strftime("%Y-%m-%d"))
    parser.add_argument("--season", type=int, default=None, help="season start year (default from date)")
    args = parser.parse_args()
    day = datetime.strptime(args.date, "%Y-%m-%d")
    season = args.season or (day.year if day.month >= 10 else day.year - 1)

    features, model, played = BuildState(season, args.date)
    slate = FetchSlate(args.date)
    strategies = DefaultStrategies()
    rows = []
    for game in slate:
        view = PreGameView(game, features, model)
        picks = {s.name: s.Decide(view) for s in strategies}
        side, prob, tier, agree, against = Recommend(view, picks)
        line = FormatLine(game)
        rows.append({**game, "line": line,
                     "home_elo": round(model.Rating(game["home_team"]), 1),
                     "away_elo": round(model.Rating(game["away_team"]), 1),
                     "home_record": f"{features.Team(game['home_team']).wins}-{features.Team(game['home_team']).losses}",
                     "away_record": f"{features.Team(game['away_team']).wins}-{features.Team(game['away_team']).losses}",
                     "home_rest_days": view["home_rest_days"], "away_rest_days": view["away_rest_days"],
                     "home_avg_margin": view["home_avg_margin"], "away_avg_margin": view["away_avg_margin"],
                     "model_home_prob": round(view["model_home_prob"], 4),
                     "picks": {k: (game[f"{v}_team"] if v else None) for k, v in picks.items()},
                     "recommended": game[f"{side}_team"], "recommended_prob": round(prob, 4),
                     "confidence": tier, "agrees": agree, "disagrees": against})

    data_through = played[-1]["game_date"] if played else None
    os.makedirs(PRED_DIR, exist_ok=True)
    with open(os.path.join(PRED_DIR, f"picks_{args.date}.json"), "w", encoding="utf-8") as fh:
        json.dump({"date": args.date, "season": season, "data_through": data_through,
                   "games_used": len(played), "games": rows}, fh, indent=2)
    lines = [f"# NBA picks {args.date} (data through {data_through}, {len(played)} games)", "",
             "| Tip ET | Game | Line | Records | Rest (A/H) | always_home | better_record | rest_adv | elo_pick | Pick | Elo prob | Conf |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["picks"]
        lines.append(f"| {r['tip_et']} | {r['away_team']} @ {r['home_team']} | {r['line']} | "
                     f"{r['away_record']} / {r['home_record']} | "
                     f"{r['away_rest_days']}/{r['home_rest_days']} | {p['always_home']} | {p['better_record'] or '-'} | "
                     f"{p['rest_advantage'] or '-'} | {p['elo_pick']} | **{r['recommended']}** | "
                     f"{r['recommended_prob'] * 100:.0f}% | {r['confidence']} |")
    text = "\n".join(lines)
    with open(os.path.join(PRED_DIR, f"picks_{args.date}.md"), "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    log_path = AppendPredictionRows(season, rows, source="PredictDay")
    print(text)
    print(f"appended {len(rows)} rows -> {log_path} (model_version={MODEL_VERSION})")


if __name__ == "__main__":
    Main()
