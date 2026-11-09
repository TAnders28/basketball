r"""
Pre-game picks for one day's NBA slate, using the same leak-free FeatureBook + EloModel + strategies
as the backtest (no new model). Only completed games dated BEFORE the target day are fed in.

    .venv\Scripts\python.exe -m Backtest.PredictDay                    # today (Eastern)
    .venv\Scripts\python.exe -m Backtest.PredictDay --date 2026-11-04 --season 2026

Needs data/season/games_<season>.json (and games_<season-1>.json to seed Elo) from GetSeasonGames.py.
Schedule comes from stats.nba.com ScoreboardV2 (GameHeader + LineScore for abbreviations / tip time).
Writes data/backtest/predictions/picks_<date>.json and .md, and appends data/predictions/<season>.jsonl.
"""

import argparse
import json
import os
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


def LoadJson(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def FetchSlate(game_date: str) -> list:
    """[{game_id, home_team, away_team, tip_et, status}] for one date, from ScoreboardV2."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    from nba_api.stats.endpoints import scoreboardv2
    sets = {r["name"]: r for r in ScoreboardSets(scoreboardv2, game_date)}
    header, lines = sets["GameHeader"], sets["LineScore"]
    abbr = {}
    for row in lines["rowSet"]:
        rec = dict(zip(lines["headers"], row))
        abbr[rec["TEAM_ID"]] = rec["TEAM_ABBREVIATION"]
    slate = []
    for row in header["rowSet"]:
        rec = dict(zip(header["headers"], row))
        slate.append({"game_id": rec["GAME_ID"], "game_date": game_date,
                      "home_team": abbr.get(rec["HOME_TEAM_ID"], rec["GAMECODE"].split("/")[1][3:]),
                      "away_team": abbr.get(rec["VISITOR_TEAM_ID"], rec["GAMECODE"].split("/")[1][:3]),
                      "tip_et": rec["GAME_STATUS_TEXT"].strip(), "season_type": "Regular Season"})
    return slate


def ScoreboardSets(endpoint, game_date: str) -> list:
    return endpoint.ScoreboardV2(game_date=game_date, timeout=60).get_dict()["resultSets"]


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
        rows.append({**game, "home_elo": round(model.Rating(game["home_team"]), 1),
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
             "| Tip ET | Game | Records | Rest (A/H) | always_home | better_record | rest_adv | elo_pick | Pick | Elo prob | Conf |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        p = r["picks"]
        lines.append(f"| {r['tip_et']} | {r['away_team']} @ {r['home_team']} | {r['away_record']} / {r['home_record']} | "
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


