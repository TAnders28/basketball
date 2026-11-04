"""
Leak-free backtest for one season. Walks games day by day (like football / baseball):

    for each game day D (chronological):
        1. every game from days before D is already in the features + model (the day barrier)
        2. build each game's pre-game view (features, model prob, odds if any - no scores) and let
           every strategy pick
        3. score D's picks (and settle bets if the game has moneylines), THEN feed D's results in

Every strategy is scored on pick accuracy. If games carry home_moneyline / away_moneyline, each
strategy also bets a flat STAKE out of its own STARTING_PURSE (reset per season).
"""

from itertools import groupby

from Backtest.Features import FeatureBook
from Backtest.Model import EloModel
from Backtest.Odds import FairProbs, Settle

STARTING_PURSE = 10_000.0
STAKE = 200.0   # flat $200 per bet = 2% of the starting purse; never more than what's left


def Completed(game: dict) -> bool:
    return game.get("home_score") is not None and game.get("away_score") is not None


def HasOdds(game: dict) -> bool:
    return game.get("home_moneyline") is not None and game.get("away_moneyline") is not None


def PreGameView(game: dict, features: FeatureBook, model: EloModel) -> dict:
    view = {"game_id": game["game_id"], "game_date": game["game_date"],
            "season_type": game.get("season_type"),
            "home_team": game["home_team"], "away_team": game["away_team"]}
    view.update(features.View(game))
    model_home = model.HomeWinProb(game["home_team"], game["away_team"])
    view["model_home_prob"], view["model_away_prob"] = model_home, 1.0 - model_home
    if HasOdds(game):
        view["home_moneyline"], view["away_moneyline"] = game["home_moneyline"], game["away_moneyline"]
        view["market_home_prob"], view["market_away_prob"] = FairProbs(game["home_moneyline"], game["away_moneyline"])
    return view


def RunSeason(games: list, strategies: list, prior_ratings: dict = None) -> dict:
    """Returns {strategy name: {"purse": final, "picks": [pick rows]}}."""
    features, model = FeatureBook(), EloModel(prior_ratings)
    books = {s.name: {"purse": STARTING_PURSE, "picks": []} for s in strategies}
    ordered = sorted((g for g in games if Completed(g)), key=lambda g: (g["game_date"], g["game_id"]))
    for _day, day_games in groupby(ordered, key=lambda g: g["game_date"]):
        day_games = list(day_games)
        views = [PreGameView(g, features, model) for g in day_games]   # decided before any result
        for strategy in strategies:
            book = books[strategy.name]
            available = book["purse"]
            for game, view in zip(day_games, views):
                side = strategy.Decide(view)
                if side is None:
                    continue
                home_won = game["home_score"] > game["away_score"]
                correct = home_won == (side == "home")
                row = {"game_id": game["game_id"], "game_date": game["game_date"],
                       "matchup": f"{game['away_team']} @ {game['home_team']}",
                       "pick": side, "pick_team": game[f"{side}_team"],
                       "model_prob": round(view[f"model_{side}_prob"], 4),
                       "score": f"{game['away_score']}-{game['home_score']}",
                       "result": "win" if correct else "loss",
                       "odds": None, "stake": 0.0, "profit": 0.0, "bankroll": round(book["purse"], 2)}
                stake = min(STAKE, available)
                if HasOdds(game) and stake > 0:
                    available -= stake
                    odds = game[f"{side}_moneyline"]
                    change = Settle(stake, odds, correct)
                    book["purse"] += change
                    row.update({"odds": odds, "stake": stake, "profit": round(change, 2),
                                "bankroll": round(book["purse"], 2)})
                book["picks"].append(row)
        for game in day_games:
            features.Apply(game)
            model.Apply(game)
    return books


def Summarize(book: dict) -> dict:
    picks = book["picks"]
    wins = sum(p["result"] == "win" for p in picks)
    bets = [p for p in picks if p["stake"]]
    staked = sum(p["stake"] for p in bets)
    profit = book["purse"] - STARTING_PURSE
    return {"picks": len(picks), "wins": wins, "losses": len(picks) - wins,
            "accuracy": round(wins / len(picks), 4) if picks else None,
            "bets": len(bets), "total_staked": round(staked, 2),
            "end": round(book["purse"], 2), "profit": round(profit, 2),
            "roi_on_staked": round(profit / staked, 4) if staked else None,
            "min_bankroll": round(min([STARTING_PURSE] + [p["bankroll"] for p in bets]), 2)}
