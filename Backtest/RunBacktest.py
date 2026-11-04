r"""
One command, from E:\Basketball\basketball:

    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2024
    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2023,2024
    .venv\Scripts\python.exe -m Backtest.RunBacktest --games-file examples\games_sample.json   # smoke test

Reads data/season/games_<season>.json (from GetSeasonGames.py); seeds Elo from games_<season-1>.json
if it exists. Prints a per-season table and writes to data/backtest/reports/:
    report_<season>.json            summary per strategy
    picks_<season>_<strategy>.csv   every pick (and bankroll path when odds are present)
    REPORT.md                       plain-language report for all seasons run
"""

import argparse
import csv
import json
import os

from Backtest.Engine import STAKE, STARTING_PURSE, RunSeason, Summarize
from Backtest.Model import CARRYOVER, HOME_COURT_ELO, K_FACTOR, SeasonFinalRatings
from Backtest.Strategies import DefaultStrategies

SEASON_DIR = os.path.join("data", "season")
REPORT_DIR = os.path.join("data", "backtest", "reports")


def LoadJson(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def RunOne(label: str, games: list, prior_games: list) -> dict:
    prior = SeasonFinalRatings(prior_games) if prior_games else None   # finished season -> no leak
    strategies = DefaultStrategies()
    books = RunSeason(games, strategies, prior)
    result = {"season": label, "games": len(games),
              "games_with_odds": sum(g.get("home_moneyline") is not None for g in games),
              "prior_season_used": bool(prior),
              "strategies": {s.name: {"description": s.description, **Summarize(books[s.name])} for s in strategies}}
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(os.path.join(REPORT_DIR, f"report_{label}.json"), "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    for name, book in books.items():
        if book["picks"]:
            with open(os.path.join(REPORT_DIR, f"picks_{label}_{name}.csv"), "w", encoding="utf-8", newline="") as fh:
                writer = csv.DictWriter(fh, fieldnames=list(book["picks"][0]))
                writer.writeheader()
                writer.writerows(book["picks"])
    return result


def TableLines(result: dict) -> list:
    lines = [f"### {result['season']} ({result['games']} games, {result['games_with_odds']} with odds, "
             f"prior-season Elo: {'yes' if result['prior_season_used'] else 'no'})", "",
             "| Strategy | Picks | W-L | Accuracy | Bets | Profit | ROI (staked) |",
             "|---|---|---|---|---|---|---|"]
    for name, s in result["strategies"].items():
        acc = f"{s['accuracy'] * 100:.1f}%" if s["accuracy"] is not None else "n/a"
        roi = f"{s['roi_on_staked'] * 100:+.1f}%" if s["roi_on_staked"] is not None else "n/a"
        profit = f"${s['profit']:+,.2f}" if s["bets"] else "n/a"
        lines.append(f"| {name} | {s['picks']} | {s['wins']}-{s['losses']} | {acc} | {s['bets']} | {profit} | {roi} |")
    return lines + [""]


def WriteMarkdown(results: list) -> str:
    lines = ["# NBA backtest", "", "## Rules in plain language", "",
             "- **Leak-free:** games are walked day by day; a game's pick only sees games from earlier days.",
             "- **Accuracy:** every strategy is scored on how often its picked team won.",
             f"- **Money (only when games carry moneylines):** each strategy starts the season with "
             f"${STARTING_PURSE:,.0f} and bets a flat ${STAKE:,.0f} per pick at the presented odds.",
             f"- **Elo model:** K={K_FACTOR:.0f}, home court {HOME_COURT_ELO:.0f} Elo, margin-of-victory multiplier, "
             f"seeded from last season with {CARRYOVER:.0%} carryover when that season's file exists.",
             "- **always_home:** pick home every game.",
             "- **better_record:** pick the better season-to-date win % (ties -> home) once both teams have 5+ games.",
             "- **rest_advantage:** pick the team with 1+ more day of rest (e.g. opponent on a back-to-back).",
             "- **elo_pick:** pick the Elo favourite.",
             "- **model_edge_5pct:** needs moneylines; pick where Elo prob beats de-vigged market prob by 5+ pts.",
             ""]
    for result in results:
        lines += TableLines(result)
    path = os.path.join(REPORT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def Main() -> None:
    parser = argparse.ArgumentParser(description="NBA pick / moneyline backtest")
    parser.add_argument("--season", default="2024", help="comma-separated start years (2024 = 2024-25)")
    parser.add_argument("--games-file", help="backtest one JSON games file instead (e.g. examples/games_sample.json)")
    args = parser.parse_args()
    if args.games_file:
        games = LoadJson(args.games_file)
        if not games:
            raise SystemExit(f"no games in {args.games_file}")
        results = [RunOne(os.path.splitext(os.path.basename(args.games_file))[0], games, [])]
    else:
        results = []
        for season in (int(s) for s in args.season.split(",") if s.strip()):
            games = LoadJson(os.path.join(SEASON_DIR, f"games_{season}.json"))
            if not games:
                raise SystemExit(f"missing data/season/games_{season}.json - run GetSeasonGames.py --season {season} first")
            results.append(RunOne(str(season), games, LoadJson(os.path.join(SEASON_DIR, f"games_{season - 1}.json"))))
    for result in results:
        print("\n".join(TableLines(result)))
    print(f"Report: {WriteMarkdown(results)}")


if __name__ == "__main__":
    Main()
