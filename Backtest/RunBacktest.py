r"""
One command, from E:\Basketball\basketball:

    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2024
    .venv\Scripts\python.exe -m Backtest.RunBacktest --season 2023,2024
    .venv\Scripts\python.exe -m Backtest.RunBacktest --tune 2022,2023 --holdout 2024
    .venv\Scripts\python.exe -m Backtest.RunBacktest --games-file examples\games_sample.json   # smoke test

Reads data/season/games_<season>.json (from GetSeasonGames.py); seeds Elo from games_<season-1>.json
if it exists. Prints a per-season table and writes to data/backtest/reports/:
    report_<season>.json            summary per strategy
    picks_<season>_<strategy>.csv   every pick (and bankroll path when odds are present)
    REPORT.md                       plain-language report for all seasons run
    holdout_<holdout>.json          (holdout mode only) tune/select/freeze/score summary
"""

import argparse
import csv
import itertools
import json
import os

from Backtest.Engine import STAKE, STARTING_PURSE, RunSeason, Summarize
from Backtest.Model import CARRYOVER, HOME_COURT_ELO, K_FACTOR, SeasonFinalRatings
from Backtest.Strategies import (
    AlwaysHome, BetterRecord, EloPick, ModelEdge, RestAdvantage, DefaultStrategies,
)

SEASON_DIR = os.path.join("data", "season")
REPORT_DIR = os.path.join("data", "backtest", "reports")
# Absolute accuracy gain below this stops further grid search (0.5 percentage points).
MIN_GAIN = 0.005


def LoadJson(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def StrategiesFromSettings(settings: dict) -> list:
    return [
        AlwaysHome(),
        BetterRecord(min_games=settings["min_games"]),
        RestAdvantage(min_gap=settings["min_gap"]),
        EloPick(),
        ModelEdge(min_edge=settings["min_edge"]),
    ]


def SettingsGrid() -> list:
    """Small, fixed grid — count of settings tried is reported (may stop early)."""
    grid = []
    for min_games, min_gap, min_edge in itertools.product((3, 5, 8), (1, 2), (0.03, 0.05, 0.08)):
        grid.append({"min_games": min_games, "min_gap": min_gap, "min_edge": min_edge})
    return grid


def RunOne(label: str, games: list, prior_games: list, strategies: list = None) -> dict:
    prior = SeasonFinalRatings(prior_games) if prior_games else None
    strategies = strategies or DefaultStrategies()
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


def WriteMarkdown(results: list, holdout_note: str = None) -> str:
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
    if holdout_note:
        lines += ["## Holdout protocol", "", holdout_note, ""]
    for result in results:
        lines += TableLines(result)
    path = os.path.join(REPORT_DIR, "REPORT.md")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return path


def MeanAccuracy(result: dict, strategy_name: str = "elo_pick") -> float:
    s = result["strategies"].get(strategy_name) or {}
    return float(s.get("accuracy") or 0.0)


def CompositeTuneScore(results: list, names: tuple = ("better_record", "rest_advantage")) -> float:
    """Mean accuracy of tunable strategies across tune seasons.

    elo_pick / always_home do not depend on SettingsGrid knobs, so they are not the
    selection objective (they are still scored on holdout for baseline comparison).
    """
    if not results:
        return 0.0
    scores = []
    for r in results:
        for name in names:
            scores.append(MeanAccuracy(r, name))
    return sum(scores) / len(scores) if scores else 0.0


def WorstSeasonRank(holdout_result: dict) -> list:
    rows = []
    for name, s in holdout_result["strategies"].items():
        rows.append({"strategy": name, "accuracy": s.get("accuracy"), "picks": s.get("picks"),
                     "n_games_season": holdout_result["games"]})
    rows.sort(key=lambda r: (r["accuracy"] is None, r["accuracy"] if r["accuracy"] is not None else 0.0))
    for i, row in enumerate(rows, 1):
        row["rank_worst_first"] = i
    return rows


def ScoreTuneSeasons(tune_seasons: list, strategies: list) -> list:
    tune_results = []
    for season in tune_seasons:
        games = LoadJson(os.path.join(SEASON_DIR, f"games_{season}.json"))
        if not games:
            raise SystemExit(f"missing data/season/games_{season}.json")
        prior = LoadJson(os.path.join(SEASON_DIR, f"games_{season - 1}.json"))
        prior_ratings = SeasonFinalRatings(prior) if prior else None
        books = RunSeason(games, strategies, prior_ratings)
        tune_results.append({
            "season": str(season), "games": len(games),
            "strategies": {s.name: Summarize(books[s.name]) for s in strategies},
        })
    return tune_results



def RunHoldout(tune_seasons: list, holdout: int, second_look: int = None,
               min_gain: float = MIN_GAIN) -> dict:
    """Tune on tune_seasons with early stop when best-score gain < min_gain; freeze; score holdout once.
    Optionally score the same frozen settings on second_look (e.g. partial current season)."""
    grid = SettingsGrid()
    trials = []
    best = None
    stopped_early = False
    for settings in grid:
        strategies = StrategiesFromSettings(settings)
        tune_results = ScoreTuneSeasons(tune_seasons, strategies)
        score = CompositeTuneScore(tune_results)
        prev_best = None if best is None else best["tune_score"]
        gain = None if prev_best is None else (score - prev_best)
        trial = {"settings": settings, "tune_score": score, "tune_score_elo_pick": CompositeTuneScore(tune_results, ("elo_pick",)),
                 "gain_vs_best_before": gain,
                 "n_per_season": {r["season"]: r["games"] for r in tune_results}}
        trials.append(trial)
        if best is None or score > best["tune_score"]:
            improved_by = None if best is None else (score - best["tune_score"])
            best = {"settings": settings, "tune_score": score, "tune_results": tune_results,
                    "tune_score_elo_pick": trial["tune_score_elo_pick"]}
            # After the first setting, stop when a new best improves by under min_gain.
            if improved_by is not None and improved_by < min_gain:
                stopped_early = True
                break
        else:
            # No new best: if last few trials also failed to beat best by min_gain, stop.
            if len(trials) >= 4:
                recent = trials[-4:]
                if all((t["tune_score"] - best["tune_score"]) < min_gain for t in recent):
                    stopped_early = True
                    break

    frozen = StrategiesFromSettings(best["settings"])
    holdout_games = LoadJson(os.path.join(SEASON_DIR, f"games_{holdout}.json"))
    if not holdout_games:
        raise SystemExit(f"missing data/season/games_{holdout}.json")
    holdout_result = RunOne(str(holdout), holdout_games,
                            LoadJson(os.path.join(SEASON_DIR, f"games_{holdout - 1}.json")),
                            strategies=frozen)

    second_result = None
    if second_look is not None:
        second_games = LoadJson(os.path.join(SEASON_DIR, f"games_{second_look}.json"))
        if not second_games:
            raise SystemExit(f"missing data/season/games_{second_look}.json")
        second_result = RunOne(f"{second_look}_partial", second_games,
                               LoadJson(os.path.join(SEASON_DIR, f"games_{second_look - 1}.json")),
                               strategies=frozen)

    frozen_tune = []
    for season in tune_seasons:
        frozen_tune.append(RunOne(
            f"{season}_tune",
            LoadJson(os.path.join(SEASON_DIR, f"games_{season}.json")),
            LoadJson(os.path.join(SEASON_DIR, f"games_{season - 1}.json")),
            strategies=frozen,
        ))

    seasons_for_n = list(tune_seasons) + [holdout]
    if second_look is not None:
        seasons_for_n.append(second_look)
    payload = {
        "mode": "holdout",
        "tune_seasons": tune_seasons,
        "holdout_season": holdout,
        "second_look_season": second_look,
        "settings_tried": len(trials),
        "settings_grid_size": len(grid),
        "stopped_early": stopped_early,
        "min_gain": min_gain,
        "selection_metric": "mean_accuracy(better_record, rest_advantage) over tune seasons",
        "selected_settings": best["settings"],
        "tune_score": best["tune_score"],
        "tune_score_elo_pick": best["tune_score_elo_pick"],
        "n_per_season": {str(s): len(LoadJson(os.path.join(SEASON_DIR, f"games_{s}.json")))
                         for s in seasons_for_n},
        "worst_season_rank_on_holdout": WorstSeasonRank(holdout_result),
        "holdout": holdout_result,
        "second_look": second_result,
        "baseline_holdout": {
            "always_home": holdout_result["strategies"].get("always_home"),
            "better_record": holdout_result["strategies"].get("better_record"),
            "elo_pick": holdout_result["strategies"].get("elo_pick"),
        },
        "trials": trials,
    }
    os.makedirs(REPORT_DIR, exist_ok=True)
    path = os.path.join(REPORT_DIR, f"holdout_{holdout}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    note = (f"Tuned on seasons {tune_seasons} ({len(trials)} settings tried"
            f"{', stopped early' if stopped_early else ''} of {len(grid)}; min_gain={min_gain}); "
            f"selection={payload['selection_metric']}; froze {best['settings']}; "
            f"scored once on holdout {holdout}"
            + (f"; second look {second_look}" if second_look is not None else "")
            + f". N/season={payload['n_per_season']}. See holdout_{holdout}.json.")
    md_results = frozen_tune + [holdout_result]
    if second_result is not None:
        md_results.append(second_result)
    WriteMarkdown(md_results, holdout_note=note)
    return payload


def Main() -> None:
    parser = argparse.ArgumentParser(description="NBA pick / moneyline backtest")
    parser.add_argument("--season", default="2024", help="comma-separated start years (2024 = 2024-25)")
    parser.add_argument("--games-file", help="backtest one JSON games file instead (e.g. examples/games_sample.json)")
    parser.add_argument("--tune", help="comma-separated seasons used only to select settings (holdout mode)")
    parser.add_argument("--holdout", type=int, help="single season scored once after freeze (holdout mode)")
    parser.add_argument("--second-look", type=int, default=None,
                        help="optional extra season scored once with frozen settings (e.g. partial current)")
    parser.add_argument("--min-gain", type=float, default=MIN_GAIN,
                        help="stop search when elo_pick tune-score gain is below this (default 0.005 = 0.5 pts)")
    args = parser.parse_args()

    if args.tune or args.holdout:
        if not args.tune or args.holdout is None:
            raise SystemExit("holdout mode needs both --tune SEASONS and --holdout SEASON")
        tune_seasons = [int(s) for s in args.tune.split(",") if s.strip()]
        payload = RunHoldout(tune_seasons, args.holdout, second_look=args.second_look, min_gain=args.min_gain)
        print(f"holdout mode: settings_tried={payload['settings_tried']}/{payload['settings_grid_size']} "
              f"stopped_early={payload['stopped_early']} selected={payload['selected_settings']} "
              f"n_per_season={payload['n_per_season']}")
        print("\n".join(TableLines(payload["holdout"])))
        if payload.get("second_look"):
            print("\n".join(TableLines(payload["second_look"])))
        print("worst-season rank (holdout, worst first):")
        for row in payload["worst_season_rank_on_holdout"]:
            acc = f"{row['accuracy'] * 100:.1f}%" if row["accuracy"] is not None else "n/a"
            print(f"  #{row['rank_worst_first']} {row['strategy']}: {acc} (picks={row['picks']})")
        print(f"Report: {os.path.join(REPORT_DIR, 'REPORT.md')}")
        print(f"Holdout JSON: {os.path.join(REPORT_DIR, f'holdout_{args.holdout}.json')}")
        return

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

