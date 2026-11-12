r"""
Informational matchup storylines for one slate (NOT predictions for the log; never writes data/predictions).

    .venv\Scripts\python.exe -m Matchups.Tonight --date 2026-11-12 --games TOR@PHI DET@WAS DAL@MIN CHA@PHX
"""

from Matchups.Cpu import LimitThreads, LowerPriority

LimitThreads()

import argparse
import json
import math
import os
from itertools import groupby

import numpy as np

from Backtest.Model import EloModel, SeasonFinalRatings
from Matchups import Data, Scoring
from Matchups.RunAll import RUN_DIR
from Matchups.Simulate import SimulateGame
from Matchups.WalkForward import LoadAll, PairShape, Walk


def EloNow(season, before_date):
    prior = Data.AllGames(season - 1)
    model = EloModel(SeasonFinalRatings(prior) if prior else None)
    for g in sorted((g for g in Data.AllGames(season) if g["game_date"] < before_date), key=lambda g: (g["game_date"], g["game_id"])):
        model.Apply(g)
    return model


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", required=True)
    parser.add_argument("--games", nargs="+", required=True)
    parser.add_argument("--report-json", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sims", type=int, default=50000)
    args = parser.parse_args()
    LowerPriority()
    rep = json.load(open(args.report_json, encoding="utf-8"))
    run = json.load(open(os.path.join(RUN_DIR, "run_2026-11-12.json"), encoding="utf-8"))
    games, players, poss = LoadAll()
    games = [g for g in games if g["game_date"] < args.date]
    data = (games, players, poss)
    def Cfg(name):
        fam, *parts = name.split("_")
        vals = {p[0]: float(p[1:]) for p in parts}
        if fam == "rapm":
            return {"model": "rapm", "lam": vals["l"], "half_life": vals["h"], "name": name}
        return {"model": "matchup", "target": "team" if fam == "matchupteam" else "player",
                "k": vals["k"], "half_life": vals["h"], "name": name}
    rapm = Walk(Cfg(run["best"]["rapm"]), data=data)
    mp = Walk(Cfg(run["best"]["matchup"]), data=data)           # offender's own points: the storyline view
    mt = Walk(Cfg(run["best"]["matchupteam"]), data=data)       # team points: the stronger predictor
    for res in (rapm, mp, mt):
        res["model"].Fit()
    book = mt["book"]
    elo = EloNow(2026, args.date)
    w_stack = np.array(rep["weights"]["stack_elo_m1_pregame"])
    sigma, h = run["best_sim"]
    last_game = max(g["game_date"] for g in games)
    name = lambda p: book.names.get(p) or str(p)
    lines = [f"Data through {last_game} (games before {args.date}). Rosters = each team's most recent game; "
             "tonight's injury report is NOT applied. Informational only: not logged, not a pick, no market comparison.", ""]
    summary = []
    for spec in args.games:
        away, home = spec.split("@")
        roster = {"home": book.PregameRoster(home, args.date), "away": book.PregameRoster(away, args.date)}
        pace = 0.5 * (book.Pace(home) + book.Pace(away))
        th, thp = mt["model"].ExpectedPoints(roster["home"], roster["away"], pace, detail=True)
        ta, tap = mt["model"].ExpectedPoints(roster["away"], roster["home"], pace, detail=True)
        rm = rapm["model"].PredictMargin(roster["home"], roster["away"])
        pe = elo.HomeWinProb(home, away)
        p_stack = float(Scoring.PredictLogistic(w_stack, np.array([[float(Scoring.Logit(pe)), rm]]))[0])
        inp = {"pace": pace, "ratio": mt["model"].AttributionRatio(),
               "home": np.array([[p, lam, PairShape(mt["model"], o, d)] for o, d, p, lam in thp]),
               "away": np.array([[p, lam, PairShape(mt["model"], o, d)] for o, d, p, lam in tap])}
        sim = SimulateGame(inp, [(sigma, h)], args.sims, seed=sum(map(ord, spec)), total_scale=run["total_scale"])[0]
        summary.append({"game": spec, "elo_home": pe, "stack_elo_m1_home": p_stack, "sim_home": sim["p_home"],
                        "rapm_margin": rm, "team_matchup_margin": th - ta, "sim": sim})
        lines += [f"### {away} @ {home}", "",
                  f"- Home ({home}) win prob: Elo {pe:.3f}; Elo+RAPM stack {p_stack:.3f}; matchup sim {sim['p_home']:.3f}.",
                  f"- Home margin: RAPM {rm:+.1f}; team-matchup model {th - ta:+.1f}; sim 10-90% {sim['margin_q10']:+.0f} to {sim['margin_q90']:+.0f}.",
                  f"- Total: sim mean {sim['total_mean']:.0f} (10-90% {sim['total_q10']:.0f} to {sim['total_q90']:.0f}).", ""]
        for off_side, def_side in (("home", "away"), ("away", "home")):
            off_team, def_team = (home, away) if off_side == "home" else (away, home)
            _, pairs = mp["model"].ExpectedPoints(roster[off_side], roster[def_side], pace, detail=True)
            top = sorted(pairs, key=lambda q: -q[2])[:4]
            lines += [f"{off_team} offense vs {def_team} defense (top pairings by expected possessions):", "",
                      "| Offender | Defender | Share of his poss | Exp poss | His pts/poss vs this D | His usual | Read |",
                      "|---|---|---|---|---|---|---|"]
            for o, d, p, lam in top:
                o_poss = roster[off_side][o] / 48.0 * pace
                base = mp["model"].OffenseRate(o)
                read = "offense" if lam > base * 1.03 else "defense" if lam < base * 0.97 else "even"
                lines.append(f"| {name(o)} | {name(d)} | {p / o_poss:.0%} | {p:.1f} | {lam:.3f} | {base:.3f} | {read} ({lam / base - 1:+.0%}) |")
            lines.append("")
    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    json.dump(summary, open(args.out.replace(".md", ".json"), "w", encoding="utf-8"), indent=1, default=float)
    print("\n".join(lines))


if __name__ == "__main__":
    Main()
