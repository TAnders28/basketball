r"""Leak-free pregame stack_global probabilities for every game from 2024-11-15 (Round 8 anchor).

One chronological walk (Backtest.StackDay.WalkMembers: each member predicts a game before seeing its result), Elo
phase-1 history, then Matchups.V3.Post.ContextStack(contextual=False): weekly refits on the trailing 365 days of games
dated strictly before each day. Same members and math as PredictDay's stack_global. Seasons 2024-25, 2025-26, 2026-27.
Writes data\inplay\stack_global.json. Does not touch data\predictions.

    .venv\Scripts\python.exe -m InPlay.StackWalk
"""

import json
import os
import sys
import time

from Backtest.StackDay import RECORD_FROM, EloHistory, MemberNames, WalkMembers
from Matchups.V2.Base import REGISTRY, LoadGames
from Matchups.V3 import Post as PP

SEASONS = [2024, 2025, 2026]


def Main():
    t0 = time.time()
    games = sorted(LoadGames(SEASONS), key=lambda g: (g["game_date"], g["game_id"]))
    approaches = {n: REGISTRY[n]() for n in MemberNames()}
    print(f"{len(games)} games, members {sorted(approaches)}", flush=True)
    walked = WalkMembers(approaches, games, record_from=RECORD_FROM, progress=lambda m: print(f"{time.time()-t0:.0f}s {m}", flush=True))
    elo = EloHistory(SEASONS, RECORD_FROM, "2099-01-01")
    rows = dict(walked["rows"])
    rows["elo_phase1"] = elo["rows"]
    out, names, _, _ = PP.ContextStack(rows, {}, contextual=False)
    res = [dict(game_id=r["game_id"], game_date=r["game_date"], season=r["season"], home=r["home"], away=r["away"],
                p_home=round(float(r["p_home"]), 5), margin_mean=float(r["margin_mean"]), home_win=r["home_win"]) for r in out]
    os.makedirs(os.path.join("data", "inplay"), exist_ok=True)
    with open(os.path.join("data", "inplay", "stack_global.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(members=names, seasons=SEASONS, record_from=RECORD_FROM, rows=res), fh)
    print(f"DONE {len(res)} rows in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    sys.exit(Main())
