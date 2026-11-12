r"""
Per-game derived cache ("reanalysis" snapshot, reading suggestion 11): possessions, stints, coaching events and box
matchups computed once per game from PBP + rotation, stored under data/matchups2/derived/<version>/<season>/.
Everything in it is a fact of that game only, so the walk still sees it only after the game's day.

  python -m Matchups.V3.Derive --seasons 2024 2025 2026 --workers 6
"""

import argparse
import os
import pickle
import time

from Matchups import Cpu
from Matchups.V2 import DataLayer as D
from Matchups.V2 import Pbp
from Matchups.V2.Base import DerivedPath, LoadGames


def One(g):
    s, gid = g["season"], g["game_id"]
    path = DerivedPath(s, gid)
    if os.path.exists(path):
        return "cached"
    rot, pbp = D.Rotation(s, gid), D.Pbp(s, gid)
    if not rot or not pbp:
        return "no_pbp"
    poss = Pbp.BuildPossessions(pbp, rot, g["home_team"], g["away_team"])
    stints = Pbp.BuildStints(pbp, rot, g["home_team"], g["away_team"], poss)
    events = Pbp.GameEvents(pbp, g["home_team"], g["away_team"])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as fh:
        pickle.dump({"has_pbp": True, "possessions": poss, "stints": stints, "events": events,
                     "matchups": D.BoxMatchups(s, gid)}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)
    return "built"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", nargs="+", type=int, default=[2024, 2025, 2026])
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    games = LoadGames(a.seasons)
    t0 = time.time()
    Cpu.LowerPriority()
    from collections import Counter
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(min(a.workers, 10), initializer=Cpu.WorkerInit) as ex:
        c = Counter(ex.map(One, games, chunksize=20))
    print(dict(c), f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
