r"""
One-command runner for V2 approaches (smoke now, full walk-forward backtests later once Tim approves).

  python -m Matchups.V2.Run --list
  python -m Matchups.V2.Run --approach kalman_team --seasons 2026 --max-days 5          # smoke
  python -m Matchups.V2.Run --approach all --seasons 2024 2025 2026 --record-from 2025-10-01 --workers 6   # later

Writes one CSV of per-game predictions per approach to logs/v2/<approach>_<tag>.csv and prints a short
summary. Single process by default; --workers runs approaches in parallel (capped at 10, 1 BLAS thread each,
BelowNormal priority via Matchups.Cpu).
"""

import argparse
import csv
import json
import os
import sys
import time

from Matchups import Cpu
import Matchups.V2.Approaches  # noqa: F401
from Matchups.V2.Base import REGISTRY, LoadGames, Walk
from Matchups.V2 import Diagnostics as G


def RunOne(args):
    name, seasons, record_from, max_days, tag, params = args
    t0 = time.time()
    games = LoadGames(seasons)
    rows = Walk(REGISTRY[name](**params), games, record_from=record_from, max_days=max_days)
    os.makedirs(os.path.join("logs", "v2"), exist_ok=True)
    path = os.path.join("logs", "v2", f"{name}_{tag}.csv")
    keys = sorted({k for r in rows for k in r if not isinstance(r[k], (list, dict))})
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    out = {"approach": name, "games": len(rows), "seconds": round(time.time() - t0, 1), "csv": path}
    ps = [(r["p_home"], r["home_win"]) for r in rows if r.get("p_home") is not None]
    out["non_finite"] = sum(1 for r in rows if "p_home" in r and not (r["p_home"] == r["p_home"]))
    if ps:
        out["p_home_min"] = round(min(p for p, _ in ps), 3)
        out["p_home_max"] = round(max(p for p, _ in ps), 3)
        if len(ps) >= 300:      # only meaningful on real backtests, never quoted from smoke runs
            out["log_loss"] = round(G.LogLoss(*zip(*ps)), 4)
            out["ece"] = round(G.Ece(*zip(*ps))[0], 4)
            mm = [(r["margin_mean"], r["margin_var"], r["margin"]) for r in rows if "margin_var" in r]
            out["coverage80"] = round(G.Coverage(*zip(*mm)), 3)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--approach", nargs="+", default=["kalman_team"])
    ap.add_argument("--seasons", nargs="+", type=int, default=[2026])
    ap.add_argument("--record-from", default=None)
    ap.add_argument("--max-days", type=int, default=None)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--tag", default="smoke")
    ap.add_argument("--params", default="{}", help="JSON dict of constructor params, e.g. prior knobs")
    a = ap.parse_args(argv)
    if a.list:
        for n, c in sorted(REGISTRY.items()):
            print(f"{n:20s} {c.idea}")
        return
    names = sorted(REGISTRY) if a.approach == ["all"] else a.approach
    jobs = [(n, a.seasons, a.record_from, a.max_days, a.tag, json.loads(a.params)) for n in names]
    Cpu.LowerPriority()
    if a.workers <= 1:
        Cpu.LimitThreads()
        results = [RunOne(j) for j in jobs]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(a.workers, Cpu.MaxWorkers(), 10), initializer=Cpu.WorkerInit) as ex:
            results = list(ex.map(RunOne, jobs))
    for r in results:
        print(json.dumps(r))


if __name__ == "__main__":
    main(sys.argv[1:])
