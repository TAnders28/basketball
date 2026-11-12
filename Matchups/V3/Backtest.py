r"""
V3 backtest orchestration (approved by Tim 2026-11-12). Stages, each resumable:

  python -m Matchups.V3.Backtest --stage base   --workers 6   # every approach (+ @oracle variants), one walk each
  python -m Matchups.V3.Backtest --stage sens   --workers 6   # 0.5x / 2x main-prior variants
  python -m Matchups.V3.Backtest --stage post                 # Elo baseline, EMOS, temperature, stacks
  python -m Matchups.V3.Backtest --stage report --out <dir>   # harness, registry, PBO, md + csv

Walk: seasons 2024 (warm-up) -> 2026, rows recorded from 2024-11-15 (2024-25 rows only train post-processing).
Selection split: 2025-26 regular season (2025-10-01 .. 2026-06-30). Forward test: 2026-27 games from 2026-10-20,
reported separately and never used to choose anything.
"""

import argparse
import json
import math
import os
import pickle
import time
from datetime import datetime

from Matchups import Cpu

Cpu.LimitThreads()          # before numpy loads: 1 BLAS thread here and in every worker

import numpy as np  # noqa: E402

ROOT = os.environ.get("V3_ROOT", os.path.join("logs", "v3"))
SMOKE = os.environ.get("V3_SMOKE") == "1"       # pipeline check only: 2 seasons, 4 approaches, separate ROOT
SEASONS = [2025, 2026] if SMOKE else [2024, 2025, 2026]
RECORD_FROM = "2025-11-15" if SMOKE else "2024-11-15"
SPLIT = ("2025-10-01", "2026-06-30")
FORWARD_FROM = "2026-10-20"
ROWS = os.path.join(ROOT, "rows")
LOG = os.path.join(ROOT, "backtest.log")
KEEP = ("game_id", "game_date", "season", "home", "away", "margin", "home_win", "margin_mean", "margin_var", "p_home",
        "term_dropped", "tau_scale", "tau_pair_hat", "home_scenarios", "away_scenarios")
SKIP = {"coach_profile"}           # minutes model: no margin distribution of its own (feeds lineup_vs_lineup)
ORACLE = ("player_kalman", "rapm_spm", "on_off", "stint_rapm")
# approach -> (constructor param, base value, how a prior-variance scale s maps onto it)
SENS = {"kalman_team": ("prior_var", 16.0, "var"), "schedule_fatigue": ("prior_scale", 1.0, "var"),
        "clutch": ("prior_scale", 1.0, "var"), "playtype_scheme": ("prior_scale", 1.0, "var"),
        "switch_hunt": ("prior_scale", 1.0, "var"), "four_factors": ("strength", 400.0, "strength"),
        "on_off": ("strength", 400.0, "strength"), "stint_rapm": ("tau", 3.0, "sd"), "pair_synergy": ("tau_pair", 1.0, "sd"),
        "scheme_assignment": ("conc", 20.0, "strength"), "player_kalman": ("prior_var", 9.0, "var"),
        "player_kalman_scen": ("prior_var", 9.0, "var"), "rapm_spm": ("tau", 3.0, "sd")}
COST = {"stacker": 9, "lineup_vs_lineup": 8, "pair_synergy": 7, "rapm_spm": 7, "size_speed": 5, "stint_rapm": 4}


def Log(msg):
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def Trial(name, params, oracle):
    tag = name + ("@oracle" if oracle else "")
    if params:
        tag += "[" + ",".join(f"{k}={v:g}" for k, v in sorted(params.items())) + "]"
    return tag


def RunJob(job):
    name, params, oracle = job
    import Matchups.V2.Approaches  # noqa: F401
    from Matchups.V2.Base import REGISTRY, LoadGames, Walk
    from Matchups.V3.Availability import OracleAvailability
    tag = Trial(name, params, oracle)
    path = os.path.join(ROWS, tag + ".pkl")
    if os.path.exists(path):
        return tag, "cached", 0.0
    t0 = time.time()
    games = LoadGames(SEASONS)
    rows = Walk(REGISTRY[name](**params), games, record_from=RECORD_FROM,
                availability=OracleAvailability if oracle else None)
    rows = [{k: r[k] for k in KEEP if k in r} for r in rows]
    os.makedirs(ROWS, exist_ok=True)
    with open(path + ".tmp", "wb") as fh:
        pickle.dump({"trial": tag, "name": name, "params": params, "oracle": oracle, "rows": rows}, fh)
    os.replace(path + ".tmp", path)
    return tag, "ok", time.time() - t0


def RunPool(jobs, workers):
    from concurrent.futures import ProcessPoolExecutor, as_completed
    jobs = sorted(jobs, key=lambda j: -COST.get(j[0], 1))
    Log(f"running {len(jobs)} jobs on {workers} workers")
    with ProcessPoolExecutor(min(workers, 10), initializer=Cpu.WorkerInit) as ex:
        futs = {ex.submit(RunJob, j): j for j in jobs}
        for f in as_completed(futs):
            try:
                tag, st, sec = f.result()
                Log(f"  {tag}: {st} {sec:.0f}s")
            except Exception as e:     # keep going; report failures
                Log(f"  FAILED {Trial(*futs[f])}: {type(e).__name__}: {e}")


def BaseJobs():
    import Matchups.V2.Approaches  # noqa: F401
    from Matchups.V2.Base import REGISTRY
    if SMOKE:
        return [("kalman_team", {}, False), ("player_kalman_scen", {}, False), ("schedule_fatigue", {}, False),
                ("rapm_spm", {}, False), ("player_kalman", {}, True)]
    jobs = [(n, {}, False) for n in sorted(REGISTRY) if n not in SKIP]
    jobs += [(n, {}, True) for n in ORACLE]
    return jobs


def SensJobs():
    jobs = []
    for n, (param, base, kind) in SENS.items():
        for s in (0.5, 2.0):
            v = base * s if kind == "var" else base * math.sqrt(s) if kind == "sd" else base / s
            jobs.append((n, {param: round(v, 6)}, False))
    return jobs


def LoadRows():
    out = {}
    for f in sorted(os.listdir(ROWS)):
        if f.endswith(".pkl"):
            with open(os.path.join(ROWS, f), "rb") as fh:
                d = pickle.load(fh)
            out[d["trial"]] = d
    return out


def Post():
    from Matchups.RunAll import EloProbs
    from Matchups.V2.Base import LoadGames
    from Matchups.V3 import Post as PP
    from Matchups.V3.Evaluate import EloRows
    runs = LoadRows()
    games = [g for g in LoadGames(SEASONS) if g["game_date"] >= RECORD_FROM]
    grow = [{"game_id": g["game_id"], "game_date": g["game_date"], "season": g["season"], "home": g["home_team"],
             "away": g["away_team"], "margin": g["home_score"] - g["away_score"],
             "home_win": int(g["home_score"] > g["away_score"])} for g in games]
    elo = EloRows(grow, EloProbs(SEASONS))
    post = {"elo_phase1": {"trial": "elo_phase1", "name": "elo_phase1", "params": {}, "oracle": False, "rows": elo, "kind": "baseline"}}
    base = {t: d for t, d in runs.items() if not d["params"] and not d["oracle"]}
    t0 = time.time()
    for t, d in base.items():
        post[t + "+emos"] = {"trial": t + "+emos", "name": d["name"], "params": {"post": "emos"}, "oracle": False,
                             "rows": PP.Emos(d["rows"]), "kind": "post"}
        post[t + "+temp"] = {"trial": t + "+temp", "name": d["name"], "params": {"post": "temperature"}, "oracle": False,
                             "rows": PP.Temperature(d["rows"]), "kind": "post"}
    Log(f"EMOS + temperature for {len(base)} approaches: {time.time() - t0:.0f}s")
    members = {t: d["rows"] for t, d in base.items()}
    members["elo_phase1"] = elo
    ctx = PP.GameContexts(LoadGames(SEASONS))
    info = {}
    for label, contextual in (("stack_ctx", True), ("stack_global", False)):
        t0 = time.time()
        rows, names, wg, wc = PP.ContextStack(members, ctx, contextual=contextual)
        post[label] = {"trial": label, "name": label, "params": {"members": len(names)}, "oracle": False, "rows": rows, "kind": "stack"}
        post[label + "+emos"] = {"trial": label + "+emos", "name": label, "params": {"post": "emos", "members": len(names)},
                                 "oracle": False, "rows": PP.Emos(rows), "kind": "stack"}
        info[label] = {"members": names, "w_global": [round(float(x), 4) for x in wg],
                       "w_context": {c: [round(float(x), 4) for x in w] for c, w in wc.items()}}
        Log(f"{label}: {time.time() - t0:.0f}s")
    with open(os.path.join(ROOT, "post.pkl"), "wb") as fh:
        pickle.dump({"post": post, "stack_info": info, "contexts": ctx}, fh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["base", "sens", "post", "report"])
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    Cpu.LowerPriority()
    Cpu.LimitThreads()
    if a.stage == "base":
        RunPool(BaseJobs(), a.workers)
    elif a.stage == "sens":
        RunPool(SensJobs(), a.workers)
    elif a.stage == "post":
        Post()
    else:
        from Matchups.V3.Report import Write
        Write(a.out)


if __name__ == "__main__":
    main()
