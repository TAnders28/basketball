r"""
Full matchup-model research run (grid -> select on tune season -> simulate -> score holdout once).

    .venv\Scripts\python.exe -m Matchups.RunAll [--workers N] [--sims 20000]

CPU cap: at most floor(cores/2) worker processes, 1 BLAS thread each, BelowNormal priority.
Splits: 2023-24 burn-in (not scored), 2024-25 TUNE, 2025-26 HOLDOUT (scored once), 2026-27 to date FORWARD.
"""

from Matchups.Cpu import LimitThreads, LowerPriority, MaxWorkers, WorkerInit

LimitThreads()

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from itertools import groupby
from multiprocessing import Pool

import numpy as np

from Backtest.Model import EloModel, SeasonFinalRatings
from Matchups import Data, Scoring
from Matchups.Simulate import SimulateChunk
from Matchups.WalkForward import LoadAll, Walk

TUNE, HOLDOUT, FORWARD = 2024, 2025, 2026
RUN_DIR = os.path.join("data", "matchups", "runs")
RAPM_GRID = [{"model": "rapm", "lam": lam, "half_life": hl, "name": f"rapm_l{lam}_h{hl}"}
             for lam in (3, 10, 30, 100) for hl in (120, 240, 480)]
MATCHUP_GRID = [{"model": "matchup", "target": tgt, "k": k, "half_life": hl,
                 "name": f"{'matchup' if tgt == 'player' else 'matchupteam'}_k{k}_h{hl}"}
                for tgt in ("player", "team") for k in (20, 60, 150, 400, 1000) for hl in (90, 180, 365)]
SIM_GRID = [(s, h) for s in (0.0, 0.03, 0.06, 0.09, 0.12, 0.15) for h in (0.0, 0.012, 0.024)]
LOG_PATH = os.path.join("logs", f"{datetime.now():%Y-%m-%d}_matchups_run.log")


def Log(msg):
    line = f"{datetime.now().astimezone().isoformat(timespec='seconds')} {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def EloProbs(seasons) -> dict:
    """elo_predictday_v1 replica: prior-season seed, day barrier, all game types fed in."""
    out = {}
    for s in seasons:
        prior = Data.AllGames(s - 1)
        model = EloModel(SeasonFinalRatings(prior) if prior else None)
        games = sorted(Data.AllGames(s), key=lambda g: (g["game_date"], g["game_id"]))
        for _d, day in groupby(games, key=lambda g: g["game_date"]):
            day = list(day)
            for g in day:
                out[g["game_id"]] = model.HomeWinProb(g["home_team"], g["away_team"])
            for g in day:
                model.Apply(g)
    return out


_DATA = None


def RunJob(cfg):
    global _DATA
    if _DATA is None:
        _DATA = LoadAll()
    res = Walk(cfg, data=_DATA)
    rows = [{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in r.items()} for r in res["rows"]]
    return cfg, rows, res["secs"]


def Column(rows, key):
    return np.array([np.nan if r.get(key) is None else float(r[key]) for r in rows])


def TuneScore(rows, feature):
    tune = [r for r in rows if r["season"] == TUNE and r.get(feature) is not None]
    x, y = Column(tune, feature), Column(tune, "home_win")
    w = Scoring.FitLogistic(x[:, None], y)
    return float(Scoring.LogLoss(Scoring.PredictLogistic(w, x[:, None]), y).mean()), len(tune)


def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--sims", type=int, default=20000)
    args = parser.parse_args()
    LowerPriority()
    os.makedirs(RUN_DIR, exist_ok=True)
    workers = MaxWorkers(args.workers)
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    Log(f"START python {sys.version.split()[0]} head {head} args {vars(args)} cores {os.cpu_count()} workers {workers} "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')}")
    t_all = time.time()
    elo = EloProbs([2023, TUNE, HOLDOUT, FORWARD])

    # ---- phase 1: bounded grid (9 RAPM + 12 matchup configs), walk-forward each ----
    jobs = RAPM_GRID + MATCHUP_GRID
    results = {}
    with Pool(workers, initializer=WorkerInit) as pool:
        for cfg, rows, secs in pool.imap_unordered(RunJob, jobs):
            results[cfg["name"]] = (cfg, rows)
            Log(f"grid {cfg['name']} done in {secs:.0f}s ({len(results)}/{len(jobs)})")
    tune_scores = {}
    for name, (cfg, rows) in results.items():
        tune_scores[name] = TuneScore(rows, "pred_margin_pregame")
    family = lambda n: n.split("_")[0]
    best = {fam: min((n for n in results if family(n) == fam), key=lambda n: tune_scores[n][0])
            for fam in ("rapm", "matchup", "matchupteam")}
    sim_family = min(("matchup", "matchupteam"), key=lambda f: tune_scores[best[f]][0])
    Log(f"tune selection {best} scores { {k: tune_scores[v] for k, v in best.items()} }")

    # ---- phase 2: simulator on the selected matchup config ----
    mm = Walk(results[best[sim_family]][0], record_sim=True)
    Log(f"simulating on {best[sim_family]}")
    sim_inputs = mm["sim_inputs"]
    tune_rows = [r for r in mm["rows"] if r["season"] == TUNE and r.get("pred_total_pregame") is not None]
    total_scale = float(np.mean([r["total"] for r in tune_rows]) / np.mean([r["pred_total_pregame"] for r in tune_rows]))
    items = list(sim_inputs.items())
    chunks = [items[i::workers * 4] for i in range(workers * 4)]
    t0 = time.time()
    sims = {}
    with Pool(workers, initializer=WorkerInit) as pool:
        for i, part in enumerate(pool.imap_unordered(SimulateChunk, [(c, SIM_GRID, args.sims, total_scale) for c in chunks])):
            sims.update(part)
            if (i + 1) % workers == 0:
                Log(f"sim chunks {i + 1}/{len(chunks)} ({time.time() - t0:.0f}s)")
    Log(f"sim done: {len(sims)} game-variants x {len(SIM_GRID)} configs x {args.sims} sims in {time.time() - t0:.0f}s")

    # ---- assemble per-game table ----
    table = {}
    player_rows = {r["game_id"]: r for r in results[best["matchup"]][1]}
    team_rows = {r["game_id"]: r for r in results[best["matchupteam"]][1]}
    for r in mm["rows"]:
        pr, tr = player_rows.get(r["game_id"], {}), team_rows.get(r["game_id"], {})
        table[r["game_id"]] = {k: r[k] for k in ("game_id", "game_date", "season", "home_team", "away_team", "home_win", "margin", "total")}
        table[r["game_id"]]["elo"] = elo.get(r["game_id"])
        for v in ("pregame", "oracle"):
            table[r["game_id"]][f"mm_margin_{v}"] = pr.get(f"pred_margin_{v}")
            table[r["game_id"]][f"mmt_margin_{v}"] = tr.get(f"pred_margin_{v}")
            table[r["game_id"]][f"sim_total_{v}"] = r.get(f"pred_total_{v}")
        for key in ("assign_l1_full", "assign_l1_prior", "assign_l1_hist", "assign_poss", "pair_ll_pair", "pair_ll_offender", "pair_ll_league"):
            table[r["game_id"]][key] = pr.get(key) if key.startswith("pair") else r.get(key)
    for r in results[best["rapm"]][1]:
        if r["game_id"] in table:
            for v in ("pregame", "oracle"):
                table[r["game_id"]][f"rapm_margin_{v}"] = r.get(f"pred_margin_{v}")
    sim_tune = {}
    for ci, cfg in enumerate(SIM_GRID):
        ps, ys = [], []
        for gid, row in table.items():
            out = sims.get((gid, "pregame"))
            if row["season"] == TUNE and out:
                ps.append(out[ci]["p_home"])
                ys.append(row["home_win"])
        sim_tune[ci] = float(Scoring.LogLoss(ps, ys).mean())
    best_sim = min(sim_tune, key=sim_tune.get)
    Log(f"sim selection {SIM_GRID[best_sim]} tune ll {sim_tune[best_sim]:.4f}")
    for gid, row in table.items():
        for v in ("pregame", "oracle"):
            out = sims.get((gid, v))
            if out:
                o = out[best_sim]
                row[f"sim_p_{v}"] = o["p_home"]
                row[f"sim_margin_sd_{v}"] = o["margin_sd"]
                row[f"sim_total_q10_{v}"], row[f"sim_total_q90_{v}"] = o["total_q10"], o["total_q90"]
                row[f"sim_margin_q10_{v}"], row[f"sim_margin_q90_{v}"] = o["margin_q10"], o["margin_q90"]
    rows = sorted(table.values(), key=lambda r: (r["game_date"], r["game_id"]))
    json.dump({"rows": rows, "best": best, "sim_family": sim_family, "tune_scores": tune_scores, "sim_grid": SIM_GRID, "sim_tune": sim_tune,
               "best_sim": SIM_GRID[best_sim], "total_scale": total_scale, "workers": workers, "sims": args.sims,
               "cores": os.cpu_count(), "head": head, "secs": time.time() - t_all,
               "configs_tried": {"rapm": len(RAPM_GRID), "matchup_player_and_team": len(MATCHUP_GRID), "sim": len(SIM_GRID)},
               "grids": {"rapm": RAPM_GRID, "matchup": MATCHUP_GRID, "sim": SIM_GRID}},
              open(os.path.join(RUN_DIR, "run_2026-11-12.json"), "w", encoding="utf-8"))
    Log(f"END {time.time() - t_all:.0f}s")


if __name__ == "__main__":
    Main()
