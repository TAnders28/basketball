r"""Round 2 (registered after the first holdout scoring; labelled as such in the registry).

A. maker2: rest the entry at the bid AND rest the exit at the target (both maker, Kalshi trade-through required).
   Stops/time exits stay taker. Kalshi only (needs trade highs/lows). Same frozen grid rows that have a target.
B. cell_surface: a Bayesian conditional-calibration surface. On the tuning split only, for each cell
   (side price bucket x game phase x side's score margin x pregame-favorite flag) take the first bar per game in the cell,
   shrink the win rate toward the cell's mean price with a Beta prior (strength K), and keep cells where the posterior
   probability that win rate > price + execution cost exceeds 0.90. On later games: first time a side enters a kept cell,
   buy at the next minute's ask and hold to settlement.

    .venv\Scripts\python.exe -m InPlay.Round2 --workers 6
"""

import argparse
import json
import math
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from InPlay.Engine import SideMid
from InPlay.Execution import Fee, TakerBuy
from InPlay.Grid import TrialId, Variants
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import (BASES, AppendMd, AppendTrials, CsvRows, Metrics, Now, SaveRes, _Init, _Run, _Short)

K_PRIOR = 30.0


def Phase(el):
    return "Q1" if el <= 720 else "Q2" if el <= 1440 else "Q3" if el <= 2160 else "Q4a" if el <= 2520 else "Q4b" if el <= 2880 else None


def Margin(sd_side):
    return "T10+" if sd_side <= -10 else "T4-9" if sd_side <= -4 else "E" if sd_side <= 3 else "L4-9" if sd_side <= 9 else "L10+"


def Cell(g, i, side):
    m = g.mid[i]
    if m is None:
        return None
    ph = Phase(g.el[i])
    if ph is None:
        return None
    p = SideMid(m, side)
    if p < 0.05 or p > 0.95:
        return None
    b = min(int(p / 0.05), 18)
    fav = int((g.pre >= 0.5) == (side == 1))
    return (b, ph, Margin(g.sd[i] * side), fav)


def Cost(p):
    return 0.005 + 0.005 + Fee(p, "kalshi_taker")  # half spread + slippage + Kalshi taker fee (hold to settle)


def FitSurface(games):
    stats = defaultdict(lambda: [0, 0.0, 0.0])  # n, sum price, sum win
    for g in games:
        seen = set()
        for i in range(g.n):
            for side in (1, -1):
                c = Cell(g, i, side)
                if c is None or c in seen:
                    continue
                seen.add(c)
                y = g.won if side == 1 else 1 - g.won
                s = stats[c]
                s[0] += 1
                s[1] += SideMid(g.mid[i], side)
                s[2] += y
    keep = {}
    for c, (n, sp, sy) in stats.items():
        if n < 15:
            continue
        pbar = sp / n
        a, b = K_PRIOR * pbar + sy, K_PRIOR * (1 - pbar) + (n - sy)
        mean = a / (a + b)
        sd = math.sqrt(a * b / ((a + b) ** 2 * (a + b + 1)))
        thr = pbar + Cost(pbar)
        prob = 0.5 * (1 + math.erf((mean - thr) / (sd * math.sqrt(2))))
        if prob > 0.90:
            keep[c] = dict(n=n, price=pbar, win=sy / n, post=mean, prob=prob)
    return keep, len(stats)


def TradeSurface(games, keep, slip=0.005):
    trades = []
    for g in games:
        done = set()
        for i in range(g.n - 1):
            for side in (1, -1):
                if side in done:
                    continue
                c = Cell(g, i, side)
                if c is None or c not in keep:
                    continue
                px = TakerBuy(g.bid[i + 1], g.ask[i + 1], side, slip)
                if px is None:
                    continue
                done.add(side)
                won = g.won if side == 1 else 1 - g.won
                trades.append({"gid": g.gid, "day": g.day, "season": g.season, "side": side, "entry": px, "exit": float(won),
                               "settled": True, "why": "settle", "el": g.el[i], "maker": False, "split": SplitOf(g.season)})
    return trades


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)
    md, rows, trials = ["## Round 2 (registered after first holdout scoring)", ""], [], []
    # ---- A. maker on both legs
    fams = ("dip_fav", "dip_any", "half_reset", "anchor", "comeback")
    grid = [(f, P) for f, P in Variants() if f in fams and (P.get("tgt") is not None or P.get("tgt_d") is not None)]
    tune = {}
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), initializer=_Init) as ex:
        for (fam, P, venue, xm, _), trades, gsp, dsp in ex.map(_Run, [(f, P, "kalshi", "maker2", ("tune",)) for f, P in grid]):
            m = Metrics(trades, gsp.get("tune", 0), dsp.get("tune", []), "kalshi_taker")
            tid = TrialId(fam, P, "kalshi", "maker2")
            tune[tid] = (fam, P, m)
            trials.append(dict(trial_id=tid, created_at=Now(), family=fam, params=P, venue="kalshi", exec="maker2",
                               tag="round2_after_first_holdout_scoring", split_scored="tune", tune={"kalshi_taker": m}))
    best = {}
    for tid, (fam, P, m) in tune.items():
        if m["n_trades"] >= 10 and (fam not in best or m["pnl_per_trade"] > best[fam][2]["pnl_per_trade"]):
            best[fam] = (tid, P, m)
    md += [f"### A. Maker on both legs (Kalshi): {len(grid)} variants on tune, best per family scored once", "",
           "| family | rule | split | trades | P/L per trade kalshi fees [95% CI] | P/L per trade Tim est. | hit |", "|---|---|---|---|---|---|---|"]
    with ProcessPoolExecutor(max_workers=min(args.workers, 10), initializer=_Init) as ex:
        jobs = [(fam, P, "kalshi", "maker2", ("tune", "holdout", "forward")) for fam, (tid, P, m) in best.items()]
        for (fam, P, venue, xm, _), trades, gsp, dsp in ex.map(_Run, jobs):
            tid = TrialId(fam, P, "kalshi", "maker2")
            res = {}
            for sp in ("tune", "holdout", "forward"):
                tr = [t for t in trades if t["split"] == sp]
                res[sp] = {b: Metrics(tr, gsp.get(sp, 0), dsp.get(sp, []), b) for b in BASES}
                k, t = res[sp]["kalshi_taker"], res[sp]["tim_est_0.02"]
                if k["n_trades"]:
                    md.append(f"| {fam} | {_Short(P)} | {sp} | {k['n_trades']} | {k['pnl_per_trade']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] | {t['pnl_per_trade']:+.4f} | {k['hit']:.2f} |")
                for b in BASES:
                    m = res[sp][b]
                    rows.append(dict(section="round2_maker2", venue="kalshi", split=sp, family=fam, trial_id=tid, fee_basis=b,
                                     metric="pnl_per_trade", value=m["pnl_per_trade"], lo=m["lo"], hi=m["hi"], n=m["n_trades"], label=_Short(P)))
            trials.append(dict(trial_id=tid, created_at=Now(), family=fam, params=P, venue="kalshi", exec="maker2",
                               tag="round2_selected_on_tune", split_scored="tune+holdout+forward", res=res))
    md.append("")
    # ---- B. Bayesian calibration surface
    G = LoadGames()
    tune_games = [g for g in G["pm"] + G["kalshi"] if SplitOf(g.season) == "tune"]
    keep, n_cells = FitSurface(tune_games)
    md += [f"### B. Bayesian calibration surface (cells fitted on {len(tune_games)} tune game-venue series; {n_cells} cells seen, "
           f"{len(keep)} kept with P(win rate > price + cost) > 0.90 under a Beta prior of strength {K_PRIOR:.0f} centred on price)", ""]
    for c, v in sorted(keep.items(), key=lambda kv: -kv[1]["prob"])[:15]:
        md.append(f"- cell price {c[0]*0.05:.2f}-{c[0]*0.05+0.05:.2f}, {c[1]}, margin {c[2]}, pregame fav={c[3]}: n={v['n']}, price {v['price']:.3f}, "
                  f"win {v['win']:.3f}, posterior {v['post']:.3f}, P={v['prob']:.2f}")
    md += ["", "| venue | split | trades | P/L per trade kalshi fees [95% CI] | P/L per trade Tim est. | hit |", "|---|---|---|---|---|---|"]
    res_b = {}
    for venue in ("kalshi", "pm"):
        trades = TradeSurface(G[venue], keep)
        res_b[venue] = {}
        for sp in ("tune", "holdout", "forward"):
            gs = [g for g in G[venue] if SplitOf(g.season) == sp]
            tr = [t for t in trades if t["split"] == sp]
            res_b[venue][sp] = {b: Metrics(tr, len(gs), sorted({g.day for g in gs}), b) for b in BASES}
            k, t = res_b[venue][sp]["kalshi_taker"], res_b[venue][sp]["tim_est_0.02"]
            if k["n_trades"]:
                md.append(f"| {venue} | {sp} | {k['n_trades']} | {k['pnl_per_trade']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] | {t['pnl_per_trade']:+.4f} | {k['hit']:.2f} |")
            for b in BASES:
                m = res_b[venue][sp][b]
                rows.append(dict(section="round2_cell_surface", venue=venue, split=sp, family="cell_surface", fee_basis=b,
                                 metric="pnl_per_trade", value=m["pnl_per_trade"], lo=m["lo"], hi=m["hi"], n=m["n_trades"],
                                 label=f"K={K_PRIOR};p>0.90;cells={len(keep)}"))
        trials.append(dict(trial_id=TrialId("cell_surface", {"K": K_PRIOR, "p": 0.9}, venue, "taker"), created_at=Now(),
                           family="cell_surface", params={"K": K_PRIOR, "prob": 0.9, "cells_kept": len(keep)}, venue=venue,
                           exec="taker", tag="round2_single_preset", split_scored="tune+holdout+forward", res=res_b[venue]))
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round2", dict(best={k: [v[0], v[1]] for k, v in best.items()}, cells={str(k): v for k, v in keep.items()}, surface=res_b))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())