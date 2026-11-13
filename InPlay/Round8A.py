r"""Round 8A (9:09 PM steer; registered on tune before scoring). Our pregame stack_global as the in-play anchor.

Model: leak-free stack_global p(home) per game (InPlay.StackWalk; every member predicts before the result, stack weights
refit weekly on earlier games only). Market pregame price = venue price as of one minute before tip (no-shift data).
Disagreement d = model - market pregame. Only games with |d| >= 0.03 (the steer's threshold; trade variants raise it).
  * Drift test: signed first-half drift = sign(d) * (price at the first bar with el >= 1440 - price at the first in-play bar).
    Under a calibrated market it is 0; positive means the live price moves toward our pregame view. Also the final-outcome
    check sign(d) * (won - market pregame), and the market-vs-model log loss on the same games.
  * Trade (taker, 1-minute latency, 0.5c slippage): at tip, buy the side our model prefers relative to the market at the
    next bar's ask; exit at the first bar with el >= X at the bid (X = 720 end of Q1, 1440 halftime) or hold to settle.
    Threshold g in (0.03, 0.05, 0.08). Selected on tune (Polymarket 2024-25 from Nov 15; Kalshi tune is too thin), scored once
    on holdout (2025-26) and forward (2026-27 to Nov 11) on both venues with both fee bases. Conservative CI.

    .venv\Scripts\python.exe -m InPlay.Round8A
"""

import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from InPlay.Align import Team
from InPlay.CiAudit import Audit
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Stats import RatioBootstrap

SLIP = 0.005


def ModelMap(games_meta):
    d = json.load(open(os.path.join("data", "inplay", "stack_global.json"), encoding="utf-8"))
    by = {(r["game_date"], Team(r["home"]), Team(r["away"])): r["p_home"] for r in d["rows"]}
    out = {}
    for gid, date, h, a in zip(games_meta.game_id, games_meta.date_et, games_meta.home, games_meta.away):
        p = by.get((date, Team(h), Team(a)))
        if p is not None:
            out[str(gid)] = p
    return out, len(d["rows"])


def FirstAt(g, x, start=0):
    return next((k for k in range(start, g.n) if g.el[k] >= x and g.mid[k] is not None), None)


def Trade(g, side, X):
    e = 1
    if g.n < 3 or g.bid[e] is None or g.ask[e] is None:
        return None
    entry = (g.ask[e] if side == 1 else 1 - g.bid[e]) + SLIP
    if entry >= 0.99:
        return None
    if X is None:
        k = None
    else:
        k = next((j for j in range(e + 1, g.n) if g.el[j] >= X and g.bid[j] is not None and g.ask[j] is not None), None)
    if k is None:
        ex, st, why = float(g.won if side == 1 else 1 - g.won), True, "settle"
    else:
        ex, st, why = max(0.01, (g.bid[k] if side == 1 else 1 - g.ask[k]) - SLIP), False, "exit"
    return {"gid": g.gid, "day": g.day, "season": g.season, "side": side, "entry": entry, "exit": ex, "settled": st, "why": why,
            "el": g.el[e], "maker": False}


def Clus(vals):
    if not vals:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"))
    ks = sorted({k for k, _ in vals})
    ix = {k: i for i, k in enumerate(ks)}
    num, den = np.zeros(len(ks)), np.zeros(len(ks))
    for k, v in vals:
        num[ix[k]] += v
        den[ix[k]] += 1
    m, lo, hi = RatioBootstrap(num, den, reps=1000)
    return dict(n=len(vals), mean=m, lo=lo, hi=hi)


def Ll(p, y):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return -(y * math.log(p) + (1 - y) * math.log(1 - p))


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    G = LoadGames(os.path.join("data", "inplay", "games_obj_s0.pkl"))
    meta = pd.read_pickle(os.path.join("data", "inplay", "games_s0.pkl"))
    model, n_rows = ModelMap(meta)
    trials, rows, res = [], [], {}
    md = ["## Round 8A: our pregame stack_global as the in-play anchor (no-shift data; registered before scoring)", "",
          f"stack_global rows: {n_rows} games from 2024-11-15 (leak-free walk). Matched to in-play games: "
          + ", ".join(f"{v} {sum(1 for g in G[v] if str(g.gid) in model)}" for v in ("kalshi", "pm")) + ".", "",
          "### Does the live price drift toward our pregame view? (games with |model - market pregame| >= 0.03)", "",
          "| venue | split | games (of matched) | median abs disagreement | signed 1st-half drift [95% game CI] | signed (win - market pre) [CI] | log loss market / model / 50-50 blend |",
          "|---|---|---|---|---|---|---|"]
    for v in ("pm", "kalshi"):
        for sp in ("tune", "holdout", "forward"):
            gs = [g for g in G[v] if SplitOf(g.season) == sp and str(g.gid) in model and g.pre is not None]
            sel, drift, fin, ll = [], [], [], [[], [], []]
            for g in gs:
                p = model[str(g.gid)]
                d = p - g.pre
                ll[0].append(Ll(g.pre, g.won))
                ll[1].append(Ll(p, g.won))
                ll[2].append(Ll(0.5 * (p + g.pre), g.won))
                if abs(d) < 0.03:
                    continue
                s = 1 if d > 0 else -1
                a, b = FirstAt(g, 0.0), FirstAt(g, 1440.0)
                sel.append(abs(d))
                if a is not None and b is not None:
                    drift.append((g.gid, s * (g.mid[b] - g.mid[a])))
                fin.append((g.gid, s * (g.won - g.pre)))
            cd, cf = Clus(drift), Clus(fin)
            res[f"drift|{v}|{sp}"] = dict(drift=cd, final=cf, n_games=len(gs), n_sel=len(sel), ll=[float(np.mean(x)) if x else None for x in ll])
            rows.append(dict(section="round8A_anchor_drift", venue=v, split=sp, family="stack_anchor", fee_basis="none", metric="signed_drift_1h",
                             value=cd["mean"], lo=cd["lo"], hi=cd["hi"], n=cd["n"]))
            rows.append(dict(section="round8A_anchor_drift", venue=v, split=sp, family="stack_anchor", fee_basis="none", metric="signed_final_vs_pre",
                             value=cf["mean"], lo=cf["lo"], hi=cf["hi"], n=cf["n"]))
            if gs:
                md.append(f"| {v} | {sp} | {len(sel)} of {len(gs)} | {np.median(sel) if sel else float('nan'):.3f} | "
                          f"{cd['mean']:+.4f} [{cd['lo']:+.4f}, {cd['hi']:+.4f}] | {cf['mean']:+.4f} [{cf['lo']:+.4f}, {cf['hi']:+.4f}] | "
                          f"{np.mean(ll[0]):.4f} / {np.mean(ll[1]):.4f} / {np.mean(ll[2]):.4f} |")
    md += ["", "### Trade toward our pregame view (taker, 1-min latency)", "",
           "| rule | venue | split | fee | trades | P/L per trade [conservative 95% CI] |", "|---|---|---|---|---|---|"]
    grid = [dict(gap=g_, exit_el=X) for g_ in (0.03, 0.05, 0.08) for X in (720, 1440, None)]

    def Run(v, sp, P):
        out = []
        for g in G[v]:
            if SplitOf(g.season) != sp or str(g.gid) not in model or g.pre is None:
                continue
            d = model[str(g.gid)] - g.pre
            if abs(d) < P["gap"]:
                continue
            t = Trade(g, 1 if d > 0 else -1, P["exit_el"])
            if t:
                out.append(t)
        return out

    tune = []
    for P in grid:
        m = Audit(Run("pm", "tune", P), "kalshi_taker")
        tune.append((P, m))
        trials.append(dict(trial_id=TrialId("stack_anchor", P, "pm", "taker"), created_at=Now(), family="stack_anchor", params=P, venue="pm",
                           exec="taker", tag="round8A_registered_before_scoring", split_scored="tune", tune=m))
        if m:
            md.append(f"| {_Short(P)} | pm | tune | kalshi_taker | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] |")
    ok = [x for x in tune if x[1] and x[1]["n"] >= 30]
    P, _ = max(ok, key=lambda x: x[1]["mean"])
    for v in ("pm", "kalshi"):
        r = {}
        for sp in ("holdout", "forward"):
            tr = Run(v, sp, P)
            for b in ("kalshi_taker", "tim_est_0.02"):
                m = Audit(tr, b)
                r[f"{sp}|{b}"] = m
                if m:
                    md.append(f"| SELECTED {_Short(P)} | {v} | {sp} | {b} | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] |")
                    rows.append(dict(section="round8A_stack_anchor", venue=v, split=sp, family="stack_anchor", trial_id=TrialId("stack_anchor", P, v, "taker"),
                                     fee_basis=b, metric="pnl_per_trade", value=m["mean"], lo=m["cons_lo"], hi=m["cons_hi"], n=m["n"], label=_Short(P)))
        res[f"trade|{v}"] = dict(params=P, **r)
        trials.append(dict(trial_id=TrialId("stack_anchor", P, v, "taker"), created_at=Now(), family="stack_anchor", params=P, venue=v, exec="taker",
                           tag="round8A_selected_on_tune", split_scored="holdout+forward", res=r))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round8A", res)
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())
