r"""Order-flow family from the Kalshi trade tape (new signal, registered before scoring; round 4).

Signal at minute bar i uses only prints with timestamp <= t_i: over the last W seconds, total taker volume V_tot and
imbalance = (taker-buy-YES volume - taker-sell-YES volume) / V_tot. If V_tot >= V_min and |imbalance| >= theta,
follow (buy the side being bought) or fade (buy the other side). Taker fill at bar i+1 (ask + 0.5c), exit at the bid
H bars later (taker), max 3 trades per game, no overlap, before 1:00 of Q4.
Tune = Kalshi 2025 playoffs (86 games); holdout = seeded random 200 games of 2025-26 (this family's holdout is fresh);
forward = 2026-27 games with a tape.

    .venv\Scripts\python.exe -m InPlay.FlowTape --workers 6
"""

import argparse
import bisect
import itertools
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from InPlay.CiAudit import Audit
from InPlay.Execution import TakerBuy, TakerSell
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Stats import Pbo
from InPlay.TradeTape import LoadTape

_D = None


def _Init():
    global _D
    G = LoadGames()
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    _D = []
    for g in G["kalshi"]:
        t = tick.get(str(g.gid))
        tp = LoadTape(t) if t else None
        if tp is not None and len(tp) > 50:
            _D.append((g, tp))
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)


def Flow(tp, t, W):
    lo, hi = bisect.bisect_right(tp[:, 0], t - W), bisect.bisect_right(tp[:, 0], t)
    if hi <= lo:
        return 0.0, 0.0
    v, s = tp[lo:hi, 2], tp[lo:hi, 3]
    tot = v.sum()
    return tot, float((v * s).sum() / tot) if tot else 0.0


def RunFlow(g, tp, P, slip=0.005):
    trades, i, last = [], 1, g.n - 1
    while i < last - 1 and len(trades) < 3:
        if g.mid[i] is None or g.el[i] < 0 or g.el[i] > 2820:
            i += 1
            continue
        tot, imb = Flow(tp, g.t[i], P["W"])
        if tot < P["vmin"] or abs(imb) < P["theta"]:
            i += 1
            continue
        side = (1 if imb > 0 else -1) * (1 if P["dir"] == "follow" else -1)
        px = TakerBuy(g.bid[i + 1], g.ask[i + 1], side, slip)
        j = i + 1 + P["H"]
        if px is None:
            i += 1
            continue
        if j > last:
            won = g.won if side == 1 else 1 - g.won
            trades.append(dict(gid=g.gid, day=g.day, season=g.season, side=side, entry=px, exit=float(won), settled=True, why="settle", el=g.el[i], maker=False))
            break
        ex = TakerSell(g.bid[j], g.ask[j], side, slip)
        k = j
        while ex is None and k < min(last, j + 3):
            k += 1
            ex = TakerSell(g.bid[k], g.ask[k], side, slip)
        if ex is None:
            won = g.won if side == 1 else 1 - g.won
            trades.append(dict(gid=g.gid, day=g.day, season=g.season, side=side, entry=px, exit=float(won), settled=True, why="settle", el=g.el[i], maker=False))
            break
        trades.append(dict(gid=g.gid, day=g.day, season=g.season, side=side, entry=px, exit=ex, settled=False, why="time", el=g.el[i], maker=False))
        i = k + 1
    return trades


def _Job(P):
    out = {}
    for g, tp in _D:
        sp = SplitOf(g.season)
        out.setdefault(sp, []).extend(RunFlow(g, tp, P))
    return P, out


def Grid():
    return [dict(W=W, theta=th, vmin=v, dir=d, H=H) for W, th, v, d, H in
            itertools.product((30, 60), (0.6, 0.8), (1000, 5000), ("follow", "fade"), (1, 3, 5))]


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tune-only", action="store_true")
    a = ap.parse_args(argv)
    grid = Grid()
    trials, rows = [], []
    with ProcessPoolExecutor(max_workers=min(a.workers, 10), initializer=_Init) as ex:
        res = list(ex.map(_Job, grid))
    tune = []
    for P, out in res:
        m = Audit(out.get("tune", []), "kalshi_taker")
        tid = TrialId("flow_tape", P, "kalshi", "taker")
        trials.append(dict(trial_id=tid, created_at=Now(), family="flow_tape", params=P, venue="kalshi", exec="taker",
                           tag="round4_registered_before_scoring", split_scored="tune", tune=m))
        tune.append((P, out, m))
    elig = [x for x in tune if x[2] and x[2]["n"] >= 30]
    best = max(elig, key=lambda x: x[2]["mean"])
    # PBO over tune days
    days = sorted({t["day"] for _, out, _ in tune for t in out.get("tune", [])})
    from InPlay.Engine import TradePnl
    perf = np.zeros((len(days), len(tune)))
    dix = {d: k for k, d in enumerate(days)}
    for j, (_, out, _) in enumerate(tune):
        for t in out.get("tune", []):
            perf[dix[t["day"]], j] += TradePnl(t, "kalshi_taker")
    pbo = Pbo(perf)[0] if len(days) >= 16 else float("nan")
    md = ["## Round 4: order-flow imbalance from the Kalshi trade tape (registered before scoring)", "",
          f"{len(grid)} variants on tune; PBO (CSCV over tune days) = {pbo:.2f}. Best tune rule scored once on the fresh tape holdout sample and forward.", "",
          "| rule | split | fee | trades | P/L per trade [conservative 95% CI] | hit |", "|---|---|---|---|---|---|"]
    for P, out, m in sorted(tune, key=lambda x: -(x[2]["mean"] if x[2] else -9))[:3]:
        md.append(f"| {_Short(P)} | tune | kalshi_taker | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] | {1 - m['losses']/m['n']:.2f} |")
    if a.tune_only:
        print("\n".join(md))
        return 0
    P, out, _ = best
    scored = {}
    for sp in ("holdout", "forward"):
        for b in ("tim_est_0.02", "kalshi_taker"):
            m = Audit(out.get(sp, []), b)
            scored[f"{sp}|{b}"] = m
            if m:
                md.append(f"| SELECTED {_Short(P)} | {sp} | {b} | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] | {1 - m['losses']/m['n']:.2f} |")
                rows.append(dict(section="round4_flow", venue="kalshi", split=sp, family="flow_tape", fee_basis=b, metric="pnl_per_trade",
                                 value=m["mean"], lo=m["cons_lo"], hi=m["cons_hi"], n=m["n"], label=_Short(P)))
    trials.append(dict(trial_id=TrialId("flow_tape", P, "kalshi", "taker"), created_at=Now(), family="flow_tape", params=P, venue="kalshi",
                       exec="taker", tag="round4_selected_on_tune", split_scored="holdout+forward", res=scored))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("flow_tape", dict(best=P, scored=scored, pbo=pbo))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())