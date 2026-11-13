r"""Round 6 (from the 8:11 PM steer; registered before scoring; holdout is fresh for both families).

A. Directional maker on Kalshi from the Polymarket gap: when the as-of Polymarket price (60 s timestamp correction) is
   >= G above the Kalshi mid, rest a Kalshi YES bid at the bid; when >= G below, rest a YES ask (i.e. sell YES / buy NO).
   Fill only on a real print through our level within the next 60 s (trade tape). Marked to mid H minutes later, Kalshi
   maker fee. Tape games only (tune 86 / holdout sample 192 / forward 170).
B. ESPN lag on a recalibrated ESPN WP. ESPN WP is biased in known contexts (see ESPN long history), so the raw
   market-vs-ESPN gap mixes lag with ESPN's bias. Recalibrate ESPN WP on ESPN-only seasons 2018-2024 (no market data):
   cells = WP bucket (0.05) x period (1-4) x whether the current leader by WP was ESPN's pregame favourite; first passage
   per game per cell; win rate shrunk toward the bucket's mean WP with strength 50. Then rerun the frozen espn_lag grid
   (18 variants, taker, 1-minute latency) with the recalibrated WP; select on tune; score once.

    .venv\Scripts\python.exe -m InPlay.Round6 --workers 6
"""

import argparse
import bisect
import glob
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

from InPlay.CiAudit import Audit
from InPlay.Engine import Game, RunGame
from InPlay.EspnLong import ToGame
from InPlay.Execution import Fee
from InPlay.Grid import TrialId, Variants
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.TradeTape import LoadTape, Score

K = 50.0


def Key(wp, per, pre):
    leader = 1 if wp >= 0.5 else -1
    fav = 1 if pre >= 0.5 else -1
    return (min(int(wp / 0.05), 19), min(max(int(per), 1), 4), int(leader == fav))


def _Fit(path):
    try:
        g = ToGame(path)
    except Exception:
        return None
    if g is None or g.season > 2024:
        return None
    seen = {}
    for i in range(g.n):
        k = Key(g.wp[i], g.per[i], g.pre)
        if k not in seen:
            seen[k] = (g.wp[i], g.won)
    return seen


def FitRecal(workers):
    paths = sorted(glob.glob(os.path.join("data", "inplay", "espn", "games", "*", "*.json")))
    paths = [p for p in paths if int(os.path.basename(os.path.dirname(p))) <= 2024]
    st = defaultdict(lambda: [0, 0.0, 0.0])
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(_Fit, paths, chunksize=25):
            if not r:
                continue
            for k, (w, y) in r.items():
                s = st[k]
                s[0] += 1
                s[1] += w
                s[2] += y
    table = {k: ((K * (sw / n) + sy) / (K + n), n, sw / n, sy / n) for k, (n, sw, sy) in st.items()}
    return table, len(paths)


def Recal(g, table):
    kw = {k: getattr(g, k) for k in Game.__slots__ if k != "n"}
    pre_espn = next((w for w in g.wp if w is not None), 0.5)
    out = []
    for w, per in zip(g.wp, g.per):
        if w is None or per is None:
            out.append(w)
            continue
        t = table.get(Key(w, per, pre_espn))
        out.append(w if t is None else w + (t[0] - t[2]))  # shift by the cell's (shrunk win rate - mean WP)
    kw["wp"] = [None if x is None else min(0.999, max(0.001, x)) for x in out]
    return Game(**kw)


def DirMaker(g, tp, G_, H, tmax=2820):
    ts, px, side = tp[:, 0], tp[:, 1], tp[:, 3]
    out = []
    for i in range(1, g.n - 1):
        if g.el[i] < 0 or g.el[i] > tmax or g.pm[i] is None or g.mid[i] is None or g.bid[i] is None or g.ask[i] is None:
            continue
        if g.pmage[i] is not None and g.pmage[i] > 120:
            continue
        gap = g.pm[i] - g.mid[i]
        if abs(gap) < G_:
            continue
        j = i + 1 + H
        mark = float(g.won) if j >= g.n else next((g.mid[k] for k in range(j, max(i, j - 5), -1) if g.mid[k] is not None), None)
        if mark is None:
            continue
        lo, hi = bisect.bisect_right(ts, g.t[i]), bisect.bisect_right(ts, g.t[i] + 60)
        if hi <= lo:
            continue
        w_px, w_sd = px[lo:hi], side[lo:hi]
        if gap > 0 and (w_px[w_sd < 0] < g.bid[i] - 1e-9).any():
            b = g.bid[i]
            out.append((g.day, mark - b - Fee(b, "kalshi_maker"), g.mid[i] - b, mark - g.mid[i]))
        if gap < 0 and (w_px[w_sd > 0] > g.ask[i] + 1e-9).any():
            a = g.ask[i]
            out.append((g.day, a - mark - Fee(a, "kalshi_maker"), a - g.mid[i], g.mid[i] - mark))
    return out


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv)
    G = LoadGames()
    trials, rows = [], []
    md = ["## Round 6 (from the 8:11 PM steer; registered before scoring)", ""]
    # ---- A
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    tg = []
    for g in G["kalshi"]:
        t = tick.get(str(g.gid))
        tp = LoadTape(t) if t else None
        if tp is not None and len(tp) > 50:
            tg.append((g, tp))
    by = {sp: [x for x in tg if SplitOf(x[0].season) == sp] for sp in ("tune", "holdout", "forward")}
    grid = [dict(gap=G_, mark_h=H) for G_ in (0.02, 0.03, 0.05) for H in (1, 5)]
    tune = []
    for P in grid:
        r = Score([f for g, tp in by["tune"] for f in DirMaker(g, tp, P["gap"], P["mark_h"])], len(by["tune"]))
        tune.append((P, r))
        trials.append(dict(trial_id=TrialId("dir_maker_pmgap", P, "kalshi", "maker"), created_at=Now(), family="dir_maker_pmgap", params=P,
                           venue="kalshi", exec="maker", tag="round6_registered_before_scoring", split_scored="tune", tune=r))
    P, _ = max([x for x in tune if x[1]["n"] >= 50], key=lambda x: x[1]["per_fill"])
    md += ["### A. Directional maker on Kalshi when Polymarket disagrees (tape-through fills, maker fee, marked to mid)", "",
           "| variant | split | fills | P/L per fill [95% CI] | half-spread earned | move after fill (signed) | P/L per game |", "|---|---|---|---|---|---|---|"]
    for Q, r in tune:
        if r["n"]:
            md.append(f"| {_Short(Q)} | tune | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
    resA = {}
    for sp in ("holdout", "forward"):
        r = Score([f for g, tp in by[sp] for f in DirMaker(g, tp, P["gap"], P["mark_h"])], len(by[sp]))
        resA[sp] = r
        md.append(f"| SELECTED {_Short(P)} | {sp} | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
        rows.append(dict(section="round6_dir_maker", venue="kalshi", split=sp, family="dir_maker_pmgap", fee_basis="kalshi_maker", metric="pnl_per_fill",
                         value=r["per_fill"], lo=r["lo"], hi=r["hi"], n=r["n"], label=_Short(P)))
    trials.append(dict(trial_id=TrialId("dir_maker_pmgap", P, "kalshi", "maker"), created_at=Now(), family="dir_maker_pmgap", params=P,
                       venue="kalshi", exec="maker", tag="round6_selected_on_tune", split_scored="holdout+forward", res=resA))
    md.append("")
    # ---- B
    table, n_paths = FitRecal(min(a.workers, 10))
    big = sorted(table.items(), key=lambda kv: -abs(kv[1][0] - kv[1][2]) * min(kv[1][1], 400))[:6]
    md += [f"### B. espn_lag on recalibrated ESPN WP (recalibration fitted on {n_paths} ESPN games, seasons 2018-2024 only)", "",
           "Largest recalibration shifts (WP bucket, period, leader is ESPN pregame favourite): " +
           "; ".join(f"{k[0]*0.05:.2f}-{k[0]*0.05+0.05:.2f} P{k[1]} fav={k[2]}: n={v[1]}, WP {v[2]:.3f} -> win {v[3]:.3f}" for k, v in big), "",
           "| venue | rule | split | fee | trades | P/L per trade [conservative 95% CI] |", "|---|---|---|---|---|---|"]
    lag = [(f, Q) for f, Q in Variants() if f == "espn_lag"]
    resB = {}
    for venue in ("pm", "kalshi"):
        games = [Recal(g, table) for g in G[venue]]
        tune_b = []
        for f, Q in lag:
            tr = [t for g in games if SplitOf(g.season) == "tune" for t in RunGame("espn_lag", Q, g)]
            m = Audit(tr, "kalshi_taker")
            tune_b.append((Q, m))
            trials.append(dict(trial_id=TrialId("espn_lag_recal", Q, venue, "taker"), created_at=Now(), family="espn_lag_recal", params=Q,
                               venue=venue, exec="taker", tag="round6_registered_before_scoring", split_scored="tune", tune=m))
        el = [x for x in tune_b if x[1] and x[1]["n"] >= (30 if venue == "pm" else 10)]
        Q, m = max(el, key=lambda x: x[1]["mean"])
        md.append(f"| {venue} | {_Short(Q)} | tune | kalshi_taker | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] |")
        for sp in ("holdout", "forward"):
            tr = [t for g in games if SplitOf(g.season) == sp for t in RunGame("espn_lag", Q, g)]
            for b in ("tim_est_0.02", "kalshi_taker"):
                mm = Audit(tr, b)
                resB[f"{venue}|{sp}|{b}"] = mm
                if mm:
                    md.append(f"| {venue} | SELECTED {_Short(Q)} | {sp} | {b} | {mm['n']} | {mm['mean']:+.4f} [{mm['cons_lo']:+.4f}, {mm['cons_hi']:+.4f}] |")
                    rows.append(dict(section="round6_espn_lag_recal", venue=venue, split=sp, family="espn_lag_recal", fee_basis=b, metric="pnl_per_trade",
                                     value=mm["mean"], lo=mm["cons_lo"], hi=mm["cons_hi"], n=mm["n"], label=_Short(Q)))
        trials.append(dict(trial_id=TrialId("espn_lag_recal", Q, venue, "taker"), created_at=Now(), family="espn_lag_recal", params=Q,
                           venue=venue, exec="taker", tag="round6_selected_on_tune", split_scored="holdout+forward",
                           res={k: v for k, v in resB.items() if k.startswith(venue)}))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round6", dict(A=resA, B=resB))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())