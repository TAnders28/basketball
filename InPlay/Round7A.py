r"""Round 7A (9:00 PM steer, priority 1; registered on tune before scoring). Polymarket timing, redone.

Three Polymarket timings, same rules, same selection:
  m60   the earlier final run: every PM history label moved 60 s earlier (games_obj.pkl).
  s0    no shift: labels taken as on time, which the on-chain audit supports (games_obj_s0.pkl).
  perg  per-game offset: for each game, the lag L in {-2..+2} minutes maximising corr(1-min ESPN WP change at bar i,
        1-min PM change at bar i+L) on the FIRST HALF only (el < 1440, >= 20 pairs, else 0), applied to the SECOND HALF
        only (bar i takes the PM value of bar i+L). ESPN is the reference because the 2024-25 tune season has no Kalshi.
        When L > 0 this uses PM labels from the future; if labels are on time (on-chain audit) that is a leak.
Rules: every registered grid variant on the PM venue (taker, 1-minute latency), plus xvenue on Kalshi (uses PM as the
signal), plus the round-6 directional maker on Kalshi off the PM gap (tape-through fills). Best variant per family is
selected on tune (Kalshi taker fee, >= 30 trades; >= 10 for Kalshi-venue families) separately for each timing, then
scored once on holdout and forward with both fee bases. CI = conservative (widest of day, game, exact binomial).

    .venv\Scripts\python.exe -m InPlay.Round7A
"""

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from InPlay.CiAudit import Audit
from InPlay.Engine import Game, RunGame
from InPlay.Grid import EngineFamily, TrialId, Variants
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Round6 import DirMaker
from InPlay.TradeTape import LoadTape, Score

SETS = (("m60", os.path.join("data", "inplay", "games_obj.pkl")), ("s0", os.path.join("data", "inplay", "games_obj_s0.pkl")))
HALF = 1440.0


def BestLag(wp, px, el, max_l=2, min_pairs=20):
    n = len(px)
    best, best_c = 0, -2.0
    for L in range(-max_l, max_l + 1):
        xs, ys = [], []
        for i in range(1, n):
            if el[i] >= HALF:
                break
            j = i + L
            if j < 1 or j >= n or el[j] >= HALF:
                continue
            a, b, c, d = wp[i], wp[i - 1], px[j], px[j - 1]
            if None in (a, b, c, d):
                continue
            xs.append(a - b)
            ys.append(c - d)
        if len(xs) < min_pairs or np.std(xs) == 0 or np.std(ys) == 0:
            continue
        c = float(np.corrcoef(xs, ys)[0, 1])
        if c > best_c:
            best, best_c = L, c
    return best


def ShiftSecondHalf(seq, el, L):
    if L == 0:
        return list(seq)
    out, n = list(seq), len(seq)
    for i in range(n):
        if el[i] >= HALF and 0 <= i + L < n:
            out[i] = seq[i + L]
    return out


def PerGame(G):
    out, lags = {"kalshi": [], "pm": []}, defaultdict(list)
    pm_lag = {}
    for g in G["pm"]:
        L = BestLag(g.wp, g.mid, g.el)
        pm_lag[g.gid] = L
        lags[SplitOf(g.season)].append(L)
        kw = {k: getattr(g, k) for k in Game.__slots__ if k != "n"}
        for c in ("mid", "bid", "ask"):
            kw[c] = ShiftSecondHalf(getattr(g, c), g.el, L)
        out["pm"].append(Game(**kw))
    for g in G["kalshi"]:
        L = pm_lag.get(g.gid)
        if L is None:
            L = BestLag(g.wp, g.pm, g.el)
        kw = {k: getattr(g, k) for k in Game.__slots__ if k != "n"}
        kw["pm"] = ShiftSecondHalf(g.pm, g.el, L)
        kw["pmage"] = ShiftSecondHalf(g.pmage, g.el, L)
        out["kalshi"].append(Game(**kw))
    return out, lags


def Jobs():
    j = [(f, P, "pm") for f, P in Variants() if f != "xvenue"]
    j += [(f, P, "kalshi") for f, P in Variants() if f == "xvenue"]
    return j


def Run(G, fam, P, venue, sp):
    return [t for g in G[venue] if SplitOf(g.season) == sp for t in RunGame(EngineFamily(fam), P, g)]


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    sets = {name: LoadGames(p) for name, p in SETS}
    sets["perg"], lags = PerGame(sets["s0"])
    trials, rows, res = [], [], {}
    md = ["## Round 7A: Polymarket timing redone (no shift, per-game offset) - registered on tune before scoring", "",
          "Per-game offset L (minutes, PM vs ESPN, first half): " + "; ".join(
              f"{sp}: " + ", ".join(f"L={L}: {np.mean(np.array(v) == L):.0%}" for L in range(-2, 3)) for sp, v in lags.items() if v), ""]
    jobs = Jobs()
    sel = {}
    for name, G in sets.items():
        tune = defaultdict(list)
        for fam, P, venue in jobs:
            m = Audit(Run(G, fam, P, venue, "tune"), "kalshi_taker")
            trials.append(dict(trial_id=TrialId(fam + "@" + name, P, venue, "taker"), created_at=Now(), family=fam, pm_timing=name,
                               params=P, venue=venue, exec="taker", tag="round7A_registered_before_scoring", split_scored="tune", tune=m))
            need = 10 if venue == "kalshi" else 30
            if m and m["n"] >= need and not fam.endswith("_ctrl"):
                tune[(fam, venue)].append((m["mean"], P, m))
        sel[name] = {k: max(v, key=lambda x: x[0]) for k, v in tune.items()}
    md += ["### Selected rule per family under each timing (taker, 1-min latency)", "",
           "| family | venue | timing | rule | tune P/L (n) | holdout P/L, Kalshi fee [cons 95% CI] (n) | holdout, $0.02 fee | forward P/L, Kalshi fee [cons CI] (n) | forward, $0.02 fee |",
           "|---|---|---|---|---|---|---|---|---|"]
    fams = sorted({k for s in sel.values() for k in s})
    for key in fams:
        for name in ("m60", "s0", "perg"):
            if key not in sel[name]:
                continue
            _, P, mt = sel[name][key]
            fam, venue = key
            cells = []
            r = {}
            for sp in ("holdout", "forward"):
                tr = Run(sets[name], fam, P, venue, sp)
                for b in ("kalshi_taker", "tim_est_0.02"):
                    m = Audit(tr, b)
                    r[f"{sp}|{b}"] = m
                    if m:
                        rows.append(dict(section="round7A_pm_timing", venue=venue, split=sp, family=fam, trial_id=TrialId(fam + "@" + name, P, venue, "taker"),
                                         fee_basis=b, metric="pnl_per_trade", value=m["mean"], lo=m["cons_lo"], hi=m["cons_hi"], n=m["n"], label=name + "|" + _Short(P)))
                mk, mt2 = r[f"{sp}|kalshi_taker"], r[f"{sp}|tim_est_0.02"]
                cells.append(f"{mk['mean']:+.4f} [{mk['cons_lo']:+.4f}, {mk['cons_hi']:+.4f}] ({mk['n']})" if mk else "no trades")
                cells.append(f"{mt2['mean']:+.4f}" if mt2 else "-")
            res[f"{fam}|{venue}|{name}"] = dict(params=P, tune=mt, **r)
            trials.append(dict(trial_id=TrialId(fam + "@" + name, P, venue, "taker"), created_at=Now(), family=fam, pm_timing=name, params=P,
                               venue=venue, exec="taker", tag="round7A_selected_on_tune", split_scored="holdout+forward", res=r))
            md.append(f"| {fam} | {venue} | {name} | {_Short(P)} | {mt['mean']:+.4f} ({mt['n']}) | " + " | ".join(cells) + " |")
    md.append("")
    # directional maker on Kalshi off the PM gap (tape fills), per timing
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    tapes = {}
    for g in sets["s0"]["kalshi"]:
        t = tick.get(str(g.gid))
        tp = LoadTape(t) if t else None
        if tp is not None and len(tp) > 50:
            tapes[g.gid] = tp
    grid = [dict(gap=G_, mark_h=H) for G_ in (0.02, 0.03, 0.05) for H in (1, 5)]
    md += ["### Slower-venue maker (Kalshi maker order when PM disagrees by >= gap; real-print fills; marked to mid H min later; Kalshi maker fee)", "",
           "| timing | selected on tune | tune per fill (n) | holdout per fill [95% CI] (n) | forward per fill [95% CI] (n) |", "|---|---|---|---|---|"]
    for name in ("m60", "s0", "perg"):
        by = {sp: [(g, tapes[g.gid]) for g in sets[name]["kalshi"] if g.gid in tapes and SplitOf(g.season) == sp] for sp in ("tune", "holdout", "forward")}
        tune = []
        for P in grid:
            r = Score([f for g, tp in by["tune"] for f in DirMaker(g, tp, P["gap"], P["mark_h"])], len(by["tune"]))
            tune.append((P, r))
            trials.append(dict(trial_id=TrialId("dir_maker_pmgap@" + name, P, "kalshi", "maker"), created_at=Now(), family="dir_maker_pmgap", pm_timing=name,
                               params=P, venue="kalshi", exec="maker", tag="round7A_registered_before_scoring", split_scored="tune", tune=r))
        ok = [x for x in tune if x[1]["n"] >= 50] or tune
        P, rt = max(ok, key=lambda x: x[1]["per_fill"] if x[1]["n"] else -9)
        out = {}
        for sp in ("holdout", "forward"):
            out[sp] = Score([f for g, tp in by[sp] for f in DirMaker(g, tp, P["gap"], P["mark_h"])], len(by[sp]))
            o = out[sp]
            rows.append(dict(section="round7A_dir_maker", venue="kalshi", split=sp, family="dir_maker_pmgap", trial_id=TrialId("dir_maker_pmgap@" + name, P, "kalshi", "maker"),
                             fee_basis="kalshi_maker", metric="pnl_per_fill", value=o["per_fill"], lo=o["lo"], hi=o["hi"], n=o["n"], label=name + "|" + _Short(P)))
        trials.append(dict(trial_id=TrialId("dir_maker_pmgap@" + name, P, "kalshi", "maker"), created_at=Now(), family="dir_maker_pmgap", pm_timing=name,
                           params=P, venue="kalshi", exec="maker", tag="round7A_selected_on_tune", split_scored="holdout+forward", res=out))
        res[f"dir_maker|{name}"] = dict(params=P, tune=rt, **out)
        f = lambda o: f"{o['per_fill']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] ({o['n']})" if o["n"] else "no fills"
        md.append(f"| {name} | {_Short(P)} | {rt['per_fill']:+.4f} ({rt['n']}) | {f(out['holdout'])} | {f(out['forward'])} |")
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round7A", res)
    print("\n".join(md))
    print("trials", len(trials))


if __name__ == "__main__":
    sys.exit(Main())
