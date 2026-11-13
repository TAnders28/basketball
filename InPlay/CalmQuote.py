r"""Steer 8:11 #3 (tune-split trial): quote only in calm stretches.

Calm = mean absolute 1-minute Kalshi mid change over the previous 10 bars <= R, AND |mid - ESPN WP| <= D, AND no ESPN WP / mid
jump > 2c in the last bar, AND no scoring in the last minute (ESPN score unchanged). Tape-through fills, Kalshi maker fee,
marked to mid H minutes later. Grid 3 x 3 x 2 = 18 on tune; best scored once on holdout tape sample and forward.

    .venv\Scripts\python.exe -m InPlay.CalmQuote
"""

import itertools
import os
import sys

import numpy as np
import pandas as pd

from InPlay.Engine import Game
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, _Short
from InPlay.TradeTape import LoadTape, Score, TapeFills


def Calm(g, R, D):
    kw = {k: getattr(g, k) for k in Game.__slots__ if k != "n"}
    bid = []
    for i in range(g.n):
        ok = i >= 11 and g.mid[i] is not None and g.wp[i] is not None and g.sd[i] is not None and g.sd[i - 1] is not None
        if ok:
            ch = [abs(g.mid[k] - g.mid[k - 1]) for k in range(i - 9, i + 1) if g.mid[k] is not None and g.mid[k - 1] is not None]
            ok = len(ch) >= 8 and np.mean(ch) <= R and abs(g.mid[i] - g.wp[i]) <= D and g.sd[i] == g.sd[i - 1]
        bid.append(g.bid[i] if ok else None)
    kw["bid"] = bid
    return Game(**kw)


def Main():
    G = LoadGames()
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    games = []
    for g in G["kalshi"]:
        t = tick.get(str(g.gid))
        tp = LoadTape(t) if t else None
        if tp is not None and len(tp) > 50:
            games.append((g, tp))
    by = {sp: [x for x in games if SplitOf(x[0].season) == sp] for sp in ("tune", "holdout", "forward")}
    grid = [dict(rv=R, dis=D, mark_h=H, pull=0.02, tmax=2160) for R, D, H in itertools.product((0.005, 0.01, 0.02), (0.03, 0.06, 0.10), (1, 5))]
    trials, tune = [], []
    for P in grid:
        r = Score([f for g, tp in by["tune"] for f in TapeFills(Calm(g, P["rv"], P["dis"]), tp, P["pull"], P["mark_h"], P["tmax"], "through")], len(by["tune"]))
        tune.append((P, r))
        trials.append(dict(trial_id=TrialId("mm_quote_calm", P, "kalshi", "maker_both"), created_at=Now(), family="mm_quote_calm", params=P, venue="kalshi",
                           exec="maker_both", tag="steer811_registered_before_scoring", split_scored="tune", tune=r))
    P, _ = max([x for x in tune if x[1]["n"] >= 100], key=lambda x: x[1]["per_fill"])
    md = ["## Calm-stretch quoting only (Kalshi; tape-through fills; registered before scoring)", "",
          "| variant | split | fills | P/L per fill, maker fee [95% CI] | half-spread earned | adverse move | P/L per game |", "|---|---|---|---|---|---|---|"]
    for Q, r in sorted(tune, key=lambda x: -x[1]["per_fill"] if x[1]["n"] else 9)[:3]:
        md.append(f"| {_Short(Q)} | tune | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
    rows, res = [], {}
    for sp in ("holdout", "forward"):
        r = Score([f for g, tp in by[sp] for f in TapeFills(Calm(g, P["rv"], P["dis"]), tp, P["pull"], P["mark_h"], P["tmax"], "through")], len(by[sp]))
        res[sp] = r
        md.append(f"| SELECTED {_Short(P)} | {sp} | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
        rows.append(dict(section="calm_quote", venue="kalshi", split=sp, family="mm_quote_calm", fee_basis="kalshi_maker", metric="pnl_per_fill",
                         value=r["per_fill"], lo=r["lo"], hi=r["hi"], n=r["n"], label=_Short(P)))
    trials.append(dict(trial_id=TrialId("mm_quote_calm", P, "kalshi", "maker_both"), created_at=Now(), family="mm_quote_calm", params=P, venue="kalshi",
                       exec="maker_both", tag="steer811_selected_on_tune", split_scored="holdout+forward", res=res))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())