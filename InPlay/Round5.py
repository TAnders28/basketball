r"""Round 5: maker quoting that also stands aside when the market disagrees with ESPN WP (registered before scoring).

Round 3B showed |mid - ESPN WP| predicts the next 10 minutes of movement (bigger disagreement, bigger moves), i.e. more
adverse selection for a resting quote. Variants: skip quoting when |mid - WP| > D, combined with the last-bar pull X.
Fill model = real trade prints strictly through our level ("tape-through", the conservative one), Kalshi maker fee,
marked to mid H minutes later. Tune = 86 tape games of the 2025 playoffs; holdout = random 192-game tape sample of
2025-26; forward = 170 games of 2026-27.

    .venv\Scripts\python.exe -m InPlay.Round5
"""

import itertools
import os
import sys

import pandas as pd

from InPlay.Engine import Game
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, _Short
from InPlay.TradeTape import LoadTape, Score, TapeFills


def Masked(g, D):
    kw = {k: getattr(g, k) for k in Game.__slots__ if k != "n"}
    kw["bid"] = [b if (b is not None and m is not None and w is not None and abs(m - w) <= D) else None
                 for b, m, w in zip(g.bid, g.mid, g.wp)]
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
    by = {sp: [(g, tp) for g, tp in games if SplitOf(g.season) == sp] for sp in ("tune", "holdout", "forward")}
    grid = [dict(dis=D, pull=X, mark_h=H, tmax=2160) for D, X, H in itertools.product((0.03, 0.06, 0.10), (None, 0.02), (1, 5))]
    trials, rows, tune = [], [], []
    for P in grid:
        fills = []
        for g, tp in by["tune"]:
            fills += TapeFills(Masked(g, P["dis"]), tp, P["pull"], P["mark_h"], P["tmax"], "through")
        r = Score(fills, len(by["tune"]))
        tid = TrialId("mm_quote_dis", P, "kalshi", "maker_both")
        trials.append(dict(trial_id=tid, created_at=Now(), family="mm_quote_dis", params=P, venue="kalshi", exec="maker_both",
                           tag="round5_registered_before_scoring", split_scored="tune", tune=r))
        tune.append((P, r))
    elig = [x for x in tune if x[1]["n"] >= 100]
    P, _ = max(elig, key=lambda x: x[1]["per_fill"])
    md = ["## Round 5: maker quoting that stands aside when |market - ESPN WP| is large (tape-through fills; registered before scoring)", "",
          f"{len(grid)} variants on tune; best tune variant scored once.", "",
          "| variant | split | fills | P/L per fill, maker fee [95% CI] | half-spread earned | adverse move | P/L per game |", "|---|---|---|---|---|---|---|"]
    for Q, r in sorted(tune, key=lambda x: -x[1]["per_fill"] if x[1]["n"] else 9)[:4]:
        md.append(f"| {_Short(Q)} | tune | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
    res = {}
    for sp in ("holdout", "forward"):
        fills = []
        for g, tp in by[sp]:
            fills += TapeFills(Masked(g, P["dis"]), tp, P["pull"], P["mark_h"], P["tmax"], "through")
        r = Score(fills, len(by[sp]))
        res[sp] = r
        md.append(f"| SELECTED {_Short(P)} | {sp} | {r['n']} | {r['per_fill']:+.4f} [{r['lo']:+.4f}, {r['hi']:+.4f}] | {r['half_spread']:+.4f} | {r['adverse']:+.4f} | {r['per_game']:+.3f} |")
        rows.append(dict(section="round5_mm_dis", venue="kalshi", split=sp, family="mm_quote_dis", fee_basis="kalshi_maker", metric="pnl_per_fill",
                         value=r["per_fill"], lo=r["lo"], hi=r["hi"], n=r["n"], label=_Short(P)))
    trials.append(dict(trial_id=TrialId("mm_quote_dis", P, "kalshi", "maker_both"), created_at=Now(), family="mm_quote_dis", params=P,
                       venue="kalshi", exec="maker_both", tag="round5_selected_on_tune", split_scored="holdout+forward", res=res))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())