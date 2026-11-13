r"""Round 7B (9:00 PM steer, priority 2; registered on tune before scoring). Queue-position fill model for quoting.

Same two-sided quote as TradeTape A (1 contract at the minute-bar YES bid and ask, kept 60 s), but our order joins the
BACK of the queue at its price. With Q contracts ahead of us, our bid fills in (t, t+60] only if (a) a taker sell prints
strictly through our bid (the level was exhausted), or (b) taker sells AT our bid total more than Q. Same for the ask
with taker buys. Kalshi history has no book depth, so Q is a registered assumption:
  fixed_50 / fixed_250 / fixed_1000 contracts, or prior5 = contracts printed at exactly that price in the previous
  5 minutes (a depth proxy), plus the earlier touch (Q=0) and through-only bounds for reference.
Variants: queue model x mark horizon H (1, 5 min) x jump filter X (none, 0.03). Selected per model on tune (>= 50 fills),
scored once on holdout and forward. Kalshi maker fee. Day-clustered 95% CI.

    .venv\Scripts\python.exe -m InPlay.Round7B
"""

import bisect
import os
import sys

import numpy as np
import pandas as pd

from InPlay.Execution import Fee
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.TradeTape import LoadTape, Score

MODELS = ("touch", "fixed_50", "fixed_250", "fixed_1000", "prior5", "through")


def Fills(g, tape, model, X, H, tmax=2160):
    ts, px, cnt, side = tape[:, 0], tape[:, 1], tape[:, 2], tape[:, 3]
    out = []
    eps = 1e-9
    for i in range(1, g.n - 1):
        el = g.el[i]
        if el < 0 or el > tmax:
            continue
        b, a, m0, m1 = g.bid[i], g.ask[i], g.mid[i], g.mid[i - 1]
        if b is None or a is None or m0 is None or a - b < 0.01:
            continue
        if X is not None:
            dm = abs(m0 - m1) if m1 is not None else 0.0
            dw = abs(g.wp[i] - g.wp[i - 1]) if g.wp[i] is not None and g.wp[i - 1] is not None else 0.0
            if max(dm, dw) > X:
                continue
        j = i + 1 + H
        if j >= g.n:
            mark = float(g.won)
        else:
            mark = next((g.mid[k] for k in range(j, max(i, j - 5), -1) if g.mid[k] is not None), None)
            if mark is None:
                continue
        lo, hi = bisect.bisect_right(ts, g.t[i]), bisect.bisect_right(ts, g.t[i] + 60)
        if hi <= lo:
            continue
        p0 = bisect.bisect_right(ts, g.t[i] - 300)
        for d, lvl in ((1, b), (-1, a)):
            wp_, wc, ws = px[lo:hi], cnt[lo:hi], side[lo:hi]
            opp = ws < 0 if d == 1 else ws > 0  # taker sells hit our bid; taker buys lift our ask
            through = (wp_[opp] < lvl - eps).any() if d == 1 else (wp_[opp] > lvl + eps).any()
            at = float(wc[opp & (np.abs(wp_ - lvl) < eps)].sum())
            if model == "through":
                fill = through
            elif model == "touch":
                fill = through or at > 0
            else:
                if model == "prior5":
                    q = float(cnt[p0:lo][np.abs(px[p0:lo] - lvl) < eps].sum())
                else:
                    q = float(model.split("_")[1])
                fill = through or at > q
            if fill:
                if d == 1:
                    out.append((g.day, mark - b - Fee(b, "kalshi_maker"), m0 - b, mark - m0))
                else:
                    out.append((g.day, a - mark - Fee(a, "kalshi_maker"), a - m0, m0 - mark))
    return out


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    G = LoadGames()
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    by = {"tune": [], "holdout": [], "forward": []}
    for g in G["kalshi"]:
        t = tick.get(str(g.gid))
        tp = LoadTape(t) if t else None
        if tp is not None and len(tp) > 50:
            by[SplitOf(g.season)].append((g, tp))
    trials, rows, res = [], [], {}
    md = ["## Round 7B: queue-position fill model for two-sided quoting (registered on tune before scoring)", "",
          f"Tape games: tune {len(by['tune'])}, holdout {len(by['holdout'])}, forward {len(by['forward'])}. P/L per fill in $ per contract, Kalshi maker fee, marked to mid H minutes later.", "",
          "| queue model | selected on tune | tune per fill (n) | holdout per fill [95% CI] (n) | holdout fills per game | forward per fill [95% CI] (n) | adverse move after fill (forward) |",
          "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        tune = []
        for H in (1, 5):
            for X in (None, 0.03):
                P = dict(queue=model, mark_h=H, jump_x=X)
                r = Score([f for g, tp in by["tune"] for f in Fills(g, tp, model, X, H)], len(by["tune"]))
                tune.append((P, r))
                trials.append(dict(trial_id=TrialId("mm_queue", P, "kalshi", "maker"), created_at=Now(), family="mm_queue", params=P, venue="kalshi",
                                   exec="maker", tag="round7B_registered_before_scoring", split_scored="tune", tune=r))
        ok = [x for x in tune if x[1]["n"] >= 50] or tune
        P, rt = max(ok, key=lambda x: x[1]["per_fill"] if x[1]["n"] else -9)
        out = {}
        for sp in ("holdout", "forward"):
            o = Score([f for g, tp in by[sp] for f in Fills(g, tp, model, P["jump_x"], P["mark_h"])], len(by[sp]))
            out[sp] = o
            rows.append(dict(section="round7B_mm_queue", venue="kalshi", split=sp, family="mm_queue", trial_id=TrialId("mm_queue", P, "kalshi", "maker"),
                             fee_basis="kalshi_maker", metric="pnl_per_fill", value=o["per_fill"], lo=o["lo"], hi=o["hi"], n=o["n"], label=_Short(P)))
        trials.append(dict(trial_id=TrialId("mm_queue", P, "kalshi", "maker"), created_at=Now(), family="mm_queue", params=P, venue="kalshi", exec="maker",
                           tag="round7B_selected_on_tune", split_scored="holdout+forward", res=out))
        res[model] = dict(params=P, tune=rt, **out)
        f = lambda o: f"{o['per_fill']:+.4f} [{o['lo']:+.4f}, {o['hi']:+.4f}] ({o['n']})" if o["n"] else "no fills"
        md.append(f"| {model} | {_Short(P)} | {rt['per_fill']:+.4f} ({rt['n']}) | {f(out['holdout'])} | {out['holdout']['n']/max(len(by['holdout']),1):.1f} | "
                  f"{f(out['forward'])} | {out['forward']['adverse']:+.4f} |")
        print(md[-1], flush=True)
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round7B", res)
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())
