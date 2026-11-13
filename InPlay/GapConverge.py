r"""Steer 8:11 #1 follow-up: how fast do 3c+ Kalshi-Polymarket gaps close, in seconds? (descriptive; registered)

Episode = first bar of a run of bars with |PM - Kalshi mid| >= 3c (in play, PM age <= 120 s). From the bar time t0:
  * Kalshi-converges time: seconds until a Kalshi print is within 1c of the PM level at t0 (Kalshi trade tape);
  * PM-converges: whether a later PM minute value is within 1c of the Kalshi mid at t0 (minute resolution only).
Both PM alignment readings are shown: H2 (labels late; the final run's -60 s shift) and H1 (labels on time; PM at bar i-1).

    .venv\Scripts\python.exe -m InPlay.GapConverge
"""

import bisect
import os
import sys

import numpy as np
import pandas as pd

from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now
from InPlay.TradeTape import LoadTape


def Episodes(g, tp, h1):
    ts, px = tp[:, 0], tp[:, 1]
    out, prev = [], False
    for i in range(2, g.n - 6):
        pm = g.pm[i - 1] if h1 else g.pm[i]
        age = g.pmage[i - 1] if h1 else g.pmage[i]
        m = g.mid[i]
        if pm is None or m is None or (age is not None and age > 120) or not (0 <= g.el[i] <= 2820):
            prev = False
            continue
        gap = pm - m
        on = abs(gap) >= 0.03
        if on and not prev:
            j = bisect.bisect_right(ts, g.t[i])
            k_sec = None
            for q in range(j, min(len(ts), j + 5000)):
                if ts[q] - g.t[i] > 600:
                    break
                if abs(px[q] - pm) <= 0.01:
                    k_sec = ts[q] - g.t[i]
                    break
            pm_min = None
            for q in range(1, 6):
                v = g.pm[i - 1 + q] if h1 else g.pm[i + q]
                if v is not None and abs(v - m) <= 0.01:
                    pm_min = q
                    break
            out.append((k_sec, pm_min))
        prev = on
    return out


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
    md = ["## Time for 3c+ Kalshi-Polymarket gaps to close (tape games; descriptive)", "",
          "| PM alignment | split | episodes | Kalshi print reaches PM level: median s | within 30 s | within 60 s | within 120 s | never in 10 min | PM reaches Kalshi mid within 1 min | within 5 min |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    rows = []
    for h1, lab in ((False, "H2: labels late, -60 s shift (final run)"), (True, "H1: labels on time")):
        for sp in ("tune", "holdout", "forward"):
            ep = [e for g, tp in games if SplitOf(g.season) == sp for e in Episodes(g, tp, h1)]
            if not ep:
                continue
            ks = np.array([e[0] if e[0] is not None else np.inf for e in ep])
            pmv = [e[1] for e in ep]
            fin = ks[np.isfinite(ks)]
            r = dict(n=len(ep), med=float(np.median(ks)) if len(fin) > len(ks) / 2 else float("inf"),
                     w30=float((ks <= 30).mean()), w60=float((ks <= 60).mean()), w120=float((ks <= 120).mean()), never=float(np.isinf(ks).mean()),
                     pm1=float(np.mean([v == 1 for v in pmv])), pm5=float(np.mean([v is not None for v in pmv])))
            md.append(f"| {lab} | {sp} | {r['n']} | {r['med']:.0f} | {r['w30']:.0%} | {r['w60']:.0%} | {r['w120']:.0%} | {r['never']:.0%} | {r['pm1']:.0%} | {r['pm5']:.0%} |")
            for k in ("med", "w30", "w60", "w120", "never", "pm1", "pm5"):
                rows.append(dict(section="gap_converge", venue="kalshi+pm", split=sp, family="describe", metric=k, value=r[k], n=r["n"], label=lab))
    md += ["", "Reading: if Kalshi usually reaches the Polymarket level within seconds, Kalshi is the slower venue and a resting Kalshi order "
           "on the Polymarket side gets filled by the move. If Polymarket reaches the Kalshi mid within a minute, Polymarket is the slower one. "
           "Which reading applies depends on the unresolved timestamp question (see the Polymarket timestamp audit).", ""]
    AppendTrials([dict(trial_id="gap-converge-seconds", created_at=Now(), family="gap_converge", tag="descriptive_registered_before_run",
                       params=dict(gap=0.03, tol=0.01), split_scored="tune,holdout,forward")])
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())