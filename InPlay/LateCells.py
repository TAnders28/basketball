r"""Steer 8:11 #4: last-2:00 leader cells split by possession and the trailer's timeouts left (registered before scoring).

Context: Q4, last 2:00 of regulation, leader up 3-8 points, leader priced 0.80-0.97. Each bar gets, from ESPN plays with
wallclock <= the bar time (no look-ahead): who has the ball (heuristic: made FG / last made FT -> other team; rebound -> that
team; turnover -> other team; else carry forward; unknown at period start) and the trailer's timeouts left (7 - used,
non-official timeouts). Cells = leader has ball (yes / no / unknown) x trailer timeouts (0-1 / 2+). First passage per game per cell.
Selection on Polymarket tune only (the steer's instruction): the cell with the largest tune (win - price - cost) where cost =
0.5c slippage + Kalshi taker fee, n >= 40 games. It is traded (buy the leader at the next minute's ask, hold to settlement)
and scored once on holdout and forward. If it qualifies (tune lower CI bound of win-price-cost above 0), it becomes the single
new dated paper-log entry (2026-11-12_late_cell); the frozen 17-rule registry and the 2-rule spec are not touched.

    .venv\Scripts\python.exe -m InPlay.LateCells
"""

import bisect
import json
import os
import sys

import numpy as np

from InPlay.Align import MonotoneWall
from InPlay.CiAudit import Audit
from InPlay.Execution import Fee, TakerBuy
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now
from InPlay.Stats import CellBootstrap

OUT = r"E:\AgentDocs\Basketbot\2026-11-12_inplay-swings"
CELLS = [(b, t) for b in ("ball", "no_ball", "unknown") for t in ("TO0-1", "TO2+")]


def States(game):
    hid, aid = str(game.get("home_id")), str(game.get("away_id"))
    other = {hid: aid, aid: hid}
    ts, st = [], []
    poss, used = None, {hid: 0, aid: 0}
    last_per = None
    for t, p in MonotoneWall(game.get("plays", [])):
        typ = (p.get("type") or "").lower()
        team = str(p.get("team")) if p.get("team") is not None else None
        if p.get("period") != last_per:
            poss, last_per = None, p.get("period")
        if "timeout" in typ and "official" not in typ and team in used:
            used[team] += 1
        if team in other:
            if "rebound" in typ:
                poss = team
            elif "turnover" in typ:
                poss = other[team]
            elif p.get("scoring") and p.get("shoot"):
                if "free throw" in typ:
                    parts = typ.split(" of ")
                    if len(parts) == 2 and parts[0].strip().endswith(parts[1].strip()[:1]):
                        poss = other[team]
                else:
                    poss = other[team]
        ts.append(t)
        st.append((poss, dict(used)))
    return ts, st, hid, aid


def Observe(g, game, mode="first"):
    ts, st, hid, aid = States(game)
    seen, out = set(), []
    for i in range(g.n - 1):
        if g.per[i] != 4 or not (2760 <= g.el[i] <= 2880) or g.sd[i] is None or g.mid[i] is None:
            continue
        a = abs(g.sd[i])
        if not (3 <= a <= 8):
            continue
        side = 1 if g.sd[i] > 0 else -1
        p = g.mid[i] if side == 1 else 1 - g.mid[i]
        if not (0.80 <= p <= 0.97):
            continue
        j = bisect.bisect_right(ts, g.t[i]) - 1
        if j < 0:
            continue
        poss, used = st[j]
        lead_id, trail_id = (hid, aid) if side == 1 else (aid, hid)
        ball = "unknown" if poss is None else ("ball" if poss == lead_id else "no_ball")
        to = "TO0-1" if 7 - used.get(trail_id, 0) <= 1 else "TO2+"
        c = (ball, to)
        if c in seen:
            continue
        seen.add(c)
        y = g.won if side == 1 else 1 - g.won
        out.append((c, i, side, p, y))
    return out


def Load(G):
    obs = {}
    for venue in ("pm", "kalshi"):
        obs[venue] = []
        for g in G[venue]:
            path = os.path.join("data", "inplay", "espn", "games", str(g.season), f"{g.gid}.json")
            if not os.path.exists(path):
                continue
            obs[venue].append((g, Observe(g, json.load(open(path, encoding="utf-8")))))
    return obs


def Calib(rows):
    S = np.zeros((len(rows), len(CELLS)))
    C = np.zeros_like(S)
    P = np.zeros_like(S)
    for gi, (g, ob) in enumerate(rows):
        for c, i, side, p, y in ob:
            j = CELLS.index(c)
            S[gi, j] += y - p - (0.005 + Fee(p, "kalshi_taker"))
            C[gi, j] += 1
            P[gi, j] += p
    pt, lo, hi = CellBootstrap(S, C, reps=1000)
    return {c: dict(n=int(C[:, j].sum()), price=float(P[:, j].sum() / max(C[:, j].sum(), 1)), net=float(pt[j]), lo=float(lo[j]), hi=float(hi[j]))
            for j, c in enumerate(CELLS)}


def Main():
    G = LoadGames()
    obs = Load(G)
    trials = [dict(trial_id=f"late-cell-{c[0]}-{c[1]}", created_at=Now(), family="late_cell_possession", params=dict(cell=c),
                   venue="pm", exec="taker_hold", tag="steer811_registered_before_scoring", split_scored="tune") for c in CELLS]
    md = ["## Last 2:00 cells by possession and trailer timeouts (leader up 3-8, priced 0.80-0.97); win - price - cost, game CIs", "",
          "Cost = 0.5c slippage + Kalshi taker fee (hold to settlement). Possession is a play-by-play heuristic (made FG/FT, rebound, turnover).", "",
          "| venue | split | cell | games | leader price | win - price - cost [95% CI] |", "|---|---|---|---|---|---|"]
    rows = []
    tune = Calib([x for x in obs["pm"] if SplitOf(x[0].season) == "tune"])
    for c, v in tune.items():
        md.append(f"| pm | tune | {c[0]}, trailer {c[1]} | {v['n']} | {v['price']:.3f} | {v['net']:+.3f} [{v['lo']:+.3f}, {v['hi']:+.3f}] |")
        rows.append(dict(section="late_cell", venue="pm", split="tune", family="late_cell_possession", metric="win_minus_price_minus_cost",
                         value=v["net"], lo=v["lo"], hi=v["hi"], n=v["n"], label=f"{c[0]}|{c[1]}"))
    best = max([c for c in CELLS if tune[c]["n"] >= 40], key=lambda c: tune[c]["net"])
    qualifies = tune[best]["lo"] > 0
    md += ["", f"Selected on tune: {best} (tune net {tune[best]['net']:+.3f} [{tune[best]['lo']:+.3f}, {tune[best]['hi']:+.3f}]); "
           f"qualifies for the paper log: {qualifies}.", "",
           "| venue | split | fee | trades | P/L per trade [conservative 95% CI] |", "|---|---|---|---|---|"]
    res = {}
    for venue in ("pm", "kalshi"):
        for sp in ("holdout", "forward"):
            tr = []
            for g, ob in obs[venue]:
                if SplitOf(g.season) != sp:
                    continue
                for c, i, side, p, y in ob:
                    if c != best:
                        continue
                    px = TakerBuy(g.bid[i + 1], g.ask[i + 1], side, 0.005)
                    if px is not None:
                        tr.append(dict(gid=g.gid, day=g.day, season=g.season, side=side, entry=px, exit=float(y), settled=True, why="settle", el=g.el[i], maker=False))
            for b in ("tim_est_0.02", "kalshi_taker"):
                m = Audit(tr, b)
                res[f"{venue}|{sp}|{b}"] = m
                if m:
                    md.append(f"| {venue} | {sp} | {b} | {m['n']} | {m['mean']:+.4f} [{m['cons_lo']:+.4f}, {m['cons_hi']:+.4f}] |")
                    rows.append(dict(section="late_cell_trade", venue=venue, split=sp, family="late_cell_possession", fee_basis=b, metric="pnl_per_trade",
                                     value=m["mean"], lo=m["cons_lo"], hi=m["cons_hi"], n=m["n"], label=f"{best[0]}|{best[1]}"))
    trials.append(dict(trial_id=f"late-cell-{best[0]}-{best[1]}-selected", created_at=Now(), family="late_cell_possession", params=dict(cell=best),
                       venue="pm+kalshi", exec="taker_hold", tag="steer811_selected_on_tune", split_scored="holdout+forward", res=res))
    if qualifies:
        entry = dict(entry="2026-11-12_late_cell", created_at=Now(), score_games_on_or_after_et="2026-11-13", family="late_cell_possession",
                     rule=f"Last 2:00 of Q4, leader up 3-8 priced 0.80-0.97, cell {best}: buy the leader at the next minute's ask, hold to settlement; first passage per game.",
                     fees=["tim_est_0.02", "kalshi_taker"], note="Paper only. Single dated addition; the frozen 17-rule registry and the 2-rule spec are unchanged.")
        with open(os.path.join(OUT, "2026-11-12_inplay-swings_paper-log-addition.json"), "w") as fh:
            json.dump(entry, fh, indent=1)
        md.append("\nAdded one dated paper-log entry: `2026-11-12_inplay-swings_paper-log-addition.json`.")
    else:
        md.append("\nNo paper-log entry added (the tune lower bound is not above zero).")
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())