r"""ESPN model win probability (2017-18 onward) as a long, frictionless proxy check.

ESPN WP is a model, not a price: nothing here is fillable. Uses per-play rows; latency = one play.

    .venv\Scripts\python.exe -m InPlay.EspnLong --workers 8
"""

import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from InPlay.Align import EspnStates, IsoTs, SeasonOf
from InPlay.Engine import Game, RunGame, TradePnl
from InPlay.Research import AppendMd, CsvRows, SaveRes
from InPlay.Stats import CellBootstrap, RatioBootstrap

RULES = [
    ("dip_fav", dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440, tgt=0.60, stop=0.30)),
    ("dip_fav", dict(pre_lo=0.55, pre_hi=0.65, entry=0.40, tmax=1440)),
    ("comeback", dict(close=0.05, tmax=2520, mode="price", lead_lo=0.75, lead_hi=0.90)),
    ("comeback", dict(close=0.05, tmax=2520, mode="pts", pts_lo=10, pts_hi=18)),
    ("comeback_ctrl", dict(mis_lo=0.10, mis_hi=0.30, leader_is="fav", tmax=2520, mode="price", lead_lo=0.75, lead_hi=0.90)),
    ("late_fav", dict(mins=6, lo=0.90, hi=0.95)),
    ("fade_run", dict(look=15, drop=0.15, tmax=2520, hold=30, max_tr=3)),
]


def ToGame(path):
    game = json.load(open(path, encoding="utf-8"))
    st = [s for s in EspnStates(game) if s["wp"] is not None]
    if len(st) < 100 or game.get("home_won") is None:
        return None
    sched = IsoTs(game.get("date"))
    wp = [min(0.999, max(0.001, s["wp"])) for s in st]
    return Game(gid=game["id"], day=path, season=SeasonOf(sched), stype=game.get("season_type"), venue="espn",
                pre=wp[0], won=int(bool(game["home_won"])), t=[s["t"] for s in st], el=[s["el"] for s in st],
                per=[s["per"] for s in st], sd=[s["hs"] - s["as"] for s in st], wp=wp, bid=wp, ask=wp, mid=wp,
                trlo=[None] * len(st), trhi=[None] * len(st), pm=[None] * len(st), pmage=[None] * len(st),
                total=None, spread=None)


def _Work(path):
    try:
        g = ToGame(path)
    except Exception:
        return None
    if g is None:
        return None
    first = {}
    fav = 1 if g.pre >= 0.5 else -1
    fp = g.pre if fav == 1 else 1 - g.pre
    for i in range(g.n):
        m = g.mid[i]
        if "head" not in first and 0.55 <= fp <= 0.65 and g.el[i] <= 1440 and (m if fav == 1 else 1 - m) <= 0.40:
            first["head"] = ((m if fav == 1 else 1 - m), g.won if fav == 1 else 1 - g.won)
        if g.el[i] <= 2520:
            leader = 1 if m >= 0.5 else -1
            lp = m if leader == 1 else 1 - m
            if 0.75 <= lp <= 0.90:
                close = abs(g.pre - 0.5)
                grp = "even" if close <= 0.05 else ("mis_leader_fav" if leader == fav else "mis_leader_dog") if 0.10 <= close <= 0.30 else None
                if grp and grp not in first:
                    first[grp] = (1 - lp, g.won if leader == -1 else 1 - g.won)
    trades = {}
    for k, (fam, P) in enumerate(RULES):
        trades[k] = [TradePnl(t, "none") for t in RunGame(fam, P, g, lat=1, slip=0.0)]
    return g.season, g.gid, first, trades


def Main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)
    paths = sorted(glob.glob(os.path.join("data", "inplay", "espn", "games", "*", "*.json")))
    by = defaultdict(lambda: defaultdict(list))
    pnl = defaultdict(lambda: defaultdict(list))
    seasons = defaultdict(int)
    with ProcessPoolExecutor(max_workers=min(args.workers, 10)) as ex:
        for r in ex.map(_Work, paths, chunksize=25):
            if r is None:
                continue
            season, gid, first, trades = r
            seasons[season] += 1
            for k, v in first.items():
                by[k][season].append(v)
            for k, v in trades.items():
                pnl[k][season].append(sum(v) if v else None)
    md = ["## ESPN model WP, long history (frictionless proxy; ESPN WP is a model, not a price)", "",
          f"Games with WP by season: {dict(sorted(seasons.items()))}", "",
          "First passage per game; win - WP with 95% CI resampling games.", "",
          "| context | seasons | games | mean WP | win rate | win - WP | 95% CI |", "|---|---|---|---|---|---|---|"]
    rows, res = [], {}
    for key, label in (("head", "fav 55-65 pregame dips <=0.40 in H1"), ("even", "trailer, even game (45-55), leader 75-90 before Q4 6:00"),
                       ("mis_leader_fav", "trailer, mismatch, favorite leads 75-90"), ("mis_leader_dog", "trailer, mismatch, underdog leads 75-90")):
        allv = [x for s in by[key].values() for x in s]
        if not allv:
            continue
        p = np.array([a for a, _ in allv])
        y = np.array([b for _, b in allv])
        pt, lo, hi = CellBootstrap((y - p)[:, None], np.ones((len(p), 1)), reps=1000)
        md.append(f"| {label} | {min(by[key])}-{max(by[key])} | {len(p)} | {p.mean():.3f} | {y.mean():.3f} | {pt[0]:+.3f} | [{lo[0]:+.3f}, {hi[0]:+.3f}] |")
        rows.append(dict(section="espn_long_calib", venue="espn", split="2018-2027", metric="win_minus_wp", value=float(pt[0]),
                         lo=float(lo[0]), hi=float(hi[0]), n=len(p), label=label))
        res[key] = dict(n=len(p), wp=float(p.mean()), win=float(y.mean()), diff=float(pt[0]), lo=float(lo[0]), hi=float(hi[0]))
        for s in sorted(by[key]):
            ps = np.array([a for a, _ in by[key][s]])
            ys = np.array([b for _, b in by[key][s]])
            rows.append(dict(section="espn_long_calib_season", venue="espn", split=str(s), metric="win_minus_wp",
                             value=float((ys - ps).mean()), n=len(ps), label=label))
    md += ["", "Frictionless rule P&L on ESPN WP (no fees, no spread, 1-play latency). Upper bound, not a tradable result.", "",
           "| rule | games traded | P&L per traded game | 95% CI (games) |", "|---|---|---|---|"]
    for k, (fam, P) in enumerate(RULES):
        vals = [v for s in pnl[k].values() for v in s if v is not None]
        if not vals:
            continue
        pt, lo, hi = RatioBootstrap(np.array(vals, dtype=float), np.ones(len(vals)))
        md.append(f"| {fam} {json.dumps(P)} | {len(vals)} | {pt:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
        rows.append(dict(section="espn_long_frictionless", venue="espn", split="2018-2027", family=fam, fee_basis="none",
                         metric="pnl_per_traded_game", value=pt, lo=lo, hi=hi, n=len(vals), label=json.dumps(P)))
        res[f"rule{k}"] = dict(fam=fam, P=P, n=len(vals), pnl=pt, lo=lo, hi=hi)
    SaveRes("espn_long", res)
    CsvRows(rows)
    AppendMd("\n".join(md))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())