r"""Round 7D (9:00 PM steer, priority 4; registered on tune before scoring). Stars who leave and do not come back.

ESPN play-by-play has no injury flag and no team-confirmation time, so this tests the observable part: after a star's
FINAL exit (a substitution out with no later participation), does the market keep drifting against his team (it was
slow to price the absence) compared with his ordinary rest exits (sub out, later returns)?
  * Star = team's top-2 players by mean points in that team's EARLIER pulled games of the same split (>= 3 prior games;
    leak-free). Kalshi trade-tape games only (tune = 2025 playoffs, holdout sample, forward).
  * Event window: game clock elapsed 0-2400 s, team price (Kalshi mid) 0.15-0.85, player did not foul out (PF < 6).
  * Drift = team-side Kalshi mid change from the first bar after the exit to +1/+5/+15 bars. Game-clustered 95% CI.
  * Trade (registered): buy the OTHER team at the next bar's ask + 0.5c, sell H minutes later at the bid - 0.5c
    (H = 5, 10, 20), both fee bases; the same trade on rest exits is the control.
Final exits include injuries, blowout rest, foul trouble and coach's decisions; a positive result would still need the
injury subset confirmed from news.

    .venv\Scripts\python.exe -m InPlay.Round7D
"""

import glob
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from InPlay.Execution import Fee
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames, SplitOf
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Stats import RatioBootstrap

SROOT = os.path.join("data", "inplay", "espn", "subs")


def Ts(s):
    return datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() if s else None


def Num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def Events(g, sub, home_id, stars):
    """Return list of (kind, team_is_home, bar index) for star exits."""
    plays = sub["plays"]
    pf = {b["id"]: Num(b.get("pf")) for b in sub["box"]}
    out = []
    for team, ids in stars.items():
        for pid in ids:
            if pf.get(pid, 0) >= 6:
                continue
            outs = [p for p in plays if "ubstitut" in (p["type"] or "") and len(p["ids"]) >= 2 and p["ids"][1] == pid]
            if not outs:
                continue
            last_part = max((p["seq"] for p in plays if pid in p["ids"]), default=-1)
            for p in outs:
                later = any(q["seq"] > p["seq"] and pid in q["ids"] for q in plays)
                kind = "final" if not later else "rest"
                w = Ts(p["wall"])
                if w is None:
                    continue
                e = next((k for k in range(g.n) if g.t[k] >= w + 60), None)
                if e is None or e + 2 >= g.n or g.el[e] < 0 or g.el[e] > 2400 or g.mid[e] is None:
                    continue
                th = team == home_id
                tp = g.mid[e] if th else 1 - g.mid[e]
                if not 0.15 <= tp <= 0.85:
                    continue
                out.append((kind, th, e))
    return out


def Drift(g, th, e, h):
    j = min(e + h, g.n - 1)
    a, b = g.mid[e], next((g.mid[k] for k in range(j, e, -1) if g.mid[k] is not None), None)
    if a is None or b is None:
        return None
    d = b - a
    return d if th else -d


def Trade(g, th, e, H, slip=0.005):
    side = -1 if th else 1  # buy the other team: home NO if the star is home
    x = e + 1
    if g.bid[x] is None or g.ask[x] is None:
        return None
    entry = (g.ask[x] if side == 1 else 1 - g.bid[x]) + slip
    y = x + H
    if y >= g.n:
        return dict(day=g.day, gid=g.gid, entry=entry, exit=float(g.won if side == 1 else 1 - g.won), settled=True)
    if g.bid[y] is None or g.ask[y] is None:
        return None
    return dict(day=g.day, gid=g.gid, entry=entry, exit=(g.bid[y] if side == 1 else 1 - g.ask[y]) - slip, settled=False)


def Pnl(t, b):
    return t["exit"] - t["entry"] - Fee(t["entry"], b) - (0 if t["settled"] else Fee(t["exit"], b))


def Clus(vals, key):
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


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    G = LoadGames()
    gk = {str(g.gid): g for g in G["kalshi"]}
    homes = {}
    for p in glob.glob(os.path.join("data", "inplay", "espn", "games", "202[5-7]", "*.json")):
        gid = os.path.basename(p)[:-5]
        if gid in gk:
            d = json.load(open(p, encoding="utf-8"))
            homes[gid] = (str(d.get("home_id")), str(d.get("away_id")))
    subs = {}
    for p in glob.glob(os.path.join(SROOT, "*.json")):
        gid = os.path.basename(p)[:-5]
        if gid in gk and gid in homes:
            subs[gid] = json.load(open(p, encoding="utf-8"))
    order = sorted(subs, key=lambda x: (gk[x].day, x))
    hist = defaultdict(lambda: defaultdict(list))  # (split, team) -> player -> points list
    ev = defaultdict(list)
    for gid in order:
        g, s = gk[gid], subs[gid]
        sp = SplitOf(g.season)
        stars = {}
        for team in homes[gid]:
            h = hist[(sp, team)]
            ng = max((len(v) for v in h.values()), default=0)
            if ng >= 3:
                avg = sorted(((np.mean(v), pid) for pid, v in h.items() if len(v) >= 2), reverse=True)[:2]
                stars[team] = [pid for _, pid in avg]
        for kind, th, e in Events(g, s, homes[gid][0], stars):
            ev[(sp, kind)].append((g, th, e))
        for b in s["box"]:
            if not b.get("dnp"):
                hist[(sp, b["team"])][b["id"]].append(Num(b.get("pts")))
    md = ["## Round 7D: star exits that do not return vs ordinary rest exits (Kalshi tape games; registered before scoring)", "",
          "Team-side Kalshi mid drift after the exit (negative = market keeps moving against the team). Game-clustered 95% CI.", "",
          "| split | exit kind | events | drift +1 min | drift +5 min | drift +15 min |", "|---|---|---|---|---|---|"]
    rows, trials, res = [], [], {}
    for sp in ("tune", "holdout", "forward"):
        for kind in ("final", "rest"):
            E = ev[(sp, kind)]
            cells = []
            for h in (1, 5, 15):
                c = Clus([(g.gid, d) for g, th, e in E for d in [Drift(g, th, e, h)] if d is not None], "gid")
                res[f"drift|{sp}|{kind}|{h}"] = c
                cells.append(f"{c['mean']:+.4f} [{c['lo']:+.4f}, {c['hi']:+.4f}]" if c["n"] else "n/a")
                rows.append(dict(section="round7D_star_exit_drift", venue="kalshi", split=sp, family="star_exit_" + kind, fee_basis="none",
                                 metric=f"drift_{h}m", value=c["mean"], lo=c["lo"], hi=c["hi"], n=c["n"]))
            md.append(f"| {sp} | {kind} | {len(E)} | " + " | ".join(cells) + " |")
    md += ["", "| rule | split | exit kind | trades | P/L per trade, Kalshi taker [95% game CI] | $0.02 fee |", "|---|---|---|---|---|---|"]
    tune = []
    for H in (5, 10, 20):
        P = dict(hold=H)
        tr = [t for g, th, e in ev[("tune", "final")] for t in [Trade(g, th, e, H)] if t]
        m = Clus([(t["gid"], Pnl(t, "kalshi_taker")) for t in tr], "gid")
        tune.append((P, m))
        trials.append(dict(trial_id=TrialId("star_exit_fade", P, "kalshi", "taker"), created_at=Now(), family="star_exit_fade", params=P, venue="kalshi",
                           exec="taker", tag="round7D_registered_before_scoring", split_scored="tune", tune=m))
        md.append(f"| hold={H} | tune | final | {m['n']} | {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | |" if m["n"] else f"| hold={H} | tune | final | 0 | n/a | |")
    P, _ = max(tune, key=lambda x: x[1]["mean"] if x[1]["n"] else -9)
    for sp in ("holdout", "forward"):
        for kind in ("final", "rest"):
            tr = [t for g, th, e in ev[(sp, kind)] for t in [Trade(g, th, e, P["hold"])] if t]
            mk = Clus([(t["gid"], Pnl(t, "kalshi_taker")) for t in tr], "gid")
            m2 = Clus([(t["gid"], Pnl(t, "tim_est_0.02")) for t in tr], "gid")
            res[f"trade|{sp}|{kind}"] = dict(kalshi_taker=mk, tim=m2)
            for b, m in (("kalshi_taker", mk), ("tim_est_0.02", m2)):
                rows.append(dict(section="round7D_star_exit_fade", venue="kalshi", split=sp, family="star_exit_fade" + ("" if kind == "final" else "_ctrl"),
                                 trial_id=TrialId("star_exit_fade", P, "kalshi", "taker"), fee_basis=b, metric="pnl_per_trade", value=m["mean"], lo=m["lo"], hi=m["hi"], n=m["n"], label=_Short(P)))
            md.append(f"| SELECTED hold={P['hold']} | {sp} | {kind} | {mk['n']} | " + (f"{mk['mean']:+.4f} [{mk['lo']:+.4f}, {mk['hi']:+.4f}] | {m2['mean']:+.4f} |" if mk["n"] else "n/a | |"))
    trials.append(dict(trial_id=TrialId("star_exit_fade", P, "kalshi", "taker"), created_at=Now(), family="star_exit_fade", params=P, venue="kalshi", exec="taker",
                       tag="round7D_selected_on_tune", split_scored="holdout+forward", res={k: v for k, v in res.items() if k.startswith("trade")}))
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round7D", res)
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())
