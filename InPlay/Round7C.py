r"""Round 7C (9:00 PM steer, priority 3). Kalshi moneyline vs Kalshi spread ladder (KXNBASPREAD), free 1-minute candles.

Data: forward-window games only (live API, 2026-09-13 to 2026-11-11), top-8 strikes by volume per game. No holdout:
those games sit behind Kalshi's historical endpoint and were not pulled in time. The forward games are therefore split
by date: games before the median date tune the trade rule ("f-early"), later games score it once ("f-late").

Per in-play minute (spread quotes as of the minute, age <= 60 s):
  * Ladder in home-margin terms: "HOME wins by over X" -> P(margin > X); "AWAY wins by over X" -> P(margin > -X) = 1 - mid.
  * Implied moneyline: fit probit(P(margin > k)) = (mu - k) / sigma across strikes with mids in [0.03, 0.97]
    (>= 2 strikes), evaluated at k = 0.5 (margins are integers, so P(home win) = P(margin > 0.5)).
  * Static arbitrage: home-by-X YES bid > moneyline YES ask (X > 0), or away-by-X YES bid + moneyline YES bid > 1.
    Counted gross and net of two Kalshi taker fees.
  * Lead-lag: corr of 1-minute changes, moneyline mid at i vs implied at i + L.
  * Trade rule (registered before scoring): when |ML mid - implied| >= g, buy the moneyline side toward the implied price
    at the next bar's ask + 0.5c, sell H minutes later at the bid - 0.5c (or settle). g in (0.03, 0.05, 0.08), H in (2, 5, 10).

    .venv\Scripts\python.exe -m InPlay.Round7C
"""

import glob
import json
import os
import sys
from collections import defaultdict
from statistics import NormalDist

import numpy as np
import pandas as pd

from InPlay.Align import AsOf
from InPlay.Execution import Fee
from InPlay.Grid import TrialId
from InPlay.Load import LoadGames
from InPlay.Research import AppendMd, AppendTrials, CsvRows, Now, SaveRes, _Short
from InPlay.Stats import RatioBootstrap

ND = NormalDist()
DROOT = os.path.join("data", "inplay", "kalshi", "derivs", "candles")


def Ladders():
    km = json.load(open(os.path.join("data", "inplay", "kalshi", "markets.json"), encoding="utf-8"))
    ev2home = {r["event_ticker"]: (r["ticker"], r["home"], r["away"]) for r in km if r["is_home"]}
    out = defaultdict(list)
    for p in glob.glob(os.path.join(DROOT, "*.json")):
        d = json.load(open(p, encoding="utf-8"))
        m = d["meta"]
        ev = m["event"].replace("KXNBASPREAD", "KXNBAGAME")
        if ev not in ev2home or m.get("strike") is None:
            continue
        kt, home, away = ev2home[ev]
        team = m["ticker"].rsplit("-", 1)[-1].rstrip("0123456789")
        if team not in (home, away):
            continue
        c = sorted(d["candles"], key=lambda x: x["t"])
        out[kt].append(dict(team_home=team == home, strike=float(m["strike"]), t=[x["t"] for x in c], c=c))
    return out


def Implied(points):
    pts = [(k, p) for k, p in points if 0.03 <= p <= 0.97]
    if len(pts) < 2:
        return None
    ks = np.array([k for k, _ in pts])
    zs = np.array([ND.inv_cdf(p) for _, p in pts])
    if np.ptp(ks) < 1:
        return None
    b, a = np.polyfit(ks, zs, 1)  # z = a + b k, with b = -1/sigma
    if b >= 0:
        return None
    return ND.cdf(a + b * 0.5)


def GameSeries(g, lad):
    imp, arb_g, arb_n = [], 0, 0
    for i in range(g.n):
        t = g.t[i]
        pts, quotes = [], []
        for L in lad:
            c = AsOf(L["t"], L["c"], t)
            if c is None or t - c["t"] > 60 or c["bid"] is None or c["ask"] is None or c["ask"] - c["bid"] > 0.10:
                continue
            mid = (c["bid"] + c["ask"]) / 2
            if L["team_home"]:
                pts.append((L["strike"], mid))
            else:
                pts.append((-L["strike"], 1 - mid))
            quotes.append((L, c))
        imp.append(Implied(pts))
        if g.bid[i] is None or g.ask[i] is None:
            continue
        for L, c in quotes:
            if L["team_home"] and L["strike"] > 0:
                edge = c["bid"] - g.ask[i]
                fees = Fee(c["bid"], "kalshi_taker") + Fee(g.ask[i], "kalshi_taker")
            elif not L["team_home"]:
                edge = c["bid"] + g.bid[i] - 1
                fees = Fee(c["bid"], "kalshi_taker") + Fee(g.bid[i], "kalshi_taker")
            else:
                continue
            if edge > 0:
                arb_g += 1
                if edge > fees:
                    arb_n += 1
    return imp, arb_g, arb_n


def Trades(g, imp, gap, H, slip=0.005):
    out, i = [], 1
    while i < g.n - 2:
        if g.el[i] < 0 or g.el[i] > 2820 or imp[i] is None or g.mid[i] is None:
            i += 1
            continue
        d = imp[i] - g.mid[i]
        if abs(d) < gap:
            i += 1
            continue
        side = 1 if d > 0 else -1
        e = i + 1
        if g.bid[e] is None or g.ask[e] is None:
            i += 1
            continue
        entry = (g.ask[e] if side == 1 else 1 - g.bid[e]) + slip
        x = e + H
        if x >= g.n:
            exitp, settled = float(g.won if side == 1 else 1 - g.won), True
        else:
            if g.bid[x] is None or g.ask[x] is None:
                i += 1
                continue
            exitp, settled = (g.bid[x] if side == 1 else 1 - g.ask[x]) - slip, False
        out.append(dict(day=g.day, gid=g.gid, entry=min(entry, 0.99), exit=exitp, settled=settled))
        i = x + 1
    return out


def Pnl(t, b):
    return t["exit"] - t["entry"] - Fee(t["entry"], b) - (0 if t["settled"] else Fee(t["exit"], b))


def Agg(trades, b):
    if not trades:
        return dict(n=0, mean=float("nan"), lo=float("nan"), hi=float("nan"))
    days = sorted({t["day"] for t in trades})
    ix = {d: k for k, d in enumerate(days)}
    num, den = np.zeros(len(days)), np.zeros(len(days))
    for t in trades:
        num[ix[t["day"]]] += Pnl(t, b)
        den[ix[t["day"]]] += 1
    m, lo, hi = RatioBootstrap(num, den, reps=1000)
    return dict(n=len(trades), mean=m, lo=lo, hi=hi)


def Main(argv=None):
    if os.name == "nt":
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
    G = LoadGames()
    meta = pd.read_pickle(os.path.join("data", "inplay", "games.pkl")).dropna(subset=["kalshi_ticker"])
    tick = dict(zip(meta.game_id.astype(str), meta.kalshi_ticker))
    lads = Ladders()
    games = [(g, lads[tick[str(g.gid)]]) for g in G["kalshi"] if str(g.gid) in tick and tick[str(g.gid)] in lads]
    days = sorted({g.day for g, _ in games})
    med = days[len(days) // 2]
    series, arb = [], [0, 0, 0]
    gap_abs, lagc = [], defaultdict(list)
    for g, lad in games:
        imp, ag, an = GameSeries(g, lad)
        series.append((g, imp))
        arb[0] += sum(1 for i in range(g.n) if g.el[i] >= 0 and imp[i] is not None)
        arb[1] += ag
        arb[2] += an
        gap_abs += [abs(imp[i] - g.mid[i]) for i in range(g.n) if imp[i] is not None and g.mid[i] is not None and 0 <= g.el[i] <= 2820]
        dm = [None if (g.mid[i] is None or g.mid[i - 1] is None) else g.mid[i] - g.mid[i - 1] for i in range(1, g.n)]
        di = [None if (imp[i] is None or imp[i - 1] is None) else imp[i] - imp[i - 1] for i in range(1, g.n)]
        for L in range(-2, 3):
            for i in range(len(dm)):
                j = i + L
                if 0 <= j < len(di) and dm[i] is not None and di[j] is not None:
                    lagc[L].append((dm[i], di[j]))
    lag_corr = {L: float(np.corrcoef(np.array(v).T)[0, 1]) if len(v) > 30 else float("nan") for L, v in lagc.items()}
    ga = np.array(gap_abs) if gap_abs else np.array([np.nan])
    trials, rows = [], []
    grid = [dict(gap=g_, hold=H) for g_ in (0.03, 0.05, 0.08) for H in (2, 5, 10)]
    tune = []
    for P in grid:
        tr = [t for g, imp in series if g.day < med for t in Trades(g, imp, P["gap"], P["hold"])]
        m = Agg(tr, "kalshi_taker")
        tune.append((P, m))
        trials.append(dict(trial_id=TrialId("ml_vs_spread", P, "kalshi", "taker"), created_at=Now(), family="ml_vs_spread", params=P, venue="kalshi",
                           exec="taker", tag="round7C_registered_before_scoring", split_scored="f-early", tune=m))
    ok = [x for x in tune if x[1]["n"] >= 20] or tune
    P, mt = max(ok, key=lambda x: x[1]["mean"] if x[1]["n"] else -9)
    late = [t for g, imp in series if g.day >= med for t in Trades(g, imp, P["gap"], P["hold"])]
    res = {b: Agg(late, b) for b in ("kalshi_taker", "tim_est_0.02", "none")}
    trials.append(dict(trial_id=TrialId("ml_vs_spread", P, "kalshi", "taker"), created_at=Now(), family="ml_vs_spread", params=P, venue="kalshi",
                       exec="taker", tag="round7C_selected_on_f-early", split_scored="f-late", res=res))
    for b, m in res.items():
        rows.append(dict(section="round7C_ml_vs_spread", venue="kalshi", split="forward_late", family="ml_vs_spread", trial_id=TrialId("ml_vs_spread", P, "kalshi", "taker"),
                         fee_basis=b, metric="pnl_per_trade", value=m["mean"], lo=m["lo"], hi=m["hi"], n=m["n"], label=_Short(P)))
    rows.append(dict(section="round7C_ml_vs_spread", venue="kalshi", split="forward", family="ml_vs_spread", metric="median_abs_gap", value=float(np.nanmedian(ga)), n=len(gap_abs)))
    md = ["## Round 7C: Kalshi moneyline vs Kalshi spread ladder (forward-window games only; registered before scoring)", "",
          f"Games with a usable spread ladder: {len(games)} (dates {days[0]} to {days[-1]}; f-early < {med} <= f-late). In-play minutes with an implied price: {arb[0]}.", "",
          f"* Gap |ML mid - spread-implied ML|: median {np.nanmedian(ga):.3f}, 90th pct {np.nanpercentile(ga, 90):.3f}, share >= 0.05: {np.nanmean(ga >= 0.05):.1%}.",
          f"* Static arbitrage minutes (strike-quote pairs): gross {arb[1]}, net of two taker fees {arb[2]}.",
          "* Lead-lag, corr(1-min ML change at i, implied change at i+L): " + ", ".join(f"L={L:+d}: {c:.3f}" for L, c in sorted(lag_corr.items())) + ".",
          "  (L > 0 larger than L < 0 would mean the spread ladder follows the moneyline.)", "",
          "| rule | split | trades | P/L per trade, Kalshi taker [95% day CI] | $0.02 fee | no fee |", "|---|---|---|---|---|---|"]
    for Q, m in tune:
        if m["n"]:
            md.append(f"| {_Short(Q)} | f-early (tune) | {m['n']} | {m['mean']:+.4f} [{m['lo']:+.4f}, {m['hi']:+.4f}] | | |")
    k, t2, n0 = res["kalshi_taker"], res["tim_est_0.02"], res["none"]
    if k["n"]:
        md.append(f"| SELECTED {_Short(P)} | f-late | {k['n']} | {k['mean']:+.4f} [{k['lo']:+.4f}, {k['hi']:+.4f}] | {t2['mean']:+.4f} | {n0['mean']:+.4f} [{n0['lo']:+.4f}, {n0['hi']:+.4f}] |")
    md.append("")
    AppendTrials(trials)
    CsvRows(rows)
    AppendMd("\n".join(md))
    SaveRes("round7C", dict(selected=P, tune=mt, res=res, lag_corr=lag_corr, arb=arb))
    print("\n".join(md))


if __name__ == "__main__":
    sys.exit(Main())
